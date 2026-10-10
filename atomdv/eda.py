"""EDA checks as plain Python (no LLM): iverilog compile, pyslang lint, testbench structure and DUT-instance
checks, and simulation of the testbench against a DUT (constant-output stubs in the loop, the reference
design when scoring).

Each check returns structured Diagnostics. The full raw tool output is kept on the CmdResult /
LintResult / SimResult for the transcript; the LLM only ever sees format_diagnostics()'s capped summary.
"""

import hashlib
import re
import subprocess
import time
from dataclasses import dataclass, field
from collections.abc import Iterable
from pathlib import Path

import pyslang
from pyslang import analysis, ast, syntax

from atomdv.spec import Port, render_header


@dataclass(frozen=True)
class Diagnostic:
    tool: str  # iverilog | slang | structure | instance | sim | agent
    category: str  # syntax | elab | lint | structure | instance | sim | no_code
    severity: str  # error | warning
    message: str
    line: int | None = None
    code: str = ""  # tool-specific id, e.g. slang's "PortWidthExpand"


@dataclass
class CmdResult:
    cmd: list[str]
    returncode: int | None
    output: str  # stdout + stderr, untruncated
    duration_s: float
    timed_out: bool = False

    @property
    def cmdline(self) -> str:
        return " ".join(self.cmd)


def run_cmd(cmd: list[str], cwd: Path, timeout: float) -> CmdResult:
    start = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return CmdResult(cmd, p.returncode, p.stdout + p.stderr, time.monotonic() - start)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return CmdResult(cmd, None, out, time.monotonic() - start, timed_out=True)


# --- iverilog -----------------------------------------------------------------------------------

_IVERILOG_CONT_RE = re.compile(r"^[^:\s][^:]*:\d+:\s+:")
_IVERILOG_LINE_RE = re.compile(r"^(?P<file>[^:\s][^:]*):(?P<line>\d+):\s*(?P<kind>error|warning|sorry)?:?\s*(?P<msg>.*)$")


def run_iverilog(files: list[Path], cwd: Path, timeout: float = 30, top: str | None = None,
                 only: str | None = None) -> tuple[CmdResult, list[Diagnostic]]:
    """Compile and elaborate with Icarus (the same compiler used for scoring), no simulation output.

    `top`: the root module (-s). `only`: keep just the diagnostics for this file name (e.g. the
    testbench, so a problem reported inside the DUT stub isn't blamed on a tb.sv line).
    """
    cmd = ["iverilog", "-g2012", *(["-s", top] if top else []), "-t", "null", *map(str, files)]
    res = run_cmd(cmd, cwd, timeout)
    if res.timed_out:
        return res, [Diagnostic("iverilog", "elab", "error", f"iverilog timed out after {timeout}s")]

    diags = []
    for raw in res.output.splitlines():
        m = _IVERILOG_LINE_RE.match(raw.strip())
        if not m or (only and Path(m["file"]).name != only):
            continue
        kind, msg = m["kind"], m["msg"].strip()
        if _IVERILOG_CONT_RE.match(raw) and diags and diags[-1].line == int(m["line"]):
            # Continuation line ("file:5:      : Expression is: ...") of the previous message.
            prev = diags.pop()
            diags.append(Diagnostic(prev.tool, prev.category, prev.severity, f"{prev.message} {msg}", prev.line))
            continue
        # The exit code decides: with exit 0 everything printed is advisory, including "sorry:"
        # limitation notices such as "constant selects in always_* processes are not currently supported".
        if res.returncode == 0 or kind == "warning":
            severity = "warning"
        else:
            severity = "error"
        category = "syntax" if "syntax error" in msg.lower() or "syntax error" in raw.lower() else "elab"
        diags.append(Diagnostic("iverilog", category, severity, msg or "syntax error", int(m["line"])))

    # iverilog prints a bare "syntax error" plus a specific message for the same line; keep the specific one.
    specific = {d.line for d in diags if d.message != "syntax error"}
    diags = [d for d in diags if not (d.message == "syntax error" and d.line in specific)]

    if res.returncode and not any(d.severity == "error" for d in diags):
        # Failed without a file:line message (e.g. "Unknown module type" summaries, missing top).
        diags.append(Diagnostic("iverilog", "elab", "error", res.output.strip() or f"exit code {res.returncode}"))
    return res, diags


# --- pyslang ------------------------------------------------------------------------------------

# Extra guidance appended to warnings promoted to errors.
_PROMOTED_HINTS = {
    "PortWidthExpand": " (treated as an error: declare the testbench signal with the port's exact width)",
    "PortWidthTruncate": " (treated as an error: declare the testbench signal with the port's exact width)",
}

_TRUNC_FROM_RE = re.compile(r"truncates from (\d+) to")


def _is_int_truncation(code: str, message: str) -> bool:
    """A truncation from 32 bits almost always comes from an unsized literal or int (`.a(0)`, `x = i`)
    and is intended, so it stays a warning even when promoted."""
    m = _TRUNC_FROM_RE.search(message)
    return code in ("WidthTruncate", "PortWidthTruncate") and m is not None and m.group(1) == "32"


_DIRECTIONS = {"In": "input", "Out": "output", "InOut": "inout", "Ref": "ref"}


@dataclass(frozen=True)
class Instance:
    """One module instance in the elaborated design and which of its ports are connected."""

    name: str  # instance name, e.g. "dut"
    module: str  # definition name, e.g. "TopModule"
    connected: tuple[str, ...]
    unconnected: tuple[str, ...]


@dataclass
class LintResult:
    report: str  # slang's own formatted output (with source carets), untruncated
    diagnostics: list[Diagnostic]
    modules: dict[str, tuple[Port, ...]] = field(default_factory=dict)  # top-level modules → ports
    instances: list[Instance] = field(default_factory=list)
    duration_s: float = 0.0


def _collect_instances(comp: ast.Compilation) -> list[Instance]:
    found = []

    def visit(sym):
        if sym.kind == ast.SymbolKind.Instance:
            conns = [(pc.port.name, pc.expression is not None) for pc in sym.portConnections]
            found.append(Instance(sym.name, sym.definition.name,
                                  tuple(n for n, ok in conns if ok), tuple(n for n, ok in conns if not ok)))
        return True

    for top in comp.getRoot().topInstances:
        top.body.visit(visit)
    return found


def lint_slang(files: list[Path], as_errors: Iterable[str] = (), only: str | None = None) -> LintResult:
    """Parse + elaborate with slang (stricter than Icarus), extract the top-level module ports and the
    module instances.

    `as_errors`: slang warning codes (e.g. "PortWidthExpand") to report as errors, so they must be fixed.
    `only`: keep just the diagnostics located in this file name (the report still has everything).
    """
    as_errors = set(as_errors)
    start = time.monotonic()
    sm = pyslang.SourceManager()
    comp = ast.Compilation()
    for f in files:
        comp.addSyntaxTree(syntax.SyntaxTree.fromFile(str(f), sm))

    engine = pyslang.DiagnosticEngine(sm)
    engine.setWarningOptions(["everything", "no-newline-eof"])
    client = pyslang.TextDiagnosticClient()
    engine.addClient(client)

    raw = list(comp.getAllDiagnostics())
    # slang 10+ runs its dataflow checks (multiple drivers, inferred latches) in a separate analysis
    # pass. Only run it on code that compiles; on broken code it would just add follow-on noise.
    if not any(d.isError() for d in raw):
        comp.freeze()
        manager = analysis.AnalysisManager()
        manager.analyze(comp)
        raw += list(manager.getDiagnostics())

    diags = []
    for d in raw:
        engine.issue(d)
        if only and Path(sm.getFileName(d.location)).name != only:
            continue
        sev = engine.getSeverity(d.code, d.location)
        code = str(d.code).removeprefix("DiagCode(").removesuffix(")")
        message = engine.formatMessage(d)
        if sev in (pyslang.DiagnosticSeverity.Error, pyslang.DiagnosticSeverity.Fatal):
            severity = "error"
        elif sev == pyslang.DiagnosticSeverity.Warning:
            severity = "warning"
            if code in as_errors and not _is_int_truncation(code, message):
                severity = "error"
                message += _PROMOTED_HINTS.get(code, " (treated as an error)")
        else:
            continue  # notes / ignored
        diags.append(Diagnostic("slang", "lint", severity, message, sm.getLineNumber(d.location), code))

    modules = {}
    for inst in comp.getRoot().topInstances:
        ports = []
        for p in inst.body.portList:
            direction = _DIRECTIONS.get(str(p.direction).split(".")[-1], str(p.direction))
            width = getattr(getattr(p, "type", None), "bitWidth", 0) or 0
            ports.append(Port(p.name, direction, width))
        modules[inst.name] = tuple(ports)

    return LintResult(client.getString(), diags, modules, _collect_instances(comp), time.monotonic() - start)


# --- testbench structure and DUT instance -------------------------------------------------------

TB_TOP = "tb"
TB_VERDICT_RE = re.compile(r"TB_RESULT:\s*(PASS|FAIL)\b")
# The watchdog's verdict: the testbench hung, which says nothing about the DUT.
TB_WATCHDOG_RE = re.compile(r"TB_RESULT:\s*FAIL\s+timeout\b")

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
# Constructs Icarus 12 rejects or only partly supports, with what to use instead.
_UNSUPPORTED = [
    (re.compile(r"\bcovergroup\b"), "covergroup", "Icarus does not support covergroups; remove them."),
    (re.compile(r"\bproperty\b|\bassert\s+property\b|\|->|\|=>"), "concurrent assertions",
     "Icarus does not support concurrent assertions (property, |->); check with procedural if statements."),
    (re.compile(r"\.\s*randomize\s*\(|\brand\s+(?:bit|logic|int)\b"), "randomize",
     "Icarus does not support randomize() / rand variables; use $urandom (or $urandom_range)."),
]


def check_tb_structure(code: str, dut_name: str | None, outputs: Iterable[str] = ()) -> list[Diagnostic]:
    """Static checks that the testbench has the parts every run relies on (no tools needed).

    `outputs`: the DUT's output port names; waiting on one of them can hang the testbench.
    """
    src = _COMMENT_RE.sub("", code)

    def err(msg: str, code: str) -> Diagnostic:
        return Diagnostic("structure", "structure", "error", msg, code=code)

    diags = []
    if not re.search(rf"^\s*module\s+{TB_TOP}\b", src, re.MULTILINE):
        diags.append(err(f"The testbench top module must be named '{TB_TOP}' (module {TB_TOP};).", "NoTbModule"))
    if dut_name:
        if re.search(rf"^\s*module\s+{dut_name}\b", src, re.MULTILINE):
            diags.append(err(f"Do not write module {dut_name}: it is the design under test and is provided "
                             "separately. Only instantiate it.", "DutDefined"))
        if not re.search(rf"\b{dut_name}\s+(?:#\s*\(.*?\)\s*)?\w+\s*\(", src, re.DOTALL):
            diags.append(err(f"The testbench never instantiates {dut_name} (e.g. {dut_name} dut (.clk(clk), ...);).",
                             "NoDutInstance"))
    if "$finish" not in src:
        diags.append(err("The testbench never calls $finish, so the simulation may never end.", "NoFinish"))
    if not ("TB_RESULT: PASS" in src and "TB_RESULT: FAIL" in src):
        diags.append(err("The testbench must print exactly one verdict line at the end: "
                         "'TB_RESULT: PASS' when no errors were found, else 'TB_RESULT: FAIL errors=<n>'.",
                         "NoVerdictText"))
    if "!==" not in src and "===" not in src:
        diags.append(err("Compare DUT outputs with !== (or ===), never with != or ==: with != an X or Z on "
                         "a DUT output is not counted as an error.", "NoCaseCompare"))
    for name in outputs:
        wait = re.search(rf"@\s*\(\s*(?:(?:pos|neg)edge\s+)?{name}\b|\bwait\s*\([^;]*\b{name}\b", src)
        if wait:
            diags.append(err(f"'{wait.group(0)}' waits on the DUT output '{name}': with a broken DUT it may never "
                             "change and the testbench hangs. Wait on clock edges or delays, then check the output.",
                             "WaitOnOutput"))
    for pattern, what, hint in _UNSUPPORTED:
        if pattern.search(src):
            diags.append(err(f"Unsupported construct ({what}): {hint}", "Unsupported"))
    return diags


def check_dut_instance(dut_name: str | None, ports: tuple[Port, ...] | None,
                       instances: list[Instance]) -> list[Diagnostic]:
    """The elaborated testbench instantiates the DUT exactly once, with every spec port connected."""
    if not dut_name:
        return []

    def err(msg: str) -> Diagnostic:
        return Diagnostic("instance", "instance", "error", msg)

    duts = [i for i in instances if i.module == dut_name]
    if len(duts) != 1:
        # Zero instances is already reported by the structure check (NoDutInstance) when it is missing
        # from the source; this catches e.g. an instance inside a module that is never elaborated.
        return [err(f"Expected exactly one instance of {dut_name} in the elaborated testbench, found {len(duts)}.")]
    dut = duts[0]
    expected = [p.name for p in ports] if ports else list(dut.connected) + list(dut.unconnected)
    return [err(f"Port '{name}' of {dut_name} ({dut.name}) is not connected. Connect every port by name.")
            for name in expected if name not in dut.connected]


# --- DUT stubs and simulation -------------------------------------------------------------------


def make_stub(module_name: str, ports: tuple[Port, ...], value: int) -> str:
    """A DUT with the spec's interface whose outputs are all 0s (value=0) or all 1s (value=1)."""
    lines = [render_header(module_name, ports)]
    lines += [f"  assign {p.name} = '{value};" for p in ports if p.direction == "output"]
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


@dataclass
class SimResult:
    compile: CmdResult
    run: CmdResult | None  # None when compilation failed
    verdicts: list[str]  # every "TB_RESULT: PASS|FAIL" printed, in order

    @property
    def status(self) -> str:
        """compile_error | sim_timeout | watchdog_timeout | no_verdict | multiple_verdicts | pass | fail"""
        if self.run is None:
            return "compile_error"
        if self.run.timed_out:
            return "sim_timeout"
        if TB_WATCHDOG_RE.search(self.run.output):
            return "watchdog_timeout"
        if not self.verdicts:
            return "no_verdict"
        if len(self.verdicts) > 1:
            return "multiple_verdicts"
        return self.verdicts[0].lower()


def parse_verdicts(output: str) -> list[str]:
    return TB_VERDICT_RE.findall(output)


def run_tb(tb: Path, dut_files: list[Path], cwd: Path, timeout: float, tag: str = "sim") -> SimResult:
    """Compile tb + DUT with Icarus and simulate (the same flow the scoring uses)."""
    vvp = f"{tag}.vvp"
    comp = run_cmd(["iverilog", "-g2012", "-s", TB_TOP, "-o", vvp, str(tb), *map(str, dut_files)], cwd, timeout)
    if comp.timed_out or comp.returncode != 0:
        return SimResult(comp, None, [])
    try:
        run = run_cmd(["vvp", "-n", vvp], cwd, timeout)
    finally:
        (cwd / vvp).unlink(missing_ok=True)
    return SimResult(comp, run, parse_verdicts(run.output))


def _tail(text: str, n: int = 15) -> str:
    lines = text.strip().splitlines()
    return "\n".join((["..."] if len(lines) > n else []) + lines[-n:]) or "(no output)"


def check_smoke(results: dict[int, SimResult], vacuity_check: bool = True) -> list[Diagnostic]:
    """Judge the testbench's runs against the constant-output stubs (keys: stub value 0 / 1).

    Every run must end on its own with exactly one verdict. With vacuity_check, at least one stub must
    FAIL: a testbench that accepts a DUT whose outputs never change checks (almost) nothing.
    """

    def err(msg: str, code: str) -> Diagnostic:
        return Diagnostic("sim", "sim", "error", msg, code=code)

    diags = []
    for value, r in results.items():
        who = f"the all-{value}s stub DUT"
        status = r.status
        if status == "compile_error":
            diags.append(err(f"Simulation build against {who} failed:\n{_tail(r.compile.output)}", "SimCompile"))
        elif status == "sim_timeout":
            diags.append(err(f"The simulation against {who} never finished (killed after the timeout). "
                             f"Last output:\n{_tail(r.run.output)}", "SimTimeout"))
        elif status == "watchdog_timeout":
            diags.append(err(f"The testbench's watchdog fired against {who}: the test never reached its end. "
                             f"Last output:\n{_tail(r.run.output)}", "WatchdogTimeout"))
        elif status == "no_verdict":
            diags.append(err(f"The simulation against {who} ended without printing a 'TB_RESULT:' line. "
                             f"Last output:\n{_tail(r.run.output)}", "NoVerdict"))
        elif status == "multiple_verdicts":
            diags.append(err(f"The simulation against {who} printed {len(r.verdicts)} 'TB_RESULT:' lines "
                             f"({', '.join(r.verdicts)}); print exactly one, at the very end.", "MultipleVerdicts"))
    if vacuity_check and not diags and all(r.status == "pass" for r in results.values()):
        values = " and ".join(f"all {v}s" for v in results)
        diags.append(err(f"The testbench reported TB_RESULT: PASS for DUTs whose outputs are constant ({values}). "
                         "A testbench that accepts these checks almost nothing.", "Vacuous"))
    return diags


# --- formatting for the LLM ---------------------------------------------------------------------


def error_signature(diags: list[Diagnostic]) -> str:
    """Stable id of the set of errors (ignoring line numbers), used to detect a stuck fix loop.

    The "code unchanged" note is excluded so an unchanged fix still counts as the same errors.
    """
    keys = sorted({f"{d.tool}:{d.code or d.message}" if d.tool == "sim" else f"{d.tool}:{d.message}"
                   for d in diags if d.severity == "error" and d.category != "no_change"})
    return hashlib.sha1("\n".join(keys).encode()).hexdigest()[:12] if keys else ""


@dataclass(frozen=True)
class FixHint:
    """How to fix a known diagnostic (tool messages say what is wrong, not how to fix it).

    Same idea as RTLFixer's hand-written compiler-error guidance: a deterministic lookup, no LLM.
    Matches on the diagnostic `code` when given (exact), else on a message regex (iverilog has no codes).
    """

    hint: str
    code: str | None = None
    pattern: re.Pattern | None = None

    def matches(self, d: Diagnostic) -> bool:
        return (self.code is not None and d.code == self.code) or (
            self.pattern is not None and self.pattern.search(d.message) is not None
        )


FIX_HINTS: list[FixHint] = [
    FixHint("A replication inside a concatenation needs its own braces: write {{24{in[7]}}, in}, not {24{in[7]}, in}.",
            pattern=re.compile(r"repeat concatenation")),
    FixHint("Declare every testbench signal (as logic, with the DUT port's width) before it is used, and check "
            "its spelling.",
            pattern=re.compile(r"Unable to bind wire/reg/memory|use of undeclared identifier|Could not find variable")),
    FixHint("Declare every variable at module level (logic [3:0] expected;) or at the very top of its "
            "begin...end block, before any statement; then assign it where needed (expected = model(a, b);). "
            "A declaration after a statement (#1; logic x = ...;) is a syntax error.",
            code="DeclarationsAtStart"),
    FixHint("Declare every variable at module level (logic [3:0] expected;) or at the very top of its "
            "begin...end block, before any statement; then assign it where needed (expected = model(a, b);). "
            "A declaration after a statement (#1; logic x = ...;) is a syntax error.",
            pattern=re.compile(r"Syntax in assignment statement l-value")),
    FixHint("A sized literal has more digits than its width (e.g. 4'b10101); fix the width or the digits. For "
            "wide values build them from smaller pieces ({4{32'hDEADBEEF}}) or use '0 / '1.",
            code="VectorLiteralOverflow"),
    FixHint("Two case items have the same value, so the second can never be taken; make every item distinct.",
            code="CaseDup"),
    FixHint("Only the ports listed in the DUT interface exist; connect them by their exact names.",
            code="PortDoesNotExist"),
    FixHint("Only the ports listed in the DUT interface exist; connect them by their exact names.",
            pattern=re.compile(r"is not a port of")),
    FixHint("Connect every DUT port by name: .clk(clk), .a(a), ... (or .* with identically named testbench signals).",
            code="UnconnectedInputPort"),
    FixHint("Declare the testbench signal connected to each DUT port with exactly that port's width.",
            code="PortWidthExpand"),
    FixHint("Declare the testbench signal connected to each DUT port with exactly that port's width.",
            code="PortWidthTruncate"),
    FixHint("An undeclared name was silently turned into a 1-bit wire, usually a typo: declare it as logic "
            "with the right width, or fix the spelling.",
            code="ImplicitNet"),
    FixHint("Never reference signals inside the DUT (dut.state): only its ports are visible. Track any state "
            "you need in the testbench's own reference model.",
            pattern=re.compile(r"Unable to bind .*dut\.|could not find member .* in 'TopModule'|"
                               r"no member named .* in .*TopModule", re.IGNORECASE)),
    FixHint("Icarus does not support this construct; rewrite it with plain procedural code (if/for/while, "
            "functions, tasks, $urandom).",
            pattern=re.compile(r"^sorry|not currently supported", re.IGNORECASE)),
    FixHint("Make sure every loop in the stimulus ends and $finish is always reached. Never wait on a DUT "
            "output to change (@(posedge out), wait (out)): a broken DUT may never change it. Wait on clock "
            "edges or delays instead, and keep a watchdog: initial begin #<limit>; "
            "$display(\"TB_RESULT: FAIL timeout\"); $finish; end",
            code="SimTimeout"),
    FixHint("The test hung until the watchdog fired. Common causes: a loop whose counter is too narrow to "
            "reach the end condition (a 2-bit 'inputs' never reaches 4 in 'inputs < 4'; loop over an int and "
            "assign the inputs from it: for (int i = 0; i < 4; i++) {a, b} = i;), and waiting on a DUT output "
            "(@(posedge out), wait (out)) instead of on clock edges or delays.",
            code="WatchdogTimeout"),
    FixHint("Never wait on a DUT output: a broken DUT may never change it, and the testbench hangs. Drive the "
            "inputs, wait a delay (#1) or a clock edge, then compare.",
            code="WaitOnOutput"),
    FixHint("At the very end, print exactly one verdict and stop: if (errors == 0) $display(\"TB_RESULT: PASS\"); "
            "else $display(\"TB_RESULT: FAIL errors=%0d\", errors); $finish;",
            code="NoVerdict"),
    FixHint("Print the verdict in exactly one place (the end of the main initial block) and call $finish right "
            "after it; the watchdog's FAIL line must also be followed by $finish.",
            code="MultipleVerdicts"),
    FixHint("Check the DUT's outputs against expected values computed by the testbench's own reference model "
            "of the spec, after every stimulus, using !==. Increment the error counter on every mismatch, and "
            "make the final verdict depend on it. Make sure the stimulus produces outputs that differ from all "
            "0s / all 1s.",
            code="Vacuous"),
]


def fix_hints(diags: list[Diagnostic]) -> list[str]:
    """The fix suggestions that apply to these diagnostics' errors, each listed once."""
    errors = [d for d in diags if d.severity == "error"]
    hints = []
    for h in FIX_HINTS:
        if h.hint not in hints and any(h.matches(d) for d in errors):
            hints.append(h.hint)
    return hints


def format_diagnostics(diags: list[Diagnostic], code: str, limit: int = 8) -> str:
    """Errors first, deduplicated, capped, each with the offending source lines, then fix hints."""
    seen, error_lines, unique = set(), set(), []
    for d in sorted(diags, key=lambda d: (d.severity != "error", d.line or 0)):
        key = (d.tool, d.message, d.line)
        if key in seen:
            continue
        seen.add(key)
        if d.severity == "error" and d.line is not None:
            # Only the first error per tool and line: the rest are parser follow-on errors
            # ("expected ';'", "unexpected '}'") caused by it, which just add noise.
            if (d.tool, d.line) in error_lines:
                continue
            error_lines.add((d.tool, d.line))
        unique.append(d)

    lines = code.splitlines()
    blocks = []
    for d in unique[:limit]:
        loc = f" line {d.line}" if d.line else ""
        block = f"[{d.severity.upper()}] {d.tool}{loc}: {d.message}"
        if d.line and 1 <= d.line <= len(lines):
            lo, hi = max(1, d.line - 2), min(len(lines), d.line + 2)
            block += "\n" + "\n".join(
                f"{'>>' if n == d.line else '  '} {n:4d} | {lines[n - 1]}" for n in range(lo, hi + 1)
            )
        blocks.append(block)

    if len(unique) > limit:
        rest = unique[limit:]
        n_err = sum(d.severity == "error" for d in rest)
        blocks.append(f"... ({n_err} more error(s), {len(rest) - n_err} more warning(s) omitted)")
    if hints := fix_hints(diags):
        blocks.append("Fix suggestions:\n" + "\n".join(f"- {h}" for h in hints))
    return "\n\n".join(blocks)
