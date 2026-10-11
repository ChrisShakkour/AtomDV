from pathlib import Path

import pytest

from atomdv.config import AgentConfig
from atomdv.eda import (
    Diagnostic,
    SimResult,
    CmdResult,
    check_dut_instance,
    check_smoke,
    check_tb_structure,
    error_signature,
    fix_hints,
    format_diagnostics,
    lint_slang,
    make_stub,
    parse_verdicts,
    run_iverilog,
    run_tb,
)
from atomdv.spec import Port

DATASET = Path(__file__).resolve().parents[2] / "verilog-eval" / "dataset_spec-to-rtl"

PORTS = (Port("clk", "input"), Port("reset", "input"), Port("q", "output", 4))

# Prob035_count1to10: decade counter 1..10, synchronous active-high reset to 1.
GOOD_TB = """module tb;
  logic clk, reset;
  logic [3:0] q;
  logic [3:0] exp_q;
  int errors = 0;

  TopModule dut (.clk(clk), .reset(reset), .q(q));

  initial clk = 0;
  always #5 clk = ~clk;

  initial begin #1000000; $display("TB_RESULT: FAIL timeout"); $finish; end

  // One clock cycle, called at a negative edge: drive, let the DUT and the model advance, check.
  task automatic step(input logic r);
    reset = r;
    @(posedge clk) exp_q = (r || exp_q == 10) ? 4'd1 : exp_q + 4'd1;
    @(negedge clk);
    if (q !== exp_q) begin
      errors++;
      if (errors <= 5) $display("t=%0t reset=%b q=%0d expected %0d", $time, r, q, exp_q);
    end
  endtask

  initial begin
    reset = 1;
    exp_q = 4'd1;
    @(negedge clk);
    step(1);
    repeat (25) step(0);
    step(1);
    repeat (200) step(($urandom % 16) == 0);
    if (errors == 0) $display("TB_RESULT: PASS");
    else $display("TB_RESULT: FAIL errors=%0d", errors);
    $finish;
  end
endmodule
"""

VACUOUS_TB = """module tb;
  logic clk = 0, reset;
  logic [3:0] q;
  int errors = 0;
  TopModule dut (.clk(clk), .reset(reset), .q(q));
  always #5 clk = ~clk;
  initial begin
    reset = 1;
    repeat (10) @(posedge clk);
    if (q !== q) errors++;
    if (errors == 0) $display("TB_RESULT: PASS");
    else $display("TB_RESULT: FAIL errors=%0d", errors);
    $finish;
  end
endmodule
"""

HANGING_TB = VACUOUS_TB.replace("repeat (10) @(posedge clk);", "@(posedge q[3]);")


def _write(tmp_path, name, code):
    p = tmp_path / name
    p.write_text(code)
    return p


def _stubs(tmp_path):
    return {v: _write(tmp_path, f"stub_{v}.sv", make_stub("TopModule", PORTS, v)) for v in (0, 1)}


def _ref_as_dut(tmp_path, problem="Prob035_count1to10"):
    ref = (DATASET / f"{problem}_ref.sv").read_text().replace("module RefModule", "module TopModule", 1)
    return _write(tmp_path, "ref_as_dut.sv", ref)


def test_stub_has_interface_and_constant_outputs(tmp_path):
    stub = make_stub("TopModule", PORTS, 1)
    assert stub.startswith("module TopModule (") and "assign q = '1;" in stub and "clk =" not in stub
    p = _write(tmp_path, "stub_1.sv", stub)
    res, diags = run_iverilog([p], tmp_path)
    assert res.returncode == 0 and diags == []


def test_good_tb_is_clean_and_fails_both_stubs(tmp_path):
    tb = _write(tmp_path, "tb.sv", GOOD_TB)
    stubs = _stubs(tmp_path)
    assert check_tb_structure(GOOD_TB, "TopModule") == []
    res, diags = run_iverilog([Path("tb.sv"), Path("stub_0.sv")], tmp_path, top="tb", only="tb.sv")
    assert res.returncode == 0 and not [d for d in diags if d.severity == "error"]
    lint = lint_slang([tb, stubs[0]], as_errors=AgentConfig().blocking_warnings, only="tb.sv")
    assert not [d for d in lint.diagnostics if d.severity == "error"]
    assert check_dut_instance("TopModule", PORTS, lint.instances) == []

    results = {v: run_tb(Path("tb.sv"), [Path(s.name)], tmp_path, 30, tag=f"smoke_{v}") for v, s in stubs.items()}
    assert {v: r.status for v, r in results.items()} == {0: "fail", 1: "fail"}
    assert check_smoke(results) == []
    assert not list(tmp_path.glob("*.vvp"))  # compiled sims are cleaned up


@pytest.mark.skipif(not DATASET.is_dir(), reason="verilog-eval dataset not cloned next to AtomDV")
def test_good_tb_passes_reference_and_catches_a_bug(tmp_path):
    _write(tmp_path, "tb.sv", GOOD_TB)
    ref = _ref_as_dut(tmp_path)
    assert run_tb(Path("tb.sv"), [Path(ref.name)], tmp_path, 30).status == "pass"
    _write(tmp_path, "buggy.sv", ref.read_text().replace("q == 10", "q == 9"))
    r = run_tb(Path("tb.sv"), [Path("buggy.sv")], tmp_path, 30)
    assert r.status == "fail" and "TB_RESULT: FAIL errors=" in r.run.output


def test_vacuous_tb_is_flagged(tmp_path):
    _write(tmp_path, "tb.sv", VACUOUS_TB)
    stubs = _stubs(tmp_path)
    results = {v: run_tb(Path("tb.sv"), [Path(s.name)], tmp_path, 30, tag=f"smoke_{v}") for v, s in stubs.items()}
    diags = check_smoke(results)
    assert [d.code for d in diags] == ["Vacuous"]
    assert check_smoke(results, vacuity_check=False) == []
    assert any("reference model" in h for h in fix_hints(diags))


def test_hanging_tb_times_out(tmp_path):
    _write(tmp_path, "tb.sv", HANGING_TB)
    stubs = _stubs(tmp_path)
    r = run_tb(Path("tb.sv"), [Path(stubs[0].name)], tmp_path, 2, tag="smoke_0")
    assert r.status == "sim_timeout"
    diags = check_smoke({0: r})
    assert [d.code for d in diags] == ["SimTimeout"]
    assert any("Never wait on a DUT output" in h for h in fix_hints(diags))


def test_verdict_parsing_and_statuses():
    assert parse_verdicts("x\nTB_RESULT: PASS\n") == ["PASS"]
    assert parse_verdicts("TB_RESULT: FAIL errors=3\nTB_RESULT: FAIL timeout") == ["FAIL", "FAIL"]
    assert parse_verdicts("all good") == []

    def sim(output, timed_out=False):
        run = CmdResult(["vvp"], None if timed_out else 0, output, 0.1, timed_out)
        return SimResult(CmdResult(["iverilog"], 0, "", 0.1), run, parse_verdicts(output))

    assert sim("TB_RESULT: PASS").status == "pass"
    assert sim("TB_RESULT: FAIL errors=1").status == "fail"
    assert sim("done").status == "no_verdict"
    assert sim("TB_RESULT: PASS\nTB_RESULT: FAIL").status == "multiple_verdicts"
    assert sim("", timed_out=True).status == "sim_timeout"
    assert SimResult(CmdResult(["iverilog"], 1, "err", 0.1), None, []).status == "compile_error"
    assert [d.code for d in check_smoke({0: sim("done"), 1: sim("TB_RESULT: PASS\nTB_RESULT: FAIL")})] == [
        "NoVerdict", "MultipleVerdicts"]
    # A FAIL on one stub is enough (e.g. a spec whose output really is constant 0).
    assert check_smoke({0: sim("TB_RESULT: PASS"), 1: sim("TB_RESULT: FAIL errors=4")}) == []


def test_structure_check():
    codes = lambda src: {d.code for d in check_tb_structure(src, "TopModule")}  # noqa: E731
    assert codes(GOOD_TB) == set()
    assert "DutDefined" in codes(GOOD_TB + "\nmodule TopModule(input clk); endmodule\n")
    assert "NoDutInstance" in codes(GOOD_TB.replace("TopModule dut", "// TopModule dut").replace(
        "(.clk(clk), .reset(reset), .q(q));", ""))
    assert "NoFinish" in codes(GOOD_TB.replace("$finish", "$stop"))
    assert "NoVerdictText" in codes(GOOD_TB.replace('"TB_RESULT: PASS"', '"ALL PASSED"'))
    assert "NoCaseCompare" in codes(GOOD_TB.replace("!==", "!="))
    assert "NoTbModule" in codes(GOOD_TB.replace("module tb;", "module test;"))
    assert "Unsupported" in codes(GOOD_TB.replace("int errors = 0;", "int errors = 0;\n  covergroup cg; endgroup"))
    # Commented-out text doesn't count.
    assert "NoFinish" in codes(GOOD_TB.replace("$finish;", "// $finish;"))


def test_bad_dut_hookup_is_reported(tmp_path):
    src = GOOD_TB.replace("logic [3:0] q;", "logic [7:0] q;").replace(
        ".reset(reset), .q(q)", ".q(q), .bogus(1'b0)")
    tb = _write(tmp_path, "tb.sv", src)
    stubs = _stubs(tmp_path)
    lint = lint_slang([tb, stubs[0]], as_errors=AgentConfig().blocking_warnings, only="tb.sv")
    errors = {d.code for d in lint.diagnostics if d.severity == "error"}
    assert {"PortDoesNotExist"} <= errors
    res, diags = run_iverilog([Path("tb.sv"), Path("stub_0.sv")], tmp_path, top="tb", only="tb.sv")
    assert any("is not a port of" in d.message for d in diags if d.severity == "error")

    # Without the bogus port: the unconnected input and the widened output are blocking slang warnings.
    src = src.replace(", .bogus(1'b0)", "")
    tb = _write(tmp_path, "tb.sv", src)
    lint = lint_slang([tb, stubs[0]], as_errors=AgentConfig().blocking_warnings, only="tb.sv")
    errors = {d.code for d in lint.diagnostics if d.severity == "error"}
    assert {"UnconnectedInputPort", "PortWidthExpand"} <= errors
    assert check_dut_instance("TopModule", PORTS, lint.instances)[0].message.startswith("Port 'reset'")
    hints = fix_hints([d for d in lint.diagnostics if d.severity == "error"])
    assert any("exactly that port's width" in h for h in hints)


def test_dut_instance_count():
    from atomdv.eda import Instance

    one = Instance("dut", "TopModule", ("clk", "reset", "q"), ())
    assert check_dut_instance("TopModule", PORTS, [one]) == []
    assert check_dut_instance("TopModule", PORTS, [])[0].message.startswith("Expected exactly one")
    assert check_dut_instance("TopModule", PORTS, [one, one])[0].message.startswith("Expected exactly one")
    assert check_dut_instance(None, PORTS, []) == []


def test_format_and_signature():
    diags = [
        Diagnostic("slang", "lint", "error", "use of undeclared identifier 'foo'", 3),
        Diagnostic("slang", "lint", "warning", "unused variable 'x'", 1),
        Diagnostic("sim", "sim", "error", "never finished. Last output:\nt=10", code="SimTimeout"),
    ]
    text = format_diagnostics(diags, "a\nb\nc\nd\n")
    assert text.index("[ERROR]") < text.index("[WARNING]") and ">>    3 | c" in text
    assert "Fix suggestions:" in text
    # Simulation errors are keyed by code, so a different output tail is still the same error.
    other = diags[:2] + [Diagnostic("sim", "sim", "error", "never finished. Last output:\nt=20", code="SimTimeout")]
    assert error_signature(diags) == error_signature(other) != ""
    assert error_signature([d for d in diags if d.severity == "warning"]) == ""


NOR_PORTS = (Port("a", "input"), Port("b", "input"), Port("out", "output"))
# From a real run: a 2-bit loop counter never reaches 4, so only the watchdog ends the simulation.
NARROW_LOOP_TB = """module tb;
  logic a, b, out;
  logic [1:0] inputs;
  int errors = 0;
  TopModule dut (.a(a), .b(b), .out(out));
  initial begin #1000000; $display("TB_RESULT: FAIL timeout"); $finish; end
  initial begin
    for (inputs = 0; inputs < 4; inputs++) begin
      {a, b} = inputs;
      #1;
      if (out !== ~(a | b)) errors++;
    end
    if (errors == 0) $display("TB_RESULT: PASS");
    else $display("TB_RESULT: FAIL errors=%0d", errors);
    $finish;
  end
endmodule
"""


def test_watchdog_verdict_is_a_hang_not_a_fail(tmp_path):
    _write(tmp_path, "tb.sv", NARROW_LOOP_TB)
    stubs = {v: _write(tmp_path, f"stub_{v}.sv", make_stub("TopModule", NOR_PORTS, v)) for v in (0, 1)}
    results = {v: run_tb(Path("tb.sv"), [Path(s.name)], tmp_path, 30, tag=f"smoke_{v}") for v, s in stubs.items()}
    assert {r.status for r in results.values()} == {"watchdog_timeout"}
    diags = check_smoke(results)
    assert {d.code for d in diags} == {"WatchdogTimeout"}
    assert any("for (int i = 0; i < 4; i++)" in h for h in fix_hints(diags))


def test_waiting_on_a_dut_output_is_flagged():
    codes = lambda src: [d.code for d in check_tb_structure(src, "TopModule", ["q"])]  # noqa: E731
    assert codes(GOOD_TB) == []
    assert codes(GOOD_TB.replace("@(negedge clk);\n    if (q", "@(posedge q);\n    if (q")) == ["WaitOnOutput"]
    assert codes(GOOD_TB.replace("@(negedge clk);\n    if (q", "wait (q == 4'd3);\n    if (q")) == ["WaitOnOutput"]
    assert codes(GOOD_TB.replace("@(negedge clk);\n    if (q", "@(q);\n    if (q")) == ["WaitOnOutput"]


def test_mid_block_declaration_gets_a_fix_hint(tmp_path):
    src = GOOD_TB.replace("    if (q !== exp_q) begin", "    logic [3:0] seen = q;\n    if (seen !== exp_q) begin")
    tb = _write(tmp_path, "tb.sv", src)
    stubs = _stubs(tmp_path)
    lint = lint_slang([tb, stubs[0]], only="tb.sv")
    res, comp = run_iverilog([Path("tb.sv"), Path("stub_0.sv")], tmp_path, top="tb", only="tb.sv")
    diags = lint.diagnostics + comp
    assert "DeclarationsAtStart" in {d.code for d in diags if d.severity == "error"}
    assert any("before any statement" in h for h in fix_hints(diags))
