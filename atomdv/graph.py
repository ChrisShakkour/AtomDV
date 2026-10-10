"""Testbench-generation graph: static Python nodes do the deterministic work, LLM nodes plan/generate/fix.

parse_spec → plan → generate → write_tb → check → route ─┬─ pass / out of budget → finalize → END
                        ▲          ▲                      ├─ errors → fix → write_tb
                        │          └──────────────────────┤
                        └────── stuck (same errors) ──────┘

The design under test is never available: check compiles and simulates the testbench against
constant-output stub DUTs built from the spec's interface. Scoring against the reference design
happens only after the agent is done (scripts/run_verilog_eval.py).
"""

import json
import re
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from atomdv import prompts
from atomdv.eda import (
    TB_TOP,
    Diagnostic,
    check_dut_instance,
    check_smoke,
    check_tb_structure,
    error_signature,
    fix_hints,
    format_diagnostics,
    lint_slang,
    make_stub,
    run_iverilog,
    run_tb,
)
from atomdv.llm import make_llm
from atomdv.spec import parse_spec
from atomdv.state import TBState, RunContext

_FENCE_RE = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)```", re.DOTALL)
_TB_MODULE_RE = re.compile(rf"^\s*module\s+{TB_TOP}\b", re.MULTILINE)
TB_FILE = f"{TB_TOP}.sv"


def extract_code(response: str) -> str | None:
    """The last fenced block that contains the full testbench module (fix replies may quote old code first)."""
    blocks = [body for _, body in _FENCE_RE.findall(response) if "endmodule" in body and _TB_MODULE_RE.search(body)]
    return blocks[-1].strip() + "\n" if blocks else None


def _call_llm(node: str, messages: list[BaseMessage], state: TBState, config: RunnableConfig, ctx: RunContext) -> AIMessage:
    llm = make_llm(ctx.config.ollama, ctx.config.node_llm(node))
    # Passing the node's config propagates the transcript callback; metadata tags the attempt.
    return llm.with_config(metadata={"attempt": state.get("attempts", 0)}).invoke(messages, config)


def _usage(msg: AIMessage) -> dict:
    u = msg.usage_metadata or {}
    return {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0), "llm_calls": 1}


# --- static nodes -------------------------------------------------------------------------------


def parse_spec_node(state: TBState, runtime: Runtime[RunContext]) -> dict:
    spec = parse_spec(state["spec"], top=state.get("top"))
    runtime.context.transcript.event(
        "state",
        "parse_spec",
        title=f"module={spec.module_name} ports={len(spec.ports) if spec.ports else None}",
        sections={"HEADER": spec.header or "(none: interface not found in spec)"},
    )
    stub_paths = None
    if spec.module_name and spec.ports:
        stub_paths = {}
        for value in (0, 1):
            path = runtime.context.workdir / f"stub_{value}.sv"
            path.write_text(make_stub(spec.module_name, spec.ports, value))
            stub_paths[value] = str(path)
    return {"module_name": spec.module_name, "ports": spec.ports, "header": spec.header,
            "stub_paths": stub_paths, "attempts": 0}


def write_tb_node(state: TBState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    code = extract_code(state["response"])
    if code is None:
        ctx.transcript.event("state", "write_tb", title="no testbench code block in response", attempt=state["attempts"])
        return {"code": None, "code_path": None, "code_unchanged": False}

    path = ctx.workdir / TB_FILE
    path.write_text(code)
    # A "fix" that returns the failing code verbatim gets called out explicitly (see check_node).
    unchanged = code.strip() == (state.get("code") or "").strip()
    title = f"wrote {path.name} ({len(code)} chars)" + (" UNCHANGED from the previous attempt" if unchanged else "")
    ctx.transcript.event("state", "write_tb", title=title, attempt=state["attempts"])
    return {"code": code, "code_path": str(path), "code_unchanged": unchanged}


def check_node(state: TBState, runtime: Runtime[RunContext]) -> dict:
    ctx, t, attempt, cfg = runtime.context, runtime.context.transcript, state["attempts"], runtime.context.config
    smoke = {}
    if not state.get("code_path"):
        diags = [Diagnostic("agent", "no_code", "error",
                            f"Your response did not contain a ```systemverilog code block with a complete "
                            f"module {TB_TOP}.")]
    else:
        path = Path(state["code_path"])
        dut = state.get("module_name")
        outputs = [p.name for p in state.get("ports") or () if p.direction == "output"]
        diags = check_tb_structure(state["code"], dut, outputs)
        t.event("tool", "check", title=f"structure check -> {'OK' if not diags else f'{len(diags)} problem(s)'}",
                attempt=attempt, sections={"OUTPUT": "\n".join(d.message for d in diags) or "OK"}, tool="structure")

        stubs = {int(v): Path(p) for v, p in (state.get("stub_paths") or {}).items()}
        if not stubs:
            # Without the interface there is no DUT to compile against; only the structure check applies.
            diags.append(Diagnostic("agent", "no_interface", "warning",
                                    "The DUT interface could not be parsed from the spec; compile, lint and "
                                    "simulation checks were skipped."))
        else:
            # Relative to the workdir so tool output shows "tb.sv:3", not a long absolute path.
            files = [Path(path.name), Path(stubs[0].name)]
            res, comp = run_iverilog(files, ctx.workdir, cfg.eda_timeout, top=TB_TOP, only=path.name)
            t.event("tool", "check", title=f"$ {res.cmdline} -> exit {res.returncode}", attempt=attempt,
                    duration_s=res.duration_s, sections={"OUTPUT": res.output or "(no output)"},
                    tool="iverilog", returncode=res.returncode)

            lint = lint_slang([path, stubs[0]], as_errors=cfg.blocking_warnings, only=path.name)
            n_err = sum(d.severity == "error" for d in lint.diagnostics)
            t.event("tool", "check", title=f"pyslang lint -> {n_err} errors, {len(lint.diagnostics) - n_err} warnings",
                    attempt=attempt, duration_s=lint.duration_s, sections={"OUTPUT": lint.report or "(clean)"},
                    tool="slang", instances=[asdict(i) for i in lint.instances])

            inst = check_dut_instance(dut, state.get("ports"), lint.instances) if n_err == 0 else []
            t.event("tool", "check", title=f"DUT instance check -> {'OK' if not inst else f'{len(inst)} problem(s)'}",
                    attempt=attempt, sections={"OUTPUT": "\n".join(d.message for d in inst) or "OK"}, tool="instance")
            diags = diags + comp + lint.diagnostics + inst

            # Simulate only a testbench that is otherwise clean: on broken code the run says nothing new.
            if not any(d.severity == "error" for d in diags):
                results = {}
                for value, stub in stubs.items():
                    r = run_tb(Path(path.name), [Path(stub.name)], ctx.workdir, cfg.sim_timeout, tag=f"smoke_{value}")
                    results[value] = r
                    smoke[f"stub_{value}"] = r.status
                    run = r.run or r.compile
                    t.event("tool", "check", title=f"smoke sim vs all-{value}s stub -> {r.status}", attempt=attempt,
                            duration_s=r.compile.duration_s + (r.run.duration_s if r.run else 0),
                            sections={"OUTPUT": run.output or "(no output)"}, tool="sim", status=r.status)
                diags = diags + check_smoke(results, cfg.vacuity_check)

        if state.get("code_unchanged") and any(d.severity == "error" for d in diags):
            diags.insert(0, Diagnostic("agent", "no_change", "error",
                                       "Your corrected testbench is identical to the one that failed, so every "
                                       "error below is still there. You must actually change the code."))

    errors = sum(d.severity == "error" for d in diags)
    text = format_diagnostics(diags, state.get("code") or "", ctx.config.max_diagnostics)
    return {
        "smoke": smoke,
        "diagnostics": diags,
        "feedback": [{"source": "check", "kind": "diagnostics", "text": text}] if diags else [],
        "history": [{
            "attempt": attempt,
            "node": state.get("response_node", "generate"),
            "signature": error_signature(diags),
            "errors": errors,
            "warnings": len(diags) - errors,
        }],
    }


def route_node(state: TBState, runtime: Runtime[RunContext]) -> dict:
    cfg = runtime.context.config
    history, attempts = state["history"], state["attempts"]
    errors = history[-1]["errors"]
    recent = [h["signature"] for h in history[-cfg.stuck_after:]]

    if errors == 0:
        route, reason = "finalize", "pass: no errors"
    elif attempts >= cfg.max_attempts:
        route, reason = "finalize", f"fail: budget exhausted ({attempts}/{cfg.max_attempts} attempts)"
    elif len(recent) == cfg.stuck_after and len(set(recent)) == 1:
        route, reason = "generate", f"stuck: same errors {cfg.stuck_after}x in a row (signature {recent[0]}), fresh restart"
    else:
        route, reason = "fix", f"{errors} error(s), attempt {attempts}/{cfg.max_attempts}"

    runtime.context.transcript.event("route", "route", title=f"-> {route}: {reason}", attempt=attempts)
    update = {"route": route, "route_reason": reason}
    if route == "generate":
        errs = [d.message for d in state["diagnostics"] if d.severity == "error"][:3]
        hints = fix_hints(state["diagnostics"])
        update["restart_hint"] = (
            "A previous testbench kept failing with these errors despite fixes:\n- "
            + "\n- ".join(errs)
            + ("\nHow to avoid them:\n- " + "\n- ".join(hints) if hints else "")
            + "\nWrite a fresh testbench that avoids the constructs that caused them."
        )
    return update


def finalize_node(state: TBState, runtime: Runtime[RunContext]) -> dict:
    ctx = runtime.context
    status = "pass" if state["history"] and state["history"][-1]["errors"] == 0 else "fail"
    result = {
        "status": status,
        "reason": state.get("route_reason"),
        "module_name": state.get("module_name"),
        "code_path": state.get("code_path"),
        "smoke": state.get("smoke", {}),
        "attempts": state["attempts"],
        "history": state["history"],
        "final_diagnostics": [asdict(d) for d in state.get("diagnostics", [])],
        "tokens": state.get("tokens", {}),
    }
    (ctx.workdir / "result.json").write_text(json.dumps(result, indent=2))
    ctx.transcript.event("state", "finalize", title=f"status={status}", sections={"RESULT": json.dumps(result, indent=2)})
    return {"status": status}


# --- LLM nodes ----------------------------------------------------------------------------------


def plan_node(state: TBState, config: RunnableConfig, runtime: Runtime[RunContext]) -> dict:
    msg = _call_llm("plan", prompts.build_plan_messages(state), state, config, runtime.context)
    return {"plan": msg.text, "tokens": _usage(msg)}


def generate_node(state: TBState, config: RunnableConfig, runtime: Runtime[RunContext]) -> dict:
    state = {**state, "attempts": state["attempts"] + 1}
    msg = _call_llm("generate", prompts.build_generate_messages(state), state, config, runtime.context)
    return {"response": msg.text, "response_node": "generate", "attempts": state["attempts"], "restart_hint": None,
            "tokens": _usage(msg)}


def fix_node(state: TBState, config: RunnableConfig, runtime: Runtime[RunContext]) -> dict:
    state = {**state, "attempts": state["attempts"] + 1}
    msg = _call_llm("fix", prompts.build_fix_messages(state), state, config, runtime.context)
    return {"response": msg.text, "response_node": "fix", "attempts": state["attempts"], "tokens": _usage(msg)}


# --- graph --------------------------------------------------------------------------------------


def build_graph():
    g = StateGraph(TBState, context_schema=RunContext)
    g.add_node("parse_spec", parse_spec_node)
    g.add_node("plan", plan_node)
    g.add_node("generate", generate_node)
    g.add_node("write_tb", write_tb_node)
    g.add_node("check", check_node)
    g.add_node("route", route_node)
    g.add_node("fix", fix_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "parse_spec")
    g.add_edge("parse_spec", "plan")
    g.add_edge("plan", "generate")
    g.add_edge("generate", "write_tb")
    g.add_edge("write_tb", "check")
    g.add_edge("check", "route")
    g.add_conditional_edges("route", lambda s: s["route"], ["finalize", "fix", "generate"])
    g.add_edge("fix", "write_tb")
    g.add_edge("finalize", END)
    return g.compile()
