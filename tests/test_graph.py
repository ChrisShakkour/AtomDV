"""Graph flow with a scripted fake LLM (no Ollama needed)."""

import json
from dataclasses import replace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import atomdv.graph as graph
from atomdv.cli import run_agent
from atomdv.config import AgentConfig

SPEC = """I would like you to implement a module named TopModule with the following
interface. All input and output ports are one bit unless otherwise specified.

 - input  a
 - input  b
 - output out

The module should implement a NOR gate.
"""
GOOD_TB = """module tb;
  logic a, b, out;
  int errors = 0;
  TopModule dut (.a(a), .b(b), .out(out));
  initial begin
    for (int i = 0; i < 4; i++) begin
      {a, b} = i[1:0];
      #1;
      if (out !== ~(a | b)) errors++;
    end
    if (errors == 0) $display("TB_RESULT: PASS");
    else $display("TB_RESULT: FAIL errors=%0d", errors);
    $finish;
  end
endmodule"""
GOOD = f"```systemverilog\n{GOOD_TB}\n```"
SYNTAX_ERR = GOOD.replace("errors++;", "errors++")
WRONG_HOOKUP = GOOD.replace(".a(a)", ".x(a)")
# Never compares against the DUT's output, so it passes the constant-output stubs too.
VACUOUS = GOOD.replace("if (out !== ~(a | b)) errors++;", "if (a !== a) errors++;")
PLAN = "- check out == ~(a | b) for all 4 input combinations"


@pytest.fixture
def scripted_llm(monkeypatch):
    """Replace the Ollama client: each LLM node call returns the next scripted response."""
    calls = []

    def install(*responses):
        fake = FakeListChatModel(responses=list(responses))

        def make_llm(ollama_cfg, node_cfg):
            calls.append(node_cfg)
            return fake

        monkeypatch.setattr(graph, "make_llm", make_llm)
        return calls

    return install


def _run(tmp_path, **cfg):
    return run_agent(SPEC, tmp_path, replace(AgentConfig(), **cfg))


def _events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]


def test_pass_first_try(tmp_path, scripted_llm):
    scripted_llm(PLAN, GOOD)
    final = _run(tmp_path)
    assert final["status"] == "pass" and final["attempts"] == 1
    assert (tmp_path / "tb.sv").read_text().startswith("module tb")
    assert final["smoke"] == {"stub_0": "fail", "stub_1": "fail"}
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "pass"


def test_fix_after_syntax_error(tmp_path, scripted_llm):
    scripted_llm(PLAN, SYNTAX_ERR, "- missing semicolon\n" + GOOD)
    final = _run(tmp_path)
    assert final["status"] == "pass" and final["attempts"] == 2
    assert [h["node"] for h in final["history"]] == ["generate", "fix"]
    assert final["history"][0]["errors"] > 0


def test_no_code_block_goes_to_fix(tmp_path, scripted_llm):
    scripted_llm(PLAN, "I will write a testbench.", GOOD)
    final = _run(tmp_path)
    assert final["status"] == "pass" and final["attempts"] == 2
    assert final["history"][0]["signature"] != ""


def test_dut_hookup_error_is_fixed(tmp_path, scripted_llm):
    scripted_llm(PLAN, WRONG_HOOKUP, GOOD)
    final = _run(tmp_path)
    assert final["status"] == "pass" and final["attempts"] == 2


def test_stuck_restarts_with_fresh_generate(tmp_path, scripted_llm):
    scripted_llm(PLAN, SYNTAX_ERR, SYNTAX_ERR, GOOD)
    final = _run(tmp_path, stuck_after=2)
    assert final["status"] == "pass"
    assert [h["node"] for h in final["history"]] == ["generate", "fix", "generate"]


def test_budget_exhausted_fails(tmp_path, scripted_llm):
    scripted_llm(PLAN, *[SYNTAX_ERR] * 10)
    final = _run(tmp_path, max_attempts=3, stuck_after=99)
    assert final["status"] == "fail" and final["attempts"] == 3
    assert "budget exhausted" in final["route_reason"]


def test_per_node_llm_settings(tmp_path, scripted_llm):
    from atomdv.config import NodeLLMConfig

    calls = scripted_llm(PLAN, GOOD)
    _run(tmp_path, nodes={"plan": NodeLLMConfig(temperature=0.3)})
    assert [c.temperature for c in calls] == [0.3, 0.0]


def test_transcript_records_everything(tmp_path, scripted_llm):
    scripted_llm(PLAN, SYNTAX_ERR, GOOD)
    _run(tmp_path)
    events = _events(tmp_path)
    llm = [e for e in events if e["kind"] == "llm"]
    assert [e["node"] for e in llm] == ["plan", "generate", "fix"]
    assert llm[1]["sections"]["RESPONSE"] == SYNTAX_ERR  # raw response, before extraction
    assert any(k.startswith("SYSTEM") for k in llm[0]["sections"])
    tools = [e["tool"] for e in events if e["kind"] == "tool"]
    # The broken testbench isn't simulated; the clean one runs against both stubs.
    assert tools == ["structure", "iverilog", "slang", "instance"] + ["structure", "iverilog", "slang", "instance",
                                                                      "sim", "sim"]
    routes = [e["title"] for e in events if e["kind"] == "route"]
    assert routes[0].startswith("-> fix") and routes[1].startswith("-> finalize")
    log = (tmp_path / "transcript.log").read_text()
    assert "$ iverilog -g2012 -s tb -t null tb.sv stub_0.sv" in log and "--- RESPONSE ---" in log


def test_graph_shape():
    mermaid = graph.build_graph().get_graph().draw_mermaid()
    for edge in ["parse_spec --> plan", "plan --> generate", "generate --> write_tb", "write_tb --> check",
                 "check --> route", "fix --> write_tb", "finalize --> __end__"]:
        assert edge in mermaid


def test_unchanged_fix_is_called_out(tmp_path, scripted_llm):
    scripted_llm(PLAN, SYNTAX_ERR, SYNTAX_ERR, GOOD)
    final = _run(tmp_path, stuck_after=99)
    assert final["status"] == "pass" and final["attempts"] == 3
    fix_prompts = [e["sections"]["HUMAN [1]"] for e in _events(tmp_path) if e["kind"] == "llm" and e["node"] == "fix"]
    assert "identical to the one that failed" not in fix_prompts[0]
    assert "identical to the one that failed" in fix_prompts[1]
    # Same compiler errors → same signature, so stuck detection still sees the repeat.
    assert final["history"][0]["signature"] == final["history"][1]["signature"]


def test_vacuous_testbench_goes_to_fix(tmp_path, scripted_llm):
    scripted_llm(PLAN, VACUOUS, GOOD)
    final = _run(tmp_path)
    assert final["status"] == "pass" and final["attempts"] == 2
    fix_prompt = next(e["sections"]["HUMAN [1]"] for e in _events(tmp_path) if e["kind"] == "llm" and e["node"] == "fix")
    assert "constant" in fix_prompt and "reference model" in fix_prompt


def test_vacuity_check_can_be_disabled(tmp_path, scripted_llm):
    scripted_llm(PLAN, VACUOUS)
    final = _run(tmp_path, vacuity_check=False)
    assert final["status"] == "pass" and final["smoke"] == {"stub_0": "pass", "stub_1": "pass"}


def test_writing_the_dut_is_an_error(tmp_path, scripted_llm):
    with_dut = GOOD.replace("```systemverilog\n", "```systemverilog\nmodule TopModule(input a, input b, output out);\n"
                            "  assign out = ~(a | b);\nendmodule\n\n")
    scripted_llm(PLAN, with_dut, GOOD)
    final = _run(tmp_path)
    assert final["status"] == "pass" and final["attempts"] == 2
    structure = [e for e in _events(tmp_path) if e["kind"] == "tool" and e["tool"] == "structure"]
    assert "Do not write module TopModule" in structure[0]["sections"]["OUTPUT"]
    assert structure[1]["sections"]["OUTPUT"] == "OK"
