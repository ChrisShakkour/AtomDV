"""Prompt builders, one per LLM node. Each takes only the state fields it needs (no shared history)."""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from atomdv.state import TBState

SYSTEM_PROMPT = """\
You are an expert design-verification engineer. From a design spec alone (the design itself is not \
available to you), you write a self-checking SystemVerilog-2012 testbench that compiles and runs with \
Icarus Verilog 12 (iverilog -g2012, vvp). It must report PASS for every correct implementation of the \
spec and FAIL for an implementation with a bug.

Rules:
- The top module is `module tb;` with no ports. Instantiate the design under test exactly once as \
`TopModule dut (...)`, connecting every port of the given interface by name to a testbench signal of \
the same width. Never write the TopModule itself.
- Only the DUT's ports are visible. Never reference signals inside it (dut.state).
- Compute the expected outputs with your own reference model of the spec inside the testbench \
(functions, tasks, or procedural code that mirrors the spec), and compare every output after every \
stimulus.
- Compare with !== so an X or Z on a DUT output counts as an error. Count errors, and $display the first \
few mismatches with the time, the inputs, the expected and the actual value.
- Unless the spec says otherwise, the design is clocked on the positive edge of clk. Generate the clock \
in the testbench (initial clk = 0; always #5 clk = ~clk;). Run the test one clock cycle at a time, each \
cycle starting at a negative edge: drive the inputs, wait for the positive edge and advance the reference \
model with those same inputs, wait for the next negative edge and compare the outputs. Do not add another \
@(negedge clk) before driving: the DUT would then advance two cycles while the model advances one.
  task automatic cycle(...);  // called at a negedge
    <drive inputs>;
    @(posedge clk) <update the model state>;
    @(negedge clk) <compare every output with !==>;
  endtask
- For combinational outputs, change the inputs, wait #1, then check.
- Declare every variable at module level or at the very top of a begin...end block, before any \
statement (never `#1; logic x = ...;`). Declare it first, assign it later.
- Loop counters: loop over an int and assign the inputs from it (for (int i = 0; i < 4; i++) begin \
{a, b} = i; ... end). A counter as narrow as the inputs never reaches the end value, so the loop \
never ends.
- Registers start unknown (X). Apply the reset described in the spec (synchronous or asynchronous, \
active level) before checking registered outputs. If there is no reset, first drive inputs that put \
every register into a known state, and only check outputs whose expected value is known.
- Stimulus: if the inputs total about 10 bits or fewer, test every input combination. Otherwise combine \
directed corner cases (all 0s, all 1s, boundaries, wrap-around, every FSM state and transition) with a \
few hundred cycles of random stimulus from $urandom with a fixed seed. Exercise reset in the middle of \
operation too.
- End with exactly one verdict line, then $finish:
  if (errors == 0) $display("TB_RESULT: PASS"); else $display("TB_RESULT: FAIL errors=%0d", errors);
  Add a watchdog: initial begin #1000000; $display("TB_RESULT: FAIL timeout"); $finish; end
- Never wait on a DUT output to change (@(posedge out), wait (out)); wait on clock edges or delays.
- Icarus limits: no classes, rand/randomize(), covergroups, concurrent assertions (property, |->), or \
interfaces. Use plain logic/int variables, arrays, functions, tasks, for/repeat/while loops and \
$urandom. Do not use `timescale.
- If the spec is ambiguous, pick the most standard interpretation and state the assumption in a comment.\
"""

_CODE_FORMAT = (
    "Reply with the complete testbench in a single ```systemverilog code block containing module tb "
    "(and nothing that defines TopModule). Do not output any other code block."
)


def _spec_section(state: TBState) -> str:
    parts = [f"## Spec of the design under test\n{state['spec'].strip()}"]
    if state.get("header"):
        parts.append(
            "## DUT interface (instantiate this module; do not implement it)\n"
            f"```systemverilog\n{state['header']}\n  ...\nendmodule\n```"
        )
    elif state.get("module_name"):
        parts.append(f"## DUT module name\n{state['module_name']}")
    return "\n\n".join(parts)


def build_plan_messages(state: TBState) -> list[BaseMessage]:
    return [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            f"{_spec_section(state)}\n\n"
            "Write a short verification plan (4-8 bullet points): the behaviors to check, how the "
            "testbench's reference model computes the expected outputs, the reset handling, when each "
            "output is sampled relative to the clock, the stimulus (exhaustive or directed + random) and "
            "the corner cases. Do not write any code yet."
        ),
    ]


def build_generate_messages(state: TBState) -> list[BaseMessage]:
    parts = [_spec_section(state)]
    if state.get("plan"):
        parts.append(f"## Verification plan\n{state['plan'].strip()}")
    if state.get("restart_hint"):
        parts.append(f"## Note\n{state['restart_hint']}")
    parts.append(f"Write the testbench.\n{_CODE_FORMAT}")
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage("\n\n".join(parts))]


def build_fix_messages(state: TBState) -> list[BaseMessage]:
    code = state.get("code")
    current = (
        f"## Current testbench\n```systemverilog\n{code.rstrip()}\n```"
        if code
        else f"## Your previous response (no usable code block was found in it)\n{state.get('response', '').strip()}"
    )
    feedback = "\n\n".join(f"### From {f['source']} ({f['kind']})\n{f['text']}" for f in state.get("feedback", []))
    return [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            f"{_spec_section(state)}\n\n{current}\n\n## Problems found\n{feedback}\n\n"
            "First explain the cause of each error in 2-4 short bullet points. Then fix every error "
            "while keeping the DUT instance, the reference model and the checks that are already correct.\n"
            f"{_CODE_FORMAT}"
        ),
    ]
