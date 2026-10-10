# AtomDV

A verification agent for chip design: it writes a self-checking SystemVerilog testbench from a design
spec. It never sees the design itself. It is built on LangGraph and a local LLM (Ollama) and is the
testbench-side companion of [AtomRTL](https://github.com/ChrisShakkour/AtomRTL), with the same
environment, graph structure and run layout.

## How it works

```
parse_spec → plan → generate → write_tb → check → route ─┬─ no errors / out of budget → finalize
   [py]      [LLM]   [LLM]      [py]      [py]    [py]   ├─ errors → fix [LLM] → write_tb
                                                         └─ same errors twice → generate again
```

- **parse_spec** reads the DUT's module name and ports from the spec and writes two stub DUTs with that
  interface. One drives every output to all 0s (`stub_0.sv`), the other to all 1s (`stub_1.sv`).
- **plan / generate / fix** (LLM, no tools) write a verification plan, then `tb.sv`. The testbench has
  a top `module tb`, one `TopModule dut (...)` instance, its own reference model of the spec, `!==`
  checks, and exactly one `TB_RESULT: PASS` or `TB_RESULT: FAIL errors=<n>` line before `$finish`.
- **check** (Python) runs, in order:
  1. A structure check: `module tb` is present, the DUT is instantiated and not defined, `$finish` and
     the verdict lines are present, the checks use `!==`, nothing waits on a DUT output, and no constructs Icarus can't handle are used.
  2. `iverilog -g2012 -s tb` on the testbench plus a stub.
  3. A pyslang lint. Unconnected or wrongly sized DUT ports and implicit nets are errors.
  4. A DUT-instance check on the elaborated design: every spec port must be connected.
  5. Smoke simulations against both stubs. Each must finish with exactly one verdict (the testbench's own watchdog firing counts as a hang),
     and **at least one must FAIL**. A testbench that passes a DUT with constant outputs checks almost nothing.

The agent never sees the reference design or verilog-eval's testbench. They are used only for scoring.

## Differences from AtomRTL

| | AtomRTL | AtomDV |
|---|---|---|
| Output | `TopModule.sv` (synthesizable RTL) | `tb.sv` (`module tb`, simulation only) |
| System prompt | RTL designer rules | DV engineer rules: reference model, cycle timing (drive on negedge, model on posedge, check on the next negedge), `!==`, exhaustive or directed+random stimulus, verdict line, watchdog, Icarus limits |
| Plan | Registers, equations, FSM | Behaviors to check, reference model, reset, sampling, stimulus, corner cases |
| Compile | `iverilog -t null TopModule.sv` | `iverilog -s tb -t null tb.sv stub_0.sv` |
| Lint errors | Latches, multiple drivers, truncation, … | DUT hookup: unconnected or wrongly sized ports, implicit nets |
| Interface check | Generated module's ports vs the spec | DUT instance connects every spec port, plus the structure check |
| Simulation in the loop | None | Against two constant-output stubs (vacuity, hangs, verdict) |
| Scoring | Hidden test + reference + generated RTL, mismatch count | Generated TB + reference (`RefModule` renamed to `TopModule`), `TB_RESULT` |

## Usage

```bash
# One spec → runs/<timestamp>/<spec-name>/
python -m atomdv.cli --spec ../verilog-eval/dataset_spec-to-rtl/Prob035_count1to10_prompt.txt --top TopModule -v

# verilog-eval benchmark (agent, then the generated testbench is run on the reference design)
python scripts/run_verilog_eval.py --problems 'Prob0[0-2]*'      # or no --problems for all 156
python scripts/run_verilog_eval.py --resume-dir runs/<timestamp>  # finish an interrupted run

# Tests (no LLM needed)
pytest
```

Each problem directory contains:
- `tb.sv`, the generated testbench, and `stub_0.sv` / `stub_1.sv`.
- `result.json`: the agent's outcome, including the smoke-sim status per stub.
- `transcript.log` / `transcript.jsonl`: every LLM call and tool run.
- For benchmark runs: `ref_as_dut.sv`, `validation.json`, `validation.log` and `status.json`.

The run directory has `command.txt`, `config.yaml` and `summary.json`.

### Scoring statuses
| Status | Meaning |
|---|---|
| `pass` | The testbench reports `TB_RESULT: PASS` on the reference design |
| `fail` | The testbench reports FAIL on the reference design: a false alarm, usually a wrong reference model or a timing mistake |
| `compile_error`, `sim_timeout`, `watchdog_timeout`, `no_verdict`, `multiple_verdicts` | The testbench doesn't build or run cleanly against the reference. `watchdog_timeout` means its own watchdog fired |
| `no_generated_tb` | The agent produced no `tb.sv` |

Passing on the reference shows the testbench has no false alarms. It does not show that the testbench
catches bugs: the stub smoke sims in the loop only rule out testbenches that check nothing.
Mutation scoring (planned) would add that.

### Known limitations
- 3 verilog-eval specs (Prob062, Prob123, Prob132) give their interface only as embedded code, not as a
  port list. For those no stubs are generated, so only the structure check runs in the loop.

## Environment

The setup is the same as AtomRTL's: conda (conda-forge) provides Python and iverilog 12.0, and pip
provides the Python packages.

- `environment.yml`: Python version, conda packages, and the pip requirements file
- `requirements.txt`: Python packages, pinned with `==`

```bash
conda env create -f environment.yml
conda activate atomdv
ollama pull devstral-small-2     # see AtomRTL's README for installing Ollama
# Benchmark dataset, cloned next to AtomDV
cd .. && git clone https://github.com/NVlabs/verilog-eval.git && cd AtomDV
```

Scripts start the Ollama server themselves if it isn't running.
