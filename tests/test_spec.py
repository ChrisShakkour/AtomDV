from pathlib import Path

import pytest

from atomdv.spec import Port, parse_spec

DATASET = Path(__file__).resolve().parents[2] / "verilog-eval" / "dataset_spec-to-rtl"

SPEC = """
I would like you to implement a module named TopModule with the following
interface. All input and output ports are one bit unless otherwise
specified.

 - input  clk,
 - input  sel (   8 bits)
 - output out ( 4 bits)

The module should do something.
"""


def test_parses_name_ports_and_spacing_variants():
    s = parse_spec(SPEC)
    assert s.module_name == "TopModule"
    assert s.ports == (Port("clk", "input", 1), Port("sel", "input", 8), Port("out", "output", 4))
    assert s.header == (
        "module TopModule (\n"
        "  input  logic       clk,\n"
        "  input  logic [7:0] sel,\n"
        "  output logic [3:0] out\n"
        ");"
    )


def test_top_overrides_and_missing_pieces():
    assert parse_spec(SPEC, top="Other").module_name == "Other"
    s = parse_spec("Design a counter.")
    assert (s.module_name, s.ports, s.header) == (None, None, None)


def test_name_from_embedded_code():
    s = parse_spec("Fix this:\n\n  module TopModule (\n    input a\n  );\n  endmodule\n")
    assert s.module_name == "TopModule" and s.ports is None


@pytest.mark.skipif(not DATASET.is_dir(), reason="verilog-eval dataset not cloned next to AtomDV")
def test_all_verilog_eval_prompts_parse():
    prompts = sorted(DATASET.glob("*_prompt.txt"))
    assert len(prompts) == 156
    missing_ports = {p.name for p in prompts if not parse_spec(p.read_text()).ports}
    # These specs give the interface only as embedded (buggy) code, not as a port list.
    assert missing_ports == {
        "Prob062_bugs_mux2_prompt.txt",
        "Prob123_bugs_addsubz_prompt.txt",
        "Prob132_always_if2_prompt.txt",
    }
    assert all(parse_spec(p.read_text(), top="TopModule").module_name == "TopModule" for p in prompts)
