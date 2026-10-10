"""Parse a design spec into a module name, a port list and a header skeleton (no LLM).

Understands verilog-eval's format:
    I would like you to implement a module named TopModule with the following interface. ...
     - input  clk
     - input  in (8 bits)
     - output out (8 bits)
Other specs fall back gracefully: missing pieces are None and the checks that need them are skipped.
"""

import re
from dataclasses import dataclass

_PORT_RE = re.compile(
    r"^\s*-\s*(input|output|inout)\s+([A-Za-z_]\w*)\s*,?\s*(?:\(\s*(\d+)\s*bits?\s*\))?\s*,?\s*$",
    re.MULTILINE,
)
_NAMED_RE = re.compile(r"\bmodule\s+named\s+([A-Za-z_]\w*)")
_CODE_MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_]\w*)\s*[(#;]", re.MULTILINE)


@dataclass(frozen=True)
class Port:
    name: str
    direction: str  # "input" | "output" | "inout"
    width: int = 1


@dataclass(frozen=True)
class Spec:
    text: str
    module_name: str | None
    ports: tuple[Port, ...] | None
    header: str | None  # SystemVerilog module header skeleton, when name and ports are known


def parse_ports(text: str) -> tuple[Port, ...] | None:
    ports = tuple(Port(name, d, int(w) if w else 1) for d, name, w in _PORT_RE.findall(text))
    return ports or None


def parse_module_name(text: str) -> str | None:
    flat = " ".join(text.split())  # the name is sometimes split across lines
    if m := _NAMED_RE.search(flat):
        return m.group(1)
    # Specs that embed existing code (e.g. "fix the bug in this module") name it there.
    if m := _CODE_MODULE_RE.search(text):
        return m.group(1)
    return None


def render_header(module_name: str, ports: tuple[Port, ...]) -> str:
    rows = []
    for p in ports:
        rng = f"[{p.width - 1}:0]" if p.width > 1 else ""
        rows.append((p.direction, rng, p.name))
    dir_w = max(len(r[0]) for r in rows)
    rng_w = max(len(r[1]) for r in rows)
    lines = [f"  {d:<{dir_w}} logic {r:<{rng_w}}{' ' if rng_w else ''}{n}" for d, r, n in rows]
    return f"module {module_name} (\n" + ",\n".join(lines) + "\n);"


def parse_spec(text: str, top: str | None = None) -> Spec:
    """`top` forces the module name (e.g. verilog-eval's testbenches always instantiate TopModule)."""
    module_name = top or parse_module_name(text)
    ports = parse_ports(text)
    header = render_header(module_name, ports) if module_name and ports else None
    return Spec(text=text, module_name=module_name, ports=ports, header=header)
