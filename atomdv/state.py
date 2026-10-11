"""Graph state and run context for the testbench-generation agent."""

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

from atomdv.config import AgentConfig
from atomdv.eda import Diagnostic
from atomdv.spec import Port
from atomdv.transcript import Transcript


class Feedback(TypedDict):
    """One piece of feedback for the next fix: from static checks now, from LLM reviewers later."""

    source: str  # e.g. "check"
    kind: str  # e.g. "diagnostics"
    text: str


class AttemptRecord(TypedDict):
    attempt: int
    node: str  # generate | fix
    signature: str  # error signature ("" when clean)
    errors: int
    warnings: int


def _add_tokens(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in a.keys() | b.keys()}


class TBState(TypedDict, total=False):
    # Input
    spec: str
    top: str | None
    # parse_spec
    module_name: str | None
    ports: tuple[Port, ...] | None
    header: str | None
    stub_paths: dict[int, str] | None  # constant-output DUT stubs (value 0 / 1) written by parse_spec
    # LLM nodes
    plan: str
    response: str  # last raw generate/fix response
    response_node: str  # which node produced `response`: generate | fix
    restart_hint: str | None  # set by route when a fresh generate replaces a stuck fix loop
    # write_tb / check
    code: str | None  # the testbench source
    code_path: str | None
    code_unchanged: bool  # the new testbench is identical to the previous attempt's
    smoke: dict[str, str]  # smoke-sim status per stub (e.g. {"stub_0": "fail"}) for the current attempt
    diagnostics: list[Diagnostic]
    feedback: list[Feedback]  # for the current attempt; the fix node renders all of it
    attempts: int
    history: Annotated[list[AttemptRecord], operator.add]
    # route / finalize
    route: str
    route_reason: str
    status: str  # pass | fail
    tokens: Annotated[dict, _add_tokens]


@dataclass
class RunContext:
    """Run-scoped, non-state inputs available to every node via `runtime.context`."""

    config: AgentConfig
    workdir: Path
    transcript: Transcript
