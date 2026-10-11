"""Shared settings. Every agent and server launch reads these, so they always agree."""

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class OllamaConfig:
    model: str = "devstral-small-2"
    # Must be identical for every client: a different num_ctx forces Ollama to reload the model.
    num_ctx: int = 16384
    # host:port, overridable with OLLAMA_HOST (e.g. to use a server on another node).
    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"))
    # Concurrent requests the server handles; each slot needs its own num_ctx of GPU memory.
    num_parallel: int = 2
    # -1 keeps the model loaded until the server stops (default would unload after 5 min idle).
    keep_alive: str = "-1"
    log_file: Path = REPO_ROOT / "logs" / "ollama.log"
    # Seconds to wait for a newly started server to answer. It doesn't answer until GPU discovery
    # finishes, which has taken 13-30s on the 3090 machine.
    startup_timeout: float = 120

    @property
    def url(self) -> str:
        return self.host if "://" in self.host else f"http://{self.host}"


DEFAULT = OllamaConfig()


@dataclass(frozen=True)
class NodeLLMConfig:
    """Per-node LLM overrides; None means use the default (model from OllamaConfig)."""

    model: str | None = None
    temperature: float | None = None
    # Max tokens to generate (Ollama num_predict); None = model default.
    max_tokens: int | None = None


@dataclass(frozen=True)
class AgentConfig:
    ollama: OllamaConfig = DEFAULT
    # Default sampling for every LLM node, overridable per node in `nodes`.
    temperature: float = 0.0
    nodes: dict[str, NodeLLMConfig] = field(default_factory=dict)
    # Total testbench attempts (first generate + fixes + restarts) before giving up.
    max_attempts: int = 6
    # Restart from a fresh generate when the same error signature repeats this many times in a row.
    stuck_after: int = 2
    # slang warnings treated as errors (must be fixed before the testbench passes): ones that almost always
    # mean a broken DUT hookup or a typo and have a mechanical fix (see FIX_HINTS in eda.py). Unconnected /
    # wrongly sized DUT ports, typo'd names (implicit nets), value-changing literals and dead case items.
    # The RTL-only codes AtomRTL blocks on (InferredLatch, MultipleAlwaysAssigns, ...) don't apply to a
    # testbench.
    blocking_warnings: tuple[str, ...] = (
        "UnconnectedInputPort", "PortWidthExpand", "PortWidthTruncate", "ImplicitNet",
        "VectorLiteralOverflow", "CaseDup",
    )
    # Max diagnostics shown to the LLM per attempt (the transcript always keeps everything).
    max_diagnostics: int = 8
    # Per-command timeout for EDA tools (compile), seconds.
    eda_timeout: float = 30
    # Per-run timeout for the smoke simulations against the stub DUTs, seconds.
    sim_timeout: float = 30
    # Require the testbench to FAIL on at least one constant-output stub DUT (all 0s / all 1s).
    vacuity_check: bool = True
    runs_dir: Path = REPO_ROOT / "runs"

    def node_llm(self, node: str) -> NodeLLMConfig:
        """Resolved LLM settings for a node: per-node overrides on top of the defaults."""
        o = self.nodes.get(node, NodeLLMConfig())
        return NodeLLMConfig(
            model=o.model or self.ollama.model,
            temperature=self.temperature if o.temperature is None else o.temperature,
            max_tokens=o.max_tokens,
        )


def load_config(path: str | Path | None = None, **overrides) -> AgentConfig:
    """AgentConfig from an optional YAML file (same keys as the dataclasses), then keyword overrides.

    YAML example:
        ollama: {model: devstral-small-2, num_ctx: 16384}
        temperature: 0.0
        max_attempts: 6
        nodes:
          plan: {temperature: 0.3}
    """
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    data = (data or {}) | {k: v for k, v in overrides.items() if v is not None}

    known = {f.name for f in dataclasses.fields(AgentConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    if "ollama" in data:
        data["ollama"] = OllamaConfig(**data["ollama"])
    if "nodes" in data:
        data["nodes"] = {name: NodeLLMConfig(**cfg) for name, cfg in data["nodes"].items()}
    if "blocking_warnings" in data:
        data["blocking_warnings"] = tuple(data["blocking_warnings"])
    if "runs_dir" in data:
        data["runs_dir"] = Path(data["runs_dir"])
    return AgentConfig(**data)


def config_to_dict(cfg: AgentConfig) -> dict:
    """Plain dict for writing the resolved config next to a run."""

    def convert(v):
        if dataclasses.is_dataclass(v):
            return {f.name: convert(getattr(v, f.name)) for f in dataclasses.fields(v)}
        if isinstance(v, dict):
            return {k: convert(x) for k, x in v.items()}
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, tuple):
            return [convert(x) for x in v]
        return v

    return convert(cfg)
