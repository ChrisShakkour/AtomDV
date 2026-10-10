"""Run the testbench-generation agent on one spec.

    python -m atomdv.cli --spec path/to/spec.txt [--top TopModule] [--config cfg.yaml] [--model NAME]
                          [--workdir DIR] [--no-server] [-v]

Creates runs/<timestamp>/<spec-name>/ with the generated tb.sv, result.json and the full transcript.
Starts the Ollama server if it isn't running (and stops it afterwards), unless --no-server.
Exit code: 0 = pass (the agent's own checks), 1 = finished without a passing testbench, 2 = crashed.
"""

import argparse
import shlex
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from atomdv.config import AgentConfig, config_to_dict, load_config
from atomdv.graph import build_graph
from atomdv.llm import ollama_server
from atomdv.state import RunContext
from atomdv.transcript import Transcript, TranscriptCallback


def new_run_dir(runs_dir: Path) -> Path:
    run_dir = runs_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_info(run_dir: Path, cfg: AgentConfig) -> None:
    (run_dir / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))


def run_agent(spec: str, workdir: Path, cfg: AgentConfig, top: str | None = None, verbose: bool = False) -> dict:
    """Run the graph on one spec inside `workdir` (expects a running Ollama server). Returns the final state."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "prompt.txt").write_text(spec)
    transcript = Transcript(workdir, echo=verbose)
    start = time.monotonic()
    try:
        return build_graph().invoke(
            {"spec": spec, "top": top},
            config={
                "callbacks": [TranscriptCallback(transcript)],
                # ~4 graph steps per attempt (fix, write_code, check, route) plus setup.
                "recursion_limit": 10 + 5 * cfg.max_attempts,
            },
            context=RunContext(config=cfg, workdir=workdir, transcript=transcript),
        )
    except Exception as e:
        transcript.event("error", "agent", title=repr(e))
        raise
    finally:
        transcript.event("state", "agent", title="run finished", duration_s=time.monotonic() - start)
        transcript.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, type=Path, help="spec text file")
    ap.add_argument("--top", help="DUT module name the testbench must instantiate (default: taken from the spec)")
    ap.add_argument("--config", type=Path, help="YAML config (see atomdv/config.py)")
    ap.add_argument("--model", help="override the Ollama model")
    ap.add_argument("--runs-dir", type=Path, help="where to create the run directory (default: runs/)")
    ap.add_argument("--workdir", type=Path, help="run directly in this directory instead of a new runs/<timestamp>/")
    ap.add_argument("--no-server", action="store_true",
                    help="only connect to an already-running Ollama server; never start or stop one "
                         "(used by the benchmark runner, which manages the shared server)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each transcript event live")
    args = ap.parse_args()

    cfg = load_config(args.config, runs_dir=args.runs_dir)
    if args.model:
        cfg = replace(cfg, ollama=replace(cfg.ollama, model=args.model))

    if args.workdir:
        workdir = args.workdir
    else:
        run_dir = new_run_dir(cfg.runs_dir)
        write_run_info(run_dir, cfg)
        workdir = run_dir / args.spec.name.removesuffix(".txt").removesuffix("_prompt")

    try:
        with nullcontext() if args.no_server else ollama_server(cfg.ollama):
            final = run_agent(args.spec.read_text(), workdir, cfg, top=args.top, verbose=args.verbose)
    except Exception:
        traceback.print_exc()
        sys.exit(2)  # crashed, distinct from 1 = finished without a passing testbench

    print(f"\n{final['status'].upper()}: {final.get('route_reason')}")
    print(f"testbench:  {final.get('code_path')}")
    print(f"transcript: {workdir / 'transcript.log'}")
    sys.exit(0 if final["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
