"""Run the agent on verilog-eval spec-to-rtl problems and score each generated testbench against the
reference design.

    python scripts/run_verilog_eval.py [--problems 'Prob0[0-2]*' ...] [--limit N] [--jobs N]
                                       [--config cfg.yaml] [--model NAME] [--resume-dir runs/<ts>]

Creates runs/<timestamp>/ (see README) with one directory per problem. Each problem runs as its own
`python -m atomdv.cli` subprocess (hard timeout, isolation) against one shared Ollama server.
The agent never sees the reference design or verilog-eval's testbench: scoring happens only after it
finishes. Scoring compiles the generated tb.sv with the problem's RefModule renamed to TopModule and
simulates it: the testbench passes when it reports TB_RESULT: PASS for the (correct) reference design.
"""

import argparse
import fnmatch
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from atomdv.cli import new_run_dir, write_run_info
from atomdv.config import REPO_ROOT, load_config
from atomdv.eda import TB_TOP, run_tb
from atomdv.llm import ServerKeeper, ollama_server

DEFAULT_DATASET = REPO_ROOT.parent / "verilog-eval" / "dataset_spec-to-rtl"
REF_MODULE_RE = re.compile(r"^(\s*module\s+)RefModule\b", re.MULTILINE)
TB_FILE = f"{TB_TOP}.sv"


def find_problems(dataset: Path, patterns: list[str] | None, limit: int | None) -> list[str]:
    names = sorted(p.name.removesuffix("_prompt.txt") for p in dataset.glob("*_prompt.txt"))
    if patterns:
        names = [n for n in names if any(fnmatch.fnmatch(n, pat) or n == pat for pat in patterns)]
    return names[:limit] if limit else names


def run_agent_subprocess(problem: str, dataset: Path, workdir: Path, config_path: Path, timeout: float) -> dict:
    cmd = [sys.executable, "-m", "atomdv.cli", "--spec", str(dataset / f"{problem}_prompt.txt"),
           "--top", "TopModule", "--config", str(config_path), "--workdir", str(workdir), "--no-server"]
    start = time.monotonic()
    workdir.mkdir(parents=True, exist_ok=True)
    with open(workdir / "agent.log", "w") as log:
        log.write(f"$ {shlex.join(cmd)}\n")
        log.flush()
        try:
            p = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            # cli exit codes: 0 = pass, 1 = finished without a passing design, 2 = crashed
            agent_status = "completed" if p.returncode in (0, 1) else "error"
        except subprocess.TimeoutExpired:
            agent_status = "timeout"
    return {"agent_status": agent_status, "agent_duration_s": round(time.monotonic() - start, 1)}


def score(problem: str, dataset: Path, workdir: Path, timeout: float = 60) -> dict:
    """Simulate the generated testbench against the reference design (after the agent is done).

    status: pass | fail (a false alarm on a correct design) | compile_error | sim_timeout |
    watchdog_timeout (the testbench's own watchdog fired) | no_verdict | multiple_verdicts | no_generated_tb
    """
    tb = workdir / TB_FILE
    if not tb.exists():
        return {"status": "no_generated_tb"}
    ref = REF_MODULE_RE.sub(r"\1TopModule", (dataset / f"{problem}_ref.sv").read_text(), count=1)
    (workdir / "ref_as_dut.sv").write_text(ref)

    r = run_tb(Path(TB_FILE), [Path("ref_as_dut.sv")], workdir, timeout, tag="validate")
    log = [f"$ {r.compile.cmdline}  -> exit {r.compile.returncode}\n{r.compile.output}"]
    if r.run:
        log.append(f"$ {r.run.cmdline}  -> exit {r.run.returncode}{' (timed out)' if r.run.timed_out else ''}\n"
                   f"{r.run.output}")
    result = {"status": r.status}
    if m := re.search(r"TB_RESULT:\s*FAIL\s+errors\s*=\s*(\d+)", r.run.output if r.run else ""):
        result["errors"] = int(m[1])
    (workdir / "validation.log").write_text("\n\n".join(log))
    (workdir / "validation.json").write_text(json.dumps(result, indent=2))
    return result


def run_problem(problem: str, dataset: Path, run_dir: Path, config_path: Path, timeout: float,
                keeper: ServerKeeper, retries: int = 1) -> dict:
    """Agent then scoring. A crashed agent (e.g. the LLM server went away) is retried from scratch."""
    workdir = run_dir / problem
    crashes = []
    for _ in range(retries + 1):
        if workdir.exists():
            shutil.rmtree(workdir)  # no leftovers from a crashed or interrupted attempt
        keeper.ensure()  # restart the shared server if it went away
        info = {"problem": problem, **run_agent_subprocess(problem, dataset, workdir, config_path, timeout)}
        if info["agent_status"] != "error":
            break
        crashes.append((workdir / "agent.log").read_text().strip().splitlines()[-1])
    if crashes:
        info["crashes"] = crashes

    result_file = workdir / "result.json"
    if result_file.exists():
        r = json.loads(result_file.read_text())
        info |= {"agent_result": r["status"], "attempts": r["attempts"], "tokens": r.get("tokens", {})}
    info |= score(problem, dataset, workdir)
    (workdir / "status.json").write_text(json.dumps(info, indent=2))
    return info


def is_done(workdir: Path) -> bool:
    """Scored, and the agent didn't crash (crashed problems are rerun on --resume-dir)."""
    status = workdir / "status.json"
    return status.exists() and ((workdir / "result.json").exists() or json.loads(status.read_text())["agent_status"] == "timeout")


def summarize(run_dir: Path, rows: list[dict]) -> dict:
    by_status = Counter(r["status"] for r in rows)
    tokens = Counter()
    for r in rows:
        tokens.update(r.get("tokens", {}))
    summary = {
        "total": len(rows),
        "passed": by_status["pass"],
        "pass_rate": round(by_status["pass"] / len(rows), 3) if rows else 0,
        "by_status": dict(by_status),
        "agent_self_check_pass": sum(r.get("agent_result") == "pass" for r in rows),
        "avg_attempts": round(sum(r.get("attempts", 0) for r in rows) / len(rows), 2) if rows else 0,
        "avg_agent_duration_s": round(sum(r["agent_duration_s"] for r in rows) / len(rows), 1) if rows else 0,
        "tokens": dict(tokens),
        "problems": sorted(rows, key=lambda r: r["problem"]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--problems", nargs="+", help="problem names or globs (default: all)")
    ap.add_argument("--limit", type=int, help="run at most N problems")
    ap.add_argument("--config", type=Path, help="YAML config (see atomdv/config.py)")
    ap.add_argument("--model", help="override the Ollama model")
    ap.add_argument("--jobs", type=int, help="problems run in parallel (default: the server's num_parallel)")
    ap.add_argument("--timeout", type=float, default=900, help="per-problem agent timeout, seconds (default 900)")
    ap.add_argument("--resume-dir", type=Path,
                    help="finish an interrupted run: skips problems already scored, reruns crashed ones")
    args = ap.parse_args()

    if not args.dataset_dir.is_dir():
        sys.exit(f"dataset not found: {args.dataset_dir} (clone verilog-eval next to AtomDV, or pass --dataset-dir)")

    if args.resume_dir:
        run_dir = args.resume_dir
        cfg = load_config(run_dir / "config.yaml")
    else:
        cfg = load_config(args.config)
        if args.model:
            cfg = replace(cfg, ollama=replace(cfg.ollama, model=args.model))
        run_dir = new_run_dir(cfg.runs_dir)
        write_run_info(run_dir, cfg)
    config_path = run_dir / "config.yaml"  # every problem uses exactly this resolved config

    problems = find_problems(args.dataset_dir, args.problems, args.limit)
    todo = [p for p in problems if not is_done(run_dir / p)]
    jobs = args.jobs or cfg.ollama.num_parallel
    print(f"run dir: {run_dir}\nmodel: {cfg.ollama.model}  problems: {len(todo)} to run "
          f"({len(problems) - len(todo)} already done)  jobs: {jobs}\n", flush=True)

    with ollama_server(cfg.ollama) as keeper:
        with ThreadPoolExecutor(jobs) as pool:
            futures = {pool.submit(run_problem, p, args.dataset_dir, run_dir, config_path, args.timeout, keeper): p
                       for p in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                extra = f"{r['errors']} errors reported" if "errors" in r else ""
                print(f"[{i:3d}/{len(todo)}] {r['problem']:<32} {r['status']:<18} agent={r.get('agent_result', r['agent_status']):<8} "
                      f"attempts={r.get('attempts', '-')}  {r['agent_duration_s']:6.1f}s  {extra}", flush=True)

    rows = [json.loads(f.read_text()) for p in problems if (f := run_dir / p / "status.json").exists()]
    s = summarize(run_dir, rows)
    print(f"\npass rate: {s['passed']}/{s['total']} = {s['pass_rate']:.1%}   by status: {s['by_status']}")
    print(f"agent self-check pass (compile+lint+structure+stub sims): {s['agent_self_check_pass']}/{s['total']}   "
          f"avg attempts: {s['avg_attempts']}")
    print(f"summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
