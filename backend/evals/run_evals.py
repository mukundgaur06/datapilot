"""Eval harness: run the golden questions through the real agent and report quality, cost, speed.

    python -m evals.run_evals --label baseline                 (from backend/, needs GEMINI_API_KEY)
    python -m evals.run_evals --label baseline --resume        (continue an interrupted run)
    python -m evals.run_evals --only cust-corr,spam-share      (a subset)
    python -m evals.run_evals --compare evals/results/A.json evals/results/B.json

Each case's result is appended to evals/results/<label>.jsonl the moment it finishes, so a
run that hits the daily quota can be resumed later without repeating (or re-paying for)
finished cases. The report is rebuilt from that file: <label>.md (readable) and
<label>.json (summary, used by --compare).

Metrics: task success %, per-check pass rates, grounding rate (expected numbers that also
appear in a tool result), avg steps / model calls / tool errors, p50/p95 latency, tokens.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.agent.llm import LLM, GeminiLLM, LLMConfigError
from app.agent.loop import run_agent
from app.agent.prompts import SYSTEM_TEMPLATE
from app.agent.retry import DailyQuotaExhausted, RetryingLLM
from app.ml.models import ModelStore
from app.store import DatasetStore
from app.tracing import Tracer
from evals.build_golden import dataset_path
from evals.grading import grade, numbers_in

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden.jsonl"
RESULTS = HERE / "results"


def load_cases(only: list[str] | None = None) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    if only:
        unknown = set(only) - {c["id"] for c in cases}
        if unknown:
            raise SystemExit(f"Unknown case id(s): {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in only]
    return cases


def run_case(case: dict[str, Any], llm: LLM) -> dict[str, Any]:
    """Run one golden case through the real agent loop and real tools; grade it."""
    store = DatasetStore()
    dataset_id = store.load_csv(dataset_path(case["dataset"]))
    tracer = Tracer(f"eval-{case['id']}")
    answer, status, partial, error = "", "error", False, ""
    tools_called, tools_ok, tool_numbers = [], [], []
    for ev in run_agent(case["question"], dataset_id, llm=llm, store=store, models=ModelStore(), tracer=tracer):
        d = ev.data
        if ev.type == "tool_call":
            tools_called.append(d["name"])
        elif ev.type == "tool_result" and not d["is_error"]:
            tools_ok.append(d["name"])
            tool_numbers.extend(numbers_in({k: v for k, v in d["result"].items() if k != "ui"}))
        elif ev.type == "answer":
            answer, status, partial = d["text"], "done", bool(d.get("partial"))
        elif ev.type in ("error", "cancelled"):
            status, error = ev.type, str(d.get("message", ""))
    summary = tracer.summary()
    run = {"answer": answer, "status": status, "partial": partial, "tools_ok": tools_ok, "tool_numbers": tool_numbers}
    return {
        "id": case["id"],
        "dataset": case["dataset"],
        "question": case["question"],
        **grade(case, run),
        "status": status,
        "error": error,
        "answer": answer,
        "tools_called": tools_called,
        "steps": summary["llm_calls"],
        "tool_errors": summary["tool_errors"],
        "latency_ms": summary["total_ms"],
        "llm_ms": summary["llm_ms"],
        "usage": summary["usage"],
    }


# ---------------------------------------------------------------- reporting


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarise(records: list[dict[str, Any]], total_cases: int) -> dict[str, Any]:
    n = len(records)
    checks = [ch for r in records for ch in r["checks"]]
    by_type: dict[str, list[bool]] = {}
    for ch in checks:
        by_type.setdefault(ch["type"], []).append(ch["pass"])
    numbers = [ch for ch in checks if ch["type"] == "number"]
    lat = [r["latency_ms"] / 1000 for r in records]
    tokens_in = [r["usage"].get("input_tokens", 0) for r in records]
    tokens_out = [r["usage"].get("output_tokens", 0) + r["usage"].get("thinking_tokens", 0) for r in records]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    return {
        "cases_run": n,
        "cases_total": total_cases,
        "success_pct": pct([r["pass"] for r in records]),
        "finished_pct": pct([r["finished"] for r in records]),
        "check_pass_pct": {k: pct(v) for k, v in sorted(by_type.items())},
        "number_found_pct": pct([c["found"] for c in numbers]),
        "grounded_pct": pct([c["grounded"] for c in numbers if c["found"]]),
        "avg_steps": round(statistics.mean(r["steps"] for r in records), 2) if n else None,
        "avg_tool_errors": round(statistics.mean(r["tool_errors"] for r in records), 2) if n else None,
        "latency_p50_s": round(percentile(lat, 0.5), 2),
        "latency_p95_s": round(percentile(lat, 0.95), 2),
        "avg_input_tokens": round(statistics.mean(tokens_in)) if n else None,
        "avg_output_tokens": round(statistics.mean(tokens_out)) if n else None,
        "total_tokens": sum(tokens_in) + sum(tokens_out),
    }


def write_report(label: str, meta: dict[str, Any], records: list[dict[str, Any]], total_cases: int) -> tuple[Path, Path]:
    summary = summarise(records, total_cases)
    json_path = RESULTS / f"{label}.json"
    json_path.write_text(json.dumps({"label": label, **meta, "summary": summary}, indent=2), encoding="utf-8")

    s = summary
    lines = [
        f"# Eval report: {label}",
        "",
        f"- Model: `{meta['model']}` · prompt sha `{meta['prompt_sha']}` · {meta['updated']}",
        f"- Cases: {s['cases_run']} of {s['cases_total']}" + (" (incomplete: resume to finish)" if s["cases_run"] < s["cases_total"] else ""),
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Task success** | **{s['success_pct']}%** |",
        f"| Finished with an answer | {s['finished_pct']}% |",
        *[f"| Check pass rate: {k} | {v}% |" for k, v in s["check_pass_pct"].items()],
        f"| Expected numbers found in answers | {s['number_found_pct']}% |",
        f"| ...of which grounded in a tool result | {s['grounded_pct']}% |",
        f"| Avg model calls (steps) per run | {s['avg_steps']} |",
        f"| Avg tool errors per run | {s['avg_tool_errors']} |",
        f"| Latency p50 / p95 | {s['latency_p50_s']}s / {s['latency_p95_s']}s |",
        f"| Avg tokens per run (in / out) | {s['avg_input_tokens']} / {s['avg_output_tokens']} |",
        f"| Total tokens | {s['total_tokens']:,} |",
        "",
        "## Cases",
        "",
        "| Case | Result | Steps | Latency | Failed checks |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        failed = [f"{c['type']}={c['expect']}" + (" (ungrounded)" if c["type"] == "number" and c["found"] and not c["grounded"] else "") for c in r["checks"] if not c["pass"]]
        if not r["finished"]:
            failed.insert(0, f"run {r['status']}: {r['error'][:80]}")
        lines.append(f"| {r['id']} | {'PASS' if r['pass'] else 'FAIL'} | {r['steps']} | {r['latency_ms'] / 1000:.1f}s | {'; '.join(failed) or '—'} |")
    md_path = RESULTS / f"{label}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def compare(a_path: Path, b_path: Path) -> str:
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (a_path, b_path))
    sa, sb = a["summary"], b["summary"]
    rows = [
        ("Task success %", "success_pct"), ("Finished %", "finished_pct"), ("Numbers found %", "number_found_pct"),
        ("Grounded %", "grounded_pct"), ("Avg steps", "avg_steps"), ("Avg tool errors", "avg_tool_errors"),
        ("Latency p50 (s)", "latency_p50_s"), ("Latency p95 (s)", "latency_p95_s"),
        ("Avg input tokens", "avg_input_tokens"), ("Avg output tokens", "avg_output_tokens"),
    ]
    out = [f"| Metric | {a['label']} | {b['label']} | Change |", "|---|---|---|---|"]
    for name, key in rows:
        va, vb = sa.get(key), sb.get(key)
        delta = f"{vb - va:+.2f}" if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else ""
        out.append(f"| {name} | {va} | {vb} | {delta} |")
    for k in sorted(set(sa["check_pass_pct"]) | set(sb["check_pass_pct"])):
        va, vb = sa["check_pass_pct"].get(k), sb["check_pass_pct"].get(k)
        delta = f"{vb - va:+.1f}" if va is not None and vb is not None else ""
        out.append(f"| Check pass: {k} % | {va} | {vb} | {delta} |")
    return "\n".join(out)


# ---------------------------------------------------------------- main


def prompt_sha() -> str:
    import hashlib

    return hashlib.sha256(SYSTEM_TEMPLATE.encode()).hexdigest()[:12]


def run(label: str, llm: LLM, model_name: str, only: list[str] | None, resume: bool, limit: int | None,
        pause_s: float = 0.0, log: Callable[[str], Any] = print) -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = load_cases(only)
    log_path = RESULTS / f"{label}.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if log_path.exists():
        if not resume:
            raise SystemExit(f"{log_path} exists. Pass --resume to continue it, or use a new --label.")
        for line in log_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("status") != "quota":  # finished cases are kept; quota-aborted ones are retried
                done[rec["id"]] = rec
    todo = [c for c in cases if c["id"] not in done][: limit or None]
    log(f"{len(done)} cases already done, running {len(todo)} with {model_name}")

    stopped = False
    with log_path.open("a", encoding="utf-8") as f:
        for i, case in enumerate(todo, 1):
            try:
                rec = run_case(case, llm)
            except DailyQuotaExhausted as e:
                log(f"[quota] {e}\nStopped. Resume later with: python -m evals.run_evals --label {label} --resume")
                stopped = True
                break
            done[case["id"]] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            log(f"[{i}/{len(todo)}] {'PASS' if rec['pass'] else 'FAIL'} {case['id']} ({rec['steps']} steps, {rec['latency_ms'] / 1000:.1f}s)")
            if pause_s and i < len(todo):
                time.sleep(pause_s)  # stay under requests-per-minute limits

    records = [done[c["id"]] for c in cases if c["id"] in done]
    meta = {"model": model_name, "prompt_sha": prompt_sha(), "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    md, js = write_report(label, meta, records, len(cases))
    log(f"report: {md}\nsummary: {js}")
    return 2 if stopped else 0


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Run DataPilot's golden-set evals.")
    p.add_argument("--label", help="Name for this run (results/<label>.*). Default: <date>-<model>.")
    p.add_argument("--model", help="Override DATAPILOT_MODEL")
    p.add_argument("--only", help="Comma-separated case ids")
    p.add_argument("--limit", type=int, help="Run at most N not-yet-done cases")
    p.add_argument("--resume", action="store_true", help="Continue an existing run with the same label")
    p.add_argument("--pause", type=float, default=0.0, help="Seconds to wait between cases (rate limits)")
    p.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"), help="Compare two run summaries")
    args = p.parse_args(argv)

    if args.compare:
        table = compare(Path(args.compare[0]), Path(args.compare[1]))
        print(table)
        return 0
    try:
        gemini = GeminiLLM(model=args.model)
    except LLMConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    label = args.label or f"{datetime.now(timezone.utc):%Y-%m-%d}-{gemini.model}"
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    return run(label, RetryingLLM(gemini), gemini.model, only, args.resume, args.limit, args.pause)


if __name__ == "__main__":
    raise SystemExit(main())
