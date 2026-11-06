"""Ask DataPilot a question from the terminal.

    python -m app.cli ../sample_data/sms_spam.csv "What share of messages are spam?"
"""

from __future__ import annotations

import argparse
import json
import sys

from app.agent.llm import GeminiLLM, LLMConfigError
from app.agent.loop import run_agent
from app.store import InvalidDataset, store


def _preview(obj: object, limit: int = 300) -> str:
    text = json.dumps(obj, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " …"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description="Ask DataPilot a question about a CSV file.")
    parser.add_argument("csv", help="Path to a CSV file")
    parser.add_argument("question", help="Question in plain English (wrap it in quotes)")
    parser.add_argument("--model", help="Override DATAPILOT_MODEL")
    parser.add_argument("--max-steps", type=int, help="Override DATAPILOT_MAX_STEPS")
    parser.add_argument("--hide-thinking", action="store_true", help="Don't print thought summaries")
    args = parser.parse_args(argv)

    try:
        llm = GeminiLLM(model=args.model)
        dataset_id = store.load_csv(args.csv)
    except (LLMConfigError, InvalidDataset, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    summary = store.schema_summary(dataset_id)
    print(f"Loaded {summary['name']}: {summary['n_rows']} rows x {summary['n_columns']} columns | model: {llm.model}\n")

    exit_code = 0
    for event in run_agent(args.question, dataset_id, llm=llm, max_steps=args.max_steps):
        d = event.data
        if event.type == "thinking" and not args.hide_thinking:
            print(f"[step {event.step}] thinking: {d['text'].strip()}\n")
        elif event.type == "tool_call":
            print(f"[step {event.step}] -> {d['name']}({_preview(d['args'], 200)})")
        elif event.type == "tool_result":
            status = "ERROR" if d["is_error"] else "ok"
            print(f"[step {event.step}] <- {status} in {d['duration_ms']} ms: {_preview(d['result'])}\n")
        elif event.type in ("answer", "error"):
            label = "ANSWER" if event.type == "answer" else "FAILED"
            body = d.get("text") or d.get("message")
            print("=" * 72 + f"\n{label}\n" + "=" * 72 + f"\n{body}\n")
            print(f"steps={d.get('steps')} elapsed={d.get('elapsed_ms')} ms tokens={d.get('usage')}")
            exit_code = 0 if event.type == "answer" else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
