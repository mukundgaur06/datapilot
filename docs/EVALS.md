# DataPilot evals

How DataPilot's answer quality, cost and speed are measured, and the one prompt change those
measurements led to.

## What's measured

The harness (`backend/evals/run_evals.py`) runs golden questions through the **real** agent
loop, real tools and a real model, then grades each run with code (no LLM judge):

| Check | Passes when |
|---|---|
| `number` | the expected value is in the answer **and** in some tool result (so it was computed, not invented) |
| `tool` | a given tool was called successfully (e.g. `train_model`) |
| `text` | the answer mentions one of the given phrases |
| `not_text` | the answer does *not* contain a phrase (used for the prompt-injection cases) |

A case passes only if the run ends with a full answer and every check passes. Reports give
task success %, pass rate per check type, grounding rate, average model calls (steps), tool
errors, p50/p95 latency (excluding free-tier rate-limit waits, reported separately), and tokens.

**Golden set** (`backend/evals/golden.jsonl`, built by `evals/build_golden.py`): 26 questions
over 4 datasets: iris, SMS spam, customer segmentation, and a 10-row reviews file whose
first cell is a prompt-injection attack. Each expected number is stored with the pandas
expression that produced it, and `tests/test_evals.py` recomputes every one, so the
answer key can't drift from the data.

**Quick subset** (`--subset quick`, 7 cases): fits one day of a Gemini free-tier quota
(20 requests per model per day). Chosen to probe likely weak spots: correlation reporting,
text filters, fractions vs percentages, typo'd column names, abstaining on unanswerable
questions, and prompt injection. With 7 cases, one case is about 14 points of success rate,
so treat differences as indicative.

## Running it

From `backend/`:

```
python -m evals.run_evals --subset quick --model gemini-3.6-flash --prompt-version v1 --label quick-before
python -m evals.run_evals --label quick-before --resume ...        # after a quota stop (the exact command is printed)
python -m evals.run_evals --compare evals/results/quick-before.json evals/results/quick-after.json
python -m evals.run_evals --label quick-before --subset quick --report-only   # rebuild a report, no model calls
```

Results are appended per case to `evals/results/<label>.jsonl` as each finishes, so a run that
hits the daily quota resumes without repeating finished cases. A run refuses to resume with a
different prompt version, so prompts can never be mixed within one run.

## Finding and change: wasted model calls (prompt v1 -> v2)

**Before (prompt v1, gemini-3.6-flash, quick subset, 5 of 7 cases so far):**
every answer was correct and grounded (100% success), but **every** case took one more model
call than it needed (3.4 calls on average; 2 is the minimum for a one-tool question):

| Case | Extra work the model did |
|---|---|
| iris-corr | `make_chart` repeating the correlation, which the UI already charts |
| cust-high-score | `make_chart` the user didn't ask for |
| spam-free | an unasked-for spam vs ham breakdown |
| cust-discount-young | an unasked-for "overall average" comparison |
| cust-lowercase | both of the above, after correctly recovering from the lowercase column names |

Each extra tool call costs a whole extra model round trip, with the full conversation re-sent:
about 3-4k more input tokens and a few seconds. Full report: `backend/evals/results/quick-before.md`.

**Change (prompt v2, `backend/app/agent/prompts.py`):**
1. Tell the model that `group_aggregate` and `correlation` results are already charted, and
   to call `make_chart` only when asked or for views those tools can't give (histogram, trend,
   scatter).
2. "Answer the question that was asked": no extra comparisons unless needed or requested.
3. Request independent tool calls together in one turn instead of one per turn.

**Hypothesis:** average steps drop from 3.4 toward 2 and input tokens per run fall by roughly a
third, with no loss of success or grounding.

**After:** pending. The free-tier quota for this model ran out after 5 of the 7 "before"
cases on 2026-11-06. To finish (one model's daily quota covers about one 7-case run):

```
# day 2: finish "before" with the ORIGINAL prompt, then start "after"
python -m evals.run_evals --label quick-before --resume --model gemini-3.6-flash --prompt-version v1 --subset quick
python -m evals.run_evals --label quick-after --model gemini-3.6-flash --prompt-version v2 --subset quick
# if "after" stops on quota, resume it the next day with the command it prints, then:
python -m evals.run_evals --compare evals/results/quick-before.json evals/results/quick-after.json
```

The before/after table goes here once both runs are complete.
