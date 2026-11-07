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
python -m evals.run_evals --subset quick --model gemini-3.6-flash --prompt-version v2 --label quick-after
python -m evals.run_evals --label quick-before --resume ...        # after a quota stop (the exact command is printed)
python -m evals.run_evals --compare evals/results/quick-before.json evals/results/quick-after.json
python -m evals.run_evals --label quick-before --subset quick --report-only   # rebuild a report, no model calls
```

Results are appended per case to `evals/results/<label>.jsonl` as each finishes, so a run that
hits the daily quota resumes without repeating finished cases. A run refuses to resume with a
different prompt version, so prompts can never be mixed within one run.

## Finding and change: wasted model calls (prompt v1 -> v2)

**Before (prompt v1, gemini-3.6-flash, quick subset):**
every answer was correct and grounded (7/7), but 6 of the 7 cases made model calls they
didn't need (3.0 calls on average; 2 is the minimum for a one-tool question):

| Case | Extra work the model did |
|---|---|
| iris-corr | `make_chart` repeating the correlation, which the UI already charts |
| cust-high-score | `make_chart` the user didn't ask for |
| spam-free | an unasked-for spam vs ham breakdown |
| cust-discount-young | an unasked-for "overall average" comparison |
| cust-lowercase | a comparison group, a chart and another breakdown (5 calls in all) |
| inject-rating | two extra breakdowns of the ratings |
| cust-unanswerable | none: answered in 1 call that the data has no email column |

Each extra tool call costs a whole extra model round trip, with the full conversation re-sent:
about 3-4k more input tokens and a few seconds. Full report: `backend/evals/results/quick-before.md`.

**Change (prompt v2, `backend/app/agent/prompts.py`):**
1. Tell the model that `group_aggregate` and `correlation` results are already charted, and
   to call `make_chart` only when asked or for views those tools can't give (histogram, trend,
   scatter).
2. "Answer the question that was asked": no extra comparisons unless needed or requested.
3. Request independent tool calls together in one turn instead of one per turn.

**Hypothesis (written before the "after" run):** average steps drop toward 2 and input tokens
per run fall by roughly a third, with no loss of success or grounding.

**Result (same model, same 7 cases; `python -m evals.run_evals --compare ...`):**

| Metric | Before (v1) | After (v2) | Change |
|---|---|---|---|
| Task success | 7/7 (100%) | 7/7 (100%) | no change |
| Expected numbers found and grounded in a tool result | 100% | 100% | no change |
| Avg model calls per question | 3.0 | 2.0 | **−33%** |
| Avg input tokens per question | 11,110 | 7,166 | **−35%** |
| Avg output tokens per question | 468 | 364 | −22% |
| Latency p50 / p95 (excl. rate-limit waits) | 11.9s / 24.1s | 11.9s / 17.7s | p95 −27% |

Per case, v2 took exactly 2 calls everywhere. The biggest saving was cust-lowercase (5 calls,
22.5k input tokens down to 2 calls, 7.7k). The one case that got more expensive was
cust-unanswerable: 1 call before, 2 after, because v2 first checked the dataset profile
before saying there is no email column. That's a defensible check, but it is an extra call.

**Verdict:** the hypothesis held. v2 answers the same questions just as correctly with a third
fewer model calls and about 35% fewer input tokens, so it ships as the default prompt
(`LATEST = "v2"`). Set `DATAPILOT_PROMPT_VERSION=v1` to go back.

**Limits of this measurement:** 7 questions on one model (gemini-3.6-flash), one run each, so
the success rate can't distinguish small differences (one case = 14 points). The efficiency
gain is consistent across cases, which makes it more trustworthy than the sample size alone
suggests. "Before" ran over two days because of the free-tier quota (cases 1-5 on 2026-11-06,
6-7 on 2026-11-07); same model and same prompt throughout, which the harness enforces.
Latencies for two "before" cases were corrected for rate-limit waits using the run log (noted
in those records). Reports: `backend/evals/results/quick-before.md`, `quick-after.md`.
