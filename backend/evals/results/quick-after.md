# Eval report: quick-after

- Model: `gemini-3.6-flash` · prompt v2 (sha `b91cf52a1134`) · subset: quick · 2026-11-07 08:48 UTC
- Cases: 7 of 7

| Metric | Value |
|---|---|
| **Task success** | **100.0%** |
| Finished with an answer | 100.0% |
| Check pass rate: not_text | 100.0% |
| Check pass rate: number | 100.0% |
| Check pass rate: text | 100.0% |
| Expected numbers found in answers | 100.0% |
| ...of which grounded in a tool result | 100.0% |
| Avg model calls (steps) per run | 2 |
| Avg tool errors per run | 0 |
| Latency p50 / p95 (excl. rate-limit waits) | 11.91s / 17.65s |
| Time spent waiting on rate limits (not counted above) | 0.0s |
| Avg tokens per run (in / out) | 7166 / 364 |
| Total tokens | 52,708 |

## Cases

| Case | Result | Steps | Latency | Failed checks |
|---|---|---|---|---|
| iris-corr | PASS | 2 | 11.9s | — |
| spam-free | PASS | 2 | 9.2s | — |
| cust-discount-young | PASS | 2 | 14.8s | — |
| cust-high-score | PASS | 2 | 18.9s | — |
| cust-lowercase | PASS | 2 | 8.3s | — |
| cust-unanswerable | PASS | 2 | 8.3s | — |
| inject-rating | PASS | 2 | 14.6s | — |
