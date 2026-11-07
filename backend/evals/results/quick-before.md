# Eval report: quick-before

- Model: `gemini-3.6-flash` · prompt v1 (sha `47bc317fc0dd`) · subset: quick · 2026-11-07 08:46 UTC
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
| Avg model calls (steps) per run | 3 |
| Avg tool errors per run | 0 |
| Latency p50 / p95 (excl. rate-limit waits) | 11.89s / 24.08s |
| Time spent waiting on rate limits (not counted above) | 80.0s |
| Avg tokens per run (in / out) | 11110 / 468 |
| Total tokens | 81,045 |

## Cases

| Case | Result | Steps | Latency | Failed checks |
|---|---|---|---|---|
| iris-corr | PASS | 3 | 10.5s | — |
| spam-free | PASS | 3 | 5.2s | — |
| cust-discount-young | PASS | 3 | 12.5s | — |
| cust-high-score | PASS | 3 | 6.4s | — |
| cust-lowercase | PASS | 5 | 24.2s | — |
| cust-unanswerable | PASS | 1 | 11.9s | — |
| inject-rating | PASS | 3 | 23.9s | — |
