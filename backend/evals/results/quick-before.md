# Eval report: quick-before

- Model: `gemini-3.6-flash` · prompt v1 (sha `47bc317fc0dd`) · subset: quick · 2026-11-06 09:38 UTC
- Cases: 5 of 7 (incomplete: resume to finish)

| Metric | Value |
|---|---|
| **Task success** | **100.0%** |
| Finished with an answer | 100.0% |
| Check pass rate: number | 100.0% |
| Check pass rate: text | 100.0% |
| Expected numbers found in answers | 100.0% |
| ...of which grounded in a tool result | 100.0% |
| Avg model calls (steps) per run | 3.4 |
| Avg tool errors per run | 0 |
| Latency p50 / p95 (excl. rate-limit waits) | 10.54s / 21.82s |
| Time spent waiting on rate limits (not counted above) | 80.0s |
| Avg tokens per run (in / out) | 12887 / 483 |
| Total tokens | 66,854 |

## Cases

| Case | Result | Steps | Latency | Failed checks |
|---|---|---|---|---|
| iris-corr | PASS | 3 | 10.5s | — |
| spam-free | PASS | 3 | 5.2s | — |
| cust-discount-young | PASS | 3 | 12.5s | — |
| cust-high-score | PASS | 3 | 6.4s | — |
| cust-lowercase | PASS | 5 | 24.2s | — |
