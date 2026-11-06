# DataPilot: Interview Study Guide

How to use this: study each concept **just before** the phase that uses it (see the phase tags),
then explain it out loud in under 60 seconds. If you can't, re-read. The "Say this" lines are
starting points. Put them in your own words; don't memorize them.

---

## Part A: Concept map

### 1. LLM fundamentals [Phase 1]
| Concept | Be able to explain | Say this (starter) |
|---|---|---|
| Tokens & context window | What a token is; why long tool outputs hurt | "Cost and latency scale with tokens, so I truncate tool results to 4 KB." |
| Temperature / sampling | Why agents use low temperature | "Tool selection should be repeatable, so temperature stays low." |
| Tool / function calling | Model emits structured `{name, args}`; *your code* executes it | "The model never runs code. It requests a tool and I validate and execute it." |
| Structured output / JSON Schema | How tool schemas are described | "My Pydantic models generate the JSON Schema the model sees." |
| Hallucination | Why it happens; grounding as mitigation | "Every number must come from a tool result, and my evals check that." |
| System prompt design | Role, rules, schema summary, few-shot | Show your actual prompt and why each section exists. |

### 2. Agents [Phase 1, 4, 6]
- **ReAct loop**: reason → act (tool) → observe → repeat. Draw it on a whiteboard.
- **Workflows vs agents**: a fixed pipeline vs the model choosing steps. Know when you'd pick a
  workflow (predictable tasks, lower cost).
- **Stop conditions**: max steps, token budget, the model answering without tool calls.
- **Error recovery**: errors as observations. Example: the model typed `Spend`, the column is `spend`, it retries.
- **Multi-agent (planner/critic)**: benefits vs added cost and latency. Don't add agents "because it's cool".
- **MCP (Model Context Protocol)**: a standard way to expose tools to any LLM client.
- **Agent security**: prompt injection (direct and indirect, via data), least-privilege tools, sandboxing.

### 3. Evals & LLMOps [Phase 4] ⭐ most asked at good companies
- Golden datasets; programmatic checks vs LLM-as-judge (judge bias, cost, calibration).
- Metrics: task success rate, steps per task, latency p50/p95, tokens/cost per run.
- Regression testing prompts: "I changed the prompt, so did anything break?"
- Tracing and observability: what you log and why. Logging PII is a risk.
- Mocking the LLM (`FakeLLM`) for deterministic unit tests.

### 4. Backend & systems [Phase 0, 2, 5]
- FastAPI: routing, dependency injection, Pydantic models, `async def` vs `def` (FastAPI runs
  sync handlers in a threadpool; pandas work is CPU-bound).
- ASGI and uvicorn; why async helps I/O (LLM calls) but not CPU (model training).
- **SSE vs WebSockets vs long polling**: direction, complexity, reconnection, proxies.
- REST design: resources (`/datasets`, `/runs`), status codes (201, 404, 422, 429).
- CORS, rate limiting, upload limits, secrets in env vars.
- Docker: image vs container, layers, why `requirements.txt` gets copied first (layer caching).
- CI: GitHub Actions running pytest on every push.

### 5. Frontend [Phase 2]
- React: state vs props, `useReducer` for an append-only event stream, `useEffect` cleanup
  (closing an `EventSource`), keys in lists.
- Rendering streams smoothly: batching updates, virtualization for long lists (concept only).
- Animation: layout animations, why you animate `transform`/`opacity` (GPU, no reflow).
- Accessibility: `aria-live` for streaming content, keyboard navigation, focus management.

### 6. Data science & ML [Phase 3] (you already started this. Go deeper.)
- pandas: dtypes, missing values, `groupby`/`agg`, memory of large frames.
- Train/validation/test splits; **k-fold and stratified CV**: why, and how many folds.
- **Data leakage**: target leakage, train-test contamination, why preprocessing goes *inside* the `Pipeline`.
- Metrics: accuracy vs precision vs recall vs F1; ROC-AUC vs PR-AUC (imbalanced data!);
  MAE vs RMSE; R².
- Baselines: "the model beats `DummyClassifier` by X". Without a baseline, accuracy means nothing.
- Class imbalance: stratification, class weights, threshold tuning.
- Feature importance: impurity-based vs **permutation importance** (and why permutation is less biased).
- KMeans (your segmentation project): choosing k (elbow, silhouette), scaling sensitivity.
- Logistic regression (your spam project): sigmoid, log-loss, regularization (C), coefficients as log-odds.

### 7. Software engineering hygiene [all phases]
- Git: small commits, branches, meaningful messages, `.gitignore` for secrets.
- Testing pyramid: unit (tools) → integration (API) → evals (agent behaviour).
- Clean code: typed functions, single responsibility, adapters and interfaces (LLM provider).

---

## Part B: 14-day plan (≈1.5–2 h study/day, alongside building)

| Day | Build | Study (before building) | Self-check |
|---|---|---|---|
| 1 (today) | Phase 0 + 1 | Tool calling, ReAct, JSON Schema, Pydantic | Draw the agent loop from memory |
| 2 | Phase 2 backend (SSE) | SSE vs WS vs polling, FastAPI async | Explain why SSE fits here |
| 3 | Phase 2 frontend | `useReducer`, `EventSource`, motion layout animations | Live-code a reducer for events |
| 4 | Phase 3 charts + train_model | CV, pipelines, leakage | Explain leakage with a concrete example |
| 5 | Phase 3 metrics + model card | Precision/recall, ROC vs PR-AUC, imbalance, baselines | Pick a metric for spam detection and defend it |
| 6 | Phase 4 tests | pytest, fixtures, mocking | Write a FakeLLM test from scratch |
| 7 | Phase 4 evals + tracing | Eval design, LLM-as-judge pitfalls | Present your before/after number |
| 8 | Phase 5 deploy | Docker, CORS, rate limiting, env secrets | Explain your Dockerfile line by line |
| 9 | Phase 5 README + demo | Storytelling (STAR) | Record the 90-second demo |
| 10 | Buffer / bug fixes | Prompt injection and agent security | Name 3 attacks on DataPilot and the defences |
| 11 | Stretch feature | MCP or sandboxing, depending on the pick | Explain the trade-offs of your pick |
| 12 | — | Mock interview: project deep-dive (Part C) | Answer all of Part C out loud |
| 13 | — | ML fundamentals revision (Section 6) | 10 random ML questions |
| 14 | Apply! | Tailor the resume line with real numbers | 10 applications sent |

**Revision method:** at the end of each day, write 3 flashcards (question → your answer). Review
them every morning (spaced repetition: Anki works well).

---

## Part C: Question bank (with answer pointers)

**Project deep-dive**
1. *Walk me through what happens when a user asks a question.*
   Upload → schema summary → system prompt → loop (LLM → tool → observation) → SSE events → UI.
2. *Why not LangChain?* You wanted to own the loop: debuggability, fewer abstractions, and you can
   explain every step. You'd consider a framework for X (for example built-in tracing) later.
3. *How do you stop the agent from hallucinating numbers?* Grounding rule in the prompt, tool-only
   numbers, eval checks, and a critic pass as the stretch goal.
4. *What happens if the model calls a tool with a wrong column?* Pydantic or the tool raises → the error becomes
   an observation → the model retries. You tested this with FakeLLM.
5. *How did you evaluate it?* Golden set of N questions, programmatic checks, metrics, and your
   before/after number. Be specific.
6. *What would break at 1,000 concurrent users?* In-memory store (move to Redis/S3), CPU-bound
   training blocking workers (task queue such as Celery/RQ), LLM rate limits (queue + backoff), cost.
7. *Security risks?* Prompt injection via CSV cells, malicious uploads (size/type), API key
   leakage, and code execution if you add the sandbox.
8. *Why SSE?* One-way stream, auto-reconnect, plain HTTP, simpler than WebSockets.
9. *Why permutation importance?* Model-agnostic, computed on held-out data, less biased toward
   high-cardinality features than impurity importance.
10. *Biggest bug you hit, and how you found it?* **Keep a "bug journal" while building** so this answer is real.

**General AI-engineer questions**
11. Agent vs workflow: when would you NOT use an agent?
12. How do you choose between a bigger and a smaller model? (Evals: accuracy vs cost vs latency.)
13. What is RAG, and when is tool use better than retrieval? (Exact computation vs fuzzy recall.)
14. How would you reduce the cost of this app by 50%? (Smaller model for simple steps, caching,
    shorter prompts, truncation.)
15. Explain precision vs recall to a product manager, using spam as the example.

---

## Part D: Your 30-second pitch (fill it in after Phase 4)
> "I built DataPilot, an agent that answers questions about any CSV by calling pandas and
> scikit-learn tools I wrote, and it streams its reasoning live. The interesting part was quality:
> I built a ___-question eval harness and found the agent ___. After I ___, task success went
> from ___% to ___%. It's deployed at ___ with tests in CI."
