# DataPilot — an Agentic Data Analyst

> Upload a CSV and ask a question in plain English. An AI agent plans the analysis, calls real
> pandas/scikit-learn tools, and streams each step live: the reasoning, the tool calls, charts and
> model metrics. It ends with a cited answer and a downloadable report.

---

## 0. Why this project (the consultant's view)

**What recruiters for AI-engineer internships screen for in 2026:**
1. You can build an **agent loop** (tool calling, not just a chat wrapper).
2. You can **measure** it (evals: success rate, latency, cost).
3. You **ship** it (a deployed URL, a clean README, a 90-second demo video).
4. You understand the **data/ML underneath**, not only the prompt.

**Why DataPilot fits you specifically:**
- It reuses what you already know: React + Vite (HackElite AI), scikit-learn pipelines (spam
  classifier, KMeans segmentation), FastAPI/Streamlit basics, and LLM APIs (Gemini).
- It closes your visible gaps: ML stuck in notebooks, AI features that are chat wrappers, and no
  evals, tests or deployment.
- It covers three job titles: **AI Engineer** (agent + evals), **Data Scientist** (sound ML
  methodology), and **SWE** (API, streaming, tests, deployment).
- You can demo it with your own datasets (`MLOps PD/3-sms_spam_sample.csv`,
  `2-customer_segmentation_sample.csv`).

**Rejected alternatives:**
- A "resume/JD matcher": every applicant has one.
- A RAG chatbot over PDFs: saturated and hard to tell apart from the rest.
- 12 small agent clones: depth beats count. In interviews you get asked about the hard parts of
  ONE project.

---

## 1. Architecture

```
┌──────────────────────────── React + Vite (frontend) ────────────────────────────┐
│  Dataset panel        │   Chat + live Agent Timeline        │   Artifacts panel   │
│  (schema, col chips)  │   (step cards stream in via SSE)    │   (charts, metrics) │
└───────────┬──────────────────────────────┬──────────────────────────────────────┘
            │ POST /datasets (upload)      │ GET /runs/{id}/stream  (Server-Sent Events)
┌───────────▼──────────────────────────────▼──────────────────────────────────────┐
│ FastAPI backend                                                                  │
│  ┌──────────────┐   ┌──────────────────────┐   ┌──────────────────────────────┐ │
│  │ Dataset store│◄──┤  Agent loop (ReAct)  ├──►│ LLM adapter (Gemini/Claude/…)│ │
│  │ (pandas DFs) │   │  max_steps, budget,  │   └──────────────────────────────┘ │
│  └──────▲───────┘   │  error→observation   │                                    │
│         │           └─────────┬────────────┘                                    │
│         │      ┌──────────────▼───────────────┐    ┌────────────────────────┐   │
│         └──────┤ Tool registry (Pydantic args)│───►│ Tracer (JSONL per run) │   │
│                │ profile · aggregate · corr · │    └────────────────────────┘   │
│                │ chart · train · evaluate     │                                 │
│                └──────────────────────────────┘                                 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Key design decisions (be ready to explain each in interviews)
| Decision | Why |
|---|---|
| **Hand-written agent loop** instead of LangChain/CrewAI | You understand and can debug every step. Frameworks are easy to add later; interviewers probe fundamentals. |
| **Constrained tools** (not "run arbitrary Python") in v1 | Safer (no code-execution sandbox needed), deterministic, testable. A sandbox is a Phase-6 stretch. |
| **Only schema + summaries go to the LLM, never raw rows** | Privacy, token cost, and resistance to prompt injection hidden in cell values. |
| **Charts returned as JSON specs**, rendered by the frontend | Interactive charts, tiny payloads, no image generation on the server. |
| **SSE** instead of WebSockets | Streaming is one-way (server → client). SSE is simpler and works over plain HTTP. |
| **Tool errors are returned to the model as observations** | The agent can fix its own mistakes (wrong column name, for example) instead of crashing. |
| **LLM behind an adapter interface** | Swap providers or models, mock the LLM in tests, compare models in evals. |

### Folder structure
```
datapilot/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app + routes
│   │   ├── store.py           # dataset sessions (id -> DataFrame)
│   │   ├── schemas.py         # Pydantic models for API + events
│   │   ├── agent/
│   │   │   ├── loop.py        # the agent loop (generator of events)
│   │   │   ├── prompts.py     # system prompt builder
│   │   │   └── llm.py         # provider adapter (Gemini / Claude / fake)
│   │   ├── tools/
│   │   │   ├── registry.py    # @tool decorator, JSON schemas, dispatch
│   │   │   ├── data_tools.py  # profile, describe_column, group_aggregate, correlation
│   │   │   ├── chart_tools.py # make_chart -> chart spec
│   │   │   └── ml_tools.py    # train_model, feature_importance, predict
│   │   └── tracing.py         # per-run JSONL traces
│   ├── evals/  (golden.jsonl, run_evals.py)
│   ├── tests/
│   └── requirements.txt
├── frontend/   (Vite + React + Tailwind + Motion + Recharts)
├── sample_data/
└── docs/
```

### The core loop (the most important ~25 lines in the project)
```python
def run_agent(question: str, dataset_id: str, max_steps: int = 8):
    """Yields events: thought | tool_call | tool_result | answer | error."""
    messages = [system(build_prompt(store.schema_summary(dataset_id))), user(question)]
    for step in range(max_steps):
        reply = llm.chat(messages, tools=registry.schemas())   # model decides: answer or call tools
        messages.append(reply.as_message())
        if not reply.tool_calls:
            yield Answer(text=reply.text, steps=step + 1)
            return
        for call in reply.tool_calls:
            yield ToolCall(name=call.name, args=call.args)
            try:
                result = registry.dispatch(call.name, dataset_id, call.args)  # Pydantic-validated
            except ToolError as e:
                result = {"error": str(e)}          # becomes an observation -> self-correction
            yield ToolResult(name=call.name, result=truncate(result))
            messages.append(tool_result_message(call.id, truncate(result)))
    yield Answer(text="Step limit reached; partial findings above.", partial=True)
```

---

## 2. Phases (goal-based, with "done when" criteria)

### Phase 0: Setup (30 min) ✅ today
- `conda create -n datapilot python=3.12 -y` then `conda activate datapilot`
- `pip install fastapi uvicorn[standard] pandas scikit-learn pydantic python-dotenv sse-starlette pytest`
  plus your LLM SDK (`google-genai` for Gemini, which you've used, or `anthropic`).
- `git init`, `.gitignore` (`.env`, `__pycache__`, `node_modules`), `.env` with the API key.
- Copy your spam and segmentation CSVs into `sample_data/`.
- **Done when:** `uvicorn app.main:app --reload` serves `GET /health → {"ok": true}`.
- 📘 *Concepts:* virtual environments, env vars and secrets, ASGI (what uvicorn is).

### Phase 1: Agent core, no UI (3–4 h) ✅ today
1. **Dataset store** (`store.py`): `load_csv(file) -> id`, `get(id) -> DataFrame`,
   `schema_summary(id)` (column, dtype, % missing, n_unique, 3 example values).
2. **Three tools** with Pydantic arg models:
   - `profile_dataset()`: shape, dtypes, missing values, basic stats.
   - `group_aggregate(group_by, column, agg)`: for example mean of `spend` by `segment`.
   - `correlation(columns?)`: top correlated pairs.
3. **Tool registry**: an `@tool` decorator that turns the function and its Pydantic model into
   the JSON schema the LLM sees.
4. **LLM adapter**: `chat(messages, tools) -> Reply(text, tool_calls)` for ONE provider, plus a
   `FakeLLM` for tests.
5. **Agent loop** as above, plus a CLI: `python -m app.cli sample_data/spam.csv "What % of messages are spam?"`
6. Endpoints: `POST /datasets` (upload) and `POST /ask` (non-streaming, returns all events).
- **Done when:** 5 hand-written questions get correct answers, and every number comes from a tool
  call (check the event log). Commit: `feat: agent core with 3 data tools`.
- 📘 *Concepts:* function/tool calling, JSON Schema, ReAct pattern, Pydantic validation, why
  max_steps exists, context-window budgeting (truncate tool output).

### Phase 2: Streaming UI (Day 2–3)
- Backend: `GET /runs/{id}/stream` sends the loop's events as SSE (`event: tool_call`, `data: {...}`).
- Frontend (Vite + React + Tailwind v4 + `motion` + Recharts):
  - **Drag-and-drop upload**, which then shows an animated schema panel with column chips (type icon, % missing bar).
  - **Agent Timeline**: each event animates in as a card ("🔧 group_aggregate(segment, spend, mean)",
    expandable to show raw args and result). A shimmer skeleton shows while the agent "thinks".
  - **Artifacts panel**: charts and tables pinned from tool results. Click a timeline card to highlight its artifact.
  - **Suggested questions** generated from the schema (one-click chips).
  - **Ctrl+K command palette**, dark/light toggle, a stop-run button (abort the SSE stream, cancel on the server).
  - Clicking a column chip inserts it into the prompt.
- **Done when:** a non-technical friend can upload a CSV and get a charted answer without help.
- 📘 *Concepts:* SSE vs WebSockets vs polling, `EventSource`, React state for streams
  (append-only event list, `useReducer`), optimistic UI, layout animations, accessibility (focus, aria-live).

### Phase 3: ML tools, the data-scientist layer (Day 4–5)
- `make_chart(kind, x, y, color?)` returns a chart spec. The backend validates column types for the chart kind.
- `train_model(target, features?, task=auto, model=auto)`:
  - Auto-detect classification vs regression. Use an sklearn `Pipeline` + `ColumnTransformer`
    (impute → scale/one-hot), **stratified k-fold cross-validation**, and compare against a baseline (DummyClassifier).
  - **Leakage guards**: drop ID-like columns (all unique values), warn if a feature correlates above 0.95 with the target.
  - Returns CV metrics (classification: accuracy, precision, recall, F1, ROC-AUC; regression: MAE, RMSE, R²) and a confusion matrix.
- `feature_importance(model_id)` with permutation importance. `predict(model_id, rows)`.
- A **model card** artifact: data used, metrics ± std, caveats (class imbalance, small n).
- **Done when:** "Build a spam detector and tell me what drives it" yields a CV-validated
  model that beats the baseline, plus an honest caveats section.
- 📘 *Concepts:* train/val/test, cross-validation, data leakage, class imbalance, precision vs recall
  trade-off, ROC-AUC, baselines, pipelines (why fit transformers only on training folds).

### Phase 4: Quality: evals, tests, tracing (Day 6–7) ⭐ the differentiator
- **Unit tests** (pytest): every tool on a tiny fixture DataFrame (deterministic). Agent loop with
  `FakeLLM` scripted replies: tool-error recovery, max-steps cutoff, unknown tool.
- **Eval harness** `evals/run_evals.py`: 20–30 golden questions over 3 datasets (iris, spam,
  segmentation), each with a programmatic check (for example `abs(answer_number - 13.4) < 0.1`, or "called `train_model`").
  Report: **task success %, avg steps, p50/p95 latency, tokens per run**, written to `evals/results/<date>.md`.
- **Tracing**: every run saved as JSONL (prompt, each tool call, latency, tokens), plus a "Trace" tab in the UI.
- Use evals to make ONE measured improvement (for example adding few-shot examples to the system prompt takes
  success from 70% to 90%). **Put this number in your resume.**
- **Done when:** `pytest` is green, the eval report exists, and one before/after improvement is documented.
- 📘 *Concepts:* eval-driven development, golden sets, LLM-as-judge (and when NOT to use it),
  mocking, observability, regression testing for prompts.

### Phase 5: Ship it (Day 8–9)
- Guardrails: upload size limit, rate limit per IP, API key only on the server, CORS locked to your frontend origin.
- Deploy: backend on Render, Railway or Hugging Face Spaces (Dockerfile; install Docker Desktop), frontend on Vercel.
- README: GIF at the top, architecture diagram, design decisions table, **eval results table**, "what I'd do next".
- A 90-second Loom demo. A LinkedIn post with the GIF and the eval number.
- **Done when:** a stranger can use the live URL, and the README shows the eval metrics.
- 📘 *Concepts:* Docker basics, CORS, rate limiting, secrets management, cold starts, CI (GitHub Actions running pytest).

### Phase 6: Stretch (pick 1–2, only after shipping)
- **Sandboxed code tool**: the agent writes pandas code and it runs in a subprocess with a timeout, no network
  and an import allow-list. A great security discussion topic.
- **Planner + critic**: a second LLM pass that checks the answer against the tool outputs (reduces hallucination).
- **MCP server**: expose your tools over the Model Context Protocol so any MCP client can use DataPilot.
- **Persistence**: MongoDB (you know it) for runs, datasets metadata, shareable run links.
- **Model comparison in evals**: same golden set across 2–3 LLMs, as a cost vs accuracy table.

---

## 3. Senior-engineer quality bar (apply in every phase)
- **Small, typed functions.** Pydantic everywhere data crosses a boundary (API, LLM, tools).
- **Deterministic core, probabilistic edge.** Tools are pure and tested; only the LLM is fuzzy, and it is isolated behind an adapter.
- **Every number in an answer must trace to a tool result.** The system prompt enforces it, and evals check it.
- **Fail soft.** Tool errors become observations. The step limit returns partial findings, never a stack trace in the UI.
- **Budget awareness.** Truncate tool outputs (≤ 50 rows / 4 KB) and log tokens per run.
- **Commit per working increment** with conventional messages (`feat:`, `fix:`, `test:`).
- **Treat data as untrusted.** Cell values can contain "ignore previous instructions...". Never give
  the agent tools whose misuse matters, and never put raw rows in the prompt.

## 4. Resume line (fill in your real numbers)
> **DataPilot**: agentic data-analysis app (FastAPI, React, scikit-learn, LLM tool calling). Built a
> ReAct agent with 7 validated tools and SSE-streamed reasoning UI; created a 25-question eval
> harness and raised task success from __% to __% via prompt and tool redesign; deployed with CI.
