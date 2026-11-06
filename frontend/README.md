# DataPilot frontend

React + TypeScript + Vite, styled with Tailwind v4, animated with `motion`, charts with Recharts.

## Run it

Start the backend first (from `backend/`):

```
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Then, from `frontend/`:

```
npm install
npm run dev
```

Open http://localhost:5173. The app talks to `http://localhost:8000` by default; set
`VITE_API_URL` (e.g. in `frontend/.env.local`) to point it elsewhere.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm test` | Unit tests (Vitest) for the reducer, artifacts, suggestions, formatting |
| `npm run build` | Type-check and production build |
| `npm run lint` | Lint with oxlint |

## How it fits together

- `src/lib/api.ts`: all network code. `streamRun()` wraps `EventSource` for the
  `GET /runs/{id}/stream` Server-Sent Events.
- `src/lib/runsReducer.ts`: one `useReducer` holds every run as an append-only list of
  timeline items built from agent events. Pure, so it's unit-tested.
- `src/lib/artifacts.ts`: picks a form for each tool result (stat tile, bar chart,
  diverging bars for correlations, or a table).
- `src/hooks/useAgentRuns.ts`: starts runs (optimistically), owns the open streams,
  and stops runs (closes the stream and cancels on the server).
