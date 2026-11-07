// Thin client for the DataPilot API. All network code lives here.

import { AGENT_EVENT_TYPES, type AgentEvent, type Dataset, type Health, type RunStatus } from './types'

// Where the API lives. Set VITE_API_URL at build time to point elsewhere (e.g. a separately
// hosted backend). Otherwise: the dev server talks to a local backend on :8000, and a
// production build talks to its own origin (the single-container deploy serves both).
export const API_URL: string =
  (import.meta.env.VITE_API_URL as string | undefined) ?? (import.meta.env.DEV ? 'http://localhost:8000' : '')

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_URL}${path}`, init)
  } catch {
    throw new ApiError(`Can't reach the DataPilot server at ${API_URL}. Is the backend running?`)
  }
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    // FastAPI puts messages in `detail`: a string for HTTPException, a list for validation errors.
    const detail = body?.detail
    const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((d) => d.msg).join('; ') : null
    throw new ApiError(message ?? `Request failed (${res.status})`)
  }
  return body as T
}

export const getHealth = () => request<Health>('/health')

export function uploadDataset(file: File): Promise<Dataset> {
  const form = new FormData()
  form.append('file', file)
  return request<Dataset>('/datasets', { method: 'POST', body: form })
}

export const createRun = (datasetId: string, question: string) =>
  request<{ run_id: string; status: RunStatus }>('/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, question }),
  })

export const cancelRun = (runId: string) => request<{ status: RunStatus }>(`/runs/${runId}/cancel`, { method: 'POST' })

export interface StreamHandlers {
  onEvent: (event: AgentEvent, id: number) => void
  onEnd: (status: RunStatus) => void
  onError: (message: string) => void
}

const MAX_RECONNECTS = 5

/**
 * Subscribe to a run's Server-Sent Events. Returns a function that closes the stream.
 *
 * EventSource reconnects by itself after a dropped connection and sends Last-Event-ID,
 * so the server replays only what we missed. We only give up after several failed
 * reconnects in a row (e.g. the backend was stopped).
 */
export function streamRun(runId: string, handlers: StreamHandlers): () => void {
  const source = new EventSource(`${API_URL}/runs/${runId}/stream`)
  let failures = 0

  source.onopen = () => {
    failures = 0
  }
  for (const type of AGENT_EVENT_TYPES) {
    source.addEventListener(type, (e) => {
      const msg = e as MessageEvent<string>
      handlers.onEvent(JSON.parse(msg.data) as AgentEvent, Number(msg.lastEventId))
    })
  }
  // The server sends `end` after the terminal event. Close, or EventSource would reconnect forever.
  source.addEventListener('end', (e) => {
    source.close()
    handlers.onEnd((JSON.parse((e as MessageEvent<string>).data) as { status: RunStatus }).status)
  })
  source.onerror = () => {
    failures += 1
    if (source.readyState === EventSource.CLOSED || failures > MAX_RECONNECTS) {
      source.close()
      handlers.onError('Lost connection to the server while the agent was working.')
    }
  }
  return () => source.close()
}

export interface TraceRecord {
  t_ms: number
  kind: 'run_start' | 'llm_call' | 'tool_call' | 'tool_result' | 'run_end'
  [key: string]: unknown
}

export interface Trace {
  run_id: string
  saved: boolean
  status: RunStatus
  records: TraceRecord[]
  summary: {
    llm_calls: number
    llm_ms: number
    tool_calls: number
    tool_errors: number
    tool_ms: number
    total_ms: number
    usage: Record<string, number>
  }
}

export const getTrace = (runId: string) => request<Trace>(`/runs/${runId}/trace`)
