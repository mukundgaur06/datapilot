// All conversation state lives in one reducer: an append-only log of runs, each holding
// the timeline items built from its stream of agent events. Pure, so it's unit-tested.

import type { AgentEvent, JsonRecord, RunStats, RunStatus } from './types'

export type ToolItem = {
  kind: 'tool'
  key: string
  step: number
  name: string
  args: JsonRecord
  result?: JsonRecord
  isError?: boolean
  durationMs?: number
}

export type TimelineItem =
  | { kind: 'thinking'; key: string; step: number; text: string }
  | ToolItem
  | { kind: 'answer'; key: string; text: string; partial: boolean; stats: RunStats }
  | { kind: 'error'; key: string; message: string }
  | { kind: 'cancelled'; key: string; message: string }

/** 'starting' = optimistic: shown before the server has even accepted the run. */
export type RunViewStatus = 'starting' | RunStatus

export interface RunView {
  localId: string
  runId: string | null
  question: string
  status: RunViewStatus
  items: TimelineItem[]
  lastEventId: number
}

export interface State {
  runs: RunView[]
}

export type Action =
  | { type: 'start'; localId: string; question: string }
  | { type: 'created'; localId: string; runId: string }
  | { type: 'event'; localId: string; id: number; event: AgentEvent }
  | { type: 'end'; localId: string; status: RunStatus }
  | { type: 'failed'; localId: string; message: string }
  | { type: 'stopped'; localId: string }
  | { type: 'reset' }

export const initialState: State = { runs: [] }

export const isActive = (run: RunView | undefined): boolean => run?.status === 'starting' || run?.status === 'running'

function stats(data: JsonRecord): RunStats {
  return { steps: data.steps as number, elapsed_ms: data.elapsed_ms as number, usage: data.usage as Record<string, number> }
}

/** Fold one agent event into the timeline. Exported for tests. */
export function applyEvent(items: TimelineItem[], event: AgentEvent, id: number): TimelineItem[] {
  const d = event.data
  switch (event.type) {
    case 'thinking':
      return [...items, { kind: 'thinking', key: `t${id}`, step: event.step, text: String(d.text ?? '') }]
    case 'tool_call':
      return [
        ...items,
        { kind: 'tool', key: String(d.key ?? `c${id}`), step: event.step, name: String(d.name), args: (d.args as JsonRecord) ?? {} },
      ]
    case 'tool_result': {
      // Fill in the matching call card (paired by the loop's stable key).
      const key = String(d.key)
      return items.map((it) =>
        it.kind === 'tool' && it.key === key
          ? { ...it, result: d.result as JsonRecord, isError: Boolean(d.is_error), durationMs: d.duration_ms as number }
          : it,
      )
    }
    case 'answer':
      return [...items, { kind: 'answer', key: `a${id}`, text: String(d.text ?? ''), partial: Boolean(d.partial), stats: stats(d) }]
    case 'error':
      return [...items, { kind: 'error', key: `e${id}`, message: String(d.message ?? 'Something went wrong.') }]
    case 'cancelled':
      return [...items, { kind: 'cancelled', key: `x${id}`, message: String(d.message ?? 'Run stopped.') }]
  }
}

function updateRun(state: State, localId: string, fn: (run: RunView) => RunView): State {
  return { runs: state.runs.map((r) => (r.localId === localId ? fn(r) : r)) }
}

export function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'start':
      return {
        runs: [
          ...state.runs,
          { localId: action.localId, runId: null, question: action.question, status: 'starting', items: [], lastEventId: -1 },
        ],
      }
    case 'created':
      return updateRun(state, action.localId, (r) => ({ ...r, runId: action.runId, status: r.status === 'starting' ? 'running' : r.status }))
    case 'event':
      return updateRun(state, action.localId, (r) => {
        // Ignore duplicates and anything arriving after the user stopped the run.
        if (action.id <= r.lastEventId || !isActive(r)) return r
        return { ...r, items: applyEvent(r.items, action.event, action.id), lastEventId: action.id }
      })
    case 'end':
      return updateRun(state, action.localId, (r) => (isActive(r) ? { ...r, status: action.status } : r))
    case 'failed':
      return updateRun(state, action.localId, (r) => ({
        ...r,
        status: 'error',
        items: [...r.items, { kind: 'error', key: `f${r.items.length}`, message: action.message }],
      }))
    case 'stopped':
      return updateRun(state, action.localId, (r) =>
        isActive(r)
          ? { ...r, status: 'cancelled', items: [...r.items, { kind: 'cancelled', key: 'stopped', message: 'Run stopped by the user.' }] }
          : r,
      )
    case 'reset':
      return initialState
  }
}
