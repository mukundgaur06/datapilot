// Trace tab: where a run's time and tokens went, as a waterfall of model and tool calls.

import { useEffect, useMemo, useState } from 'react'

import { getTrace, type Trace } from '../lib/api'
import { formatNumber } from '../lib/format'
import type { RunView } from '../lib/runsReducer'
import { buildWaterfall, formatMs } from '../lib/trace'
import { AlertIcon } from './Icons'

const KIND_COLOR = { model: 'var(--series-1)', tool: 'var(--series-2)' } as const

export function TracePanel({ runs }: { runs: RunView[] }) {
  const traced = runs.filter((r) => r.runId)
  const [chosen, setChosen] = useState<string | null>(null)
  const run = traced.find((r) => r.localId === chosen) ?? traced.at(-1)
  const [state, setState] = useState<{ key: string; trace: Trace | null; error: string | null } | null>(null)

  // Refetch when the run or its status changes (a finished run has its full trace).
  // Each response is stored with the key it was fetched for and only shown while that key is
  // current, so a slow, stale response can never overwrite what's on screen.
  const key = run ? `${run.runId}:${run.status}` : ''
  useEffect(() => {
    if (!run?.runId) return
    getTrace(run.runId).then(
      (trace) => setState({ key, trace, error: null }),
      (e: Error) => setState({ key, trace: null, error: e.message }),
    )
  }, [key, run?.runId])

  const trace = state?.key === key ? state.trace : null
  const waterfall = useMemo(() => (trace ? buildWaterfall(trace.records) : null), [trace])

  if (!run) {
    return <p className="px-2 py-10 text-center text-sm text-ink-muted">Ask a question to see how the agent spent its time and tokens.</p>
  }

  return (
    <div className="flex flex-col gap-3 text-sm">
      {traced.length > 1 && (
        <label className="flex flex-col gap-1 text-xs text-ink-muted">
          Run
          <select
            value={run.localId}
            onChange={(e) => setChosen(e.target.value)}
            className="rounded-lg border border-hairline bg-surface px-2 py-1.5 text-sm text-ink"
          >
            {traced.map((r) => (
              <option key={r.localId} value={r.localId}>
                {r.question.length > 60 ? `${r.question.slice(0, 59)}…` : r.question}
              </option>
            ))}
          </select>
        </label>
      )}
      {state?.key === key && state.error && <p className="text-xs text-critical">{state.error}</p>}
      {!trace && !state?.error && <div className="shimmer h-24 rounded-lg" aria-label="Loading trace" />}
      {trace && waterfall && (
        <>
          <dl className="grid grid-cols-3 gap-2">
            <Stat label="Total time" value={formatMs(trace.summary.total_ms)} />
            <Stat label="Model time" value={`${Math.round((100 * trace.summary.llm_ms) / Math.max(trace.summary.total_ms, 1))}%`} />
            <Stat label="Tool time" value={formatMs(trace.summary.tool_ms)} />
            <Stat label="Model calls" value={String(trace.summary.llm_calls)} />
            <Stat label="Tool calls" value={`${trace.summary.tool_calls}${trace.summary.tool_errors ? ` (${trace.summary.tool_errors} failed)` : ''}`} />
            <Stat
              label="Tokens in / out"
              value={`${formatNumber(trace.summary.usage.input_tokens ?? 0)} / ${formatNumber((trace.summary.usage.output_tokens ?? 0) + (trace.summary.usage.thinking_tokens ?? 0))}`}
            />
          </dl>

          <div className="flex gap-3 text-[11px] text-ink-2" aria-hidden="true">
            {(['model', 'tool'] as const).map((k) => (
              <span key={k} className="flex items-center gap-1.5">
                <span className="size-2.5 rounded-sm" style={{ background: KIND_COLOR[k] }} /> {k === 'model' ? 'Model call' : 'Tool call'}
              </span>
            ))}
          </div>

          <ol className="flex flex-col gap-1.5" aria-label="Timeline of model and tool calls">
            {waterfall.spans.map((s, i) => {
              const left = (100 * s.start) / waterfall.totalMs
              const width = Math.max((100 * s.duration) / waterfall.totalMs, 0.6)
              return (
                <li key={i} className="grid grid-cols-[minmax(0,7.5rem)_minmax(0,1fr)] items-center gap-2">
                  <span className="flex min-w-0 items-center gap-1 text-xs" title={s.label}>
                    {s.isError && <AlertIcon className="size-3 shrink-0 text-critical" />}
                    <span className={`truncate ${s.kind === 'tool' ? 'font-mono text-[11px]' : ''}`}>{s.label}</span>
                  </span>
                  <span className="relative h-5 rounded bg-surface-2" title={`${s.label}: ${formatMs(s.duration)}, ${s.detail}`}>
                    <span
                      className="absolute inset-y-0.5 rounded-sm"
                      style={{ left: `${Math.min(left, 99.4)}%`, width: `${width}%`, background: s.isError ? 'var(--critical)' : KIND_COLOR[s.kind] }}
                    />
                  </span>
                  <span className="sr-only">
                    {s.label} started at {formatMs(s.start)}, took {formatMs(s.duration)}, {s.detail}
                  </span>
                  <span className="col-start-2 -mt-1 text-[11px] text-ink-muted tabular">
                    {formatMs(s.duration)} · {s.detail}
                  </span>
                </li>
              )
            })}
          </ol>
          <p className="text-[11px] text-ink-muted">
            Run {trace.run_id}.{' '}
            {trace.saved ? 'Saved as JSONL in backend/traces/ (tool results there are capped previews).' : 'Kept in memory only (file tracing is off).'}
          </p>
        </>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-hairline bg-surface px-2 py-1.5">
      <dt className="text-[10px] text-ink-muted">{label}</dt>
      <dd className="mt-0.5 text-sm font-semibold tabular">{value}</dd>
    </div>
  )
}
