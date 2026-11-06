// Turn trace records into a waterfall: one span per model call or tool call, placed in time.
// Records carry t_ms (time since run start, at the END of the operation) and duration_ms.

import type { TraceRecord } from './api'

export interface Span {
  kind: 'model' | 'tool'
  label: string
  start: number
  duration: number
  detail: string
  isError: boolean
}

export interface Waterfall {
  totalMs: number
  spans: Span[]
}

const n = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : 0)

export function buildWaterfall(records: TraceRecord[]): Waterfall {
  const spans: Span[] = []
  for (const r of records) {
    const duration = n(r.duration_ms)
    const start = Math.max(0, n(r.t_ms) - duration)
    if (r.kind === 'llm_call') {
      const usage = (r.usage ?? {}) as Record<string, number>
      const calls = n(r.n_tool_calls)
      const outcome = r.error ? 'failed' : calls ? `asked for ${calls} tool${calls > 1 ? 's' : ''}` : 'answered'
      const tokens = n(usage.input_tokens) + n(usage.output_tokens) + n(usage.thinking_tokens)
      spans.push({
        kind: 'model',
        label: `Model call ${n(r.step)}`,
        start,
        duration,
        detail: `${outcome}${tokens ? ` · ${tokens.toLocaleString('en-US')} tokens` : ''}`,
        isError: Boolean(r.error),
      })
    } else if (r.kind === 'tool_result') {
      spans.push({
        kind: 'tool',
        label: String(r.name),
        start,
        duration,
        detail: r.is_error ? 'error, sent back to the model' : `${n(r.result_chars).toLocaleString('en-US')} chars returned`,
        isError: Boolean(r.is_error),
      })
    }
  }
  const end = records.find((r) => r.kind === 'run_end')
  const totalMs = Math.max(n(end?.t_ms), ...spans.map((s) => s.start + s.duration), 1)
  return { totalMs, spans }
}

export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10_000 ? 0 : 1)}s` : `${Math.round(ms)} ms`
}
