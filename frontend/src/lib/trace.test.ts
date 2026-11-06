import { describe, expect, it } from 'vitest'

import type { TraceRecord } from './api'
import { buildWaterfall, formatMs } from './trace'

const rec = (r: Partial<TraceRecord> & Pick<TraceRecord, 'kind' | 't_ms'>) => r as TraceRecord

describe('buildWaterfall', () => {
  it('places model and tool spans by start time and describes them', () => {
    const w = buildWaterfall([
      rec({ kind: 'run_start', t_ms: 0 }),
      rec({ kind: 'llm_call', t_ms: 1200, step: 1, duration_ms: 1150, usage: { input_tokens: 900, output_tokens: 40 }, n_tool_calls: 1 }),
      rec({ kind: 'tool_call', t_ms: 1201, name: 'group_aggregate' }),
      rec({ kind: 'tool_result', t_ms: 1230, name: 'group_aggregate', duration_ms: 29, is_error: false, result_chars: 1534 }),
      rec({ kind: 'llm_call', t_ms: 2400, step: 2, duration_ms: 1160, usage: { input_tokens: 1200, output_tokens: 80 }, n_tool_calls: 0 }),
      rec({ kind: 'run_end', t_ms: 2402 }),
    ])
    expect(w.totalMs).toBe(2402)
    expect(w.spans.map((s) => [s.kind, s.label, s.start])).toEqual([
      ['model', 'Model call 1', 50],
      ['tool', 'group_aggregate', 1201],
      ['model', 'Model call 2', 1240],
    ])
    expect(w.spans[0].detail).toBe('asked for 1 tool · 940 tokens')
    expect(w.spans[1].detail).toBe('1,534 chars returned')
    expect(w.spans[2].detail).toBe('answered · 1,280 tokens')
  })

  it('marks failed model calls and tool errors', () => {
    const w = buildWaterfall([
      rec({ kind: 'tool_result', t_ms: 10, name: 'correlation', duration_ms: 5, is_error: true }),
      rec({ kind: 'llm_call', t_ms: 30, step: 2, duration_ms: 20, usage: {}, error: 'rate limit' }),
    ])
    expect(w.spans.map((s) => s.isError)).toEqual([true, true])
    expect(w.spans[1].detail).toBe('failed')
  })
})

describe('formatMs', () => {
  it('uses ms below a second, seconds above', () => {
    expect([formatMs(29.4), formatMs(1240), formatMs(16500)]).toEqual(['29 ms', '1.2s', '17s'])
  })
})
