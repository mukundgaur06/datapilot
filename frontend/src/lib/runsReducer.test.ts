import { describe, expect, it } from 'vitest'

import { initialState, reducer, type State, type ToolItem } from './runsReducer'
import type { AgentEvent } from './types'

const ev = (type: AgentEvent['type'], data: AgentEvent['data'] = {}, step = 1): AgentEvent => ({ type, step, data })

function play(...events: AgentEvent[]): State {
  let s = reducer(initialState, { type: 'start', localId: 'r1', question: 'q' })
  s = reducer(s, { type: 'created', localId: 'r1', runId: 'srv1' })
  events.forEach((event, id) => {
    s = reducer(s, { type: 'event', localId: 'r1', id, event })
  })
  return s
}

describe('runs reducer', () => {
  it('starts optimistically, then becomes running once the server accepts', () => {
    let s = reducer(initialState, { type: 'start', localId: 'r1', question: 'q' })
    expect(s.runs[0].status).toBe('starting')
    s = reducer(s, { type: 'created', localId: 'r1', runId: 'srv1' })
    expect(s.runs[0]).toMatchObject({ status: 'running', runId: 'srv1' })
  })

  it('pairs tool_result with its tool_call by key', () => {
    const s = play(
      ev('tool_call', { key: '1.0', name: 'group_aggregate', args: { group_by: ['label'] } }),
      ev('tool_result', { key: '1.0', name: 'group_aggregate', result: { n_groups: 2 }, is_error: false, duration_ms: 3 }),
    )
    const items = s.runs[0].items
    expect(items).toHaveLength(1)
    expect(items[0]).toMatchObject({ kind: 'tool', name: 'group_aggregate', result: { n_groups: 2 }, isError: false, durationMs: 3 })
  })

  it('ignores replayed duplicates (same or older event id)', () => {
    let s = play(ev('thinking', { text: 'a' }))
    s = reducer(s, { type: 'event', localId: 'r1', id: 0, event: ev('thinking', { text: 'a' }) })
    expect(s.runs[0].items).toHaveLength(1)
  })

  it('records the answer and the end status', () => {
    let s = play(ev('answer', { text: '50%', partial: false, steps: 2, elapsed_ms: 900 }))
    s = reducer(s, { type: 'end', localId: 'r1', status: 'done' })
    expect(s.runs[0].status).toBe('done')
    expect(s.runs[0].items[0]).toMatchObject({ kind: 'answer', text: '50%', stats: { steps: 2 } })
  })

  it('stop marks the run cancelled and drops late events', () => {
    let s = play(ev('tool_call', { key: '1.0', name: 'profile_dataset', args: {} }))
    s = reducer(s, { type: 'stopped', localId: 'r1' })
    s = reducer(s, { type: 'event', localId: 'r1', id: 5, event: ev('answer', { text: 'late' }) })
    s = reducer(s, { type: 'end', localId: 'r1', status: 'done' })
    const run = s.runs[0]
    expect(run.status).toBe('cancelled')
    expect(run.items.map((i) => i.kind)).toEqual(['tool', 'cancelled'])
    expect((run.items[0] as ToolItem).result).toBeUndefined()
  })

  it('network failures become an error item', () => {
    const s = reducer(reducer(initialState, { type: 'start', localId: 'r1', question: 'q' }), {
      type: 'failed',
      localId: 'r1',
      message: 'Server down',
    })
    expect(s.runs[0].status).toBe('error')
    expect(s.runs[0].items[0]).toMatchObject({ kind: 'error', message: 'Server down' })
  })

  it('reset clears the conversation', () => {
    expect(reducer(play(ev('thinking', { text: 'x' })), { type: 'reset' })).toEqual(initialState)
  })
})
