import { describe, expect, it } from 'vitest'

import { artifactFromTool } from './artifacts'
import type { ToolItem } from './runsReducer'

const tool = (name: string, args: ToolItem['args'], result: ToolItem['result'], isError = false): ToolItem => ({
  kind: 'tool',
  key: '1.0',
  step: 1,
  name,
  args,
  result,
  isError,
})

describe('artifactFromTool', () => {
  it('turns a one-column group count into a bar chart with percentages', () => {
    const a = artifactFromTool(
      'x',
      tool('group_aggregate', { group_by: ['label'], agg: 'count' }, {
        agg: 'count', column: null, group_by: ['label'], n_rows_matched: 4, n_rows_total: 4, n_groups: 2, truncated: false,
        rows: [{ label: 'spam', value: 2, pct_of_rows: 50 }, { label: 'ham', value: 2, pct_of_rows: 50 }],
      }),
    )
    expect(a).toMatchObject({ kind: 'bar', title: 'Rows by label', subtitle: 'All 4 rows' })
    expect(a?.kind === 'bar' && a.data).toEqual([{ label: 'spam', value: 2, pct: 50 }, { label: 'ham', value: 2, pct: 50 }])
  })

  it('turns an ungrouped filtered aggregate into a stat tile that names the filter', () => {
    const a = artifactFromTool(
      'x',
      tool('group_aggregate', { agg: 'mean', column: 'Income', filters: [{ column: 'Age', op: '>', value: 40 }] }, {
        agg: 'mean', column: 'Income', n_rows_matched: 490, n_rows_total: 1200, value: 83.8296,
      }),
    )
    expect(a).toMatchObject({ kind: 'stat', title: 'Average of Income', value: '83.83' })
    expect(a?.subtitle).toContain('Age > 40')
    expect(a?.subtitle).toContain('490 of 1,200')
  })

  it('uses a table for two grouping columns', () => {
    const a = artifactFromTool(
      'x',
      tool('group_aggregate', { group_by: ['a', 'b'] }, {
        agg: 'count', group_by: ['a', 'b'], n_rows_matched: 2, n_rows_total: 2, n_groups: 2, rows: [{ a: 1, b: 2, value: 1, pct_of_rows: 50 }],
      }),
    )
    expect(a).toMatchObject({ kind: 'table', columns: ['a', 'b', 'value', 'pct_of_rows'] })
  })

  it('turns correlation pairs with a target into diverging bars labelled by the other column', () => {
    const a = artifactFromTool(
      'x',
      tool('correlation', { target: 'Score' }, { method: 'pearson', pairs: [{ a: 'Score', b: 'Visits', r: 0.77 }, { a: 'Days', b: 'Score', r: -0.71 }] }),
    )
    expect(a?.kind === 'diverging' && a.data).toEqual([{ label: 'Visits', value: 0.77 }, { label: 'Days', value: -0.71 }])
  })

  it('pins nothing for errors, empty results or unknown tools', () => {
    expect(artifactFromTool('x', tool('correlation', {}, { error: 'bad' }, true))).toBeNull()
    expect(artifactFromTool('x', tool('group_aggregate', { group_by: ['a'] }, { agg: 'count', rows: [], n_rows_total: 3 }))).toBeNull()
    expect(artifactFromTool('x', tool('mystery', {}, { value: 1 }))).toBeNull()
    expect(artifactFromTool('x', { ...tool('profile_dataset', {}, undefined) })).toBeNull()
  })
})
