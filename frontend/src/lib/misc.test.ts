import { describe, expect, it } from 'vitest'

import { filterCommands, type Command } from './commands'
import { formatNumber, formatToolCall } from './format'
import { suggestQuestions } from './suggestions'
import type { ColumnSummary, DatasetSummary } from './types'

const col = (name: string, semantic_type: ColumnSummary['semantic_type'], n_unique: number): ColumnSummary => ({
  name, semantic_type, n_unique, dtype: 'x', missing_pct: 0, examples: [],
})
const summary = (columns: ColumnSummary[]): DatasetSummary => ({ name: 'd.csv', n_rows: 100, n_columns: columns.length, columns })

describe('suggestQuestions', () => {
  it('uses a low-cardinality category and numeric columns, never IDs', () => {
    const qs = suggestQuestions(summary([col('Customer_ID', 'id', 100), col('Segment', 'categorical', 4), col('Spend', 'numeric', 80), col('Age', 'numeric', 40)]))
    expect(qs).toEqual([
      'What share of rows falls in each Segment?',
      'What is the average Spend for each Segment?',
      'Which columns are most strongly correlated with Spend?',
      'Give me an overview of this dataset and any data-quality issues.',
    ])
    expect(qs.join(' ')).not.toContain('Customer_ID')
  })

  it('always offers at least an overview', () => {
    expect(suggestQuestions(summary([col('message', 'text', 100)]))).toEqual(['Give me an overview of this dataset and any data-quality issues.'])
  })
})

describe('format', () => {
  it('formats numbers readably', () => {
    expect(formatNumber(1284)).toBe('1,284')
    expect(formatNumber(0.77142)).toBe('0.77')
    expect(formatNumber(12_900_000)).toBe('12.9M')
    expect(formatNumber(null)).toBe('—')
  })

  it('formats tool calls on one line, skipping empty args', () => {
    expect(formatToolCall('group_aggregate', { group_by: ['label'], agg: 'count', filters: [], column: null })).toBe('group_aggregate(group_by=label, agg=count)')
    expect(formatToolCall('group_aggregate', { filters: [{ column: 'Age', op: '>', value: 40 }] })).toBe('group_aggregate(where Age > 40)')
  })
})

describe('filterCommands', () => {
  const cmds: Command[] = [
    { id: 'a', group: 'View', label: 'Switch to dark mode', run: () => {} },
    { id: 'b', group: 'Data', label: 'Upload a CSV', run: () => {} },
  ]
  it('matches every word in any order, case-insensitively', () => {
    expect(filterCommands(cmds, 'DARK switch').map((c) => c.id)).toEqual(['a'])
    expect(filterCommands(cmds, 'data').map((c) => c.id)).toEqual(['b'])
    expect(filterCommands(cmds, '')).toHaveLength(2)
  })
})
