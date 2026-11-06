// Starter questions built from the schema, so a first-time user never faces a blank box.
// Deterministic (no LLM call): instant, free, and every suggestion is answerable by our tools.

import type { ColumnSummary, DatasetSummary } from './types'

export function suggestQuestions(summary: DatasetSummary, max = 4): string[] {
  const usable = summary.columns.filter((c) => c.missing_pct < 100)
  // Good grouping columns have a handful of distinct values; IDs and free text don't.
  const groups = usable.filter((c) => c.semantic_type === 'categorical' && c.n_unique >= 2 && c.n_unique <= 12)
  const numbers = usable.filter((c) => c.semantic_type === 'numeric' && c.n_unique > 2)
  const byFewestGroups = (a: ColumnSummary, b: ColumnSummary) => a.n_unique - b.n_unique

  const out: string[] = []
  const [group] = [...groups].sort(byFewestGroups)
  if (group) out.push(`What share of rows falls in each ${group.name}?`)
  if (group && numbers[0]) out.push(`What is the average ${numbers[0].name} for each ${group.name}?`)
  if (numbers.length >= 2) out.push(`Which columns are most strongly correlated with ${numbers[0].name}?`)
  if (numbers.length >= 2 && !group) {
    out.push(`How many rows have ${numbers[1].name} above its median, and what is their average ${numbers[0].name}?`)
  }
  out.push('Give me an overview of this dataset and any data-quality issues.')
  return out.slice(0, max)
}
