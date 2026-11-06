// Turn tool results into things worth pinning: charts, stat tiles, tables.
// The backend returns data, never pictures; the form is picked here by the data's job:
//   one value -> stat tile · values per category -> bar chart · signed r -> diverging bars ·
//   multi-dimensional -> table.

import { AGG_LABEL, describeFilters, formatNumber } from './format'
import type { ToolItem } from './runsReducer'
import type { JsonRecord } from './types'

export interface BarDatum {
  label: string
  value: number
  pct?: number
}

export type Artifact =
  | { kind: 'bar'; id: string; title: string; subtitle: string; valueLabel: string; data: BarDatum[] }
  | { kind: 'diverging'; id: string; title: string; subtitle: string; data: BarDatum[] }
  | { kind: 'stat'; id: string; title: string; subtitle: string; value: string; detail?: string }
  | { kind: 'table'; id: string; title: string; subtitle: string; columns: string[]; rows: JsonRecord[] }

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

function aggTitle(agg: string, column: string | null | undefined): string {
  if (agg === 'count') return column ? `Non-empty ${column} values` : 'Number of rows'
  return `${AGG_LABEL[agg] ?? agg} of ${column}`
}

function groupAggregate(id: string, args: JsonRecord, r: JsonRecord): Artifact | null {
  const agg = String(r.agg ?? args.agg ?? 'count')
  const column = (r.column ?? args.column) as string | null
  const groupBy = (r.group_by ?? args.group_by ?? []) as string[]
  const where = describeFilters(args.filters)
  const scope = where ? `Rows where ${where} · ${formatNumber(r.n_rows_matched)} of ${formatNumber(r.n_rows_total)}` : `All ${formatNumber(r.n_rows_total)} rows`

  if (groupBy.length === 0) {
    if (!isNum(r.value)) return null
    const detail = agg === 'count' && isNum(r.pct_of_rows) ? `${formatNumber(r.pct_of_rows)}% of all rows` : undefined
    return { kind: 'stat', id, title: aggTitle(agg, column), subtitle: scope, value: formatNumber(r.value), detail }
  }

  const rows = (r.rows ?? []) as JsonRecord[]
  if (rows.length === 0) return null
  const more = r.truncated ? ` · top ${rows.length} of ${formatNumber(r.n_groups)} groups` : ''

  if (groupBy.length === 1 && rows.every((row) => isNum(row.value))) {
    const key = groupBy[0]
    return {
      kind: 'bar',
      id,
      title: agg === 'count' ? `Rows by ${key}` : `${aggTitle(agg, column)} by ${key}`,
      subtitle: scope + more,
      valueLabel: agg === 'count' ? 'Rows' : (AGG_LABEL[agg] ?? agg),
      data: rows.map((row) => ({
        label: row[key] == null ? '(missing)' : String(row[key]),
        value: row.value as number,
        pct: isNum(row.pct_of_rows) ? row.pct_of_rows : undefined,
      })),
    }
  }

  const columns = [...groupBy, 'value', ...(rows.some((row) => 'pct_of_rows' in row) ? ['pct_of_rows'] : [])]
  return { kind: 'table', id, title: `${aggTitle(agg, column)} by ${groupBy.join(' and ')}`, subtitle: scope + more, columns, rows }
}

function correlation(id: string, args: JsonRecord, r: JsonRecord): Artifact | null {
  const pairs = (r.pairs ?? []) as { a: string; b: string; r: number }[]
  if (pairs.length === 0) return null
  const target = args.target as string | undefined
  return {
    kind: 'diverging',
    id,
    title: target ? `Correlation with ${target}` : 'Strongest correlations',
    subtitle: `${r.method === 'spearman' ? 'Spearman' : 'Pearson'} r, from −1 to +1 · not causation`,
    data: pairs.map((p) => ({
      label: target ? (p.a === target ? p.b : p.a) : `${p.a} × ${p.b}`,
      value: p.r,
    })),
  }
}

function profile(id: string, r: JsonRecord): Artifact | null {
  const cols = (r.columns ?? []) as JsonRecord[]
  if (cols.length === 0) return null
  const rows = cols.map((c) => {
    const top = (c.top_values as { value: string; count: number }[] | undefined)?.[0]
    return {
      column: c.name,
      type: c.dtype,
      missing: c.missing,
      mean: c.mean ?? null,
      median: c.median ?? null,
      'most common': top ? `${top.value} (${formatNumber(top.count)})` : null,
    }
  })
  return {
    kind: 'table',
    id,
    title: 'Dataset profile',
    subtitle: `${formatNumber(r.n_rows)} rows × ${formatNumber(r.n_columns)} columns · ${formatNumber(r.duplicate_rows)} duplicate rows`,
    columns: ['column', 'type', 'missing', 'mean', 'median', 'most common'],
    rows,
  }
}

/** Artifact for a finished, successful tool call, or null if there's nothing worth pinning. */
export function artifactFromTool(id: string, item: ToolItem): Artifact | null {
  if (!item.result || item.isError) return null
  switch (item.name) {
    case 'group_aggregate':
      return groupAggregate(id, item.args, item.result)
    case 'correlation':
      return correlation(id, item.args, item.result)
    case 'profile_dataset':
      return profile(id, item.result)
    default:
      return null
  }
}
