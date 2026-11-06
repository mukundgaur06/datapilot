import type { JsonRecord } from './types'

const full = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 })
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 })

/** 1,284 · 0.77 · 12.9M: full digits below a million, compact above. */
export function formatNumber(value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return value == null ? '—' : String(value)
  return Math.abs(value) >= 1_000_000 ? compact.format(value) : full.format(value)
}

interface Filter {
  column: string
  op: string
  value: unknown
}

export function describeFilters(filters: unknown): string {
  if (!Array.isArray(filters) || filters.length === 0) return ''
  return (filters as Filter[])
    .map((f) => `${f.column} ${f.op} ${Array.isArray(f.value) ? `[${f.value.join(', ')}]` : String(f.value)}`)
    .join(' and ')
}

/** One-line, human-readable tool call: group_aggregate(group_by=label, agg=count). */
export function formatToolCall(name: string, args: JsonRecord): string {
  const parts = Object.entries(args)
    .filter(([, v]) => v !== null && v !== undefined && !(Array.isArray(v) && v.length === 0))
    .map(([k, v]) => {
      if (k === 'filters') return `where ${describeFilters(v)}`
      if (Array.isArray(v)) return `${k}=${v.join(', ')}`
      return `${k}=${String(v)}`
    })
  return `${name}(${parts.join(', ')})`
}

export const AGG_LABEL: Record<string, string> = {
  count: 'Count',
  mean: 'Average',
  median: 'Median',
  sum: 'Total',
  min: 'Minimum',
  max: 'Maximum',
  std: 'Std. deviation',
  nunique: 'Distinct values',
}
