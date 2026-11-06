// Turn tool results into things worth pinning: charts, stat tiles, tables, model cards.
// The backend returns data, never pictures; the form is picked here by the data's job:
//   one value -> stat tile · values per category -> bar chart · signed r -> diverging bars ·
//   trend -> line · relationship -> scatter · distribution -> histogram ·
//   multi-dimensional -> table · trained model -> model card.

import { AGG_LABEL, describeFilters, formatNumber } from './format'
import type { ToolItem } from './runsReducer'
import type { JsonRecord } from './types'

export interface BarDatum {
  label: string
  value: number
  pct?: number
}

export interface Metric {
  name: string
  mean: number
  std: number
  baseline: number
}

export interface ModelCard {
  modelId: string
  task: 'classification' | 'regression'
  target: string
  model: string
  nRows: number
  validation: string
  primary: string
  comparedOn: string
  metricsNote?: string
  metrics: Metric[]
  beatsBaseline: boolean
  baselineModel: string
  features: { numeric: string[]; categorical: string[]; text: string[] }
  dropped: { column: string; reason: string }[]
  confusion?: { labels: string[]; matrix: number[][] }
  caveats: string[]
}

export interface TermGroup {
  label: string
  items: BarDatum[]
}

type Base = { id: string; title: string; subtitle: string }

export type Artifact =
  | (Base & { kind: 'bar'; valueLabel: string; data: BarDatum[] })
  | (Base & { kind: 'diverging'; data: BarDatum[] })
  | (Base & { kind: 'stat'; value: string; detail?: string })
  | (Base & { kind: 'table'; columns: string[]; rows: JsonRecord[] })
  | (Base & { kind: 'line'; xLabel: string; yLabel: string; xType: 'date' | 'number'; data: { x: string | number; y: number }[] })
  | (Base & { kind: 'scatter'; xLabel: string; yLabel: string; series: { name: string; points: [number, number][] }[] })
  | (Base & { kind: 'histogram'; xLabel: string; data: { x0: number; x1: number; y: number }[] })
  | (Base & { kind: 'model'; card: ModelCard })
  | (Base & { kind: 'importance'; features: BarDatum[]; terms: TermGroup[] })

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

const MODEL_NAMES: Record<string, string> = {
  logistic_regression: 'Logistic regression',
  random_forest: 'Random forest',
  ridge_regression: 'Ridge regression',
}
export const modelName = (m: string) => MODEL_NAMES[m] ?? m.replace(/_/g, ' ')

function chart(id: string, r: JsonRecord): Artifact | null {
  const ui = r.ui as JsonRecord | undefined
  if (!ui) return null
  const title = String(ui.title ?? 'Chart')
  const used = `${formatNumber(ui.n_rows_used)} rows`
  const xLabel = String(ui.x_label ?? '')
  const yLabel = String(ui.y_label ?? '')
  switch (ui.kind) {
    case 'bar': {
      const data = ((ui.data ?? []) as { x: string; y: number }[]).filter((d) => isNum(d.y)).map((d) => ({ label: d.x, value: d.y }))
      if (!data.length) return null
      return { kind: 'bar', id, title, subtitle: used + (ui.truncated ? ` · top ${data.length} groups` : ''), valueLabel: yLabel, data }
    }
    case 'line': {
      const data = ((ui.data ?? []) as { x: string | number; y: number }[]).filter((d) => isNum(d.y))
      if (!data.length) return null
      return { kind: 'line', id, title, subtitle: used, xLabel, yLabel, xType: ui.x_type === 'date' ? 'date' : 'number', data }
    }
    case 'scatter': {
      const series = (ui.series ?? []) as { name: string; points: [number, number][] }[]
      if (!series.some((s) => s.points.length)) return null
      return { kind: 'scatter', id, title, subtitle: used + (ui.sampled ? ' · random sample of 500 points' : ''), xLabel, yLabel, series }
    }
    case 'histogram': {
      const data = (ui.data ?? []) as { x0: number; x1: number; y: number }[]
      if (!data.length) return null
      return { kind: 'histogram', id, title, subtitle: used, xLabel, data }
    }
    default:
      return null
  }
}

type MetricStats = Record<string, { mean: number; std: number }>

function modelCard(id: string, r: JsonRecord): Artifact | null {
  const metrics = r.metrics as MetricStats | undefined
  if (!metrics) return null
  const base = (r.baseline as { model: string; metrics: MetricStats } | undefined) ?? { model: '', metrics: {} }
  const cm = r.confusion_matrix as { labels: string[]; matrix: number[][] } | undefined
  const card: ModelCard = {
    modelId: String(r.model_id),
    task: r.task === 'regression' ? 'regression' : 'classification',
    target: String(r.target),
    model: String(r.model),
    nRows: Number(r.n_rows_used),
    validation: String(r.validation ?? ''),
    primary: String(r.primary_metric),
    comparedOn: String(r.baseline_compared_on ?? r.primary_metric),
    metricsNote: r.metrics_note as string | undefined,
    metrics: Object.entries(metrics).map(([name, m]) => ({ name, mean: m.mean, std: m.std, baseline: base.metrics[name]?.mean ?? NaN })),
    beatsBaseline: Boolean(r.beats_baseline),
    baselineModel: base.model,
    features: (r.features as ModelCard['features']) ?? { numeric: [], categorical: [], text: [] },
    dropped: (r.dropped_features as ModelCard['dropped']) ?? [],
    confusion: cm ? { labels: cm.labels, matrix: cm.matrix } : undefined,
    caveats: (r.caveats as string[]) ?? [],
  }
  return {
    kind: 'model',
    id,
    title: `Model: predicting ${card.target}`,
    subtitle: `${modelName(card.model)} · ${formatNumber(card.nRows)} rows · ${card.validation}`,
    card,
  }
}

function importance(id: string, r: JsonRecord): Artifact | null {
  const features = ((r.features ?? []) as { feature: string; importance: number }[]).map((f) => ({ label: f.feature, value: f.importance }))
  const t = r.top_terms as JsonRecord | null | undefined
  const toItems = (xs: unknown) => ((xs ?? []) as { term: string; weight: number }[]).map((x) => ({ label: x.term, value: Math.abs(x.weight) }))
  const terms: TermGroup[] = []
  if (t?.toward) {
    for (const [cls, xs] of Object.entries(t.toward as JsonRecord)) terms.push({ label: `Words pushing toward "${cls}"`, items: toItems(xs) })
  }
  if (t?.increase) {
    terms.push({ label: 'Words that raise the prediction', items: toItems(t.increase) }, { label: 'Words that lower it', items: toItems(t.decrease) })
  }
  if (t?.kind === 'magnitude') terms.push({ label: 'Most influential words', items: toItems(t.terms) })
  if (!features.length && !terms.length) return null
  return { kind: 'importance', id, title: 'What drives the model', subtitle: String(r.method ?? ''), features, terms: terms.filter((g) => g.items.length) }
}

function predictions(id: string, args: JsonRecord, r: JsonRecord): Artifact | null {
  const preds = (r.predictions ?? []) as { prediction: unknown; probabilities?: Record<string, number> }[]
  if (!preds.length) return null
  const inputs = (args.rows ?? []) as JsonRecord[]
  const inputCols = [...new Set(inputs.flatMap((row) => Object.keys(row)))]
  const hasProba = Boolean(preds[0].probabilities)
  const rows = preds.map((p, i) => ({
    ...inputs[i],
    prediction: p.prediction,
    ...(p.probabilities ? { confidence: `${Math.round(Math.max(...Object.values(p.probabilities)) * 100)}%` } : {}),
  }))
  return {
    kind: 'table',
    id,
    title: `Predicted ${String(r.target)}`,
    subtitle: `${preds.length} row${preds.length > 1 ? 's' : ''} · model ${String(r.model_id)}`,
    columns: [...inputCols, 'prediction', ...(hasProba ? ['confidence'] : [])],
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
    case 'make_chart':
      return chart(id, item.result)
    case 'train_model':
      return modelCard(id, item.result)
    case 'feature_importance':
      return importance(id, item.result)
    case 'predict':
      return predictions(id, item.args, item.result)
    default:
      return null
  }
}
