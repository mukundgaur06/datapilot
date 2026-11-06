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

describe('Phase 3 artifacts', () => {
  it('builds a model card with metrics next to the baseline', () => {
    const a = artifactFromTool(
      'm',
      tool('train_model', { target: 'label' }, {
        model_id: 'm_1', task: 'classification', target: 'label', model: 'logistic_regression', n_rows_used: 380,
        validation: '5-fold stratified cross-validation', primary_metric: 'f1', baseline_compared_on: 'roc_auc',
        metrics: { f1: { mean: 0.98, std: 0.01 }, roc_auc: { mean: 0.999, std: 0.001 } },
        baseline: { model: 'always predicts the most common class', metrics: { f1: { mean: 0, std: 0 }, roc_auc: { mean: 0.5, std: 0 } } },
        beats_baseline: true, features: { numeric: [], categorical: [], text: ['message'] }, dropped_features: [],
        confusion_matrix: { labels: ['ham', 'spam'], matrix: [[188, 2], [6, 184]] }, caveats: ['Small sample.'],
      }),
    )
    expect(a).toMatchObject({ kind: 'model', title: 'Model: predicting label' })
    expect(a?.subtitle).toBe('Logistic regression · 380 rows · 5-fold stratified cross-validation')
    if (a?.kind !== 'model') throw new Error('expected model card')
    expect(a.card.metrics).toEqual([
      { name: 'f1', mean: 0.98, std: 0.01, baseline: 0 },
      { name: 'roc_auc', mean: 0.999, std: 0.001, baseline: 0.5 },
    ])
    expect(a.card.confusion?.matrix[1]).toEqual([6, 184])
  })

  it('turns feature importance into column bars and per-class word groups', () => {
    const a = artifactFromTool(
      'i',
      tool('feature_importance', { model_id: 'm_1' }, {
        method: 'permutation', features: [{ feature: 'message', importance: 0.49, std: 0.04 }],
        top_terms: { kind: 'direction', toward: { spam: [{ term: 'claim', weight: 1.4 }], ham: [{ term: 'meeting', weight: 1.1 }] } },
      }),
    )
    if (a?.kind !== 'importance') throw new Error('expected importance')
    expect(a.terms.map((g) => g.label)).toEqual(['Words pushing toward "spam"', 'Words pushing toward "ham"'])
    expect(a.terms[0].items).toEqual([{ label: 'claim', value: 1.4 }])
  })

  it('maps make_chart specs to chart artifacts, using only the ui data', () => {
    const scatter = artifactFromTool('s', tool('make_chart', { kind: 'scatter' }, {
      chart: 'scatter', summary: {}, ui: { kind: 'scatter', title: 'b vs a', x_label: 'a', y_label: 'b', n_rows_used: 3, sampled: false, series: [{ name: 'b', points: [[1, 2], [2, 3]] }] },
    }))
    expect(scatter).toMatchObject({ kind: 'scatter', title: 'b vs a', xLabel: 'a', yLabel: 'b' })
    const hist = artifactFromTool('h', tool('make_chart', { kind: 'histogram' }, {
      ui: { kind: 'histogram', title: 'Distribution of Age', x_label: 'Age', n_rows_used: 8, data: [{ x0: 20, x1: 30, y: 3 }] },
    }))
    expect(hist).toMatchObject({ kind: 'histogram', xLabel: 'Age' })
    expect(artifactFromTool('x', tool('make_chart', {}, { chart: 'bar' }))).toBeNull() // no ui payload
  })

  it('shows predictions next to the inputs with a confidence column', () => {
    const a = artifactFromTool('p', tool('predict', { model_id: 'm_1', rows: [{ message: 'free prize' }] }, {
      model_id: 'm_1', target: 'label', predictions: [{ prediction: 'spam', probabilities: { spam: 0.93, ham: 0.07 } }],
    }))
    expect(a).toMatchObject({ kind: 'table', title: 'Predicted label', columns: ['message', 'prediction', 'confidence'] })
    expect(a?.kind === 'table' && a.rows[0]).toEqual({ message: 'free prize', prediction: 'spam', confidence: '93%' })
  })
})
