// Model card (from train_model) and "what drives the model" (from feature_importance).
// Built to be honest at a glance: the verdict vs the baseline comes first, every metric
// shows its fold-to-fold spread, and the caveats are always visible, never collapsed.

import type { ModelCard as Card, TermGroup, BarDatum } from '../lib/artifacts'
import { formatNumber } from '../lib/format'
import { HBarChart } from './Charts'
import { AlertIcon, CheckIcon } from './Icons'

const METRIC_LABEL: Record<string, string> = {
  f1: 'F1',
  precision: 'Precision',
  recall: 'Recall',
  accuracy: 'Accuracy',
  roc_auc: 'ROC-AUC',
  f1_macro: 'F1 (macro)',
  precision_macro: 'Precision (macro)',
  recall_macro: 'Recall (macro)',
  roc_auc_ovr: 'ROC-AUC (one-vs-rest)',
  r2: 'R²',
  mae: 'Mean absolute error',
  rmse: 'Root mean squared error',
}
const LOWER_IS_BETTER = new Set(['mae', 'rmse'])
const label = (m: string) => METRIC_LABEL[m] ?? m
const fmt = (v: number) => (Number.isFinite(v) ? (Math.abs(v) >= 100 ? formatNumber(v) : v.toFixed(3)) : '—')

export function ModelCardView({ card }: { card: Card }) {
  const compared = card.metrics.find((m) => m.name === card.comparedOn)
  return (
    <div className="flex flex-col gap-4 text-sm">
      {/* Verdict: icon + words, never colour alone */}
      <div
        className={`flex items-start gap-2 rounded-lg border p-2.5 ${card.beatsBaseline ? 'border-good/40 bg-good/5' : 'border-critical/40 bg-critical/5'}`}
      >
        {card.beatsBaseline ? <CheckIcon className="mt-0.5 size-4 shrink-0 text-good" /> : <AlertIcon className="mt-0.5 size-4 shrink-0 text-critical" />}
        <p>
          <span className="font-medium">{card.beatsBaseline ? 'Beats the baseline' : 'Does not clearly beat the baseline'}</span>
          {compared && (
            <span className="text-ink-2">
              {' '}
              on {label(compared.name)}: {fmt(compared.mean)} vs {fmt(compared.baseline)} ({card.baselineModel})
            </span>
          )}
        </p>
      </div>

      <section aria-label="Cross-validated metrics">
        <table className="w-full text-left text-xs">
          <thead className="text-ink-muted">
            <tr>
              <th scope="col" className="py-1 font-medium">
                Metric
              </th>
              <th scope="col" className="py-1 text-right font-medium">
                Model
              </th>
              <th scope="col" className="py-1 text-right font-medium">
                Baseline
              </th>
            </tr>
          </thead>
          <tbody className="tabular">
            {card.metrics.map((m) => (
              <tr key={m.name} className="border-t border-hairline">
                <td className={`py-1.5 ${m.name === card.primary ? 'font-semibold' : ''}`}>
                  {label(m.name)}
                  {LOWER_IS_BETTER.has(m.name) && <span className="text-ink-muted"> (lower is better)</span>}
                </td>
                <td className="py-1.5 text-right whitespace-nowrap">
                  {fmt(m.mean)} <span className="text-ink-muted">± {fmt(m.std)}</span>
                </td>
                <td className="py-1.5 text-right text-ink-2">{fmt(m.baseline)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-1 text-[11px] text-ink-muted">
          ± is the spread across cross-validation folds.
          {card.metricsNote && ` ${card.metricsNote.charAt(0).toUpperCase() + card.metricsNote.slice(1)}.`}
        </p>
      </section>

      {card.confusion && <ConfusionMatrix labels={card.confusion.labels} matrix={card.confusion.matrix} />}

      <section aria-label="Features" className="text-xs">
        <p className="text-ink-muted">Features used</p>
        <p className="mt-0.5">
          {[...card.features.numeric, ...card.features.categorical, ...card.features.text.map((t) => `${t} (text)`)].join(', ') || '—'}
        </p>
        {card.dropped.length > 0 && (
          <>
            <p className="mt-2 text-ink-muted">Dropped automatically</p>
            <ul className="mt-0.5 list-disc pl-4">
              {card.dropped.map((d) => (
                <li key={d.column}>
                  <span className="font-medium">{d.column}</span>: {d.reason}
                </li>
              ))}
            </ul>
          </>
        )}
      </section>

      {card.caveats.length > 0 && (
        <section aria-label="Caveats" className="rounded-lg bg-surface-2 p-2.5 text-xs">
          <p className="flex items-center gap-1.5 font-medium">
            <AlertIcon className="size-3.5" /> Caveats
          </p>
          <ul className="mt-1.5 flex list-disc flex-col gap-1 pl-4 text-ink-2">
            {card.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </section>
      )}
      <p className="text-[11px] text-ink-muted">Model id: {card.modelId}</p>
    </div>
  )
}

/** Rows = actual, columns = predicted. Shade = share of the actual class (one hue, light to dark), number always printed. */
function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  return (
    <section aria-label="Confusion matrix">
      <p className="mb-1.5 text-xs text-ink-muted">Confusion matrix (out-of-fold predictions)</p>
      <table className="tabular text-xs">
        <thead>
          <tr>
            <th scope="col" className="px-1.5 pb-1 text-left font-normal text-ink-muted">
              actual ↓ / predicted →
            </th>
            {labels.map((l) => (
              <th key={l} scope="col" className="px-1.5 pb-1 font-medium">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1
            return (
              <tr key={labels[i]}>
                <th scope="row" className="px-1.5 text-left font-medium">
                  {labels[i]}
                </th>
                {row.map((v, j) => {
                  const share = v / total
                  return (
                    <td
                      key={j}
                      title={`${labels[i]} predicted as ${labels[j]}: ${v} (${Math.round(share * 100)}% of ${labels[i]})`}
                      className="min-w-14 rounded-md border-2 border-surface px-2 py-1.5 text-center"
                      style={{
                        background: `color-mix(in oklab, var(--series-1) ${Math.round(8 + share * 80)}%, var(--surface))`,
                        color: share > 0.55 ? '#ffffff' : 'var(--ink)',
                      }}
                    >
                      {formatNumber(v)}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </section>
  )
}

export function ImportanceView({ features, terms }: { features: BarDatum[]; terms: TermGroup[] }) {
  const showColumns = features.length > 1 // one input column (e.g. just the text) says nothing; skip the chart
  return (
    <div className="flex flex-col gap-4">
      {showColumns && (
        <section>
          <p className="mb-1 text-xs text-ink-muted">Drop in score when each column is shuffled</p>
          <HBarChart data={features} valueLabel="Importance" />
        </section>
      )}
      {terms.map((g) => (
        <section key={g.label}>
          <p className="mb-1 text-xs text-ink-muted">{g.label}</p>
          <HBarChart data={g.items} valueLabel="Weight" />
        </section>
      ))}
      <p className="text-[11px] text-ink-muted">Shows what the model relies on, not what causes the outcome.</p>
    </div>
  )
}
