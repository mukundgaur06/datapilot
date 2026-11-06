import { motion } from 'motion/react'

import { formatNumber } from '../lib/format'
import type { DatasetSummary } from '../lib/types'
import { TypeGlyph } from './Icons'

interface Props {
  summary: DatasetSummary
  onColumnClick: (name: string) => void
}

/** Column chips: type glyph, name, and a bar for % missing. Clicking one inserts it into the question. */
export function SchemaPanel({ summary, onColumnClick }: Props) {
  return (
    <section aria-labelledby="schema-heading">
      <h2 id="schema-heading" className="truncate text-sm font-semibold" title={summary.name}>
        {summary.name}
      </h2>
      <p className="mt-0.5 text-xs text-ink-muted">
        {formatNumber(summary.n_rows)} rows · {summary.n_columns} columns · click a column to use it
      </p>
      <ul className="mt-3 flex flex-col gap-1.5">
        {summary.columns.map((col, i) => (
          <motion.li
            key={col.name}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: Math.min(i * 0.03, 0.4), duration: 0.25 }}
          >
            <button
              type="button"
              onClick={() => onColumnClick(col.name)}
              title={`${col.semantic_type} · ${col.dtype} · ${col.n_unique} distinct · e.g. ${col.examples.join(', ')}`}
              aria-label={`Insert column ${col.name} (${col.semantic_type}, ${col.missing_pct}% missing) into the question`}
              className="group flex w-full items-center gap-2 rounded-lg border border-hairline bg-surface px-2 py-1.5 text-left transition-colors hover:border-accent/50 hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-accent"
            >
              <TypeGlyph type={col.semantic_type} />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[13px] font-medium">{col.name}</span>
                <span className="mt-1 flex items-center gap-1.5">
                  {/* Missing-values meter: track + fill, with the number beside it (never colour alone). */}
                  <span className="h-1 flex-1 overflow-hidden rounded-full bg-surface-2">
                    <span
                      className={`block h-full rounded-full ${col.missing_pct > 20 ? 'bg-critical' : 'bg-accent'}`}
                      style={{ width: `${Math.max(col.missing_pct, col.missing_pct > 0 ? 2 : 0)}%` }}
                    />
                  </span>
                  <span className="tabular w-16 shrink-0 text-right text-[10px] text-ink-muted">
                    {col.missing_pct === 0 ? 'no missing' : `${col.missing_pct}% missing`}
                  </span>
                </span>
              </span>
            </button>
          </motion.li>
        ))}
      </ul>
    </section>
  )
}
