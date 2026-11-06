import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import type { Artifact } from '../lib/artifacts'
import type { JsonRecord } from '../lib/types'
import { DataTable, DivergingChart, HBarChart, HistogramChart, ScatterPlot, TrendChart } from './Charts'
import { ChartIcon, TableIcon } from './Icons'
import { ImportanceView, ModelCardView } from './ModelCard'

interface Props {
  artifacts: Artifact[]
  selected: string | null
}

export function ArtifactsPanel({ artifacts, selected }: Props) {
  if (artifacts.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-sm text-ink-muted">
        <ChartIcon className="size-8" />
        Charts and tables from the agent's work will appear here.
      </div>
    )
  }
  return (
    <ul className="flex flex-col gap-3" aria-label="Artifacts">
      <AnimatePresence initial={false}>
        {artifacts.map((a) => (
          <motion.li key={a.id} layout initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.25 }}>
            <ArtifactCard artifact={a} selected={a.id === selected} />
          </motion.li>
        ))}
      </AnimatePresence>
    </ul>
  )
}

function ArtifactCard({ artifact, selected }: { artifact: Artifact; selected: boolean }) {
  const ref = useRef<HTMLElement>(null)
  const [view, setView] = useState<'chart' | 'table'>('chart')
  const chartable = ['bar', 'diverging', 'line', 'scatter', 'histogram'].includes(artifact.kind)

  useEffect(() => {
    if (selected) ref.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [selected])

  return (
    <article
      ref={ref}
      className={`rounded-xl border bg-surface p-3.5 transition-shadow ${selected ? 'border-accent ring-2 ring-accent/60' : 'border-hairline'}`}
    >
      <header className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold">{artifact.title}</h3>
          <p className="mt-0.5 text-xs text-ink-muted">{artifact.subtitle}</p>
        </div>
        {chartable && (
          <div className="flex shrink-0 rounded-lg border border-hairline p-0.5" role="group" aria-label="View as">
            {(['chart', 'table'] as const).map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setView(v)}
                aria-pressed={view === v}
                aria-label={v === 'chart' ? 'Chart view' : 'Table view'}
                className={`grid size-6 place-items-center rounded-md ${view === v ? 'bg-surface-2 text-ink' : 'text-ink-muted hover:text-ink'}`}
              >
                {v === 'chart' ? <ChartIcon className="size-3.5" /> : <TableIcon className="size-3.5" />}
              </button>
            ))}
          </div>
        )}
      </header>
      <div className="mt-3">
        {artifact.kind === 'stat' && <StatTile value={artifact.value} detail={artifact.detail} />}
        {artifact.kind === 'table' && <DataTable columns={artifact.columns} rows={artifact.rows} />}
        {artifact.kind === 'bar' &&
          (view === 'chart' ? (
            <HBarChart data={artifact.data} valueLabel={artifact.valueLabel} />
          ) : (
            <DataTable columns={['label', 'value', ...(artifact.data.some((d) => d.pct != null) ? ['pct'] : [])]} rows={artifact.data as unknown as JsonRecord[]} />
          ))}
        {artifact.kind === 'diverging' &&
          (view === 'chart' ? <DivergingChart data={artifact.data} /> : <DataTable columns={['label', 'value']} rows={artifact.data as unknown as JsonRecord[]} />)}
        {artifact.kind === 'line' &&
          (view === 'chart' ? (
            <TrendChart data={artifact.data} xLabel={artifact.xLabel} yLabel={artifact.yLabel} />
          ) : (
            <DataTable columns={['x', 'y']} rows={artifact.data.map((d) => ({ x: d.x, y: d.y }))} />
          ))}
        {artifact.kind === 'scatter' &&
          (view === 'chart' ? (
            <ScatterPlot series={artifact.series} xLabel={artifact.xLabel} yLabel={artifact.yLabel} />
          ) : (
            <DataTable
              columns={['group', artifact.xLabel, artifact.yLabel]}
              rows={artifact.series.flatMap((s) => s.points.map(([x, y]) => ({ group: s.name, [artifact.xLabel]: x, [artifact.yLabel]: y })))}
            />
          ))}
        {artifact.kind === 'histogram' &&
          (view === 'chart' ? (
            <HistogramChart data={artifact.data} xLabel={artifact.xLabel} />
          ) : (
            <DataTable columns={['from', 'to', 'rows']} rows={artifact.data.map((d) => ({ from: d.x0, to: d.x1, rows: d.y }))} />
          ))}
        {artifact.kind === 'model' && <ModelCardView card={artifact.card} />}
        {artifact.kind === 'importance' && <ImportanceView features={artifact.features} terms={artifact.terms} />}
      </div>
    </article>
  )
}

function StatTile({ value, detail }: { value: string; detail?: string }) {
  return (
    <div>
      <p className="text-4xl font-semibold tracking-tight">{value}</p>
      {detail && <p className="mt-1 text-sm text-ink-2">{detail}</p>}
    </div>
  )
}

