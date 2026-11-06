import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { Artifact, BarDatum } from '../lib/artifacts'
import { formatNumber } from '../lib/format'
import type { JsonRecord } from '../lib/types'
import { ChartIcon, TableIcon } from './Icons'

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
  const chartable = artifact.kind === 'bar' || artifact.kind === 'diverging'

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

// ---------------------------------------------------------------- charts

const ROW_PX = 28
const axisTick = { fill: 'var(--ink-muted)', fontSize: 11 }
const categoryTick = { fill: 'var(--ink-2)', fontSize: 12 }

/** Category labels get as much room as the longest needs, within limits. */
const labelWidth = (data: BarDatum[]) => Math.min(150, Math.max(48, Math.max(...data.map((d) => d.label.length)) * 7 + 8))

interface TipProps {
  active?: boolean
  payload?: { payload: BarDatum }[]
}

function ChartTooltip({ active, payload, valueLabel, signed }: TipProps & { valueLabel: string; signed?: boolean }) {
  const d = active ? payload?.[0]?.payload : undefined
  if (!d) return null
  return (
    <div className="rounded-lg border border-hairline bg-surface px-2.5 py-1.5 text-xs shadow-md">
      <p className="font-medium">{d.label}</p>
      <p className="tabular text-ink-2">
        {valueLabel}: {signed && d.value > 0 ? '+' : ''}
        {formatNumber(d.value)}
        {d.pct != null && ` (${formatNumber(d.pct)}%)`}
      </p>
    </div>
  )
}

function HBarChart({ data, valueLabel }: { data: BarDatum[]; valueLabel: string }) {
  const showLabels = data.length <= 12 // label every bar tip only while that stays readable
  return (
    <div role="img" aria-label={`Bar chart: ${data.map((d) => `${d.label} ${formatNumber(d.value)}`).join(', ')}`}>
      <ResponsiveContainer width="100%" height={data.length * ROW_PX + 32}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: showLabels ? 56 : 12, bottom: 4, left: 4 }} barCategoryGap={6}>
          <CartesianGrid horizontal={false} stroke="var(--grid)" />
          <XAxis type="number" tick={axisTick} tickFormatter={formatNumber} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="label" width={labelWidth(data)} tick={categoryTick} axisLine={{ stroke: 'var(--axis)' }} tickLine={false} interval={0} />
          <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<ChartTooltip valueLabel={valueLabel} />} isAnimationActive={false} />
          <Bar dataKey="value" fill="var(--series-1)" radius={[0, 4, 4, 0]} maxBarSize={20}>
            {showLabels && (
              <LabelList
                dataKey="value"
                position="right"
                formatter={(v: unknown) => formatNumber(v)}
                style={{ fill: 'var(--ink-2)', fontSize: 11, fontVariantNumeric: 'tabular-nums' }}
              />
            )}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

const signed = (v: unknown) => (typeof v === 'number' ? `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(2)}` : '')

interface LabelProps {
  x?: number | string
  y?: number | string
  width?: number | string
  height?: number | string
  value?: unknown
}

/** Positive r: label at the bar tip. Negative r: label just right of the zero line, so it can
 *  never run into the category names on the left, however long the bar is. */
function DivergingLabel({ x = 0, y = 0, width = 0, height = 0, value }: LabelProps) {
  // A bar's right edge is its tip when r > 0, and the zero line when r < 0.
  const right = Math.max(Number(x), Number(x) + Number(width))
  return (
    <text x={right + 6} y={Number(y) + Number(height) / 2} dy="0.35em" fontSize={11} fill="var(--ink-2)" style={{ fontVariantNumeric: 'tabular-nums' }}>
      {signed(value)}
    </text>
  )
}

/** Correlation r on a fixed −1…+1 scale around a zero line: blue positive, red negative, sign printed too. */
function DivergingChart({ data }: { data: BarDatum[] }) {
  return (
    <div role="img" aria-label={`Correlation chart: ${data.map((d) => `${d.label} ${signed(d.value)}`).join(', ')}`}>
      <ResponsiveContainer width="100%" height={data.length * ROW_PX + 32}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 44, bottom: 4, left: 4 }} barCategoryGap={6}>
          <CartesianGrid horizontal={false} stroke="var(--grid)" />
          <XAxis type="number" domain={[-1, 1]} ticks={[-1, -0.5, 0, 0.5, 1]} tick={axisTick} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="label" width={labelWidth(data)} tick={categoryTick} axisLine={false} tickLine={false} interval={0} />
          <ReferenceLine x={0} stroke="var(--axis)" />
          <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<ChartTooltip valueLabel="r" signed />} isAnimationActive={false} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20}>
            {data.map((d) => (
              <Cell key={d.label} fill={d.value >= 0 ? 'var(--diverge-pos)' : 'var(--diverge-neg)'} />
            ))}
            <LabelList dataKey="value" content={(props) => <DivergingLabel {...(props as LabelProps)} />} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---------------------------------------------------------------- table

function DataTable({ columns, rows }: { columns: string[]; rows: JsonRecord[] }) {
  return (
    <div className="max-h-80 overflow-auto rounded-lg border border-hairline">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-surface-2">
          <tr>
            {columns.map((c) => (
              <th key={c} scope="col" className="px-2.5 py-1.5 font-medium text-ink-2">
                {c.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-t border-hairline">
              {columns.map((c) => (
                <td key={c} className={`px-2.5 py-1.5 ${typeof row[c] === 'number' ? 'tabular text-right' : ''}`}>
                  {typeof row[c] === 'number' ? formatNumber(row[c]) : row[c] == null ? '—' : String(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
