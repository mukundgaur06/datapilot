// Chart components. Colour roles come from CSS variables (index.css), so every chart
// follows the light/dark theme; the palette was validated for colour-blind separation.

import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'

import type { BarDatum } from '../lib/artifacts'
import { formatNumber } from '../lib/format'
import type { JsonRecord } from '../lib/types'

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

export function HBarChart({ data, valueLabel }: { data: BarDatum[]; valueLabel: string }) {
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
export function DivergingChart({ data }: { data: BarDatum[] }) {
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

export function DataTable({ columns, rows }: { columns: string[]; rows: JsonRecord[] }) {
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

// ---------------------------------------------------------------- line, scatter, histogram

const CHART_H = 240
const margin = { top: 8, right: 16, bottom: 20, left: 4 }
const axisLabel = (value: string, position: 'insideBottom' | 'insideLeft') => ({
  value,
  position,
  offset: position === 'insideBottom' ? -12 : 10,
  angle: position === 'insideLeft' ? -90 : 0,
  style: { fill: 'var(--ink-muted)', fontSize: 11, textAnchor: 'middle' as const },
})

function SimpleTip({ active, payload, label, lines }: { active?: boolean; payload?: { payload: JsonRecord }[]; label?: unknown; lines: (d: JsonRecord, label?: unknown) => string[] }) {
  const d = active ? payload?.[0]?.payload : undefined
  if (!d) return null
  const [head, ...rest] = lines(d, label)
  return (
    <div className="rounded-lg border border-hairline bg-surface px-2.5 py-1.5 text-xs shadow-md">
      <p className="font-medium">{head}</p>
      {rest.map((l) => (
        <p key={l} className="tabular text-ink-2">
          {l}
        </p>
      ))}
    </div>
  )
}

export function TrendChart({ data, xLabel, yLabel }: { data: { x: string | number; y: number }[]; xLabel: string; yLabel: string }) {
  return (
    <div role="img" aria-label={`Line chart of ${yLabel} over ${xLabel}, ${data.length} points, from ${formatNumber(data[0]?.y)} to ${formatNumber(data.at(-1)?.y)}`}>
      <ResponsiveContainer width="100%" height={CHART_H}>
        <LineChart data={data} margin={margin}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey="x" tick={axisTick} tickLine={false} axisLine={{ stroke: 'var(--axis)' }} minTickGap={24} label={axisLabel(xLabel, 'insideBottom')} />
          <YAxis tick={axisTick} tickFormatter={formatNumber} tickLine={false} axisLine={false} width={48} />
          {/* Crosshair: a hairline that snaps to the nearest x, so readers aim at a position, not a 2px line. */}
          <Tooltip
            cursor={{ stroke: 'var(--ink-muted)', strokeWidth: 1 }}
            isAnimationActive={false}
            content={<SimpleTip lines={(d) => [String(d.x), `${yLabel}: ${formatNumber(d.y)}`]} />}
          />
          <Line type="monotone" dataKey="y" stroke="var(--series-1)" strokeWidth={2} dot={false} activeDot={{ r: 4, stroke: 'var(--surface)', strokeWidth: 2 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// Fixed slot order (never cycled); "Other" is a neutral grey, never a 4th hue.
const SERIES_COLORS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)']
const seriesColor = (name: string, i: number) => (name === 'Other' ? 'var(--series-other)' : SERIES_COLORS[i] ?? 'var(--series-other)')

export function ScatterPlot({ series, xLabel, yLabel }: { series: { name: string; points: [number, number][] }[]; xLabel: string; yLabel: string }) {
  const sets = series.map((s) => ({ name: s.name, data: s.points.map(([x, y]) => ({ x, y, group: s.name })) }))
  const n = series.reduce((k, s) => k + s.points.length, 0)
  return (
    <div role="img" aria-label={`Scatter plot of ${yLabel} against ${xLabel}, ${n} points${series.length > 1 ? `, coloured by ${series.map((s) => s.name).join(', ')}` : ''}`}>
      <ResponsiveContainer width="100%" height={CHART_H + (series.length > 1 ? 24 : 0)}>
        <ScatterChart margin={margin}>
          <CartesianGrid stroke="var(--grid)" />
          <XAxis type="number" dataKey="x" name={xLabel} tick={axisTick} tickFormatter={formatNumber} tickLine={false} axisLine={{ stroke: 'var(--axis)' }} domain={['auto', 'auto']} label={axisLabel(xLabel, 'insideBottom')} />
          <YAxis type="number" dataKey="y" name={yLabel} tick={axisTick} tickFormatter={formatNumber} tickLine={false} axisLine={false} width={48} domain={['auto', 'auto']} />
          <Tooltip
            isAnimationActive={false}
            cursor={{ stroke: 'var(--ink-muted)', strokeDasharray: '0' }}
            content={<SimpleTip lines={(d) => [String(d.group), `${xLabel}: ${formatNumber(d.x)}`, `${yLabel}: ${formatNumber(d.y)}`]} />}
          />
          {series.length > 1 && <Legend verticalAlign="top" height={24} iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, color: 'var(--ink-2)' }} />}
          {sets.map((s, i) => (
            // r=4 markers with a 2px surface ring keep overlapping points legible.
            <Scatter key={s.name} name={s.name} data={s.data} fill={seriesColor(s.name, i)} fillOpacity={0.75} stroke="var(--surface)" strokeWidth={1.5} shape="circle" isAnimationActive={false} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

export function HistogramChart({ data, xLabel }: { data: { x0: number; x1: number; y: number }[]; xLabel: string }) {
  const bins = data.map((d) => ({ ...d, label: `${formatNumber(d.x0)}–${formatNumber(d.x1)}` }))
  const peak = bins.reduce((a, b) => (b.y > a.y ? b : a), bins[0])
  return (
    <div role="img" aria-label={`Histogram of ${xLabel}: ${bins.length} bins, most values between ${peak?.label} (${peak?.y} rows)`}>
      <ResponsiveContainer width="100%" height={CHART_H}>
        <BarChart data={bins} margin={margin} barCategoryGap={2}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey="x0" tick={axisTick} tickFormatter={formatNumber} tickLine={false} axisLine={{ stroke: 'var(--axis)' }} minTickGap={20} label={axisLabel(xLabel, 'insideBottom')} />
          <YAxis tick={axisTick} tickFormatter={formatNumber} tickLine={false} axisLine={false} width={40} allowDecimals={false} />
          <Tooltip
            cursor={{ fill: 'var(--surface-2)' }}
            isAnimationActive={false}
            content={<SimpleTip lines={(d) => [`${xLabel} ${String(d.label)}`, `${formatNumber(d.y)} rows`]} />}
          />
          <Bar dataKey="y" fill="var(--series-1)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
