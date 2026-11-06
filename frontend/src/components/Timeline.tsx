import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import Markdown from 'react-markdown'

import { formatNumber, formatToolCall } from '../lib/format'
import { isActive, type RunView, type TimelineItem, type ToolItem } from '../lib/runsReducer'
import { AlertIcon, ChartIcon, CheckIcon, ChevronIcon, SparkIcon, StopIcon, ToolIcon } from './Icons'

interface Props {
  runs: RunView[]
  artifactIds: Set<string>
  selectedArtifact: string | null
  onSelectArtifact: (id: string) => void
}

const enter = { initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.25 } }

export function Timeline({ runs, artifactIds, selectedArtifact, onSelectArtifact }: Props) {
  return (
    <ol className="flex flex-col gap-8">
      {runs.map((run) => (
        <li key={run.localId}>
          <RunSection run={run} artifactIds={artifactIds} selectedArtifact={selectedArtifact} onSelectArtifact={onSelectArtifact} />
        </li>
      ))}
    </ol>
  )
}

function RunSection({ run, artifactIds, selectedArtifact, onSelectArtifact }: { run: RunView } & Omit<Props, 'runs'>) {
  const last = run.items.at(-1)
  const waitingOnModel = isActive(run) && !(last?.kind === 'tool' && !last.result)

  return (
    <section aria-label={`Question: ${run.question}`}>
      <motion.p {...enter} className="ml-auto w-fit max-w-[85%] rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-[15px] text-white">
        {run.question}
      </motion.p>
      <ol className="mt-4 flex flex-col gap-2">
        <AnimatePresence initial={false}>
          {run.items.map((item) => (
            <motion.li key={item.key} layout {...enter}>
              <ItemCard
                item={item}
                artifactId={`${run.localId}:${item.key}`}
                artifactIds={artifactIds}
                selectedArtifact={selectedArtifact}
                onSelectArtifact={onSelectArtifact}
              />
            </motion.li>
          ))}
          {waitingOnModel && (
            <motion.li key="skeleton" layout {...enter} exit={{ opacity: 0 }}>
              <Skeleton label={run.items.length === 0 ? 'Reading the schema and planning…' : 'Thinking about the results…'} />
            </motion.li>
          )}
        </AnimatePresence>
      </ol>
    </section>
  )
}

function Skeleton({ label }: { label: string }) {
  return (
    <div className="rounded-xl border border-hairline bg-surface p-3" aria-hidden="true">
      <div className="flex items-center gap-2 text-xs text-ink-muted">
        <SparkIcon className="size-3.5 animate-pulse motion-reduce:animate-none" />
        {label}
      </div>
      <div className="mt-2 flex flex-col gap-1.5">
        <div className="shimmer h-2.5 w-3/4 rounded" />
        <div className="shimmer h-2.5 w-1/2 rounded" />
      </div>
    </div>
  )
}

function ItemCard({ item, artifactId, artifactIds, selectedArtifact, onSelectArtifact }: { item: TimelineItem; artifactId: string } & Omit<Props, 'runs'>) {
  switch (item.kind) {
    case 'thinking':
      return <ThinkingCard text={item.text} />
    case 'tool':
      return (
        <ToolCard
          item={item}
          hasArtifact={artifactIds.has(artifactId)}
          selected={selectedArtifact === artifactId}
          onShowArtifact={() => onSelectArtifact(artifactId)}
        />
      )
    case 'answer':
      return (
        <div className="rounded-xl border border-hairline bg-surface p-4 shadow-sm">
          {item.partial && (
            <p className="mb-2 flex items-center gap-1.5 text-xs font-medium text-ink-2">
              <AlertIcon className="size-3.5" /> Partial answer
            </p>
          )}
          <div className="markdown text-[15px] leading-relaxed">
            <Markdown>{item.text}</Markdown>
          </div>
          <p className="tabular mt-3 text-[11px] text-ink-muted">
            {item.stats.steps} steps · {((item.stats.elapsed_ms ?? 0) / 1000).toFixed(1)}s
            {item.stats.usage?.input_tokens != null &&
              ` · ${formatNumber((item.stats.usage.input_tokens ?? 0) + (item.stats.usage.output_tokens ?? 0))} tokens`}
          </p>
        </div>
      )
    case 'error':
      return (
        <div role="alert" className="flex gap-2 rounded-xl border border-critical/40 bg-critical/5 p-3 text-sm">
          <AlertIcon className="mt-0.5 size-4 shrink-0 text-critical" />
          <span>{item.message}</span>
        </div>
      )
    case 'cancelled':
      return (
        <div className="flex items-center gap-2 rounded-xl border border-hairline p-3 text-sm text-ink-2">
          <StopIcon className="size-4" /> {item.message}
        </div>
      )
  }
}

function ThinkingCard({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded-xl px-1 text-sm text-ink-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex items-center gap-1.5 rounded text-xs text-ink-muted hover:text-ink focus-visible:outline-2 focus-visible:outline-accent"
      >
        <ChevronIcon className={`size-3 transition-transform ${open ? 'rotate-90' : ''}`} />
        <SparkIcon className="size-3.5" /> Reasoning
      </button>
      {open && <div className="markdown mt-1.5 border-l-2 border-hairline pl-3 text-[13px]"><Markdown>{text}</Markdown></div>}
    </div>
  )
}

function ToolCard({ item, hasArtifact, selected, onShowArtifact }: { item: ToolItem; hasArtifact: boolean; selected: boolean; onShowArtifact: () => void }) {
  const [open, setOpen] = useState(false)
  const pending = !item.result
  return (
    <div
      className={[
        'rounded-xl border bg-surface transition-shadow',
        item.isError ? 'border-critical/40' : 'border-hairline',
        selected ? 'ring-2 ring-accent' : '',
      ].join(' ')}
    >
      <div className="flex items-center gap-2 p-2.5">
        <span className={`grid size-7 shrink-0 place-items-center rounded-lg ${item.isError ? 'bg-critical/10 text-critical' : 'bg-accent/10 text-accent-ink'}`}>
          <ToolIcon className="size-4" />
        </span>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-1.5 rounded text-left focus-visible:outline-2 focus-visible:outline-accent"
        >
          <code className="truncate font-mono text-[12.5px]">{formatToolCall(item.name, item.args)}</code>
          <ChevronIcon className={`size-3 shrink-0 text-ink-muted transition-transform ${open ? 'rotate-90' : ''}`} />
        </button>
        <span className="tabular shrink-0 text-[11px] text-ink-muted">
          {pending ? (
            <span className="inline-flex items-center gap-1">
              <span className="size-1.5 animate-pulse rounded-full bg-accent motion-reduce:animate-none" /> running
            </span>
          ) : item.isError ? (
            <span className="inline-flex items-center gap-1 text-critical">
              <AlertIcon className="size-3" /> error, sent back to the AI
            </span>
          ) : (
            <span className="inline-flex items-center gap-1">
              <CheckIcon className="size-3 text-good" /> {item.durationMs} ms
            </span>
          )}
        </span>
        {hasArtifact && (
          <button
            type="button"
            onClick={onShowArtifact}
            aria-label="Show this result in the artifacts panel"
            title="Show chart"
            className="grid size-7 shrink-0 place-items-center rounded-lg text-ink-2 hover:bg-surface-2 hover:text-ink focus-visible:outline-2 focus-visible:outline-accent"
          >
            <ChartIcon className="size-4" />
          </button>
        )}
      </div>
      {open && (
        <div className="grid grid-cols-[minmax(0,1fr)] gap-2 border-t border-hairline p-2.5 text-[12px]">
          <Json label="Arguments" value={item.args} />
          {item.result && <Json label={item.isError ? 'Error' : 'Result'} value={item.result} />}
        </div>
      )}
    </div>
  )
}

function Json({ label, value }: { label: string; value: unknown }) {
  return (
    <div>
      <p className="mb-1 text-[11px] font-medium text-ink-muted">{label}</p>
      <pre className="max-h-64 overflow-auto rounded-lg bg-surface-2 p-2 font-mono text-[11.5px] leading-snug">{JSON.stringify(value, null, 2)}</pre>
    </div>
  )
}
