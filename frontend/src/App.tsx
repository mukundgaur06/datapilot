import { MotionConfig } from 'motion/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ArtifactsPanel } from './components/ArtifactsPanel'
import { AskBar } from './components/AskBar'
import { CommandPalette } from './components/CommandPalette'
import { AlertIcon, MoonIcon, SunIcon } from './components/Icons'
import { SchemaPanel } from './components/SchemaPanel'
import { Timeline } from './components/Timeline'
import { UploadDropzone } from './components/UploadDropzone'
import { useAgentRuns } from './hooks/useAgentRuns'
import { useTheme } from './hooks/useTheme'
import { ApiError, getHealth, uploadDataset } from './lib/api'
import { artifactFromTool, type Artifact } from './lib/artifacts'
import type { Command } from './lib/commands'
import { isActive, type RunView } from './lib/runsReducer'
import { suggestQuestions } from './lib/suggestions'
import type { Dataset, Health } from './lib/types'

/** Short status line for screen readers (aria-live), so streaming progress is announced, not just shown. */
function liveStatus(run: RunView | undefined): string {
  if (!run) return ''
  const last = run.items.at(-1)
  if (isActive(run)) return last?.kind === 'tool' && !last.result ? `Running ${last.name}` : 'Thinking'
  if (run.status === 'done') return 'Answer ready'
  if (run.status === 'cancelled') return 'Run stopped'
  return 'The run failed'
}

export default function App() {
  const [theme, toggleTheme] = useTheme()
  const [health, setHealth] = useState<Health | null | 'down'>(null)
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('')
  const [selectedArtifact, setSelectedArtifact] = useState<string | null>(null)
  const [paletteOpen, setPaletteOpen] = useState(false)

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const timelineEnd = useRef<HTMLDivElement>(null)

  const { runs, running, ask, stop } = useAgentRuns(dataset?.dataset_id ?? null)

  useEffect(() => {
    getHealth().then(setHealth, () => setHealth('down'))
  }, [])

  // Ctrl+K / Cmd+K opens the command palette from anywhere.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen((o) => !o)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Keep the newest timeline card in view while a run streams in.
  const itemCount = runs.reduce((n, r) => n + r.items.length + 1, 0)
  useEffect(() => {
    timelineEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [itemCount])

  const upload = useCallback(async (file: File) => {
    setUploading(true)
    setUploadError(null)
    try {
      setDataset(await uploadDataset(file))
      setSelectedArtifact(null)
      requestAnimationFrame(() => textareaRef.current?.focus())
    } catch (e) {
      setUploadError(e instanceof ApiError ? e.message : 'Upload failed.')
    } finally {
      setUploading(false)
    }
  }, [])

  const submit = useCallback(
    (question: string) => {
      const q = question.trim()
      if (!q || running) return
      setPrompt('')
      void ask(q)
    },
    [ask, running],
  )

  const insertColumn = useCallback(
    (name: string) => {
      const el = textareaRef.current
      const start = el?.selectionStart ?? prompt.length
      const end = el?.selectionEnd ?? prompt.length
      const before = prompt.slice(0, start)
      const insert = `${before && !before.endsWith(' ') ? ' ' : ''}${name} `
      setPrompt(before + insert + prompt.slice(end))
      requestAnimationFrame(() => {
        el?.focus()
        const caret = before.length + insert.length
        el?.setSelectionRange(caret, caret)
      })
    },
    [prompt],
  )

  // Artifacts: one per successful tool result that's worth pinning, newest first.
  const artifacts = useMemo(() => {
    const out: Artifact[] = []
    for (const run of runs) {
      for (const item of run.items) {
        if (item.kind !== 'tool') continue
        const a = artifactFromTool(`${run.localId}:${item.key}`, item)
        if (a) out.push(a)
      }
    }
    return out.reverse()
  }, [runs])
  const artifactIds = useMemo(() => new Set(artifacts.map((a) => a.id)), [artifacts])

  const suggestions = useMemo(() => (dataset ? suggestQuestions(dataset.summary) : []), [dataset])

  const commands: Command[] = useMemo(
    () => [
      ...suggestions.map((q, i) => ({ id: `ask-${i}`, group: 'Ask', label: q, run: () => submit(q) })),
      ...(running ? [{ id: 'stop', group: 'Run', label: 'Stop the current run', run: stop }] : []),
      { id: 'focus', group: 'Go to', label: 'Question box', run: () => textareaRef.current?.focus() },
      { id: 'upload', group: 'Data', label: dataset ? 'Upload a different CSV' : 'Upload a CSV', run: () => fileInputRef.current?.click() },
      { id: 'theme', group: 'View', label: theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode', run: toggleTheme },
    ],
    [suggestions, running, stop, dataset, theme, toggleTheme, submit],
  )

  const latest = runs.at(-1)
  const llmMissing = health !== null && health !== 'down' && !health.llm_configured

  return (
    <MotionConfig reducedMotion="user">
      <div className="flex h-full flex-col">
        <header className="flex shrink-0 items-center gap-3 border-b border-hairline bg-surface px-4 py-2.5">
          <img src="/favicon.svg" alt="" className="size-6" />
          <h1 className="text-[15px] font-semibold">DataPilot</h1>
          <span className="hidden text-xs text-ink-muted sm:inline">your AI data analyst</span>
          <div className="ml-auto flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setPaletteOpen(true)}
              className="hidden items-center gap-2 rounded-lg border border-hairline px-2.5 py-1 text-xs text-ink-muted hover:text-ink focus-visible:outline-2 focus-visible:outline-accent sm:flex"
            >
              Commands <kbd className="font-mono">Ctrl K</kbd>
            </button>
            <button
              type="button"
              onClick={toggleTheme}
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              className="grid size-8 place-items-center rounded-lg text-ink-2 hover:bg-surface-2 hover:text-ink focus-visible:outline-2 focus-visible:outline-accent"
            >
              {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
            </button>
          </div>
        </header>

        {(health === 'down' || llmMissing) && (
          <div role="alert" className="flex items-center gap-2 border-b border-critical/30 bg-critical/5 px-4 py-2 text-sm">
            <AlertIcon className="size-4 shrink-0 text-critical" />
            {health === 'down'
              ? 'The DataPilot server is not reachable. Start the backend (uvicorn app.main:app) and reload this page.'
              : 'The server has no Gemini API key, so questions will fail. Add GEMINI_API_KEY to backend/.env and restart it.'}
          </div>
        )}

        {!dataset ? (
          <main className="grid flex-1 place-items-center overflow-y-auto p-6">
            <div className="w-full max-w-xl text-center">
              <h2 className="text-2xl font-semibold tracking-tight">Ask your data anything</h2>
              <p className="mt-2 text-ink-2">
                Upload a CSV, ask a question in plain English, and watch the agent compute the answer step by step.
              </p>
              <div className="mt-8">
                <UploadDropzone onFile={upload} busy={uploading} error={uploadError} inputRef={fileInputRef} />
              </div>
            </div>
          </main>
        ) : (
          <div className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[minmax(0,1fr)_auto] lg:grid-cols-[260px_minmax(0,1fr)_380px] lg:grid-rows-1">
            <aside className="hidden min-h-0 flex-col gap-4 overflow-y-auto border-r border-hairline p-4 lg:flex">
              <SchemaPanel summary={dataset.summary} onColumnClick={insertColumn} />
              <UploadDropzone onFile={upload} busy={uploading} error={uploadError} compact inputRef={fileInputRef} />
            </aside>

            <main className="flex min-h-0 flex-col">
              <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-8">
                <div className="mx-auto max-w-2xl">
                  {runs.length === 0 ? (
                    <div className="py-16 text-center">
                      <h2 className="text-lg font-semibold">What would you like to know?</h2>
                      <p className="mt-1 text-sm text-ink-2">
                        Try a suggestion below, or click a column on the left to use it in your question.
                      </p>
                    </div>
                  ) : (
                    <Timeline runs={runs} artifactIds={artifactIds} selectedArtifact={selectedArtifact} onSelectArtifact={setSelectedArtifact} />
                  )}
                  <div ref={timelineEnd} />
                </div>
              </div>
              <div className="shrink-0 px-4 pb-3 sm:px-8">
                <div className="mx-auto max-w-2xl">
                  <AskBar
                    value={prompt}
                    onChange={setPrompt}
                    onSubmit={() => submit(prompt)}
                    onStop={stop}
                    running={running}
                    disabled={!dataset}
                    suggestions={runs.length === 0 ? suggestions : []}
                    onSuggestion={submit}
                    textareaRef={textareaRef}
                  />
                </div>
              </div>
            </main>

            <aside className="max-h-[40vh] min-h-0 overflow-y-auto border-t border-hairline bg-page p-4 lg:max-h-none lg:border-t-0 lg:border-l" aria-label="Artifacts panel">
              <h2 className="mb-3 text-sm font-semibold">Artifacts</h2>
              <ArtifactsPanel artifacts={artifacts} selected={selectedArtifact} />
            </aside>
          </div>
        )}

        <p className="sr-only" aria-live="polite">
          {liveStatus(latest)}
        </p>
        <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} commands={commands} />
      </div>
    </MotionConfig>
  )
}
