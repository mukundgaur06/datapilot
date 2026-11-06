import { SendIcon, StopIcon } from './Icons'

interface Props {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  onStop: () => void
  running: boolean
  disabled: boolean
  suggestions: string[]
  onSuggestion: (q: string) => void
  textareaRef: React.RefObject<HTMLTextAreaElement | null>
}

export function AskBar({ value, onChange, onSubmit, onStop, running, disabled, suggestions, onSuggestion, textareaRef }: Props) {
  const canSend = !disabled && !running && value.trim().length > 0

  return (
    <div>
      {suggestions.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-1.5" aria-label="Suggested questions">
          {suggestions.map((q) => (
            <button
              key={q}
              type="button"
              disabled={running || disabled}
              onClick={() => onSuggestion(q)}
              className="rounded-full border border-hairline bg-surface px-3 py-1 text-xs text-ink-2 transition-colors hover:border-accent/50 hover:text-ink focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50"
            >
              {q}
            </button>
          ))}
        </div>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (canSend) onSubmit()
        }}
        className="flex items-end gap-2 rounded-2xl border border-hairline bg-surface p-2 shadow-sm focus-within:border-accent/60"
      >
        <label htmlFor="question" className="sr-only">
          Ask a question about your data
        </label>
        <textarea
          id="question"
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends, Shift+Enter adds a new line.
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              if (canSend) onSubmit()
            }
          }}
          rows={1}
          disabled={disabled}
          maxLength={2000}
          placeholder={disabled ? 'Upload a CSV to start asking questions' : 'Ask a question about your data…'}
          className="field-sizing-content max-h-40 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] outline-none placeholder:text-ink-muted disabled:cursor-not-allowed"
        />
        {running ? (
          <button
            type="button"
            onClick={onStop}
            className="flex h-9 shrink-0 items-center gap-1.5 rounded-xl border border-hairline px-3 text-sm font-medium hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-accent"
          >
            <StopIcon className="size-4" /> Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!canSend}
            aria-label="Ask"
            className="grid size-9 shrink-0 place-items-center rounded-xl bg-accent text-white transition-opacity hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-40"
          >
            <SendIcon className="size-4" />
          </button>
        )}
      </form>
      <p className="mt-1.5 text-center text-[11px] text-ink-muted">
        Enter to ask · Shift+Enter for a new line · <kbd className="font-mono">Ctrl K</kbd> for commands
      </p>
    </div>
  )
}
