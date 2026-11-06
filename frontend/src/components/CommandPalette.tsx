import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { filterCommands, type Command } from '../lib/commands'

interface Props {
  open: boolean
  onClose: () => void
  commands: Command[]
}

export function CommandPalette({ open, onClose, commands }: Props) {
  // Mounted only while open, so the query and selection start fresh every time.
  return <AnimatePresence>{open && <PaletteDialog onClose={onClose} commands={commands} />}</AnimatePresence>
}

function PaletteDialog({ onClose, commands }: Omit<Props, 'open'>) {
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const dialogRef = useRef<HTMLDivElement>(null)
  const results = useMemo(() => filterCommands(commands, query), [commands, query])

  useEffect(() => {
    const returnFocus = document.activeElement as HTMLElement | null
    inputRef.current?.focus()
    const dialog = dialogRef.current
    return () => {
      // Give focus back to where the user was, unless the chosen command moved it somewhere on purpose.
      const now = document.activeElement
      if (!now || now === document.body || dialog?.contains(now)) returnFocus?.focus?.()
    }
  }, [])

  const choose = (cmd: Command | undefined) => {
    if (!cmd) return
    onClose()
    cmd.run()
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setIndex((i) => (results.length ? (i + 1) % results.length : 0))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setIndex((i) => (results.length ? (i - 1 + results.length) % results.length : 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      choose(results[index])
    } else if (e.key === 'Escape') {
      e.preventDefault()
      onClose()
    } else if (e.key === 'Tab') {
      e.preventDefault() // keep focus inside the dialog
    }
  }

  return (
        <motion.div
          className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 px-4 pt-[15vh]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.12 }}
          onMouseDown={(e) => e.target === e.currentTarget && onClose()}
        >
          <motion.div
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-label="Command palette"
            className="w-full max-w-lg overflow-hidden rounded-2xl border border-hairline bg-surface shadow-2xl"
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.15 }}
            onKeyDown={onKeyDown}
          >
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value)
                setIndex(0)
              }}
              placeholder="Type a command or search…"
              role="combobox"
              aria-expanded="true"
              aria-controls="palette-list"
              aria-activedescendant={results[index] ? `cmd-${results[index].id}` : undefined}
              className="w-full border-b border-hairline bg-transparent px-4 py-3.5 text-[15px] outline-none placeholder:text-ink-muted"
            />
            <ul id="palette-list" role="listbox" className="max-h-80 overflow-y-auto p-1.5">
              {results.length === 0 && <li className="px-3 py-6 text-center text-sm text-ink-muted">No matching commands</li>}
              {results.map((cmd, i) => (
                <li
                  key={cmd.id}
                  id={`cmd-${cmd.id}`}
                  role="option"
                  aria-selected={i === index}
                  onMouseMove={() => setIndex(i)}
                  onClick={() => choose(cmd)}
                  className={`flex cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-sm ${i === index ? 'bg-surface-2' : ''}`}
                >
                  <span className="w-16 shrink-0 text-[11px] text-ink-muted">{cmd.group}</span>
                  <span className="min-w-0 flex-1 truncate">{cmd.label}</span>
                  {cmd.hint && <kbd className="shrink-0 rounded border border-hairline px-1.5 font-mono text-[10px] text-ink-muted">{cmd.hint}</kbd>}
                </li>
              ))}
            </ul>
          </motion.div>
        </motion.div>
  )
}
