import { useRef, useState } from 'react'

import { UploadIcon } from './Icons'

interface Props {
  onFile: (file: File) => void
  busy: boolean
  error: string | null
  compact?: boolean
  inputRef?: React.RefObject<HTMLInputElement | null>
}

/** Drag a CSV in, or click / press Enter to pick one. */
export function UploadDropzone({ onFile, busy, error, compact = false, inputRef }: Props) {
  const [over, setOver] = useState(false)
  const localRef = useRef<HTMLInputElement>(null)
  const ref = inputRef ?? localRef

  const take = (files: FileList | null) => {
    const file = files?.[0]
    if (file) onFile(file)
  }

  return (
    <div>
      <button
        type="button"
        onClick={() => ref.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          take(e.dataTransfer.files)
        }}
        disabled={busy}
        aria-describedby={error ? 'upload-error' : undefined}
        className={[
          'group flex w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed text-center transition-colors',
          'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent',
          compact ? 'px-3 py-3' : 'px-6 py-12',
          over ? 'border-accent bg-accent/5' : 'border-hairline hover:border-accent/60 hover:bg-surface-2',
          busy ? 'cursor-wait opacity-60' : 'cursor-pointer',
        ].join(' ')}
      >
        <UploadIcon className={compact ? 'size-4 text-ink-2' : 'size-8 text-accent'} />
        <span className={compact ? 'text-xs text-ink-2' : 'text-base font-medium'}>
          {busy ? 'Reading your file…' : compact ? 'Upload a different CSV' : 'Drop a CSV file here, or click to choose one'}
        </span>
        {!compact && <span className="text-sm text-ink-muted">Up to 10 MB. The AI sees column summaries and computed results, never the full table.</span>}
      </button>
      <input
        ref={ref}
        type="file"
        accept=".csv,text/csv"
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => {
          take(e.target.files)
          e.target.value = '' // allow re-uploading the same file
        }}
      />
      {error && (
        <p id="upload-error" role="alert" className="mt-2 text-sm text-critical">
          {error}
        </p>
      )}
    </div>
  )
}
