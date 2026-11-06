// Inline SVG icons (stroke = currentColor), so they follow text colour in both themes.

import type { SemanticType } from '../lib/types'

type P = { className?: string }
const base = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round' } as const

const Svg = ({ className = 'size-4', children }: P & { children: React.ReactNode }) => (
  <svg viewBox="0 0 24 24" className={className} aria-hidden="true" {...base}>
    {children}
  </svg>
)

export const UploadIcon = (p: P) => (
  <Svg {...p}>
    <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
  </Svg>
)
export const SunIcon = (p: P) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </Svg>
)
export const MoonIcon = (p: P) => (
  <Svg {...p}>
    <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z" />
  </Svg>
)
export const StopIcon = (p: P) => (
  <Svg {...p}>
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </Svg>
)
export const SendIcon = (p: P) => (
  <Svg {...p}>
    <path d="M5 12h14m0 0-6-6m6 6-6 6" />
  </Svg>
)
export const ToolIcon = (p: P) => (
  <Svg {...p}>
    <path d="M14.7 6.3a4 4 0 0 0-5.4 5.2L3 17.8V21h3.2l6.3-6.3a4 4 0 0 0 5.2-5.4l-2.6 2.6-2.4-.6-.6-2.4 2.6-2.6Z" />
  </Svg>
)
export const ChevronIcon = (p: P) => (
  <Svg {...p}>
    <path d="m9 6 6 6-6 6" />
  </Svg>
)
export const SparkIcon = (p: P) => (
  <Svg {...p}>
    <path d="M12 3v4m0 10v4M3 12h4m10 0h4M6 6l2.5 2.5m7 7L18 18M6 18l2.5-2.5m7-7L18 6" />
  </Svg>
)
export const ChartIcon = (p: P) => (
  <Svg {...p}>
    <path d="M4 20h16M7 16v-4m5 4V8m5 8v-6" />
  </Svg>
)
export const TableIcon = (p: P) => (
  <Svg {...p}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M3 10h18M9 10v10" />
  </Svg>
)
export const AlertIcon = (p: P) => (
  <Svg {...p}>
    <path d="M12 9v4m0 4h.01M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
  </Svg>
)
export const CheckIcon = (p: P) => (
  <Svg {...p}>
    <path d="m5 12 5 5L20 7" />
  </Svg>
)

/** Small glyph per column type: shape + text, so type never depends on colour alone. */
export function TypeGlyph({ type }: { type: SemanticType }) {
  const label: Record<SemanticType, string> = {
    numeric: '#',
    categorical: 'Ab',
    text: '¶',
    datetime: '◷',
    boolean: '✓✗',
    id: 'ID',
    empty: '∅',
  }
  return (
    <span
      className="inline-flex h-5 min-w-5 items-center justify-center rounded bg-surface-2 px-1 font-mono text-[10px] font-semibold text-ink-2"
      aria-hidden="true"
    >
      {label[type]}
    </span>
  )
}
