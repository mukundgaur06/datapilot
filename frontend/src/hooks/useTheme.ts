import { useCallback, useState } from 'react'

export type Theme = 'light' | 'dark'
const KEY = 'datapilot-theme'

// index.html applies the saved theme before React loads; this hook just reads and flips it.
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => (document.documentElement.classList.contains('dark') ? 'dark' : 'light'))

  const toggle = useCallback(() => {
    setTheme((current) => {
      const next: Theme = current === 'dark' ? 'light' : 'dark'
      document.documentElement.classList.toggle('dark', next === 'dark')
      try {
        localStorage.setItem(KEY, next)
      } catch {
        /* private mode: the theme just won't persist */
      }
      return next
    })
  }, [])

  return [theme, toggle]
}
