export interface Command {
  id: string
  label: string
  group: string
  hint?: string
  run: () => void
}

/** Every word typed must appear somewhere in the group or label (order-free, case-insensitive). */
export function filterCommands(commands: Command[], query: string): Command[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  return commands.filter((c) => words.every((w) => `${c.group} ${c.label}`.toLowerCase().includes(w)))
}
