// Shapes shared with the FastAPI backend (see backend/app/main.py and agent/loop.py).

export type SemanticType = 'numeric' | 'boolean' | 'datetime' | 'categorical' | 'text' | 'id' | 'empty'

export interface ColumnSummary {
  name: string
  dtype: string
  semantic_type: SemanticType
  missing_pct: number
  n_unique: number
  examples: string[]
}

export interface DatasetSummary {
  name: string
  n_rows: number
  n_columns: number
  columns: ColumnSummary[]
}

export interface Dataset {
  dataset_id: string
  summary: DatasetSummary
}

export type AgentEventType = 'thinking' | 'tool_call' | 'tool_result' | 'answer' | 'error' | 'cancelled'
export const AGENT_EVENT_TYPES: AgentEventType[] = ['thinking', 'tool_call', 'tool_result', 'answer', 'error', 'cancelled']

export interface RunStats {
  steps?: number
  elapsed_ms?: number
  usage?: Record<string, number>
}

export interface AgentEvent {
  type: AgentEventType
  step: number
  data: Record<string, unknown>
}

export type RunStatus = 'running' | 'done' | 'error' | 'cancelled'

export interface Health {
  ok: boolean
  model: string
  llm_configured: boolean
}

export type JsonRecord = Record<string, unknown>
