import { useCallback, useEffect, useReducer, useRef } from 'react'

import { ApiError, cancelRun, createRun, streamRun } from '../lib/api'
import { initialState, isActive, reducer } from '../lib/runsReducer'

/**
 * Owns the runs reducer and every open EventSource.
 *
 * ask():  optimistic. The run appears (with a skeleton) before the server answers.
 * stop(): closes the stream immediately (the UI feels instant) and tells the server to
 *         cancel, so the agent stops spending tokens too.
 */
export function useAgentRuns(datasetId: string | null) {
  const [state, dispatch] = useReducer(reducer, initialState)
  const streams = useRef(new Map<string, () => void>()) // localId -> close()
  const counter = useRef(0)
  const stoppedEarly = useRef(new Set<string>())

  const closeStream = (localId: string) => {
    streams.current.get(localId)?.()
    streams.current.delete(localId)
  }

  // Close every stream when the component unmounts (or the dataset changes).
  useEffect(() => {
    const open = streams.current
    return () => {
      open.forEach((close) => close())
      open.clear()
    }
  }, [datasetId])

  // A new dataset starts a fresh conversation.
  useEffect(() => {
    dispatch({ type: 'reset' })
  }, [datasetId])

  const ask = useCallback(
    async (question: string) => {
      if (!datasetId) return
      const localId = `run-${++counter.current}`
      dispatch({ type: 'start', localId, question })
      try {
        const { run_id } = await createRun(datasetId, question)
        if (stoppedEarly.current.delete(localId)) {
          // Stop was clicked before the server replied: cancel the run it just started.
          cancelRun(run_id).catch(() => {})
          return
        }
        dispatch({ type: 'created', localId, runId: run_id })
        const close = streamRun(run_id, {
          onEvent: (event, id) => dispatch({ type: 'event', localId, event, id }),
          onEnd: (status) => {
            dispatch({ type: 'end', localId, status })
            streams.current.delete(localId)
          },
          onError: (message) => {
            dispatch({ type: 'failed', localId, message })
            streams.current.delete(localId)
          },
        })
        streams.current.set(localId, close)
      } catch (e) {
        dispatch({ type: 'failed', localId, message: e instanceof ApiError ? e.message : 'Could not start the run.' })
      }
    },
    [datasetId],
  )

  const active = state.runs.find(isActive)

  const stop = useCallback(() => {
    if (!active) return
    closeStream(active.localId)
    dispatch({ type: 'stopped', localId: active.localId })
    if (active.runId) cancelRun(active.runId).catch(() => {}) // best effort: the UI has already stopped
    else stoppedEarly.current.add(active.localId) // ask() cancels it once the server returns an id
  }, [active])

  return { runs: state.runs, running: Boolean(active), ask, stop }
}
