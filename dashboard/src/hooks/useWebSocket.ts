// dashboard/src/hooks/useWebSocket.ts
import { useEffect, useRef } from 'react'
import { useBrainStore } from '../stores/brainState'

/**
 * Fetch current brain state from REST API on mount.
 * This ensures neuromodulators/mode/regions are correct after refresh,
 * without waiting for the next processing cycle to emit WS events.
 */
async function fetchInitialState() {
  try {
    const res = await fetch('/api/state')
    if (!res.ok) return
    const data = await res.json()
    if (data.error) return
    const store = useBrainStore.getState()
    if (data.neuromodulators) {
      store.setNeuromodulators(data.neuromodulators)
    }
    if (data.network_mode) {
      store.setNetworkMode(data.network_mode.toLowerCase().replace(' ', '_'))
    }
    if (data.regions) {
      for (const [name, level] of Object.entries(data.regions)) {
        store.setRegionActivation(name, level as number, (level as number) > 0.1 ? 'active' : 'inactive')
      }
    }
  } catch {
    // API not available yet — will get state via WS events
  }
  // Fetch memory stats so the flow bar shows current counts on load
  try {
    const res = await fetch('/api/memory/stats')
    if (res.ok) {
      const stats = await res.json()
      if (!stats.error) {
        useBrainStore.getState().setMemoryFlow(stats)
      }
    }
  } catch { /* ignore */ }
  await useBrainStore.getState().refreshWorkspaces()
}

export function useWebSocket(url: string) {
  const wsRef = useRef<WebSocket | null>(null)
  const store = useBrainStore()

  useEffect(() => {
    // Load persisted state immediately on mount (before WS connects)
    fetchInitialState()
    useBrainStore.getState().loadChatHistory()

    const connect = () => {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => store.setConnected(true)
      ws.onclose = () => {
        store.setConnected(false)
        setTimeout(connect, 3000)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data)
          handleEvent(data)
        } catch {}
      }
    }

    const handleEvent = (data: { type: string; payload: Record<string, any>; ts: number }) => {
      store.addEvent(data)

      switch (data.type) {
        case 'region_activation':
          store.setRegionActivation(data.payload.region, data.payload.level, data.payload.mode || 'active')
          break
        case 'network_switch':
          store.setNetworkMode(data.payload.to?.toLowerCase().replace(' ', '_') || 'default_mode')
          break
        case 'neuromodulator':
          store.setNeuromodulators(data.payload as any)
          break
        case 'memory_flow':
          store.setMemoryFlow(data.payload as any)
          break
        case 'signal_flow':
          store.addSignalFlow({
            source: data.payload.source,
            target: data.payload.target,
            signal_type: data.payload.signal_type,
            strength: data.payload.strength,
          })
          break
        case 'region_processing':
          store.addThinkingStep({
            region: data.payload.region,
            phase: data.payload.phase,
            summary: data.payload.summary,
            details: data.payload.details,
            ts: data.ts || Date.now(),
          })
          break
        case 'broadcast': {
          const msg = data.payload?.content
          const source = data.payload?.origin
          if (msg && typeof msg === 'string' && !msg.startsWith('tool_') && msg !== 'action_executed') {
            const s = useBrainStore.getState()
            if (source === 'broca_refined') {
              // Broca refined the response — update the last brain message in-place
              s.updateLastBrainMessage(msg)
              s.setLastResponse(msg)
            } else {
              // PFC response — show immediately, don't wait for Broca
              const last = s.chatMessages[s.chatMessages.length - 1]
              if (!last || last.role !== 'brain' || last.text !== msg) {
                const steps = [...s.thinkingSteps]
                s.addChatMessage({ role: 'brain', text: msg, thinkingSteps: steps.length > 0 ? steps : undefined, ts: Date.now() })
                s.setLastResponse(msg)
                s.setChatLoading(false)
              }
            }
          }
          break
        }
        case 'workspace_changed': {
          const payload = data.payload
          const state = useBrainStore.getState()
          const next = state.workspaces.find((workspace) => workspace.id === payload.workspace_id)
          if (next) state.setCurrentWorkspace(next)
          else state.refreshWorkspaces()
          break
        }
        case 'clarification_requested': {
          const payload = data.payload
          const state = useBrainStore.getState()
          const questionText = typeof payload.question === 'string'
            ? payload.question
            : Array.isArray(payload.questions)
              ? payload.questions[0]
              : ''
          const question = {
            id: String(payload.id || payload.question_id || `ws-${data.ts || Date.now()}`),
            question_id: payload.question_id,
            question: questionText,
            severity: (payload.severity || 'moderate') as 'minor' | 'moderate' | 'severe',
            workspace_id: String(payload.workspace_id || 'personal'),
            context_input: payload.context_input,
            raised_by: payload.raised_by,
          }
          if (question.question) state.addOpenQuestion(question)
          if (question.severity === 'severe') state.setLastResponseBlock('block', question)
          break
        }
        case 'contradiction_detected':
          useBrainStore.getState().addContradiction(data.payload as any)
          break
        case 'ontology_proposal':
          useBrainStore.getState().addOntologyProposal(data.payload as any)
          break
        case 'question_answered':
          useBrainStore.getState().removeOpenQuestion(String(data.payload.question_id || data.payload.id))
          break
        case 'contradiction_resolved':
          useBrainStore.getState().removeContradiction(String(data.payload.contradiction_id || data.payload.id))
          break
        case 'proposal_decided':
          useBrainStore.getState().removeOntologyProposal(String(data.payload.proposal_id || data.payload.id))
          break
      }
    }

    connect()
    return () => { wsRef.current?.close() }
  }, [url])
}
