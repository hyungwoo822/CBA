// dashboard/src/stores/brainState.ts
import { create } from 'zustand'

export interface RegionState {
  level: number
  mode: string // 'active' | 'high_activity' | 'inactive'
}

export interface BrainEvent {
  type: string
  payload: Record<string, any>
  ts: number
}

export interface SignalFlowEvent {
  id: string
  source: string
  target: string
  signal_type: string
  strength: number
  timestamp: number
}

export interface Particle {
  id: string
  source: string
  target: string
  signal_type: string
  progress: number
  speed: number
  color: string
  delay: number
}

export interface Workspace {
  id: string
  name: string
  description?: string
  decay_policy: 'none' | 'slow' | 'normal'
  node_count?: number
  edge_count?: number
  pending_count?: number
  blocking_count?: number
}

export interface OpenQuestion {
  id: string
  question_id?: string
  question: string
  severity: 'minor' | 'moderate' | 'severe'
  workspace_id: string
  context_input?: string
  raised_by?: string
}

export interface Contradiction {
  id: string
  contradiction_id?: string
  subject?: string
  subject_node?: string
  value_a: string
  value_b: string
  severity: 'minor' | 'moderate' | 'severe'
  workspace_id: string
  source_a?: string
  source_b?: string
  epistemic_source_a?: string
  epistemic_source_b?: string
}

export interface OntologyProposal {
  id: string
  proposal_id?: string
  kind: 'node_type' | 'relation_type'
  proposed_name: string
  confidence: string
  workspace_id: string
  source_snippet?: string
  definition?: string
}

export interface ModelInfo {
  id: string
  vendor: string
  available: boolean
  reason?: string
}

interface BrainState {
  regions: Record<string, RegionState>
  networkMode: string
  neuromodulators: { dopamine: number; norepinephrine: number; serotonin: number; acetylcholine: number; cortisol: number; epinephrine: number; gaba: number }
  memoryFlow: { sensory: number; working: number; staging: number; episodic: number; semantic: number; procedural: number }
  events: BrainEvent[]
  connected: boolean
  signalFlows: SignalFlowEvent[]
  particles: Particle[]
  lastResponse: string | null
  responseTimestamp: number
  lastResponseMode: 'normal' | 'append' | 'block'
  lastBlockQuestion: OpenQuestion | null
  isAudioMode: boolean
  audioState: 'idle' | 'listening' | 'processing' | 'done'
  selectedRegion: string | null
  memoryPanelOpen: boolean
  regionScreenPositions: Record<string, { x: number; y: number }>
  chatInputText: string
  attachedFiles: { name: string; type: string; size: number; url: string }[]
  chatMessages: { role: 'user' | 'brain'; text: string; files?: { name: string; type: string; url: string }[]; thinkingSteps?: BrainState['thinkingSteps']; ts: number }[]
  thinkingSteps: { region: string; phase: string; summary: string; details?: Record<string, any>; ts: number }[]
  isDragOver: boolean
  chatLoading: boolean
  interactionMode: 'question' | 'expression'
  setInteractionMode: (mode: 'question' | 'expression') => void
  profileModalOpen: boolean
  currentWorkspace: Workspace | null
  workspaces: Workspace[]
  openQuestions: OpenQuestion[]
  contradictions: Contradiction[]
  ontologyProposals: OntologyProposal[]
  defaultModel: string
  availableModels: ModelInfo[]

  setRegionScreenPositions: (positions: Record<string, { x: number; y: number }>) => void
  setRegionActivation: (region: string, level: number, mode: string) => void
  setNetworkMode: (mode: string) => void
  setNeuromodulators: (nm: BrainState['neuromodulators']) => void
  setMemoryFlow: (mf: BrainState['memoryFlow']) => void
  addEvent: (event: BrainEvent) => void
  setConnected: (c: boolean) => void
  addSignalFlow: (flow: Omit<SignalFlowEvent, 'id' | 'timestamp'>) => void
  setParticles: (particles: Particle[]) => void
  setLastResponse: (response: string) => void
  setLastResponseBlock: (mode: 'normal' | 'append' | 'block', q?: OpenQuestion) => void
  setAudioMode: (active: boolean) => void
  setAudioState: (state: BrainState['audioState']) => void
  setSelectedRegion: (region: string | null) => void
  setMemoryPanelOpen: (open: boolean) => void
  setChatInputText: (text: string) => void
  addAttachedFile: (file: { name: string; type: string; size: number; url: string }) => void
  removeAttachedFile: (name: string) => void
  clearAttachedFiles: () => void
  addChatMessage: (msg: BrainState['chatMessages'][0]) => void
  updateLastBrainMessage: (text: string) => void
  addThinkingStep: (step: BrainState['thinkingSteps'][0]) => void
  clearThinkingSteps: () => void
  setDragOver: (v: boolean) => void
  setChatLoading: (v: boolean) => void
  setProfileModalOpen: (v: boolean) => void
  setCurrentWorkspace: (ws: Workspace | null) => void
  setWorkspaces: (list: Workspace[]) => void
  refreshWorkspaces: () => Promise<void>
  removeOpenQuestion: (id: string) => void
  removeContradiction: (id: string) => void
  removeOntologyProposal: (id: string) => void
  addOpenQuestion: (q: OpenQuestion) => void
  addContradiction: (c: Contradiction) => void
  addOntologyProposal: (p: OntologyProposal) => void
  refreshCuration: (ws_id: string) => Promise<void>
  setModelInventory: (defaultModel: string, models: ModelInfo[]) => void
  refreshModelInventory: () => Promise<void>
  loadChatHistory: () => Promise<void>
  submitChat: () => Promise<void>
  submitAudio: (audioBlob: Blob) => Promise<void>
}

export const useBrainStore = create<BrainState>((set) => ({
  regions: {
    prefrontal_cortex: { level: 0, mode: 'inactive' },
    acc: { level: 0, mode: 'inactive' },
    amygdala: { level: 0, mode: 'inactive' },
    basal_ganglia: { level: 0, mode: 'inactive' },
    cerebellum: { level: 0, mode: 'inactive' },
    thalamus: { level: 0, mode: 'inactive' },
    hypothalamus: { level: 0, mode: 'inactive' },
    hippocampus: { level: 0, mode: 'inactive' },
    salience_network: { level: 0, mode: 'inactive' },
    visual_cortex: { level: 0, mode: 'inactive' },
    auditory_cortex_l: { level: 0, mode: 'inactive' },
    auditory_cortex_r: { level: 0, mode: 'inactive' },
    wernicke: { level: 0, mode: 'inactive' },
    broca: { level: 0, mode: 'inactive' },
    brainstem: { level: 0, mode: 'inactive' },
    vta: { level: 0, mode: 'inactive' },
    corpus_callosum: { level: 0, mode: 'inactive' },
    angular_gyrus: { level: 0, mode: 'inactive' },
    medial_pfc: { level: 0, mode: 'inactive' },
    tpj: { level: 0, mode: 'inactive' },
    insula: { level: 0, mode: 'inactive' },
  },
  networkMode: 'default_mode',
  neuromodulators: { dopamine: 0.5, norepinephrine: 0.5, serotonin: 0.5, acetylcholine: 0.5, cortisol: 0.5, epinephrine: 0.5, gaba: 0.5 },
  memoryFlow: { sensory: 0, working: 0, staging: 0, episodic: 0, semantic: 0, procedural: 0 },
  events: [],
  connected: false,
  signalFlows: [],
  particles: [],
  lastResponse: null,
  responseTimestamp: 0,
  lastResponseMode: 'normal',
  lastBlockQuestion: null,
  isAudioMode: false,
  audioState: 'idle' as const,
  selectedRegion: null,
  regionScreenPositions: {},
  memoryPanelOpen: true,
  chatInputText: '',
  attachedFiles: [],
  chatMessages: [],
  thinkingSteps: [],
  isDragOver: false,
  chatLoading: false,
  interactionMode: 'question',
  setInteractionMode: (mode) => set({ interactionMode: mode }),
  profileModalOpen: false,
  currentWorkspace: null,
  workspaces: [],
  openQuestions: [],
  contradictions: [],
  ontologyProposals: [],
  defaultModel: '',
  availableModels: [],

  setRegionActivation: (region, level, mode) =>
    set((s) => ({ regions: { ...s.regions, [region]: { level, mode } } })),
  setNetworkMode: (mode) => set({ networkMode: mode }),
  setNeuromodulators: (nm) => set({ neuromodulators: nm }),
  setMemoryFlow: (mf) => set({ memoryFlow: mf }),
  addEvent: (event) =>
    set((s) => ({ events: [...s.events.slice(-499), event] })),
  setConnected: (c) => set({ connected: c }),
  addSignalFlow: (flow) => set((s) => {
    const event: SignalFlowEvent = {
      ...flow,
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      timestamp: Date.now(),
    }

    const SIGNAL_COLORS: Record<string, string> = {
      'EXTERNAL_INPUT': '#06b6d4',
      'text_input': '#06b6d4',
      'TEXT_INPUT': '#06b6d4',
      'image_input': '#f59e0b',
      'IMAGE_INPUT': '#f59e0b',
      'audio_input': '#10b981',
      'AUDIO_INPUT': '#10b981',
      'PLAN': '#3b82f6',
      'ACTION_SELECTED': '#22c55e',
      'EMOTIONAL_TAG': '#f43f5e',
      'GWT_BROADCAST': '#eab308',
      'ENCODE': '#a855f7',
      'RETRIEVE': '#8b5cf6',
      'ACTION_RESULT': '#14b8a6',
    }

    const count = Math.max(1, Math.min(5, Math.round(flow.strength * 5)))
    const speed = 0.5 + flow.strength * 1.0
    const color = SIGNAL_COLORS[flow.signal_type] || '#ffffff'

    const newParticles: Particle[] = Array.from({ length: count }, (_, i) => ({
      id: `${event.id}-p${i}`,
      source: flow.source,
      target: flow.target,
      signal_type: flow.signal_type,
      progress: 0,
      speed,
      color,
      delay: i * 0.1,
    }))

    return {
      signalFlows: [...s.signalFlows.slice(-49), event],
      particles: [...s.particles, ...newParticles].slice(-150),
    }
  }),
  setParticles: (particles) => set({ particles }),
  setLastResponse: (response) => set({ lastResponse: response, responseTimestamp: Date.now(), lastResponseMode: 'normal', lastBlockQuestion: null }),
  setLastResponseBlock: (mode, q) => set({ lastResponseMode: mode, lastBlockQuestion: q || null, responseTimestamp: Date.now() }),
  setAudioMode: (active) => set({ isAudioMode: active }),
  setAudioState: (audioState) => set({ audioState }),
  setSelectedRegion: (region) => set({ selectedRegion: region }),
  setRegionScreenPositions: (positions) => set({ regionScreenPositions: positions }),
  setMemoryPanelOpen: (open) => set({ memoryPanelOpen: open }),
  setChatInputText: (text) => set({ chatInputText: text }),
  addAttachedFile: (file) => set((s) => ({ attachedFiles: [...s.attachedFiles, file] })),
  removeAttachedFile: (name) => set((s) => ({ attachedFiles: s.attachedFiles.filter((f) => f.name !== name) })),
  clearAttachedFiles: () => set({ attachedFiles: [] }),
  addChatMessage: (msg) => {
    set((s) => ({ chatMessages: [...s.chatMessages, msg] }))
    // Persist to server DB (fire-and-forget)
    fetch('/api/chat/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(msg),
    }).catch(() => {})
  },
  updateLastBrainMessage: (text) => {
    set((s) => {
      const msgs = [...s.chatMessages]
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].role === 'brain') {
          msgs[i] = { ...msgs[i], text }
          break
        }
      }
      return { chatMessages: msgs }
    })
    // Persist updated message
    fetch('/api/chat/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role: 'brain', text, ts: Date.now(), updated: true }),
    }).catch(() => {})
  },
  addThinkingStep: (step) => set((s) => ({ thinkingSteps: [...s.thinkingSteps.slice(-19), step] })),
  clearThinkingSteps: () => set({ thinkingSteps: [] }),
  setDragOver: (v) => set({ isDragOver: v }),
  setChatLoading: (v) => set({ chatLoading: v }),
  setProfileModalOpen: (v) => set({ profileModalOpen: v }),
  setCurrentWorkspace: (ws) => set({ currentWorkspace: ws }),
  setWorkspaces: (list) => set({ workspaces: list }),
  refreshWorkspaces: async () => {
    try {
      const listResponse = await fetch('/api/workspaces')
      const listData = await listResponse.json()
      const workspaces = listData.workspaces || []
      const currentResponse = await fetch('/api/workspaces/current?session_id=default')
      const currentData = await currentResponse.json()
      set({
        workspaces,
        currentWorkspace: currentData.workspace || workspaces[0] || null,
      })
    } catch {}
  },
  addOpenQuestion: (q) => set((s) => {
    const id = q.id || q.question_id
    if (!id || s.openQuestions.some((item) => item.id === id || item.question_id === id)) return {}
    return { openQuestions: [...s.openQuestions, { ...q, id }] }
  }),
  addContradiction: (c) => set((s) => {
    const id = c.id || c.contradiction_id
    if (!id || s.contradictions.some((item) => item.id === id || item.contradiction_id === id)) return {}
    return { contradictions: [...s.contradictions, { ...c, id }] }
  }),
  addOntologyProposal: (p) => set((s) => {
    const id = p.id || p.proposal_id
    if (!id || s.ontologyProposals.some((item) => item.id === id || item.proposal_id === id)) return {}
    return { ontologyProposals: [...s.ontologyProposals, { ...p, id }] }
  }),
  removeOpenQuestion: (id) => set((s) => ({ openQuestions: s.openQuestions.filter((q) => q.id !== id && q.question_id !== id) })),
  removeContradiction: (id) => set((s) => ({ contradictions: s.contradictions.filter((c) => c.id !== id && c.contradiction_id !== id) })),
  removeOntologyProposal: (id) => set((s) => ({ ontologyProposals: s.ontologyProposals.filter((p) => p.id !== id && p.proposal_id !== id) })),
  refreshCuration: async (ws_id) => {
    try {
      const [questions, contradictions, proposals] = await Promise.all([
        fetch(`/api/questions/${ws_id}`).then((r) => r.json()),
        fetch(`/api/contradictions/${ws_id}`).then((r) => r.json()),
        fetch(`/api/ontology/${ws_id}/proposals`).then((r) => r.json()),
      ])
      set({
        openQuestions: questions.questions || [],
        contradictions: contradictions.contradictions || [],
        ontologyProposals: proposals.proposals || [],
      })
    } catch {}
  },
  setModelInventory: (defaultModel, models) => set({ defaultModel, availableModels: models }),
  refreshModelInventory: async () => {
    try {
      const response = await fetch('/api/llm/providers')
      const data = await response.json()
      set({
        defaultModel: data.default_model || '',
        availableModels: data.available || [],
      })
    } catch {}
  },
  loadChatHistory: async () => {
    try {
        const res = await fetch('/api/chat/history')
        const data = await res.json()
        if (data.messages && data.messages.length > 0) {
            set({ chatMessages: data.messages })
        }
    } catch {}
  },
  submitChat: async () => {
    const s = useBrainStore.getState()
    const text = s.chatInputText.trim()
    if (!text && s.attachedFiles.length === 0) return

    const msgFiles = s.attachedFiles.map((f) => ({ name: f.name, type: f.type, url: f.url }))
    s.addChatMessage({ role: 'user', text, files: msgFiles.length > 0 ? msgFiles : undefined, ts: Date.now() })

    set({ chatLoading: true, chatInputText: '' })
    useBrainStore.getState().clearThinkingSteps()

    try {
      const fd = new FormData()
      fd.append('text', text)
      fd.append('mode', useBrainStore.getState().interactionMode)

      // Convert blob URLs back to File objects for upload
      for (const af of s.attachedFiles) {
        try {
          const resp = await fetch(af.url)
          const blob = await resp.blob()
          fd.append('files', blob, af.name)
        } catch { /* skip failed file */ }
      }

      const res = await fetch('/api/process', {
        method: 'POST',
        body: fd,  // No Content-Type header — browser sets multipart boundary
      })
      const data = await res.json()
      const responseText = data.response || data.error || 'No response'
      // Only add HTTP response if WS broadcast didn't already deliver it.
      // WS broadcast sets chatLoading=false — if it's already false, WS handled it.
      if (useBrainStore.getState().chatLoading) {
        useBrainStore.getState().setLastResponse(responseText)
        const steps = [...useBrainStore.getState().thinkingSteps]
        useBrainStore.getState().addChatMessage({ role: 'brain', text: responseText, thinkingSteps: steps.length > 0 ? steps : undefined, ts: Date.now() })
      }
    } catch {
      useBrainStore.getState().setLastResponse('Connection error')
      useBrainStore.getState().addChatMessage({ role: 'brain', text: 'Connection error', ts: Date.now() })
    }
    set({ chatLoading: false })
    useBrainStore.getState().clearAttachedFiles()
  },
  submitAudio: async (audioBlob: Blob) => {
    const s = useBrainStore.getState()
    if (s.chatLoading) return

    s.addChatMessage({ role: 'user', text: '\u{1F3A4} Voice message', ts: Date.now() })
    set({ chatLoading: true, isAudioMode: false, audioState: 'processing' })
    useBrainStore.getState().clearThinkingSteps()

    try {
      const fd = new FormData()
      fd.append('text', '')
      fd.append('files', audioBlob, 'recording.webm')

      const res = await fetch('/api/process', { method: 'POST', body: fd })
      const data = await res.json()
      const responseText = data.response || data.error || 'No response'
      useBrainStore.getState().setLastResponse(responseText)
      const steps = [...useBrainStore.getState().thinkingSteps]
      useBrainStore.getState().addChatMessage({ role: 'brain', text: responseText, thinkingSteps: steps.length > 0 ? steps : undefined, ts: Date.now() })
    } catch {
      useBrainStore.getState().setLastResponse('Connection error')
      useBrainStore.getState().addChatMessage({ role: 'brain', text: 'Connection error', ts: Date.now() })
    }
    set({ chatLoading: false, audioState: 'idle' })
  },
}))
