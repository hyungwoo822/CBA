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
  profileModalOpen: boolean

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
  setAudioMode: (active: boolean) => void
  setAudioState: (state: BrainState['audioState']) => void
  setSelectedRegion: (region: string | null) => void
  setMemoryPanelOpen: (open: boolean) => void
  setChatInputText: (text: string) => void
  addAttachedFile: (file: { name: string; type: string; size: number; url: string }) => void
  removeAttachedFile: (name: string) => void
  clearAttachedFiles: () => void
  addChatMessage: (msg: BrainState['chatMessages'][0]) => void
  addThinkingStep: (step: BrainState['thinkingSteps'][0]) => void
  clearThinkingSteps: () => void
  setDragOver: (v: boolean) => void
  setChatLoading: (v: boolean) => void
  setProfileModalOpen: (v: boolean) => void
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
  profileModalOpen: false,

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
  setLastResponse: (response) => set({ lastResponse: response, responseTimestamp: Date.now() }),
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
  addThinkingStep: (step) => set((s) => ({ thinkingSteps: [...s.thinkingSteps.slice(-19), step] })),
  clearThinkingSteps: () => set({ thinkingSteps: [] }),
  setDragOver: (v) => set({ isDragOver: v }),
  setChatLoading: (v) => set({ chatLoading: v }),
  setProfileModalOpen: (v) => set({ profileModalOpen: v }),
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
    if ((!text && s.attachedFiles.length === 0) || s.chatLoading) return

    const msgFiles = s.attachedFiles.map((f) => ({ name: f.name, type: f.type, url: f.url }))
    s.addChatMessage({ role: 'user', text, files: msgFiles.length > 0 ? msgFiles : undefined, ts: Date.now() })

    set({ chatLoading: true, chatInputText: '' })
    useBrainStore.getState().clearThinkingSteps()

    try {
      const fd = new FormData()
      fd.append('text', text)

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
      useBrainStore.getState().setLastResponse(responseText)
      // Attach accumulated thinking steps to the brain message for later review
      const steps = [...useBrainStore.getState().thinkingSteps]
      useBrainStore.getState().addChatMessage({ role: 'brain', text: responseText, thinkingSteps: steps.length > 0 ? steps : undefined, ts: Date.now() })
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
