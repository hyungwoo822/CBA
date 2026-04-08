// dashboard/src/App.tsx
import { useCallback, useRef, useEffect, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { BrainScene } from './components/BrainScene'
import { HUD } from './components/HUD'
import { MemoryFlowBar } from './components/MemoryFlowBar'
import { EventLog } from './components/EventLog'
import { ChatInput, addFilesToStore } from './components/ChatInput'
import { RegionDetailPanel } from './components/RegionDetailPanel'
import { BrainResponseBubble } from './components/BrainResponseBubble'
import { RegionBubbles } from './components/RegionBubble'
import { MemoryPanel } from './components/MemoryPanel'
import { ProfileEditModal } from './components/ProfileEditModal'
import { ChannelToggle } from './components/ChannelToggle'
import KnowledgeGraphModal from './components/KnowledgeGraphModal'
import { useWebSocket } from './hooks/useWebSocket'
import { resetAllPositions } from './hooks/useDraggable'
import { useBrainStore } from './stores/brainState'
import './App.css'

function CloningScoreBar() {
  const [score, setScore] = useState<any>(null)
  const msgCount = useBrainStore((s) => s.chatMessages.length)

  useEffect(() => {
    fetch('/api/cloning-score').then(r => r.json()).then(setScore).catch(() => {})
  }, [msgCount])

  if (!score || (!score.user_graph_size && !score.agent_graph_size)) return null

  const pct = Math.round((score.cloning_score || 0) * 100)
  const userSize = score.user_graph_size || 0
  const agentSize = score.agent_graph_size || 0
  const nodeOvl = Math.round((score.node_recall ?? score.node_overlap ?? 0) * 100)
  const relOvl = Math.round((score.relation_recall ?? score.relation_overlap ?? 0) * 100)
  const edgeOvl = Math.round((score.edge_recall ?? score.edge_overlap ?? 0) * 100)

  return (
    <div style={{
      position: 'absolute', left: '50%', transform: 'translateX(-50%)',
      bottom: 'calc(var(--chat-height, 140px) + 52px)',
      zIndex: 20, display: 'flex', alignItems: 'center', gap: 10,
      background: 'rgba(10,10,20,0.85)', backdropFilter: 'blur(8px)',
      border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8,
      padding: '6px 14px', fontSize: '10px', color: 'rgba(226,232,240,0.7)',
    }}>
      {/* User side */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#22d3ee' }} />
        <span style={{ color: '#22d3ee', fontWeight: 600 }}>User</span>
        <span>{userSize}</span>
      </div>

      {/* Sync bar */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
        <span style={{ fontSize: '11px', fontWeight: 700, color: pct > 60 ? '#fbbf24' : pct > 30 ? '#fb923c' : '#f87171' }}>
          {pct}% sync
        </span>
        <div style={{ width: 120, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.08)', overflow: 'hidden', display: 'flex' }}>
          <div style={{ width: `${pct}%`, height: '100%', borderRadius: 2,
            background: pct > 60 ? 'linear-gradient(90deg, #22d3ee, #a78bfa)' : pct > 30 ? '#fb923c' : '#f87171',
            transition: 'width 0.5s ease' }} />
        </div>
        <span style={{ fontSize: '8px', opacity: 0.5 }}>
          node {nodeOvl}% / rel {relOvl}% / edge {edgeOvl}%
        </span>
      </div>

      {/* Agent side */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <span style={{ color: '#a78bfa', fontWeight: 600 }}>Agent</span>
        <span>{agentSize}</span>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#a78bfa' }} />
      </div>
    </div>
  )
}

function EditableUserName() {
  const [name, setName] = useState('CloneJIN')
  const [editing, setEditing] = useState(false)
  const [saving, setSaving] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const openModal = useBrainStore((s) => s.setProfileModalOpen)

  // Load name from profile on mount
  useEffect(() => {
    fetch('/api/profile').then(r => r.json()).then(data => {
      if (data['USER.md']) {
        const m = data['USER.md'].match(/\*\*Name\*\*:\s*(.*)/)
        if (m) {
          const val = m[1].trim()
          if (val && !val.startsWith('(')) setName(val)
        }
      }
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [editing])

  const saveName = useCallback(async (newName: string) => {
    const trimmed = newName.trim()
    if (!trimmed) { setEditing(false); return }
    setName(trimmed)
    setEditing(false)
    setSaving(true)
    try {
      const res = await fetch('/api/profile')
      const data = await res.json()
      let userMd = data['USER.md'] || ''
      if (userMd.match(/\*\*Name\*\*:/)) {
        userMd = userMd.replace(/(\*\*Name\*\*:\s*)(.*)/, `$1${trimmed}`)
      }
      await fetch('/api/profile', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'USER.md': userMd }),
      })
    } catch { /* silent */ }
    setSaving(false)
  }, [])

  return (
    <div className="nav-user-area">
      <button className="nav-edit-btn" onClick={() => openModal(true)} title="Edit profile">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
        </svg>
      </button>
      {editing ? (
        <input
          ref={inputRef}
          className="nav-user-input"
          value={name}
          onChange={e => setName(e.target.value)}
          onBlur={() => saveName(name)}
          onKeyDown={e => { if (e.key === 'Enter') saveName(name); if (e.key === 'Escape') setEditing(false) }}
        />
      ) : (
        <div
          className={`nav-user ${saving ? 'saving' : ''}`}
          onClick={() => setEditing(true)}
          title="Click to edit name"
        >
          {name}
        </div>
      )}
    </div>
  )
}

function App() {
  // Auto-detect WebSocket URL: same host as the page
  const wsUrl = `ws://${window.location.host}/ws`
  useWebSocket(wsUrl)

  const isDragOver = useBrainStore((s) => s.isDragOver)
  const setDragOver = useBrainStore((s) => s.setDragOver)
  const dragCounter = useRef(0)

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    dragCounter.current++
    if (e.dataTransfer.types.includes('Files')) setDragOver(true)
  }, [setDragOver])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    dragCounter.current--
    if (dragCounter.current <= 0) {
      dragCounter.current = 0
      setDragOver(false)
    }
  }, [setDragOver])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    dragCounter.current = 0
    setDragOver(false)
    if (e.dataTransfer.files.length > 0) {
      addFilesToStore(e.dataTransfer.files)
    }
  }, [setDragOver])

  return (
    <div
      className="app"
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {isDragOver && (
        <div className="drop-overlay">
          <div className="drop-overlay-inner">
            <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
            </svg>
            <span>Drop files here</span>
            <span className="drop-hint">PDF, Images, Videos</span>
          </div>
        </div>
      )}
      <nav className="top-nav">
        <div className="nav-left">
          <div className="nav-logo">CBA - Clone Brain Agent</div>
          <button className="nav-reset" onClick={resetAllPositions} title="Reset panel positions">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
              <path d="M3 3v5h5"/>
            </svg>
          </button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <ChannelToggle />
          <EditableUserName />
        </div>
      </nav>
      <div className="brain-viewport">
        <Canvas camera={{ position: [0, 20, 80], fov: 50 }}>
          <ambientLight intensity={1.0} />
          <pointLight position={[60, 80, 80]} intensity={1.8} />
          <pointLight position={[-60, -40, 60]} intensity={0.6} color="#a0c0ff" />
          <pointLight position={[0, 40, -60]} intensity={0.5} color="#ffe0c0" />
          <hemisphereLight args={['#c0d0e0', '#0a0a14', 0.4]} />
          <BrainScene />
          <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        </Canvas>
      </div>
      <CloningScoreBar />
      <div className="left-column">
        <HUD />
        <EventLog />
      </div>
      <BrainResponseBubble />
      <MemoryPanel />
      <RegionBubbles />
      <RegionDetailPanel />
      <ChatInput />
      <MemoryFlowBar />
      <ProfileEditModal />
      <KnowledgeGraphModal />
    </div>
  )
}

export default App
