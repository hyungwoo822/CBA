import { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import { useBrainStore } from '../stores/brainState'
import { useDraggable } from '../hooks/useDraggable'
import ForceGraph2D from 'react-force-graph-2d'
const TABS = ['Chat', 'Working', 'Staging', 'Episodic', 'Semantic', 'Procedural'] as const
type Tab = typeof TABS[number]

/** Fetch data on mount + refetch after each chat message. NO polling — stops constant refresh. */
function useAutoFetch<T>(url: string) {
  const [data, setData] = useState<T | null>(null)
  const msgCount = useBrainStore((s) => s.chatMessages.length)

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(url)
      setData(await res.json())
    } catch { /* ignore */ }
  }, [url])

  useEffect(() => { fetchData() }, [fetchData, msgCount])

  return { data, refetch: fetchData }
}

export function MemoryPanel() {
  const isOpen = useBrainStore((s) => s.memoryPanelOpen)
  const setOpen = useBrainStore((s) => s.setMemoryPanelOpen)

  const [tab, setTab] = useState<Tab>('Chat')
  const { pos, onMouseDown: onDragMouseDown } = useDraggable('memory-panel', window.innerWidth - 390, 82)
  const bodyRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    (window as any).__memoryPanelSetTab = (t: Tab) => {
      setTab(t)
      if (!isOpen) setOpen(true)
    }
    return () => { delete (window as any).__memoryPanelSetTab }
  }, [isOpen, setOpen])

  if (!isOpen) return null

  return (
    <div
      className="memory-panel"
      style={{ left: pos.x, top: pos.y }}
      onMouseDown={onDragMouseDown}
    >
      <div className="mp-header" data-drag-handle style={{ cursor: 'grab' }}>
        <span className="mp-title">Brain Memory</span>
        <button className="mp-close" onClick={() => setOpen(false)}>&times;</button>
      </div>
      <div className="mp-tabs">
        {TABS.map((t) => (
          <button
            key={t}
            className={`mp-tab${tab === t ? ' active' : ''}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="mp-body" ref={bodyRef}>
        {tab === 'Chat' && <ChatTab />}
        {tab === 'Working' && <WorkingTab />}
        {tab === 'Staging' && <StagingTab />}
        {tab === 'Episodic' && <EpisodicTab />}
        {tab === 'Semantic' && <SemanticTab />}
        {tab === 'Procedural' && <ProceduralTab />}
      </div>
    </div>
  )
}

/* ============================================================
   Chat Tab
   ============================================================ */
function ThinkingStepsExpander({ steps }: { steps: { region: string; phase: string; summary: string }[] }) {
  const [open, setOpen] = useState(false)
  if (!steps || steps.length === 0) return null
  return (
    <div className="thinking-expander">
      <button className="thinking-toggle" onClick={() => setOpen(!open)}>
        {open ? '▼' : '▶'} Neural Activity ({steps.length} steps)
      </button>
      {open && (
        <div className="mp-thinking-steps expanded">
          {steps.map((step, i) => (
            <div key={i} className="thinking-step">
              <span className="thinking-region">{step.region.replace(/_/g, ' ')}</span>
              <span className="thinking-summary">{step.summary}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ChatTab() {
  const scrollRef = useRef<HTMLDivElement>(null)
  const messages = useBrainStore((s) => s.chatMessages)
  const chatInputText = useBrainStore((s) => s.chatInputText)
  const setChatInputText = useBrainStore((s) => s.setChatInputText)
  const attachedFiles = useBrainStore((s) => s.attachedFiles)
  const loading = useBrainStore((s) => s.chatLoading)
  const thinkingSteps = useBrainStore((s) => s.thinkingSteps)
  const submitChat = useBrainStore((s) => s.submitChat)

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages.length, chatInputText, thinkingSteps.length])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    await submitChat()
  }

  return (
    <div className="mp-chat-wrap">
      <div className="mp-scroll" ref={scrollRef}>
        {messages.length === 0 && !chatInputText && <div className="mp-empty">No messages yet. Type something below.</div>}
        {messages.map((m, i) => (
          <div key={i} className={`mp-chat-msg ${m.role}`}>
            <div className="mp-chat-role">{m.role === 'user' ? 'You' : 'Brain'}</div>
            {m.files && m.files.length > 0 && (
              <div className="mp-chat-files">
                {m.files.map((f, fi) => (
                  <div key={fi} className="mp-chat-file-chip">
                    {f.type.startsWith('image/') ? (
                      <img src={f.url} alt={f.name} className="mp-chat-file-thumb" />
                    ) : (
                      <span className="mp-chat-file-icon">
                        {f.type === 'application/pdf' ? 'PDF' : 'VID'}
                      </span>
                    )}
                    <span className="mp-chat-file-name">{f.name}</span>
                  </div>
                ))}
              </div>
            )}
            {m.text && <div className="mp-chat-text">{m.text}</div>}
            {m.role === 'brain' && m.thinkingSteps && m.thinkingSteps.length > 0 && (
              <ThinkingStepsExpander steps={m.thinkingSteps} />
            )}
          </div>
        ))}
        {(chatInputText || attachedFiles.length > 0) && (
          <div className="mp-chat-msg user typing">
            <div className="mp-chat-role">You</div>
            {attachedFiles.length > 0 && (
              <div className="mp-chat-files">
                {attachedFiles.map((f, fi) => (
                  <div key={fi} className="mp-chat-file-chip preview">
                    {f.type.startsWith('image/') ? (
                      <img src={f.url} alt={f.name} className="mp-chat-file-thumb" />
                    ) : (
                      <span className="mp-chat-file-icon">
                        {f.type === 'application/pdf' ? 'PDF' : 'VID'}
                      </span>
                    )}
                    <span className="mp-chat-file-name">{f.name}</span>
                  </div>
                ))}
              </div>
            )}
            {chatInputText && (
              <div className="mp-chat-text">
                {chatInputText}
                <span className="typing-cursor">|</span>
              </div>
            )}
          </div>
        )}
        {loading && thinkingSteps.length > 0 && (
          <div className="mp-chat-msg brain thinking">
            <div className="mp-chat-role">Brain (thinking...)</div>
            <div className="mp-thinking-steps">
              {thinkingSteps.map((step, i) => (
                <div key={i} className="thinking-step" style={{ opacity: i === thinkingSteps.length - 1 ? 1 : 0.5 }}>
                  <span className="thinking-region">{step.region.replace(/_/g, ' ')}</span>
                  <span className="thinking-summary">{step.summary}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      <form className="mp-chat-input" onSubmit={handleSubmit}>
        <input
          className="mp-chat-field"
          type="text"
          value={chatInputText}
          onChange={(e) => setChatInputText(e.target.value)}
          placeholder={loading ? 'Processing...' : 'Ask anything...'}
          disabled={loading}
        />
        <button
          type="submit"
          className="mp-chat-send"
          disabled={loading || (!chatInputText.trim() && attachedFiles.length === 0)}
        >
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="19" x2="12" y2="5"/>
            <polyline points="5 12 12 5 19 12"/>
          </svg>
        </button>
      </form>
    </div>
  )
}

/* ============================================================
   Working Memory Tab — bucket visualization
   ============================================================ */
function WorkingTab() {
  const { data } = useAutoFetch<any>('/api/memory/working')

  if (!data) return <div className="mp-empty">Loading...</div>
  if (data.error) return <div className="mp-empty">Connect backend to view working memory</div>

  const capacities: Record<string, number> = data.capacities || { phonological: 4, visuospatial: 3, episodic_buffer: 4 }
  const slots = ['phonological', 'visuospatial', 'episodic_buffer']
  const slotLabels: Record<string, string> = {
    phonological: 'Phonological Loop',
    visuospatial: 'Visuospatial Sketchpad',
    episodic_buffer: 'Episodic Buffer',
  }
  const slotColors: Record<string, string> = {
    phonological: '#3b82f6',
    visuospatial: '#8b5cf6',
    episodic_buffer: '#06b6d4',
  }

  return (
    <div className="mp-scroll">
      <div className="mp-subtitle">Baddeley Working Memory Model</div>
      {slots.map((slot) => {
        const items = (data.items || []).filter((it: any) => it.slot === slot)
        const cap = capacities[slot] || 4
        return (
          <div key={slot} className="wm-bucket">
            <div className="wm-bucket-header">
              <span className="wm-bucket-label">{slotLabels[slot]}</span>
              <span className="wm-bucket-count" style={{ color: slotColors[slot] }}>
                {items.length}/{cap}
              </span>
            </div>
            <div className="wm-bucket-rows">
              {Array.from({ length: cap }).map((_, i) => (
                <div
                  key={i}
                  className={`wm-row ${i < items.length ? 'filled' : 'empty'}`}
                  style={{ borderLeftColor: i < items.length ? slotColors[slot] : 'rgba(255,255,255,0.08)' }}
                >
                  {i < items.length ? (
                    <span className="wm-row-text">{items[i].content}</span>
                  ) : (
                    <span className="wm-row-empty">empty slot</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ============================================================
   Staging Tab
   ============================================================ */
function StagingTab() {
  const { data } = useAutoFetch<any>('/api/memory/staging')

  if (!data) return <div className="mp-empty">Loading...</div>
  if (data.error) return <div className="mp-empty">Connect backend to view staging memory</div>

  return (
    <div className="mp-scroll">
      <div className="mp-subtitle">Hippocampal Staging — awaiting consolidation</div>
      {(data.items || []).length === 0 && <div className="mp-empty">No staging memories</div>}
      {(data.items || []).map((item: any) => (
        <div key={item.id} className="mp-memory-card">
          <div className="mp-memory-content">{item.content}</div>
          <div className="mp-memory-meta">
            <span className="mp-strength-bar">
              <span style={{ width: `${(item.strength || 0) * 100}%`, background: '#34d399' }} />
            </span>
            <span className="mp-meta-text">str: {(item.strength || 0).toFixed(2)}</span>
            {item.source_modality && (
              <span className="mp-meta-text">{item.source_modality}</span>
            )}
            {item.emotional_tag?.arousal > 0.3 && (
              <span className="mp-emotion-tag" style={{ color: item.emotional_tag.valence > 0 ? '#22c55e' : '#ef4444' }}>
                {item.emotional_tag.valence > 0 ? '+ ' : '- '}
                a:{item.emotional_tag.arousal.toFixed(1)}
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

/* ============================================================
   Episodic Memory Tab — vertical timeline, oldest at bottom
   ============================================================ */
function EpisodicTab() {
  const { data } = useAutoFetch<any>('/api/memory/episodic')
  const [selected, setSelected] = useState<string | null>(null)

  if (!data) return <div className="mp-empty">Loading...</div>
  if (data.error) return <div className="mp-empty">Connect backend to view episodic memory</div>

  const episodes = data.episodes || []
  if (episodes.length === 0) return <div className="mp-scroll"><div className="mp-empty">No episodic memories yet</div></div>

  const maxStrength = Math.max(...episodes.map((ep: any) => ep.strength || 0), 0.01)

  // newest first (top), oldest at bottom
  const sorted = [...episodes].reverse()

  return (
    <div className="mp-scroll">
      <div className="mp-subtitle">Episodic Timeline — {episodes.length} memories (newest first)</div>
      <div className="ep-vertical-timeline">
        {sorted.map((ep: any, i: number) => {
          const strength = (ep.strength || 0) / maxStrength
          const valence = ep.emotional_tag?.valence || 0
          const arousal = ep.emotional_tag?.arousal || 0
          const color = valence > 0.2 ? '#4ade80' : valence < -0.2 ? '#f43f5e' : '#64748b'
          const dotSize = 8 + strength * 8
          const opacity = 0.35 + strength * 0.65
          const isSelected = selected === ep.id

          return (
            <div
              key={ep.id}
              className={`ep-vt-item ${isSelected ? 'selected' : ''}`}
              onClick={() => setSelected(isSelected ? null : ep.id)}
            >
              {/* Timeline spine */}
              <div className="ep-vt-spine">
                <div
                  className="ep-vt-dot"
                  style={{
                    width: dotSize,
                    height: dotSize,
                    backgroundColor: color,
                    opacity,
                    boxShadow: arousal > 0.5 ? `0 0 ${arousal * 10}px ${color}` : 'none',
                  }}
                />
                {i < sorted.length - 1 && <div className="ep-vt-line" />}
              </div>
              {/* Content */}
              <div className="ep-vt-content">
                <div className="ep-vt-summary" style={{ opacity }}>
                  {ep.content.length > 80 ? ep.content.slice(0, 80) + '...' : ep.content}
                </div>
                {isSelected && (
                  <div className="ep-vt-detail">
                    <div className="ep-vt-full-text">{ep.content}</div>
                    <div className="mp-memory-meta">
                      <span className="mp-strength-bar">
                        <span style={{ width: `${(ep.strength || 0) * 100}%`, background: color }} />
                      </span>
                      <span className="mp-meta-text">str: {(ep.strength || 0).toFixed(2)}</span>
                      <span className="mp-meta-text">access: {ep.access_count || 0}</span>
                      {arousal > 0 && (
                        <span className="mp-emotion-tag" style={{ color }}>
                          v:{valence.toFixed(1)} a:{arousal.toFixed(1)}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ============================================================
   Semantic Memory Tab — force-directed knowledge graph
   Nodes colored by category, centered layout, no auto-refresh jitter
   ============================================================ */
const CATEGORY_COLORS: Record<string, string> = {
  PREFERENCE: '#f43f5e',
  ACTION: '#3b82f6',
  ATTRIBUTE: '#8b5cf6',
  SOCIAL: '#f97316',
  CAUSAL: '#eab308',
  SPATIAL: '#06b6d4',
  TEMPORAL: '#34d399',
  IDENTITY: '#ec4899',
  GENERAL: '#64748b',
}

/** Mini graph + table for a single origin (user or agent) */
function OriginGraphPanel({ label, color, relations, graphW }: {
  label: string; color: string; relations: any[]; graphW: number
}) {
  const graphRef = useRef<any>(null)
  const [hovered, setHovered] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  const graphData = useMemo(() => {
    if (!relations.length) return { nodes: [], links: [] }
    const nodeSet = new Map<string, { id: string; connections: number; category: string }>()
    const links: any[] = []
    for (const rel of relations) {
      if (!nodeSet.has(rel.source)) nodeSet.set(rel.source, { id: rel.source, connections: 0, category: rel.category || 'GENERAL' })
      if (!nodeSet.has(rel.target)) nodeSet.set(rel.target, { id: rel.target, connections: 0, category: rel.category || 'GENERAL' })
      nodeSet.get(rel.source)!.connections++
      nodeSet.get(rel.target)!.connections++
      links.push({ source: rel.source, target: rel.target, label: rel.relation, weight: rel.weight || 0.5, category: rel.category || 'GENERAL' })
    }
    const nodes = Array.from(nodeSet.values()).map(n => ({ ...n, val: 1.5 + n.connections * 0.6 }))
    return { nodes, links }
  }, [relations])

  useEffect(() => {
    if (!graphRef.current || !graphData.nodes.length) return
    try {
      graphRef.current.d3Force('charge')?.strength(-25)
      graphRef.current.d3Force('link')?.distance(35)
      setTimeout(() => graphRef.current?.zoomToFit?.(400, 15), 1500)
    } catch {}
  }, [graphData])

  const filtered = useMemo(() => {
    if (!search.trim()) return relations
    const q = search.toLowerCase()
    return relations.filter((r: any) => r.source.includes(q) || r.relation.includes(q) || r.target.includes(q))
  }, [relations, search])

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '4px 8px', background: 'rgba(0,0,0,0.15)' }}>
        <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
        <span style={{ fontSize: '10px', fontWeight: 600, color }}>{label}</span>
        <span style={{ fontSize: '9px', color: 'rgba(226,232,240,0.5)' }}>
          {graphData.nodes.length} nodes, {graphData.links.length} edges
        </span>
      </div>
      {graphData.nodes.length > 0 ? (
        <ForceGraph2D
          key={`kg-${label}-${graphData.nodes.length}`}
          ref={graphRef}
          width={graphW || 340}
          height={150}
          graphData={graphData}
          nodeRelSize={3.5}
          nodeColor={(node: any) => CATEGORY_COLORS[node.category] || color}
          nodeLabel={(node: any) => `${node.id} (${node.connections})`}
          nodeCanvasObjectMode={() => 'after'}
          nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D) => {
            ctx.font = `${node.id === hovered ? 'bold 8' : '6.5'}px Inter, sans-serif`
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            ctx.fillStyle = node.id === hovered ? '#fff' : 'rgba(226,232,240,0.7)'
            ctx.fillText(node.id, node.x, node.y + (node.val || 1) * 3.5 + 3)
          }}
          linkColor={() => color + '40'}
          linkWidth={(link: any) => 0.4 + (link.weight || 0.5) * 1.5}
          linkDirectionalArrowLength={2.5}
          linkDirectionalArrowRelPos={1}
          linkLabel={(link: any) => `${link.label} (${(link.weight || 0).toFixed(2)})`}
          onNodeHover={(node: any) => setHovered(node?.id || null)}
          backgroundColor="transparent"
          cooldownTicks={80}
          d3AlphaDecay={0.04}
          d3VelocityDecay={0.3}
          warmupTicks={40}
        />
      ) : (
        <div className="mp-empty" style={{ height: 80, fontSize: '10px' }}>No {label.toLowerCase()} relations yet</div>
      )}
      {/* Scrollable table */}
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <input className="kg-search" type="text" placeholder="Search..." value={search}
          onChange={(e) => setSearch(e.target.value)} style={{ fontSize: '9px', padding: '2px 6px' }} />
        <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
          <table className="kg-table">
            <thead><tr><th>Source</th><th>Relation</th><th>Target</th><th>W</th><th>Cat</th></tr></thead>
            <tbody>
              {filtered.length === 0 && <tr><td colSpan={5} className="kg-table-empty">No relations</td></tr>}
              {filtered.map((rel: any, i: number) => (
                <tr key={i}>
                  <td className="kg-td-source">{rel.source}</td>
                  <td className="kg-td-rel">{rel.relation}</td>
                  <td className="kg-td-target">{rel.target}</td>
                  <td className="kg-td-weight">{(rel.weight || 0).toFixed(2)}</td>
                  <td><span className="kg-td-cat" style={{ color: CATEGORY_COLORS[rel.category] || '#64748b' }}>{(rel.category || 'GEN').slice(0, 4)}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function SemanticTab() {
  const { data, refetch } = useAutoFetch<any>('/api/memory/semantic')
  const graphWrapRef = useRef<HTMLDivElement>(null)
  const [graphW, setGraphW] = useState(350)
  const [cloningScore, setCloningScore] = useState<any>(null)

  useEffect(() => {
    fetch('/api/cloning-score').then(r => r.json()).then(setCloningScore).catch(() => {})
  }, [data])

  useEffect(() => {
    if (!graphWrapRef.current) return
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setGraphW(e.contentRect.width)
    })
    ro.observe(graphWrapRef.current)
    return () => ro.disconnect()
  }, [])

  const { userRels, agentRels } = useMemo(() => {
    const rels = data?.relations || []
    return {
      userRels: rels.filter((r: any) => r.origin === 'user_input'),
      agentRels: rels.filter((r: any) =>
        r.origin === 'agent_about_user' || r.origin === 'agent_response'),
    }
  }, [data])

  if (!data) return <div className="mp-empty">Loading...</div>
  if (data.error) return <div className="mp-empty">Connect backend to view knowledge graph</div>

  return (
    <div className="kg-split" ref={graphWrapRef} style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header with cloning score */}
      <div className="kg-header" style={{ flexShrink: 0 }}>
        <span className="mp-subtitle" style={{ margin: 0 }}>Semantic Memory</span>
        {cloningScore && cloningScore.cloning_score > 0 && (
          <span style={{ fontSize: '10px', color: '#fbbf24', marginLeft: 6, fontWeight: 600 }}
                title={`Node recall: ${((cloningScore.node_recall ?? cloningScore.node_overlap ?? 0) * 100).toFixed(0)}% | Rel recall: ${((cloningScore.relation_recall ?? cloningScore.relation_overlap ?? 0) * 100).toFixed(0)}% | Edge recall: ${((cloningScore.edge_recall ?? cloningScore.edge_overlap ?? 0) * 100).toFixed(0)}% | User: ${cloningScore.user_graph_size} | Agent: ${cloningScore.agent_graph_size}`}>
            Cloning {(cloningScore.cloning_score * 100).toFixed(0)}%
          </span>
        )}
        <button className="kg-refresh-btn" onClick={refetch} title="Refresh">↻</button>
      </div>

      {/* Top half: User graph */}
      <OriginGraphPanel label="User Graph" color="#22d3ee" relations={userRels} graphW={graphW} />

      {/* Bottom half: Agent's understanding of user */}
      <OriginGraphPanel label="Agent's User Model" color="#a78bfa" relations={agentRels} graphW={graphW} />
    </div>
  )
}

/* ============================================================
   Procedural Memory Tab
   ============================================================ */
function ProceduralTab() {
  const { data } = useAutoFetch<any>('/api/memory/procedural')

  if (!data) return <div className="mp-empty">Loading...</div>
  if (data.error) return <div className="mp-empty">Connect backend to view procedural memory</div>

  return (
    <div className="mp-scroll">
      {(data.procedures || []).length === 0 && <div className="mp-empty">No procedures learned yet</div>}
      {(data.procedures || []).map((proc: any) => (
        <div key={proc.id} className="mp-proc-card">
          <div className="mp-proc-trigger">{proc.trigger_pattern}</div>
          {proc.strategy && <div className="mp-proc-strategy">{proc.strategy}</div>}
          <div className="mp-proc-meta">
            <span className={`mp-stage-badge ${proc.stage}`}>{proc.stage}</span>
            <span className="mp-meta-text">
              {proc.execution_count}x | {((proc.success_rate || 0) * 100).toFixed(0)}%
            </span>
          </div>
          <div className="proc-progress">
            <div className="proc-progress-track">
              <div
                className={`proc-progress-fill ${proc.stage}`}
                style={{
                  width: proc.stage === 'autonomous' ? '100%'
                    : proc.stage === 'associative' ? '60%'
                    : `${Math.min(100, (proc.execution_count || 0) / 10 * 30)}%`
                }}
              />
            </div>
            <div className="proc-progress-labels">
              <span className={proc.stage === 'cognitive' ? 'active' : ''}>cognitive</span>
              <span className={proc.stage === 'associative' ? 'active' : ''}>associative</span>
              <span className={proc.stage === 'autonomous' ? 'active' : ''}>autonomous</span>
            </div>
          </div>
          <div className="mp-proc-actions">
            {(proc.action_sequence || []).map((a: any, i: number) => (
              <div key={i} className="mp-proc-step">{i + 1}. {a.tool || a.type || JSON.stringify(a).slice(0, 40)}</div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
