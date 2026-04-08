import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

const COMMUNITY_COLORS = [
  '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6',
  '#06b6d4', '#ec4899', '#14b8a6', '#f97316', '#6366f1',
]

const CONFIDENCE_OPACITY: Record<string, string> = {
  EXTRACTED: 'cc',
  INFERRED: '80',
  AMBIGUOUS: '40',
}

interface KGNode {
  id: string
  label: string
  community: number
  val?: number
}

interface KGLink {
  source: string
  target: string
  relation: string
  confidence: string
  weight: number
}

interface KGData {
  nodes: KGNode[]
  edges: KGLink[]
  communities: Record<string, { members: string[]; cohesion: number }>
  hubs: { id: string; label: string; edges: number }[]
}

export default function KnowledgeGraphPanel({ width, height }: { width: number; height: number }) {
  const [data, setData] = useState<KGData | null>(null)
  const [hovered, setHovered] = useState<string | null>(null)
  const graphRef = useRef<any>(null)

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch('/api/memory/knowledge-graph')
      if (res.ok) setData(await res.json())
    } catch {}
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 30_000)
    return () => clearInterval(interval)
  }, [fetchData])

  const graphData = useMemo(() => {
    if (!data || !data.nodes.length) return { nodes: [], links: [] }
    const hubIds = new Set((data.hubs || []).map(h => h.id))
    const nodes = data.nodes.map(n => ({
      ...n,
      val: hubIds.has(n.id) ? 4.0 : 1.5,
      color: COMMUNITY_COLORS[n.community % COMMUNITY_COLORS.length] || '#94a3b8',
    }))
    const links = data.edges.map(e => ({
      source: e.source,
      target: e.target,
      relation: e.relation,
      confidence: e.confidence,
      weight: e.weight,
    }))
    return { nodes, links }
  }, [data])

  useEffect(() => {
    if (!graphRef.current || !graphData.nodes.length) return
    try {
      graphRef.current.d3Force('charge')?.strength(-30)
      graphRef.current.d3Force('link')?.distance(40)
      setTimeout(() => graphRef.current?.zoomToFit?.(400, 20), 1500)
    } catch {}
  }, [graphData])

  const communityCount = data ? Object.keys(data.communities || {}).length : 0
  const hubCount = data?.hubs?.length || 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{
        display: 'flex', gap: 8, padding: '4px 8px',
        background: 'rgba(0,0,0,0.15)', fontSize: '9px', color: 'rgba(226,232,240,0.7)',
      }}>
        <span>{graphData.nodes.length} concepts</span>
        <span>{graphData.links.length} relations</span>
        <span>{communityCount} communities</span>
        <span style={{ color: '#f59e0b' }}>{hubCount} hubs</span>
      </div>
      {graphData.nodes.length > 0 ? (
        <ForceGraph2D
          ref={graphRef}
          width={width || 340}
          height={(height || 300) - 25}
          graphData={graphData}
          nodeRelSize={3.5}
          nodeColor={(node: any) => node.color || '#94a3b8'}
          nodeLabel={(node: any) => `${node.label} (community ${node.community})`}
          nodeCanvasObjectMode={() => 'after'}
          nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D) => {
            const isHub = node.val > 3
            ctx.font = `${isHub ? 'bold 8' : '6.5'}px Inter, sans-serif`
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            ctx.fillStyle = node.id === hovered ? '#fff' : 'rgba(226,232,240,0.7)'
            ctx.fillText(node.label || node.id, node.x, node.y + (node.val || 1) * 3.5 + 3)
          }}
          linkColor={(link: any) => {
            const base = '#94a3b8'
            const opacity = CONFIDENCE_OPACITY[link.confidence] || '60'
            return base + opacity
          }}
          linkWidth={(link: any) => 0.3 + (link.weight || 0.5) * 1.2}
          linkDirectionalArrowLength={2.5}
          linkDirectionalArrowRelPos={1}
          linkLabel={(link: any) => `${link.relation} [${link.confidence}] (${(link.weight || 0).toFixed(2)})`}
          onNodeHover={(node: any) => setHovered(node?.id || null)}
          backgroundColor="transparent"
          cooldownTicks={80}
          d3AlphaDecay={0.04}
          d3VelocityDecay={0.3}
          warmupTicks={40}
        />
      ) : (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          height: 80, fontSize: '10px', color: 'rgba(226,232,240,0.4)',
        }}>
          No knowledge graph data yet
        </div>
      )}
    </div>
  )
}
