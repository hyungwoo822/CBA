import { useState } from 'react'
import { useDraggable } from '../hooks/useDraggable'
import KnowledgeGraphPanel from './KnowledgeGraphPanel'

const MODAL_W = 520
const MODAL_H = 420

export default function KnowledgeGraphModal() {
  const [open, setOpen] = useState(false)
  const { pos, onMouseDown } = useDraggable(
    'knowledge-graph-modal',
    Math.round((window.innerWidth - MODAL_W) / 2),
    Math.round((window.innerHeight - MODAL_H) / 2),
  )

  return (
    <>
      {/* Toggle button — positioned above the brain */}
      <button
        onClick={() => setOpen(v => !v)}
        title="Knowledge Graph"
        style={{
          position: 'fixed',
          top: 12,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 900,
          background: open ? 'rgba(139,92,246,0.25)' : 'rgba(15,23,42,0.7)',
          border: open ? '1px solid rgba(139,92,246,0.5)' : '1px solid rgba(148,163,184,0.2)',
          borderRadius: 8,
          padding: '6px 10px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          color: open ? '#c4b5fd' : 'rgba(226,232,240,0.7)',
          fontSize: 11,
          fontFamily: 'Inter, system-ui, sans-serif',
          backdropFilter: 'blur(8px)',
          transition: 'all 0.2s ease',
        }}
      >
        {/* Brain/thought icon */}
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2a7 7 0 0 0-7 7c0 2.38 1.19 4.47 3 5.74V17a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 0 0-7-7z"/>
          <path d="M9 21h6"/>
          <path d="M10 17v4"/>
          <path d="M14 17v4"/>
        </svg>
        Knowledge
      </button>

      {/* Floating modal */}
      {open && (
        <div
          onMouseDown={onMouseDown}
          style={{
            position: 'fixed',
            left: pos.x,
            top: pos.y,
            width: MODAL_W,
            height: MODAL_H,
            zIndex: 950,
            background: 'rgba(15,23,42,0.92)',
            border: '1px solid rgba(139,92,246,0.3)',
            borderRadius: 10,
            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
            backdropFilter: 'blur(12px)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          {/* Title bar — draggable */}
          <div
            data-drag-handle
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '6px 10px',
              background: 'rgba(139,92,246,0.12)',
              borderBottom: '1px solid rgba(139,92,246,0.15)',
              cursor: 'grab',
              userSelect: 'none',
              flexShrink: 0,
            }}
          >
            <span style={{ fontSize: 11, fontWeight: 600, color: '#c4b5fd', fontFamily: 'Inter, system-ui, sans-serif' }}>
              Knowledge Graph
            </span>
            <button
              onClick={() => setOpen(false)}
              style={{
                background: 'none',
                border: 'none',
                color: 'rgba(226,232,240,0.5)',
                cursor: 'pointer',
                fontSize: 16,
                lineHeight: 1,
                padding: '0 2px',
              }}
            >
              &times;
            </button>
          </div>
          {/* Graph content */}
          <div style={{ flex: 1, minHeight: 0 }}>
            <KnowledgeGraphPanel width={MODAL_W - 2} height={MODAL_H - 30} />
          </div>
        </div>
      )}
    </>
  )
}
