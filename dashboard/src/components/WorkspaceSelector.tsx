import { useState } from 'react'
import { useWorkspace } from '../hooks/useWorkspace'

export function WorkspaceSelector() {
  const { current, workspaces, setCurrent } = useWorkspace()
  const [open, setOpen] = useState(false)

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <button
        onClick={() => setOpen((value) => !value)}
        style={{
          background: 'rgba(20,184,166,0.14)',
          border: '1px solid rgba(20,184,166,0.35)',
          borderRadius: 4,
          padding: '4px 10px',
          color: '#5eead4',
          fontSize: 11,
          cursor: 'pointer',
        }}
        data-testid="workspace-selector-btn"
      >
        {current?.name || 'Select workspace'} <span style={{ fontSize: 8 }}>v</span>
      </button>
      {open && (
        <div
          role="menu"
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            marginTop: 4,
            minWidth: 180,
            background: 'rgba(15,23,42,0.96)',
            border: '1px solid rgba(20,184,166,0.35)',
            borderRadius: 4,
            padding: 4,
            zIndex: 1000,
          }}
        >
          {workspaces.map((workspace) => (
            <button
              key={workspace.id}
              onClick={async () => {
                await setCurrent(workspace.id)
                setOpen(false)
              }}
              style={{
                display: 'block',
                width: '100%',
                padding: '4px 8px',
                cursor: 'pointer',
                fontSize: 11,
                textAlign: 'left',
                color: workspace.id === current?.id ? '#5eead4' : 'rgba(226,232,240,0.78)',
                background: workspace.id === current?.id ? 'rgba(20,184,166,0.16)' : 'transparent',
                border: 'none',
                borderRadius: 3,
              }}
            >
              {workspace.name}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
