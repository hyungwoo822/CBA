import { useState } from 'react'
import { useWorkspace } from '../hooks/useWorkspace'

export function WorkspaceSelector() {
  const { current, workspaces, setCurrent } = useWorkspace()
  const [open, setOpen] = useState(false)
  const options = current && !workspaces.some((workspace) => workspace.id === current.id)
    ? [current, ...workspaces]
    : workspaces

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <button
        type="button"
        className="top-chip"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        data-testid="workspace-selector-btn"
      >
        {current?.name || 'Select workspace'}
        <span className="top-chip-caret">v</span>
      </button>
      {open && (
        <div role="menu" className="top-chip-menu">
          {options.map((workspace) => (
            <button
              key={workspace.id}
              type="button"
              className={`top-chip-menu-item${workspace.id === current?.id ? ' active' : ''}`}
              onClick={async () => {
                await setCurrent(workspace.id)
                setOpen(false)
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
