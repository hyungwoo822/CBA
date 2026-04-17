import { useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useWorkspace } from '../hooks/useWorkspace'
import type { Workspace } from '../stores/brainState'

const PERSONAL_WORKSPACE: Workspace = {
  id: 'personal',
  name: 'Personal Knowledge',
  decay_policy: 'normal',
}

function normalizeWorkspace(workspace: Workspace): Workspace {
  if (workspace.id !== PERSONAL_WORKSPACE.id) return workspace
  return {
    ...workspace,
    name: PERSONAL_WORKSPACE.name,
    decay_policy: workspace.decay_policy || PERSONAL_WORKSPACE.decay_policy,
  }
}

export function WorkspaceSelector() {
  const { current, workspaces, setCurrent } = useWorkspace()
  const [open, setOpen] = useState(false)
  const [menuPosition, setMenuPosition] = useState({ left: 0, top: 0, width: 200 })
  const buttonRef = useRef<HTMLButtonElement>(null)
  const normalizedWorkspaces = workspaces.map(normalizeWorkspace)
  const displayCurrent = current
    ? normalizeWorkspace(current)
    : normalizedWorkspaces.find((workspace) => workspace.id === PERSONAL_WORKSPACE.id)
    || PERSONAL_WORKSPACE
  const options = [
    displayCurrent,
    ...normalizedWorkspaces.filter((workspace) => workspace.id !== displayCurrent.id),
  ]

  const toggleOpen = () => {
    const rect = buttonRef.current?.getBoundingClientRect()
    if (rect) {
      setMenuPosition({
        left: rect.left,
        top: rect.bottom + 6,
        width: Math.max(200, rect.width),
      })
    }
    setOpen((value) => !value)
  }

  const handleSelect = async (workspace: Workspace) => {
    setOpen(false)
    ;(window as any).__openKnowledgeGraph?.({ workspaceId: workspace.id })
    if (workspace.id === displayCurrent.id) return
    await setCurrent(workspace.id)
  }

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <button
        ref={buttonRef}
        type="button"
        className="top-chip"
        aria-expanded={open}
        onClick={toggleOpen}
        data-testid="workspace-selector-btn"
      >
        {displayCurrent.name}
        <span className="top-chip-caret">v</span>
      </button>
      {open && createPortal(
        <div
          role="menu"
          className="top-chip-menu"
          style={{
            left: menuPosition.left,
            top: menuPosition.top,
            minWidth: menuPosition.width,
          }}
        >
          {options.map((workspace) => (
            <button
              key={workspace.id}
              type="button"
              className={`top-chip-menu-item${workspace.id === displayCurrent.id ? ' active' : ''}`}
              onClick={() => { void handleSelect(workspace) }}
            >
              {workspace.name}
            </button>
          ))}
          {!options.length && (
            <div className="top-chip-menu-empty">No workspaces loaded</div>
          )}
        </div>,
        document.body,
      )}
    </div>
  )
}
