import { useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useWorkspace } from '../hooks/useWorkspace'

export function WorkspaceSelector() {
  const { current, workspaces, setCurrent } = useWorkspace()
  const [open, setOpen] = useState(false)
  const [menuPosition, setMenuPosition] = useState({ left: 0, top: 0, width: 200 })
  const buttonRef = useRef<HTMLButtonElement>(null)
  const options = current && !workspaces.some((workspace) => workspace.id === current.id)
    ? [current, ...workspaces]
    : workspaces

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
        {current?.name || 'Select workspace'}
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
              className={`top-chip-menu-item${workspace.id === current?.id ? ' active' : ''}`}
              onClick={async () => {
                await setCurrent(workspace.id)
                setOpen(false)
              }}
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
