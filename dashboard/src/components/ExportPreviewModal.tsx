import type { CSSProperties } from 'react'
import { lazy, Suspense, useEffect, useState } from 'react'
import { useModalDrag } from '../hooks/useModalDrag'
import { useBrainStore } from '../stores/brainState'

const MonacoEditor = lazy(() => import('@monaco-editor/react').then((module) => ({ default: module.default })))

interface Filters {
  never_decay_only: boolean
  min_importance: number
  min_confidence: string
  include_raw_vault: boolean
}

export function ExportPreviewModal({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const workspaces = useBrainStore((s) => s.workspaces)
  const currentWorkspace = useBrainStore((s) => s.currentWorkspace)
  const [workspaceId, setWorkspaceId] = useState(currentWorkspace?.id || 'personal')
  const [filters, setFilters] = useState<Filters>({
    never_decay_only: false,
    min_importance: 0,
    min_confidence: 'PROVISIONAL',
    include_raw_vault: false,
  })
  const [json, setJson] = useState('{}')
  const { style, onMouseDown } = useModalDrag('export-preview-modal', 60, 80)

  useEffect(() => {
    if (!currentWorkspace?.id) return
    setWorkspaceId(currentWorkspace.id)
  }, [currentWorkspace?.id])

  useEffect(() => {
    if (!open) return
    fetch('/api/export/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_id: workspaceId, filters }),
    }).then((response) => response.json())
      .then((data) => setJson(JSON.stringify(data, null, 2)))
      .catch(() => {})
  }, [open, workspaceId, filters])

  if (!open) return null

  const options = workspaces.length ? workspaces : currentWorkspace ? [currentWorkspace] : []
  const modalStyle: CSSProperties = {
    ...style,
    width: 'min(900px, 92vw)',
    height: 'min(720px, 82vh)',
  }

  return (
    <div className="kl-modal" style={modalStyle} onMouseDown={onMouseDown}>
      <div className="kl-modal-header" data-drag-handle>
        <span className="kl-modal-title">Export Preview (MCP JSON)</span>
        <button
          type="button"
          className="kl-modal-close"
          aria-label="Close"
          onClick={onClose}
        >
          &times;
        </button>
      </div>
      <div className="kl-modal-toolbar">
        <select
          value={workspaceId}
          onChange={(event) => setWorkspaceId(event.target.value)}
          aria-label="workspace"
        >
          {options.map((workspace) => (
            <option key={workspace.id} value={workspace.id}>{workspace.name}</option>
          ))}
          {!options.length && <option value="personal">personal</option>}
        </select>
        <label>
          <input
            type="checkbox"
            checked={filters.never_decay_only}
            aria-label="never_decay_only"
            onChange={(event) => setFilters((current) => ({ ...current, never_decay_only: event.target.checked }))}
          />
          never_decay_only
        </label>
        <label>
          min_importance:
          <input
            type="number"
            min={0}
            max={1}
            step={0.1}
            value={filters.min_importance}
            onChange={(event) => setFilters((current) => ({ ...current, min_importance: Number(event.target.value) }))}
            style={{ width: 58 }}
          />
        </label>
        <label>
          min_confidence:
          <select
            value={filters.min_confidence}
            onChange={(event) => setFilters((current) => ({ ...current, min_confidence: event.target.value }))}
          >
            {['PROVISIONAL', 'STABLE', 'CANONICAL', 'USER_GROUND_TRUTH'].map((confidence) => (
              <option key={confidence} value={confidence}>{confidence}</option>
            ))}
          </select>
        </label>
        <label>
          <input
            type="checkbox"
            checked={filters.include_raw_vault}
            onChange={(event) => setFilters((current) => ({ ...current, include_raw_vault: event.target.checked }))}
          />
          include_raw_vault
        </label>
        <button
          type="button"
          className="kl-modal-btn"
          onClick={() => { void navigator.clipboard?.writeText(json) }}
        >
          Copy
        </button>
        <button
          type="button"
          className="kl-modal-btn primary"
          onClick={() => {
            const blob = new Blob([json], { type: 'application/json' })
            const url = URL.createObjectURL(blob)
            const anchor = document.createElement('a')
            anchor.href = url
            anchor.download = `cba-export-${workspaceId}.json`
            anchor.click()
            URL.revokeObjectURL(url)
          }}
        >
          Download
        </button>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <Suspense fallback={<pre style={{ padding: 10, fontSize: 10, color: 'var(--text-primary)' }}>{json}</pre>}>
          <MonacoEditor
            height="100%"
            language="json"
            value={json}
            options={{ readOnly: true, minimap: { enabled: false }, fontSize: 11 }}
            theme="vs-dark"
          />
        </Suspense>
      </div>
    </div>
  )
}
