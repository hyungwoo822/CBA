import type { CSSProperties } from 'react'
import { lazy, Suspense, useEffect, useState } from 'react'
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

  return (
    <div style={{
      position: 'fixed',
      inset: 40,
      zIndex: 1500,
      background: 'rgba(15,23,42,0.97)',
      border: '1px solid rgba(20,184,166,0.38)',
      borderRadius: 8,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      <div style={{
        padding: '8px 12px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'rgba(20,184,166,0.14)',
      }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: '#5eead4' }}>
          Export Preview (MCP JSON)
        </span>
        <button onClick={onClose} style={{
          background: 'none',
          border: 'none',
          color: '#94a3b8',
          cursor: 'pointer',
          fontSize: 18,
        }}>
          &times;
        </button>
      </div>
      <div style={{
        display: 'flex',
        gap: 12,
        padding: 8,
        alignItems: 'center',
        borderBottom: '1px solid rgba(148,163,184,0.15)',
        flexWrap: 'wrap',
      }}>
        <select value={workspaceId} onChange={(event) => setWorkspaceId(event.target.value)} style={{ padding: '3px 6px', fontSize: 11 }}>
          {options.map((workspace) => (
            <option key={workspace.id} value={workspace.id}>{workspace.name}</option>
          ))}
          {!options.length && <option value="personal">personal</option>}
        </select>
        <label style={labelStyle}>
          <input
            type="checkbox"
            checked={filters.never_decay_only}
            aria-label="never_decay_only"
            onChange={(event) => setFilters((current) => ({ ...current, never_decay_only: event.target.checked }))}
          />
          never_decay_only
        </label>
        <label style={labelStyle}>
          min_importance:
          <input
            type="number"
            min={0}
            max={1}
            step={0.1}
            value={filters.min_importance}
            onChange={(event) => setFilters((current) => ({ ...current, min_importance: Number(event.target.value) }))}
            style={{ width: 50, marginLeft: 4 }}
          />
        </label>
        <label style={labelStyle}>
          min_confidence:
          <select
            value={filters.min_confidence}
            onChange={(event) => setFilters((current) => ({ ...current, min_confidence: event.target.value }))}
            style={{ marginLeft: 4, padding: '2px 4px', fontSize: 10 }}
          >
            {['PROVISIONAL', 'STABLE', 'CANONICAL', 'USER_GROUND_TRUTH'].map((confidence) => (
              <option key={confidence} value={confidence}>{confidence}</option>
            ))}
          </select>
        </label>
        <label style={labelStyle}>
          <input
            type="checkbox"
            checked={filters.include_raw_vault}
            onChange={(event) => setFilters((current) => ({ ...current, include_raw_vault: event.target.checked }))}
          />
          include_raw_vault
        </label>
        <button onClick={() => { void navigator.clipboard?.writeText(json) }} style={buttonStyle('#22c55e')}>Copy</button>
        <button
          onClick={() => {
            const blob = new Blob([json], { type: 'application/json' })
            const url = URL.createObjectURL(blob)
            const anchor = document.createElement('a')
            anchor.href = url
            anchor.download = `cba-export-${workspaceId}.json`
            anchor.click()
            URL.revokeObjectURL(url)
          }}
          style={buttonStyle('#60a5fa')}
        >
          Download
        </button>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <Suspense fallback={<pre style={{ padding: 10, fontSize: 10, color: '#e2e8f0' }}>{json}</pre>}>
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

const labelStyle: CSSProperties = {
  fontSize: 10,
  color: '#cbd5e1',
  display: 'flex',
  alignItems: 'center',
  gap: 3,
}

const buttonStyle = (background: string): CSSProperties => ({
  padding: '3px 10px',
  fontSize: 10,
  background,
  color: '#0a0a14',
  border: 'none',
  borderRadius: 3,
  cursor: 'pointer',
})
