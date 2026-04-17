import type { CSSProperties } from 'react'
import { useEffect, useState } from 'react'
import { useModalDrag } from '../hooks/useModalDrag'

interface SourceMeta {
  id: string
  workspace_id: string
  kind: string
  uri?: string
  mime_type?: string
  sha256: string
  integrity_valid: number
  ingested_at: string
}

export function RawVaultPanel({
  sourceId,
  onClose,
}: {
  sourceId: string | null
  onClose: () => void
}) {
  const [meta, setMeta] = useState<SourceMeta | null>(null)
  const [text, setText] = useState('')
  const { style, onMouseDown } = useModalDrag('raw-vault-panel', window.innerWidth - 380, 80)

  useEffect(() => {
    if (!sourceId) return
    fetch(`/api/sources/${sourceId}`).then((response) => response.json()).then(setMeta).catch(() => {})
    fetch(`/api/sources/${sourceId}/text`).then((response) => response.json()).then((data) => {
      setText(data.text || '')
    }).catch(() => {})
  }, [sourceId])

  if (!sourceId) return null

  const modalStyle: CSSProperties = {
    ...style,
    width: 360,
    height: 'calc(100vh - 160px)',
    maxHeight: 640,
  }

  return (
    <div className="kl-modal" style={modalStyle} onMouseDown={onMouseDown}>
      <div className="kl-modal-header" data-drag-handle>
        <span className="kl-modal-title">Raw Vault Source</span>
        <button
          type="button"
          className="kl-modal-close"
          aria-label="Close"
          onClick={onClose}
        >
          &times;
        </button>
      </div>
      <div className="kl-modal-body">
        {meta && (
          <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 8, lineHeight: 1.7 }}>
            <div><b>ID:</b> {meta.id}</div>
            <div><b>Kind:</b> {meta.kind}</div>
            <div>
              <b>SHA256:</b>{' '}
              <span style={{ fontFamily: 'var(--font-mono)' }}>{meta.sha256.slice(0, 16)}...</span>
            </div>
            <div>
              <b>Integrity:</b>{' '}
              {meta.integrity_valid
                ? 'valid'
                : <span style={{ color: 'var(--red)' }}>FAILED</span>}
            </div>
            {meta.mime_type?.startsWith('image/') && meta.integrity_valid ? (
              <img
                src={`/api/sources/${meta.id}/raw`}
                style={{ maxWidth: '100%', marginTop: 6, borderRadius: 6 }}
                alt="source preview"
              />
            ) : null}
          </div>
        )}
        {text && (
          <pre style={{
            fontSize: 10,
            color: 'var(--text-primary)',
            whiteSpace: 'pre-wrap',
            background: 'rgba(0,0,0,0.3)',
            padding: 10,
            borderRadius: 8,
            border: '1px solid var(--border)',
          }}>
            {text}
          </pre>
        )}
      </div>
    </div>
  )
}
