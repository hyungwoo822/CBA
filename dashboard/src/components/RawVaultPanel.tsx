import { useEffect, useState } from 'react'

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

  useEffect(() => {
    if (!sourceId) return
    fetch(`/api/sources/${sourceId}`).then((response) => response.json()).then(setMeta).catch(() => {})
    fetch(`/api/sources/${sourceId}/text`).then((response) => response.json()).then((data) => {
      setText(data.text || '')
    }).catch(() => {})
  }, [sourceId])

  if (!sourceId) return null

  return (
    <div style={{
      position: 'fixed',
      right: 12,
      top: 60,
      bottom: 12,
      width: 340,
      zIndex: 780,
      background: 'rgba(15,23,42,0.96)',
      border: '1px solid rgba(34,197,94,0.3)',
      borderRadius: 8,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      <div style={{
        padding: '6px 10px',
        background: 'rgba(34,197,94,0.12)',
        display: 'flex',
        justifyContent: 'space-between',
      }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: '#86efac' }}>Raw Vault Source</span>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}>
          &times;
        </button>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: 10 }}>
        {meta && (
          <div style={{ fontSize: 10, color: '#cbd5e1', marginBottom: 8 }}>
            <div><b>ID:</b> {meta.id}</div>
            <div><b>Kind:</b> {meta.kind}</div>
            <div><b>SHA256:</b> <span style={{ fontFamily: 'monospace' }}>{meta.sha256.slice(0, 16)}...</span></div>
            <div><b>Integrity:</b> {meta.integrity_valid ? 'valid' : <span style={{ color: '#f87171' }}>FAILED</span>}</div>
            {meta.mime_type?.startsWith('image/') && meta.integrity_valid ? (
              <img
                src={`/api/sources/${meta.id}/raw`}
                style={{ maxWidth: '100%', marginTop: 6, borderRadius: 3 }}
                alt="source preview"
              />
            ) : null}
          </div>
        )}
        {text && (
          <pre style={{
            fontSize: 10,
            color: '#e2e8f0',
            whiteSpace: 'pre-wrap',
            background: 'rgba(0,0,0,0.3)',
            padding: 8,
            borderRadius: 3,
          }}>
            {text}
          </pre>
        )}
      </div>
    </div>
  )
}
