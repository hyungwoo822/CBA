import { useEffect, useState } from 'react'
import { useBrainStore } from '../stores/brainState'

type Stage = 'triage' | 'extract' | 'temporal' | 'refine'

export function ModelSelector({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const defaultModel = useBrainStore((s) => s.defaultModel)
  const models = useBrainStore((s) => s.availableModels)
  const refreshModelInventory = useBrainStore((s) => s.refreshModelInventory)
  const [selection, setSelection] = useState<Record<Stage, string>>({
    triage: 'auto',
    extract: 'auto',
    temporal: 'auto',
    refine: 'auto',
  })

  useEffect(() => {
    if (!open) return
    void refreshModelInventory()
  }, [open, refreshModelInventory])

  if (!open) return null

  const stages: Stage[] = ['triage', 'extract', 'temporal', 'refine']

  return (
    <div style={{
      position: 'fixed',
      right: 12,
      top: 60,
      width: 340,
      zIndex: 790,
      background: 'rgba(15,23,42,0.97)',
      border: '1px solid rgba(20,184,166,0.32)',
      borderRadius: 8,
      padding: 12,
      display: 'flex',
      flexDirection: 'column',
      gap: 10,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: '#5eead4' }}>
          LLM Model per Stage
        </span>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}>
          &times;
        </button>
      </div>
      <div style={{ fontSize: 9, color: '#94a3b8' }}>
        Default (auto): <span style={{ fontFamily: 'monospace', color: '#e2e8f0' }}>{defaultModel}</span>
      </div>
      {stages.map((stage) => (
        <label key={stage} style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <span style={{ fontSize: 10, color: '#cbd5e1', textTransform: 'capitalize' }}>{stage}_model</span>
          <select
            aria-label={`${stage}_model`}
            value={selection[stage]}
            onChange={(event) => setSelection((current) => ({ ...current, [stage]: event.target.value }))}
            style={{
              padding: '3px 6px',
              fontSize: 11,
              background: 'rgba(0,0,0,0.3)',
              color: '#e2e8f0',
              border: '1px solid rgba(148,163,184,0.22)',
            }}
          >
            <option value="auto">auto ({defaultModel})</option>
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.id} {model.available ? '' : `[${model.reason || 'unavailable'}]`}
              </option>
            ))}
          </select>
        </label>
      ))}
      <div style={{ fontSize: 9, color: 'rgba(148,163,184,0.72)', marginTop: 4 }}>
        Selections apply next extraction call. Unavailable vendors may fail at runtime.
      </div>
    </div>
  )
}
