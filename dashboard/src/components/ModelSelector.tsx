import type { CSSProperties } from 'react'
import { useEffect, useState } from 'react'
import { useModalDrag } from '../hooks/useModalDrag'
import { useBrainStore } from '../stores/brainState'

type Stage = 'triage' | 'extract' | 'temporal' | 'refine'

const STAGES: Stage[] = ['triage', 'extract', 'temporal', 'refine']

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
  const { style, onMouseDown } = useModalDrag('model-selector', window.innerWidth - 380, 80)

  useEffect(() => {
    if (!open) return
    void refreshModelInventory()
  }, [open, refreshModelInventory])

  if (!open) return null

  const modalStyle: CSSProperties = {
    ...style,
    width: 360,
  }

  return (
    <div className="kl-modal" style={modalStyle} onMouseDown={onMouseDown}>
      <div className="kl-modal-header" data-drag-handle>
        <span className="kl-modal-title">LLM Model per Stage</span>
        <button
          type="button"
          className="kl-modal-close"
          aria-label="Close"
          onClick={onClose}
        >
          &times;
        </button>
      </div>
      <div className="kl-modal-body" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
          Default (auto): <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{defaultModel}</span>
        </div>
        {STAGES.map((stage) => (
          <label key={stage} style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <span style={{ fontSize: 10, color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
              {stage}_model
            </span>
            <select
              aria-label={`${stage}_model`}
              value={selection[stage]}
              onChange={(event) => setSelection((current) => ({ ...current, [stage]: event.target.value }))}
              style={{
                padding: '4px 8px',
                fontSize: 11,
                background: 'rgba(255,255,255,0.04)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border)',
                borderRadius: 6,
              }}
            >
              <option value="auto">auto (&rarr; {defaultModel})</option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.id} {model.available ? '' : `[${model.reason || 'unavailable'}]`}
                </option>
              ))}
            </select>
          </label>
        ))}
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
          Selections apply next extraction call. Unavailable vendors may fail at runtime.
        </div>
      </div>
    </div>
  )
}
