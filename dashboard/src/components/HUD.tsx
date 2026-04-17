import { useState } from 'react'
import { useBrainStore } from '../stores/brainState'
import { REGION_CONFIG } from '../constants/brainRegions'

const MODE_STYLES: Record<string, { label: string; bg: string; color: string; border: string }> = {
  executive_control: { label: 'ECN', bg: 'rgba(249,115,22,0.12)', color: '#f97316', border: 'rgba(249,115,22,0.15)' },
  default_mode: { label: 'DMN', bg: 'rgba(96,165,250,0.12)', color: '#60a5fa', border: 'rgba(96,165,250,0.15)' },
}

const NEURO_BARS = [
  { key: 'dopamine' as const, label: 'Dopamine (DA)', gradient: 'linear-gradient(90deg, #a855f7, #c084fc)' },
  { key: 'norepinephrine' as const, label: 'Norepinephrine (NE)', gradient: 'linear-gradient(90deg, #f97316, #fb923c)' },
  { key: 'serotonin' as const, label: 'Serotonin (5-HT)', gradient: 'linear-gradient(90deg, #3b82f6, #60a5fa)' },
  { key: 'acetylcholine' as const, label: 'Acetylcholine (ACh)', gradient: 'linear-gradient(90deg, #22c55e, #4ade80)' },
  { key: 'cortisol' as const, label: 'Cortisol (CORT)', gradient: 'linear-gradient(90deg, #ef4444, #f87171)' },
  { key: 'epinephrine' as const, label: 'Epinephrine (EPI)', gradient: 'linear-gradient(90deg, #eab308, #facc15)' },
  { key: 'gaba' as const, label: 'GABA', gradient: 'linear-gradient(90deg, #06b6d4, #0891b2)' },
]

export function HUD() {
  const nm = useBrainStore((s) => s.neuromodulators)
  const mode = useBrainStore((s) => s.networkMode)
  const regions = useBrainStore((s) => s.regions)
  const connected = useBrainStore((s) => s.connected)
  const selectedRegion = useBrainStore((s) => s.selectedRegion)
  const setSelectedRegion = useBrainStore((s) => s.setSelectedRegion)
  const [showRegions, setShowRegions] = useState(true)

  const modeStyle = MODE_STYLES[mode] || { label: 'CRE', bg: 'rgba(139,92,246,0.12)', color: '#8b5cf6', border: 'rgba(139,92,246,0.15)' }
  const activeCount = Object.values(regions).filter((r) => r.level > 0.1).length

  return (
    <div className="hud">
      <div className="status-row">
        <div
          className="status-dot"
          style={{
            background: connected ? '#22c55e' : '#ef4444',
            boxShadow: connected ? '0 0 8px rgba(34,197,94,0.5)' : '0 0 8px rgba(239,68,68,0.5)',
          }}
        />
        <span className="status-label">{connected ? 'Connected' : 'Disconnected'}</span>
        <div
          className="mode-badge"
          style={{ background: modeStyle.bg, color: modeStyle.color, border: `1px solid ${modeStyle.border}` }}
        >
          {modeStyle.label}
        </div>
      </div>

      <div className="neuro-grid">
        {[0, 2, 4, 6].map((startIdx) => (
          <div key={startIdx} className="neuro-pair">
            {NEURO_BARS.slice(startIdx, startIdx + 2).map((bar) => {
              const val = Math.round((nm[bar.key] ?? 0.5) * 100)
              return (
                <div key={bar.key} className="neuro-item">
                  <span className="neuro-label">{bar.label} ({val})</span>
                  <div className="neuro-bar-track">
                    <div className="neuro-bar-fill" style={{ width: `${val}%`, background: bar.gradient }} />
                  </div>
                </div>
              )
            })}
          </div>
        ))}
      </div>

      <div
        className="region-toggle"
        onClick={() => setShowRegions(!showRegions)}
      >
        <span>Regions ({activeCount}/{Object.keys(regions).length})</span>
        <span style={{ fontSize: 10 }}>{showRegions ? 'Hide' : 'Show'}</span>
      </div>

      {showRegions && (
        <div className="region-list">
          {Object.entries(regions).map(([name, state]) => {
            const isActive = state.level > 0.1
            const cfg = REGION_CONFIG[name]
            const isSelected = selectedRegion === name
            return (
              <div
                key={name}
                className={`region-list-item clickable${isSelected ? ' selected' : ''}`}
                onClick={() => setSelectedRegion(isSelected ? null : name)}
              >
                <div
                  className="region-indicator"
                  style={{
                    background: isActive ? cfg?.color : 'rgba(100,100,100,0.2)',
                    boxShadow: isActive ? `0 0 4px ${cfg?.color}` : 'none',
                  }}
                />
                <span className={`region-name${isActive ? ' active' : ''}${isSelected ? ' selected' : ''}`}>
                  {name.replace(/_/g, ' ')}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
