import { useBrainStore } from '../stores/brainState'
import { REGION_CONFIG, REGION_INFO, REGION_DESCRIPTIONS } from '../constants/brainRegions'
import { useDraggable } from '../hooks/useDraggable'

const ALL_REGIONS = Object.keys(REGION_CONFIG)

export function RegionDetailPanel() {
  const selectedRegion = useBrainStore((s) => s.selectedRegion)
  const region = useBrainStore((s) => selectedRegion ? s.regions[selectedRegion] : null)
  const events = useBrainStore((s) => s.events)
  const setSelectedRegion = useBrainStore((s) => s.setSelectedRegion)
  const { pos, onMouseDown } = useDraggable('region-detail', window.innerWidth - 710, 82)

  if (!selectedRegion || !region) return null

  const config = REGION_CONFIG[selectedRegion]
  const info = REGION_INFO[selectedRegion]
  const desc = REGION_DESCRIPTIONS[selectedRegion]
  const statusText = desc?.[region.mode] || desc?.active || 'Idle'

  const regionEvents = events.filter((evt) => {
    const p = evt.payload
    return p.region === selectedRegion || p.source === selectedRegion || p.target === selectedRegion
  }).slice(-30).reverse()

  return (
    <div
      className="region-detail-panel"
      style={{ left: pos.x, top: pos.y }}
      onMouseDown={onMouseDown}
    >
      <div className="rdp-header" data-drag-handle>
        <div className="rdp-color-dot" style={{ background: config?.color, width: 7, height: 7 }} />
        <span className="rdp-header-title">{info?.fullName || selectedRegion.replace(/_/g, ' ')}</span>
        <div className="rdp-change-wrap" onMouseDown={(e) => e.stopPropagation()}>
          <select
            className="rdp-change-select"
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
          >
            {ALL_REGIONS.map((r) => (
              <option key={r} value={r}>
                {REGION_INFO[r]?.fullName || r.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
          <span className="rdp-change-icon">&#8645;</span>
        </div>
        <button className="rdp-close" onClick={() => setSelectedRegion(null)}>&times;</button>
      </div>

      <div className="rdp-body">
        <div className="rdp-status">
          <span className="rdp-level-label">Activation</span>
          <div className="rdp-level-bar">
            <div className="rdp-level-fill" style={{ width: `${Math.round(region.level * 100)}%`, background: config?.color }} />
          </div>
          <span className="rdp-level-value">{(region.level * 100).toFixed(0)}%</span>
        </div>

        <div className="rdp-status-text" style={{ color: config?.color }}>{statusText}</div>

        <div className="rdp-section">
          <div className="rdp-section-title">Role</div>
          <div className="rdp-section-body">{info?.role}</div>
        </div>

        <div className="rdp-section">
          <div className="rdp-section-title">Mechanism</div>
          <div className="rdp-section-body">{info?.mechanism}</div>
        </div>

        <div className="rdp-section">
          <div className="rdp-section-title">Event Log ({regionEvents.length})</div>
          <div className="rdp-events">
            {regionEvents.length === 0 && <div className="rdp-no-events">No events yet</div>}
            {regionEvents.map((evt, i) => (
              <div key={i} className="rdp-event-item">
                <span className="rdp-event-type">{evt.type.replace(/_/g, ' ')}</span>
                <span className="rdp-event-payload">{JSON.stringify(evt.payload).slice(0, 60)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
