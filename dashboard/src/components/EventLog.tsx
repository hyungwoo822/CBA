import { useEffect, useRef } from 'react'
import { useBrainStore } from '../stores/brainState'

const TAG_COLORS: Record<string, string> = {
  region_activation: '#3b82f6',
  network_switch: '#ef4444',
  routing_event: '#4ade80',
  memory_event: '#06b6d4',
  memory_flow: '#c084fc',
  neuromodulator: '#f97316',
  broadcast: '#fbbf24',
  region_io: '#14b8a6',
  signal_flow: '#4ade80',
}

export function EventLog() {
  const events = useBrainStore((s) => s.events)
  const connected = useBrainStore((s) => s.connected)
  const recent = events.slice(-30).reverse()
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0
  }, [events.length])

  return (
    <div className="event-log">
      <div className="event-log-header">
        <div className="dot" style={connected ? {} : { background: '#ef4444', boxShadow: '0 0 8px rgba(239,68,68,0.5)' }} />
        <span>Event Stream</span>
        <span style={{ marginLeft: 'auto', fontSize: 9, color: connected ? '#22c55e' : '#ef4444' }}>
          {connected ? 'LIVE' : 'OFF'}
        </span>
      </div>
      <div className="event-log-body" ref={scrollRef}>
        {recent.length === 0 && (
          <div style={{ color: '#64748b', fontSize: 11, lineHeight: 1.8, padding: '8px 0' }}>
            {connected
              ? 'Waiting for brain activity...'
              : (
                <>
                  <div>Backend not connected.</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'rgba(148,163,184,0.4)', marginTop: 4 }}>
                    $ brain-agent dashboard --port 3000
                  </div>
                </>
              )}
          </div>
        )}
        {recent.map((evt, i) => (
          <div key={i} className="event-item">
            <div
              className="event-dot"
              style={{
                background: TAG_COLORS[evt.type] || '#94a3b8',
                boxShadow: `0 0 4px ${TAG_COLORS[evt.type] || '#94a3b8'}`,
              }}
            />
            <div className="event-tag">
              <span className="label" style={{ color: TAG_COLORS[evt.type] || '#94a3b8' }}>
                {evt.type.replace(/_/g, ' ').slice(0, 14)}
              </span>
              <span className="payload">{JSON.stringify(evt.payload).slice(0, 50)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
