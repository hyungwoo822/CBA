import type { CSSProperties } from 'react'
import type { Contradiction } from '../stores/brainState'

export function ContradictionCard({
  c,
  onResolve,
}: {
  c: Contradiction
  onResolve: (id: string, choice: 'A' | 'B' | 'BOTH' | 'DISMISS') => void | Promise<void>
}) {
  const subject = c.subject || c.subject_node || 'unknown'

  return (
    <div style={{
      padding: 10,
      marginBottom: 8,
      borderRadius: 6,
      background: 'rgba(15,23,42,0.62)',
      border: '1px solid rgba(239,68,68,0.25)',
    }}>
      <div style={{ fontSize: 11, color: '#e2e8f0', marginBottom: 6 }}>
        <b>{subject}</b> <span style={{ color: '#94a3b8' }}>severity: {c.severity}</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 8 }}>
        {[
          { label: 'A', value: c.value_a, src: c.source_a, epi: c.epistemic_source_a },
          { label: 'B', value: c.value_b, src: c.source_b, epi: c.epistemic_source_b },
        ].map((side) => (
          <div key={side.label} style={{
            padding: 6,
            borderRadius: 4,
            background: 'rgba(0,0,0,0.22)',
            border: '1px solid rgba(148,163,184,0.15)',
          }}>
            <div style={{ fontSize: 10, color: '#94a3b8' }}>{side.label}</div>
            <div style={{ fontSize: 12, color: '#e2e8f0', marginTop: 2 }}>{side.value}</div>
            {side.epi && (
              <div style={{
                marginTop: 4,
                display: 'inline-block',
                padding: '1px 5px',
                borderRadius: 2,
                fontSize: 9,
                background: 'rgba(20,184,166,0.16)',
                color: '#5eead4',
              }}>
                {side.epi}
              </div>
            )}
            {side.src && (
              <div style={{
                fontSize: 9,
                color: 'rgba(148,163,184,0.72)',
                marginTop: 3,
                fontStyle: 'italic',
              }}>
                "{side.src}"
              </div>
            )}
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        <button onClick={() => { void onResolve(c.id, 'A') }} style={buttonStyle('#22c55e')}>Choose A</button>
        <button onClick={() => { void onResolve(c.id, 'B') }} style={buttonStyle('#22c55e')}>Choose B</button>
        <button onClick={() => { void onResolve(c.id, 'BOTH') }} style={buttonStyle('#60a5fa')}>Both</button>
        <button onClick={() => { void onResolve(c.id, 'DISMISS') }} style={buttonStyle('#6b7280')}>Dismiss</button>
      </div>
    </div>
  )
}

const buttonStyle = (background: string): CSSProperties => ({
  padding: '3px 8px',
  fontSize: 10,
  background,
  color: '#0a0a14',
  border: 'none',
  borderRadius: 3,
  cursor: 'pointer',
})
