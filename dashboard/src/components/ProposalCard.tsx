import type { OntologyProposal } from '../stores/brainState'

export function ProposalCard({
  p,
  onDecide,
}: {
  p: OntologyProposal
  onDecide: (id: string, approve: boolean) => void | Promise<void>
}) {
  return (
    <div style={{
      padding: 10,
      marginBottom: 8,
      borderRadius: 6,
      background: 'rgba(15,23,42,0.62)',
      border: '1px solid rgba(20,184,166,0.28)',
    }}>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
        <span style={{
          padding: '1px 6px',
          borderRadius: 3,
          fontSize: 9,
          background: 'rgba(20,184,166,0.16)',
          color: '#5eead4',
        }}>
          {p.kind}
        </span>
        <b style={{ fontSize: 12, color: '#e2e8f0' }}>{p.proposed_name}</b>
        <span style={{ fontSize: 9, color: '#94a3b8' }}>({p.confidence})</span>
      </div>
      {p.definition && (
        <div style={{
          fontSize: 10,
          color: 'rgba(226,232,240,0.72)',
          marginBottom: 4,
          fontFamily: 'monospace',
        }}>
          {p.definition}
        </div>
      )}
      {p.source_snippet && (
        <div style={{
          fontSize: 10,
          color: 'rgba(148,163,184,0.82)',
          marginBottom: 6,
          fontStyle: 'italic',
        }}>
          "{p.source_snippet}"
        </div>
      )}
      <div style={{ display: 'flex', gap: 6 }}>
        <button
          onClick={() => { void onDecide(p.id, true) }}
          style={{
            padding: '3px 10px',
            fontSize: 10,
            background: '#22c55e',
            color: '#0a0a14',
            border: 'none',
            borderRadius: 3,
            cursor: 'pointer',
          }}
        >
          Approve
        </button>
        <button
          onClick={() => { void onDecide(p.id, false) }}
          style={{
            padding: '3px 10px',
            fontSize: 10,
            background: '#ef4444',
            color: '#0a0a14',
            border: 'none',
            borderRadius: 3,
            cursor: 'pointer',
          }}
        >
          Reject
        </button>
      </div>
    </div>
  )
}
