import { useEffect, useState } from 'react'

interface ChainEntry {
  edge_id: string
  subject: string
  relation: string
  target: string
  valid_from: string
  valid_to: string | null
  superseded_by: string | null
  confidence: string
}

export function TimelineView({
  workspaceId,
  subject,
}: {
  workspaceId: string
  subject: string
}) {
  const [chain, setChain] = useState<ChainEntry[]>([])

  useEffect(() => {
    fetch(`/api/memory/timeline?workspace_id=${workspaceId}&subject=${encodeURIComponent(subject)}`)
      .then((response) => response.json())
      .then((data) => setChain(data.chain || []))
      .catch(() => {})
  }, [workspaceId, subject])

  if (!chain.length) {
    return (
      <div style={{ fontSize: 10, color: '#94a3b8', padding: 10 }}>
        No temporal history for "{subject}"
      </div>
    )
  }

  const timestamps = chain.flatMap((entry) => [
    Date.parse(entry.valid_from),
    entry.valid_to ? Date.parse(entry.valid_to) : Date.now(),
  ])
  const min = Math.min(...timestamps)
  const max = Math.max(...timestamps, min + 1)

  return (
    <div style={{ padding: 10 }}>
      <div style={{ fontSize: 11, color: '#e2e8f0', marginBottom: 8 }}>
        Timeline for <b>{subject}</b>
      </div>
      <div style={{ position: 'relative', minHeight: chain.length * 28 + 20 }}>
        {chain.map((entry, index) => {
          const from = Date.parse(entry.valid_from)
          const to = entry.valid_to ? Date.parse(entry.valid_to) : Date.now()
          const left = ((from - min) / (max - min)) * 100
          const width = Math.max(2, ((to - from) / (max - min)) * 100)
          return (
            <div
              key={entry.edge_id}
              style={{
                position: 'absolute',
                top: index * 28,
                left: `${left}%`,
                width: `${width}%`,
                height: 22,
                background: entry.valid_to ? 'rgba(148,163,184,0.35)' : 'rgba(34,197,94,0.5)',
                border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: 3,
                fontSize: 9,
                color: '#e2e8f0',
                padding: '2px 6px',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
              }}
              title={`${entry.relation} -> ${entry.target} [${entry.valid_from} -> ${entry.valid_to || 'now'}]`}
            >
              {entry.relation} -&gt; {entry.target}
            </div>
          )
        })}
      </div>
    </div>
  )
}
