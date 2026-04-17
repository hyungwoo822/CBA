import { useState } from 'react'
import type { OpenQuestion } from '../stores/brainState'

const SEVERITY_STYLE: Record<string, { bg: string; color: string }> = {
  severe: { bg: 'rgba(239,68,68,0.15)', color: '#f87171' },
  moderate: { bg: 'rgba(245,158,11,0.15)', color: '#fbbf24' },
  minor: { bg: 'rgba(96,165,250,0.15)', color: '#60a5fa' },
}

export function QuestionCard({
  q,
  onAnswer,
}: {
  q: OpenQuestion
  onAnswer: (id: string, answer: string) => void | Promise<void>
}) {
  const [answer, setAnswer] = useState('')
  const style = SEVERITY_STYLE[q.severity] || SEVERITY_STYLE.minor

  return (
    <div style={{
      padding: 10,
      marginBottom: 8,
      borderRadius: 6,
      background: 'rgba(15,23,42,0.62)',
      border: `1px solid ${style.color}40`,
    }}>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
        <span style={{
          padding: '1px 6px',
          borderRadius: 3,
          fontSize: 9,
          background: style.bg,
          color: style.color,
          textTransform: 'uppercase',
        }}>
          {q.severity}
        </span>
        {q.raised_by && (
          <span style={{ fontSize: 10, color: 'rgba(226,232,240,0.7)' }}>
            {q.raised_by}
          </span>
        )}
      </div>
      <div style={{ fontSize: 12, color: '#e2e8f0', marginBottom: 6 }}>{q.question}</div>
      {q.context_input && (
        <div style={{
          fontSize: 10,
          color: 'rgba(148,163,184,0.82)',
          marginBottom: 6,
          fontStyle: 'italic',
        }}>
          ctx: "{q.context_input}"
        </div>
      )}
      <textarea
        value={answer}
        onChange={(event) => setAnswer(event.target.value)}
        placeholder="Your answer..."
        rows={2}
        style={{
          width: '100%',
          padding: 6,
          fontSize: 11,
          background: 'rgba(0,0,0,0.3)',
          color: '#e2e8f0',
          border: '1px solid rgba(148,163,184,0.22)',
          borderRadius: 3,
          boxSizing: 'border-box',
        }}
      />
      <button
        onClick={() => {
          if (!answer.trim()) return
          void onAnswer(q.id, answer)
          setAnswer('')
        }}
        style={{
          marginTop: 6,
          padding: '4px 10px',
          fontSize: 11,
          background: '#22c55e',
          color: '#0a0a14',
          border: 'none',
          borderRadius: 3,
          cursor: 'pointer',
        }}
      >
        Submit
      </button>
    </div>
  )
}
