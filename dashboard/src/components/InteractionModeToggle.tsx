import { useBrainStore } from '../stores/brainState'

export function InteractionModeToggle() {
  const mode = useBrainStore((s) => s.interactionMode)
  const setMode = useBrainStore((s) => s.setInteractionMode)

  return (
    <div style={{
      position: 'fixed',
      top: '12%',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 901,
      display: 'flex',
      borderRadius: 20,
      overflow: 'hidden',
      border: '1px solid rgba(148,163,184,0.15)',
      background: 'rgba(10,10,20,0.7)',
      backdropFilter: 'blur(12px)',
      fontSize: 12,
      fontFamily: 'inherit',
      userSelect: 'none',
    }}>
      <button
        onClick={() => setMode('question')}
        style={{
          padding: '5px 14px',
          border: 'none',
          cursor: 'pointer',
          background: mode === 'question' ? 'rgba(249,115,22,0.25)' : 'transparent',
          color: mode === 'question' ? '#f97316' : 'rgba(226,232,240,0.5)',
          fontWeight: mode === 'question' ? 600 : 400,
          transition: 'all 0.2s ease',
          boxShadow: mode === 'question' ? '0 0 12px rgba(249,115,22,0.2)' : 'none',
        }}
      >
        Question
      </button>
      <button
        onClick={() => setMode('expression')}
        style={{
          padding: '5px 14px',
          border: 'none',
          cursor: 'pointer',
          background: mode === 'expression' ? 'rgba(96,165,250,0.25)' : 'transparent',
          color: mode === 'expression' ? '#60a5fa' : 'rgba(226,232,240,0.5)',
          fontWeight: mode === 'expression' ? 600 : 400,
          transition: 'all 0.2s ease',
          boxShadow: mode === 'expression' ? '0 0 12px rgba(96,165,250,0.2)' : 'none',
        }}
      >
        Expression
      </button>
    </div>
  )
}
