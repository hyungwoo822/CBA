import { useEffect, useState } from 'react'
import { useCurationInbox } from '../hooks/useCurationInbox'
import { ContradictionCard } from './ContradictionCard'
import { ProposalCard } from './ProposalCard'
import { QuestionCard } from './QuestionCard'

type Tab = 'questions' | 'contradictions' | 'proposals'

export function CurationInbox({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const {
    openQuestions,
    contradictions,
    ontologyProposals,
    answerQuestion,
    resolveContradiction,
    decideProposal,
  } = useCurationInbox()
  const [tab, setTab] = useState<Tab>('questions')

  useEffect(() => {
    ;(window as any).__openCurationInbox = () => {
      ;(window as any).__setInboxOpen?.(true)
    }
    return () => {
      delete (window as any).__openCurationInbox
    }
  }, [])

  if (!open) return null

  const tabs = [
    {
      key: 'questions',
      label: 'Questions',
      count: openQuestions.length,
      severe: openQuestions.some((question) => question.severity === 'severe'),
    },
    {
      key: 'contradictions',
      label: 'Contradictions',
      count: contradictions.length,
      severe: contradictions.some((contradiction) => contradiction.severity === 'severe'),
    },
    {
      key: 'proposals',
      label: 'Proposals',
      count: ontologyProposals.length,
      severe: false,
    },
  ] as const

  return (
    <div style={{
      position: 'fixed',
      right: 12,
      top: 60,
      bottom: 12,
      width: 380,
      zIndex: 800,
      background: 'rgba(15,23,42,0.96)',
      border: '1px solid rgba(96,165,250,0.3)',
      borderRadius: 8,
      backdropFilter: 'blur(10px)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '6px 10px',
        background: 'rgba(20,184,166,0.14)',
      }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: '#5eead4' }}>
          Curation Inbox
        </span>
        <button onClick={onClose} style={{
          background: 'none',
          border: 'none',
          color: '#94a3b8',
          cursor: 'pointer',
          fontSize: 16,
          padding: 0,
        }}>
          &times;
        </button>
      </div>
      <div style={{ display: 'flex', borderBottom: '1px solid rgba(148,163,184,0.15)' }}>
        {tabs.map((item) => (
          <button
            key={item.key}
            onClick={() => setTab(item.key)}
            style={{
              flex: 1,
              padding: '6px 4px',
              fontSize: 10,
              background: tab === item.key ? 'rgba(20,184,166,0.14)' : 'transparent',
              color: tab === item.key ? '#5eead4' : '#94a3b8',
              border: 'none',
              borderBottom: tab === item.key ? '2px solid #14b8a6' : '2px solid transparent',
              cursor: 'pointer',
            }}
          >
            {item.label}{' '}
            <span style={{
              padding: '1px 5px',
              borderRadius: 8,
              fontSize: 9,
              background: item.severe ? '#ef4444' : 'rgba(148,163,184,0.25)',
              color: item.severe ? '#fff' : '#94a3b8',
            }}>
              {item.count}
            </span>
          </button>
        ))}
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: 8 }}>
        {tab === 'questions' && openQuestions.map((question) => (
          <QuestionCard key={question.id} q={question} onAnswer={answerQuestion} />
        ))}
        {tab === 'contradictions' && contradictions.map((contradiction) => (
          <ContradictionCard
            key={contradiction.id}
            c={contradiction}
            onResolve={resolveContradiction}
          />
        ))}
        {tab === 'proposals' && ontologyProposals.map((proposal) => (
          <ProposalCard key={proposal.id} p={proposal} onDecide={decideProposal} />
        ))}
        {tab === 'questions' && openQuestions.length === 0 && <Empty />}
        {tab === 'contradictions' && contradictions.length === 0 && <Empty />}
        {tab === 'proposals' && ontologyProposals.length === 0 && <Empty />}
      </div>
    </div>
  )
}

function Empty() {
  return (
    <div style={{
      color: 'rgba(148,163,184,0.55)',
      fontSize: 10,
      padding: 20,
      textAlign: 'center',
    }}>
      Nothing pending here.
    </div>
  )
}
