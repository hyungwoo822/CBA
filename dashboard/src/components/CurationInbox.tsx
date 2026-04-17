import type { CSSProperties } from 'react'
import { useEffect, useState } from 'react'
import { useModalDrag } from '../hooks/useModalDrag'
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
  const { style, onMouseDown } = useModalDrag('curation-inbox', window.innerWidth - 400, 80)

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

  const modalStyle: CSSProperties = {
    ...style,
    width: 380,
    height: 'calc(100vh - 160px)',
    maxHeight: 680,
  }

  return (
    <div className="kl-modal" style={modalStyle} onMouseDown={onMouseDown}>
      <div className="kl-modal-header" data-drag-handle>
        <span className="kl-modal-title">Curation Inbox</span>
        <button
          type="button"
          className="kl-modal-close"
          aria-label="Close"
          onClick={onClose}
        >
          &times;
        </button>
      </div>
      <div className="kl-modal-tabs">
        {tabs.map((item) => (
          <button
            key={item.key}
            type="button"
            className={`kl-modal-tab${tab === item.key ? ' active' : ''}`}
            onClick={() => setTab(item.key)}
          >
            {item.label}
            <span className={`kl-modal-tab-count${item.severe ? ' severe' : ''}`}>
              {item.count}
            </span>
          </button>
        ))}
      </div>
      <div className="kl-modal-body">
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
      color: 'var(--text-muted)',
      fontSize: 11,
      padding: 20,
      textAlign: 'center',
    }}>
      Nothing pending here.
    </div>
  )
}
