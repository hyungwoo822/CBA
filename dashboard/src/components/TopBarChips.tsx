import { useBrainStore } from '../stores/brainState'
import { WorkspaceSelector } from './WorkspaceSelector'

export function TopBarChips() {
  const openQuestionCount = useBrainStore((s) => s.openQuestions.length)
  const severeCount = useBrainStore((s) =>
    s.openQuestions.filter((q) => q.severity === 'severe').length +
    s.contradictions.filter((c) => c.severity === 'severe').length
  )

  return (
    <div className="top-chip-row">
      <WorkspaceSelector />
      <button
        type="button"
        className="top-chip"
        onClick={() => { (window as any).__setInboxOpen?.(true) }}
        title="Curation Inbox"
      >
        Inbox
        <span
          data-testid="inbox-count"
          className={`top-chip-count${severeCount > 0 ? ' severe' : ''}`}
        >
          {openQuestionCount}
        </span>
      </button>
      <button
        type="button"
        className="top-chip"
        onClick={() => { (window as any).__setExportOpen?.(true) }}
        title="Export preview"
      >
        Export
      </button>
      <button
        type="button"
        className="top-chip"
        onClick={() => { (window as any).__setModelSelectorOpen?.(true) }}
        title="LLM models"
      >
        Models
      </button>
    </div>
  )
}
