import type { MouseEvent } from 'react'
import { useBrainStore } from '../stores/brainState'
import { WorkspaceSelector } from './WorkspaceSelector'

export function TopBarChips() {
  const openQuestionCount = useBrainStore((s) => s.openQuestions.length)
  const severeCount = useBrainStore((s) =>
    s.openQuestions.filter((q) => q.severity === 'severe').length +
    s.contradictions.filter((c) => c.severity === 'severe').length
  )
  const openModels = (event: MouseEvent<HTMLButtonElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    ;(window as any).__setModelSelectorOpen?.({
      open: true,
      anchor: {
        x: Math.max(8, Math.min(rect.right + 8, window.innerWidth - 372)),
        y: Math.max(8, rect.top),
      },
    })
  }

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
        onClick={openModels}
        title="LLM models"
      >
        Models
        <svg
          className="top-chip-icon"
          data-testid="models-icon"
          width="13"
          height="13"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M12 3v18" />
          <path d="M5 8h14" />
          <path d="M5 16h14" />
          <circle cx="8" cy="8" r="2" />
          <circle cx="16" cy="16" r="2" />
        </svg>
      </button>
    </div>
  )
}
