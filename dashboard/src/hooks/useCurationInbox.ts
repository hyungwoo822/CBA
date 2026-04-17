import { useEffect } from 'react'
import { useBrainStore } from '../stores/brainState'

export function useCurationInbox() {
  const currentWorkspaceId = useBrainStore((s) => s.currentWorkspace?.id)
  const refresh = useBrainStore((s) => s.refreshCuration)
  const openQuestions = useBrainStore((s) => s.openQuestions)
  const contradictions = useBrainStore((s) => s.contradictions)
  const ontologyProposals = useBrainStore((s) => s.ontologyProposals)

  useEffect(() => {
    if (currentWorkspaceId) void refresh(currentWorkspaceId)
  }, [currentWorkspaceId, refresh])

  const answerQuestion = async (id: string, answer: string) => {
    useBrainStore.getState().removeOpenQuestion(id)
    await fetch(`/api/questions/${id}/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answer, answer_source: 'inbox' }),
    })
  }

  const resolveContradiction = async (
    id: string,
    resolution: 'A' | 'B' | 'BOTH' | 'DISMISS',
  ) => {
    useBrainStore.getState().removeContradiction(id)
    const path = resolution === 'DISMISS'
      ? `/api/contradictions/${id}/dismiss`
      : `/api/contradictions/${id}/resolve`
    await fetch(path, {
      method: 'POST',
      headers: resolution === 'DISMISS' ? {} : { 'Content-Type': 'application/json' },
      body: resolution === 'DISMISS'
        ? undefined
        : JSON.stringify({
          resolution,
          resolved_by: 'user',
          resolution_confidence: 'USER_GROUND_TRUTH',
        }),
    })
  }

  const decideProposal = async (id: string, approve: boolean) => {
    useBrainStore.getState().removeOntologyProposal(id)
    await fetch(`/api/ontology/proposals/${id}/${approve ? 'approve' : 'reject'}`, {
      method: 'POST',
    })
  }

  return {
    openQuestions,
    contradictions,
    ontologyProposals,
    answerQuestion,
    resolveContradiction,
    decideProposal,
  }
}
