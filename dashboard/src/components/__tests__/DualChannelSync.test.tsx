import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useBrainStore } from '../../stores/brainState'
import { CurationInbox } from '../CurationInbox'

beforeEach(() => {
  globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }) as any
  useBrainStore.setState({
    currentWorkspace: { id: 'w1', name: 'W1', decay_policy: 'normal' },
    openQuestions: [{ id: 'q1', question: 'why', severity: 'moderate', workspace_id: 'w1' }],
    contradictions: [],
    ontologyProposals: [],
  })
})

describe('Dual-channel sync', () => {
  it('question_answered WS event removes card from inbox', async () => {
    render(<CurationInbox open={true} onClose={() => {}} />)
    expect(screen.getByText(/why/)).toBeInTheDocument()
    act(() => {
      useBrainStore.getState().removeOpenQuestion('q1')
    })
    await waitFor(() => {
      expect(screen.queryByText(/why/)).not.toBeInTheDocument()
    })
  })

  it('inbox answer triggers optimistic removal', async () => {
    render(<CurationInbox open={true} onClose={() => {}} />)
    fireEvent.change(screen.getByPlaceholderText(/Your answer/), { target: { value: 'x' } })
    fireEvent.click(screen.getByText('Submit'))
    await waitFor(() => {
      expect(useBrainStore.getState().openQuestions).toHaveLength(0)
    })
  })
})
