import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useBrainStore } from '../../stores/brainState'
import { CurationInbox } from '../CurationInbox'

beforeEach(() => {
  globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }) as any
  useBrainStore.setState({
    currentWorkspace: null,
    openQuestions: [
      { id: 'q1', question: 'why', severity: 'moderate', workspace_id: 'w1' },
    ],
    contradictions: [
      { id: 'c1', subject: 's', value_a: 'a', value_b: 'b', severity: 'severe', workspace_id: 'w1' },
    ],
    ontologyProposals: [
      { id: 'p1', kind: 'node_type', proposed_name: 'Foo', confidence: 'PROVISIONAL', workspace_id: 'w1' },
    ],
  })
})

describe('CurationInbox', () => {
  it('renders 3 tabs with counts', () => {
    render(<CurationInbox open={true} onClose={() => {}} />)
    expect(screen.getByText(/Questions/)).toHaveTextContent('1')
    expect(screen.getByText(/Contradictions/)).toHaveTextContent('1')
    expect(screen.getByText(/Proposals/)).toHaveTextContent('1')
  })

  it('switches tabs', () => {
    render(<CurationInbox open={true} onClose={() => {}} />)
    fireEvent.click(screen.getByText(/Contradictions/))
    expect(screen.getByText(/Choose A/)).toBeInTheDocument()
  })

  it('submitting question answer triggers API call and removes card', async () => {
    render(<CurationInbox open={true} onClose={() => {}} />)
    fireEvent.change(screen.getByPlaceholderText(/Your answer/), { target: { value: 'because' } })
    fireEvent.click(screen.getByText('Submit'))
    await waitFor(() => {
      const call = (globalThis.fetch as any).mock.calls.find(
        (item: any[]) => item[0] === '/api/questions/q1/answer',
      )
      expect(call).toBeTruthy()
    })
    expect(useBrainStore.getState().openQuestions).toHaveLength(0)
  })
})
