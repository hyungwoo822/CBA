import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useBrainStore } from '../../stores/brainState'
import { ExportPreviewModal } from '../ExportPreviewModal'

beforeEach(() => {
  useBrainStore.setState({
    currentWorkspace: { id: 'w1', name: 'W1', decay_policy: 'normal' },
    workspaces: [{ id: 'w1', name: 'W1', decay_policy: 'normal' }],
  })
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ workspace: { id: 'w1' }, facts: [], ontology: {} }),
  }) as any
})

describe('ExportPreviewModal', () => {
  it('fetches on open', async () => {
    render(<ExportPreviewModal open={true} onClose={() => {}} />)
    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/export/preview',
        expect.objectContaining({ method: 'POST' }),
      )
    })
  })

  it('refetches when never_decay_only toggled', async () => {
    render(<ExportPreviewModal open={true} onClose={() => {}} />)
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    const callsBefore = (globalThis.fetch as any).mock.calls.length
    fireEvent.click(screen.getByLabelText(/never_decay_only/i))
    await waitFor(() => {
      expect((globalThis.fetch as any).mock.calls.length).toBeGreaterThan(callsBefore)
    })
  })

  it('Copy button writes JSON to clipboard', async () => {
    const writeText = vi.fn()
    Object.assign(navigator, { clipboard: { writeText } })
    render(<ExportPreviewModal open={true} onClose={() => {}} />)
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    fireEvent.click(screen.getByText(/Copy/))
    await waitFor(() => {
      expect(writeText).toHaveBeenCalled()
    })
  })
})

describe('ExportPreviewModal frame', () => {
  it('renders a draggable header with a data-drag-handle attribute', async () => {
    render(<ExportPreviewModal open={true} onClose={() => {}} />)
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    const header = document.querySelector('.kl-modal-header')
    expect(header).toBeTruthy()
    expect(header?.hasAttribute('data-drag-handle')).toBe(true)
  })

  it('renders an accessible close button that fires onClose', async () => {
    const onClose = vi.fn()
    render(<ExportPreviewModal open={true} onClose={onClose} />)
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    const closeBtn = screen.getByRole('button', { name: /close/i })
    closeBtn.click()
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})
