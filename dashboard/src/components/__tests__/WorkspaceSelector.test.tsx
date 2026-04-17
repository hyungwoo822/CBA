import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useBrainStore } from '../../stores/brainState'
import { WorkspaceSelector } from '../WorkspaceSelector'

const fetchMock = vi.fn()

beforeEach(() => {
  fetchMock.mockReset()
  globalThis.fetch = fetchMock as any
  useBrainStore.setState({
    workspaces: [
      { id: 'personal', name: 'Personal Knowledge', decay_policy: 'normal' },
      { id: 'w1', name: 'Billing', decay_policy: 'none' },
    ],
    currentWorkspace: { id: 'personal', name: 'Personal Knowledge', decay_policy: 'normal' },
  })
})

describe('WorkspaceSelector', () => {
  it('shows current workspace name', () => {
    render(<WorkspaceSelector />)
    expect(screen.getByText(/Personal Knowledge/)).toBeInTheDocument()
  })

  it('switches workspace via PUT /api/workspaces/current', async () => {
    fetchMock.mockResolvedValue({ ok: true, json: async () => ({ workspaces: [] }) })
    render(<WorkspaceSelector />)
    fireEvent.click(screen.getByText(/Personal Knowledge/))
    fireEvent.click(screen.getByText(/Billing/))
    await waitFor(() => {
      const putCall = fetchMock.mock.calls.find((call) => call[1]?.method === 'PUT')
      expect(putCall?.[0]).toBe('/api/workspaces/current')
      expect(JSON.parse(putCall?.[1].body).workspace_id).toBe('w1')
    })
  })
})
