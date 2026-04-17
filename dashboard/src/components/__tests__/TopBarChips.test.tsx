import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useBrainStore } from '../../stores/brainState'
import { TopBarChips } from '../TopBarChips'

describe('TopBarChips', () => {
  beforeEach(() => {
    useBrainStore.setState({
      openQuestions: [{ id: '1', question: 'q', severity: 'minor' } as any],
      contradictions: [],
      ontologyProposals: [],
      currentWorkspace: { id: 'personal', name: 'personal' } as any,
      workspaces: [{ id: 'personal', name: 'personal' } as any],
      refreshWorkspaces: vi.fn(async () => {}),
    })
    ;(window as any).__setInboxOpen = vi.fn()
    ;(window as any).__setExportOpen = vi.fn()
    ;(window as any).__setModelSelectorOpen = vi.fn()
  })

  afterEach(() => {
    delete (window as any).__setInboxOpen
    delete (window as any).__setExportOpen
    delete (window as any).__setModelSelectorOpen
  })

  it('renders workspace, inbox, export, and models chips in a single row', () => {
    const { container } = render(<TopBarChips />)
    expect(container.querySelector('.top-chip-row')).toBeTruthy()
    expect(screen.getByTestId('workspace-selector-btn')).toBeTruthy()
    expect(screen.getByTestId('workspace-selector-btn')).toHaveTextContent('Personal Knowledge')
    expect(screen.getByRole('button', { name: /Inbox/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Export/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Models/i })).toBeTruthy()
  })

  it('shows the inbox count as a chip badge', () => {
    render(<TopBarChips />)
    expect(screen.getByTestId('inbox-count')).toHaveTextContent('1')
  })

  it('renders a model icon next to Models', () => {
    render(<TopBarChips />)
    expect(screen.getByTestId('models-icon')).toBeInTheDocument()
  })

  it('opens inbox / export / models modals via window hooks', () => {
    render(<TopBarChips />)
    fireEvent.click(screen.getByRole('button', { name: /Inbox/i }))
    expect((window as any).__setInboxOpen).toHaveBeenCalledWith(true)
    fireEvent.click(screen.getByRole('button', { name: /Export/i }))
    expect((window as any).__setExportOpen).toHaveBeenCalledWith(true)
    fireEvent.click(screen.getByRole('button', { name: /Models/i }))
    expect((window as any).__setModelSelectorOpen).toHaveBeenCalledWith(
      expect.objectContaining({
        open: true,
        anchor: expect.objectContaining({ x: expect.any(Number), y: expect.any(Number) }),
      }),
    )
  })
})
