import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ModelSelector } from '../ModelSelector'

beforeEach(() => {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      default_model: 'openai/gpt-4o-mini',
      available: [
        { id: 'openai/gpt-4o-mini', vendor: 'openai', available: true },
        { id: 'ollama/llama3', vendor: 'ollama', available: false, reason: 'ollama not reachable' },
      ],
    }),
  }) as any
})

describe('ModelSelector', () => {
  it('renders 4 dropdowns', async () => {
    render(<ModelSelector open={true} onClose={() => {}} />)
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalledWith('/api/llm/providers'))
    expect(screen.getByLabelText(/triage/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/extract/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/temporal/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/refine/i)).toBeInTheDocument()
  })

  it('shows unavailable vendor model with reason', async () => {
    render(<ModelSelector open={true} onClose={() => {}} />)
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    expect(screen.getAllByText(/ollama not reachable/).length).toBeGreaterThan(0)
  })

  it('uses dropdown options from fetch', async () => {
    render(<ModelSelector open={true} onClose={() => {}} />)
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    const optionTexts = screen.getAllByRole('option').map((option) => option.textContent)
    expect(optionTexts.filter((text) => text?.includes('gpt-4o-mini')).length).toBeGreaterThan(0)
    expect(optionTexts.filter((text) => text?.includes('ollama/llama3')).length).toBeGreaterThan(0)
  })
})
