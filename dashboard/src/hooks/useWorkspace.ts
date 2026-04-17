import { useEffect } from 'react'
import { useBrainStore } from '../stores/brainState'

export function useWorkspace() {
  const refresh = useBrainStore((s) => s.refreshWorkspaces)
  const current = useBrainStore((s) => s.currentWorkspace)
  const workspaces = useBrainStore((s) => s.workspaces)

  useEffect(() => { void refresh() }, [refresh])

  const setCurrent = async (workspaceId: string) => {
    await fetch('/api/workspaces/current', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: 'default', workspace_id: workspaceId }),
    })
    await refresh()
  }

  return { current, workspaces, setCurrent, refresh }
}
