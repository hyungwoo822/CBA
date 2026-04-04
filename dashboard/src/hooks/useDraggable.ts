import { useState, useRef, useCallback, useEffect } from 'react'

const savedPositions: Record<string, { x: number; y: number }> = {}
let resetCounter = 0
const resetListeners: (() => void)[] = []

/** Call this to reset ALL draggable panels to their default positions */
export function resetAllPositions() {
  for (const key of Object.keys(savedPositions)) {
    delete savedPositions[key]
  }
  resetCounter++
  resetListeners.forEach((fn) => fn())
}

export function useDraggable(id: string, defaultX: number, defaultY: number) {
  const [pos, setPos] = useState(savedPositions[id] || { x: defaultX, y: defaultY })
  const dragging = useRef(false)
  const dragStart = useRef({ x: 0, y: 0 })
  const posStart = useRef({ x: 0, y: 0 })

  // Listen for global reset
  useEffect(() => {
    const listener = () => setPos({ x: defaultX, y: defaultY })
    resetListeners.push(listener)
    return () => {
      const idx = resetListeners.indexOf(listener)
      if (idx >= 0) resetListeners.splice(idx, 1)
    }
  }, [defaultX, defaultY])

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (!(e.target as HTMLElement).closest('[data-drag-handle]')) return
    e.preventDefault()
    dragging.current = true
    dragStart.current = { x: e.clientX, y: e.clientY }
    posStart.current = { ...pos }

    const onMouseMove = (me: MouseEvent) => {
      if (!dragging.current) return
      const newPos = {
        x: Math.max(0, Math.min(window.innerWidth - 100, posStart.current.x + (me.clientX - dragStart.current.x))),
        y: Math.max(0, Math.min(window.innerHeight - 50, posStart.current.y + (me.clientY - dragStart.current.y))),
      }
      setPos(newPos)
      savedPositions[id] = newPos
    }
    const onMouseUp = () => {
      dragging.current = false
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
  }, [pos, id])

  return { pos, onMouseDown }
}
