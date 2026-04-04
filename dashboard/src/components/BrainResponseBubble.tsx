import { useEffect, useState, useRef, useCallback } from 'react'
import { useBrainStore } from '../stores/brainState'

// Persist position across responses
let savedPos: { x: number; y: number } | null = null

export function BrainResponseBubble() {
  const lastResponse = useBrainStore((s) => s.lastResponse)
  const responseTimestamp = useBrainStore((s) => s.responseTimestamp)
  const [visible, setVisible] = useState(false)
  const [fading, setFading] = useState(false)
  const [pos, setPos] = useState(savedPos || { x: window.innerWidth * 0.22, y: window.innerHeight * 0.15 })
  const dragging = useRef(false)
  const dragStart = useRef({ x: 0, y: 0 })
  const posStart = useRef({ x: 0, y: 0 })

  useEffect(() => {
    if (!lastResponse) return
    setVisible(true)
    setFading(false)
    // Use saved position if available
    if (savedPos) setPos(savedPos)

    const fadeTimer = setTimeout(() => setFading(true), 8000)
    const hideTimer = setTimeout(() => setVisible(false), 8500)

    return () => {
      clearTimeout(fadeTimer)
      clearTimeout(hideTimer)
    }
  }, [lastResponse, responseTimestamp])

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragging.current = true
    dragStart.current = { x: e.clientX, y: e.clientY }
    posStart.current = { ...pos }

    const onMouseMove = (me: MouseEvent) => {
      if (!dragging.current) return
      const newPos = {
        x: posStart.current.x + (me.clientX - dragStart.current.x),
        y: posStart.current.y + (me.clientY - dragStart.current.y),
      }
      setPos(newPos)
      savedPos = newPos
    }

    const onMouseUp = () => {
      dragging.current = false
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
  }, [pos])

  if (!visible || !lastResponse) return null

  return (
    <div
      className={`brain-response-bubble${fading ? ' fading' : ''}`}
      style={{
        position: 'absolute',
        left: pos.x,
        top: pos.y,
        cursor: 'grab',
        zIndex: 15,
      }}
      onMouseDown={onMouseDown}
    >
      <div className="br-indicator">
        <div className="br-dot" />
        <span className="br-label">Brain</span>
      </div>
      <div className="br-text">{lastResponse}</div>
    </div>
  )
}
