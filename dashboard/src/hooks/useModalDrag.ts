import type { CSSProperties } from 'react'
import { useDraggable } from './useDraggable'

/**
 * Wrapper around useDraggable that returns the style object and drag handler
 * needed for a .kl-modal. The modal root gets { style, onMouseDown }; the
 * header gets data-drag-handle so dragging starts only from the header.
 */
export function useModalDrag(id: string, defaultX: number, defaultY: number) {
  const { pos, onMouseDown } = useDraggable(id, defaultX, defaultY)
  const style: CSSProperties = { left: pos.x, top: pos.y }
  return { style, onMouseDown }
}
