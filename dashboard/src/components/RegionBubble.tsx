import { useState, useRef, useCallback } from 'react'
import { useBrainStore } from '../stores/brainState'
import { REGION_CONFIG, REGION_DESCRIPTIONS } from '../constants/brainRegions'

// Persisted drag offsets per region (relative to projected position)
const savedOffsets: Record<string, { x: number; y: number }> = {}

export function RegionBubbles() {
  const regions = useBrainStore((s) => s.regions)
  const screenPositions = useBrainStore((s) => s.regionScreenPositions)

  return (
    <>
      {Object.keys(REGION_CONFIG).map((name) => {
        const region = regions[name]
        if (!region || region.level <= 0.1) return null
        const config = REGION_CONFIG[name]
        const descriptions = REGION_DESCRIPTIONS[name]
        const text = descriptions?.[region.mode] || descriptions?.active || ''
        if (!text || !config) return null
        const screenPos = screenPositions[name]
        if (!screenPos) return null
        return (
          <DraggableBubble
            key={name}
            id={name}
            color={config.color}
            name={name}
            text={text}
            baseX={screenPos.x}
            baseY={screenPos.y}
          />
        )
      })}
    </>
  )
}

function DraggableBubble({ id, color, name, text, baseX, baseY }: {
  id: string; color: string; name: string; text: string; baseX: number; baseY: number
}) {
  const [offset, setOffset] = useState(savedOffsets[id] || { x: 0, y: -30 })
  const dragging = useRef(false)
  const dragStart = useRef({ x: 0, y: 0 })
  const offsetStart = useRef({ x: 0, y: 0 })

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragging.current = true
    dragStart.current = { x: e.clientX, y: e.clientY }
    offsetStart.current = { ...offset }

    const onMouseMove = (me: MouseEvent) => {
      if (!dragging.current) return
      const newOffset = {
        x: offsetStart.current.x + (me.clientX - dragStart.current.x),
        y: offsetStart.current.y + (me.clientY - dragStart.current.y),
      }
      setOffset(newOffset)
      savedOffsets[id] = newOffset
    }

    const onMouseUp = () => {
      dragging.current = false
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
  }, [offset, id])

  return (
    <div
      className="region-bubble"
      style={{
        position: 'absolute',
        left: baseX + offset.x,
        top: baseY + offset.y,
        borderColor: `${color}40`,
        cursor: 'grab',
        zIndex: 15,
        transform: 'translateX(-50%)',
      }}
      onMouseDown={onMouseDown}
    >
      <div className="rb-name" style={{ color }}>{name.replace(/_/g, ' ')}</div>
      <div className="rb-text">{text}</div>
    </div>
  )
}
