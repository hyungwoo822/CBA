import { useState, useEffect, useRef } from 'react'
import { useBrainStore } from '../stores/brainState'

const STAGES = [
  { key: 'working' as const, label: 'Working', cls: 'mem-working', tab: 'Working', desc: 'Active processing slots — phonological(4) + visuospatial(3) + episodic_buffer(4) = 11 (Baddeley 2000, Cowan 2001)' },
  { key: 'staging' as const, label: 'Staging', cls: 'mem-staging', tab: 'Staging', desc: 'Hippocampal fast encoding — awaiting consolidation' },
  { key: 'episodic' as const, label: 'Episodic', cls: 'mem-episodic', tab: 'Episodic', desc: 'Episodic memory — temporal event records with forgetting curves' },
  { key: 'semantic' as const, label: 'Semantic', cls: 'mem-semantic', tab: 'Semantic', desc: 'Long-term knowledge — facts, relations, concepts' },
  { key: 'procedural' as const, label: 'Procedural', cls: 'mem-procedural', tab: 'Procedural', desc: 'Learned action sequences — BG + cerebellum (Fitts 1967)' },
]

export function MemoryFlowBar() {
  const mf = useBrainStore((s) => s.memoryFlow)
  const [sensoryFlash, setSensoryFlash] = useState(false)
  const prevSensory = useRef(mf.sensory)

  // Flash sensory modal when sensory count changes
  useEffect(() => {
    if (mf.sensory !== prevSensory.current && mf.sensory > 0) {
      setSensoryFlash(true)
      const timer = setTimeout(() => setSensoryFlash(false), 2000)
      prevSensory.current = mf.sensory
      return () => clearTimeout(timer)
    }
    prevSensory.current = mf.sensory
  }, [mf.sensory])

  const handleClick = (tab: string) => {
    ;(window as any).__memoryPanelSetTab?.(tab)
  }

  return (
    <div className="memory-bar-wrap">
      {/* Sensory flash modal */}
      {sensoryFlash && (
        <div className="sensory-flash">
          <span className="sensory-flash-dot" />
          <span>Sensory input received ({mf.sensory})</span>
        </div>
      )}
      <div className="memory-bar">
        {STAGES.flatMap((stage, i) => [
          i > 0 && <span key={`arrow-${i}`} className="mem-arrow">›</span>,
          <span
            key={stage.key}
            className={`mem-stage ${stage.cls}`}
            title={stage.desc}
            onClick={() => handleClick(stage.tab)}
            style={{ cursor: 'pointer' }}
          >
            <span className="mem-dot" />
            <span className="mem-label">{stage.label}</span>
            <span className="mem-count">
              {stage.key === 'working' ? `${mf[stage.key]}/11` : (mf as any)[stage.key] ?? 0}
            </span>
          </span>,
        ])}
      </div>
    </div>
  )
}
