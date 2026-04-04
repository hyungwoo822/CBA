import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import { useBrainStore } from '../stores/brainState'
import { POSITIONS } from '../constants/brainRegions'
import * as THREE from 'three'

const CONNECTIONS: [string, string][] = [
  // Pipeline path
  ['thalamus', 'amygdala'],
  ['amygdala', 'salience_network'],
  ['salience_network', 'prefrontal_cortex'],
  ['prefrontal_cortex', 'acc'],
  ['acc', 'basal_ganglia'],
  ['basal_ganglia', 'cerebellum'],
  ['cerebellum', 'hippocampus'],
  // Anatomical connections
  ['thalamus', 'hippocampus'],
  ['thalamus', 'cerebellum'],
  ['prefrontal_cortex', 'hypothalamus'],
  ['hippocampus', 'prefrontal_cortex'],
  ['salience_network', 'amygdala'],
  ['amygdala', 'hippocampus'],
  ['thalamus', 'prefrontal_cortex'],
  ['hypothalamus', 'thalamus'],
  // Visual pathway
  ['visual_cortex', 'thalamus'],
  ['visual_cortex', 'hippocampus'],
  // Auditory pathway
  ['auditory_cortex_l', 'wernicke'],
  ['auditory_cortex_r', 'amygdala'],
  // Language circuit
  ['wernicke', 'prefrontal_cortex'],
  ['prefrontal_cortex', 'broca'],
  ['wernicke', 'broca'],
  // Subcortical
  ['brainstem', 'thalamus'],
  ['vta', 'basal_ganglia'],
  ['vta', 'prefrontal_cortex'],
  ['brainstem', 'hypothalamus'],
]

export function computeCurvePoints(
  from: [number, number, number],
  to: [number, number, number],
  segments: number = 32,
): [number, number, number][] {
  const fromV = new THREE.Vector3(...from)
  const toV = new THREE.Vector3(...to)
  const mid = new THREE.Vector3().addVectors(fromV, toV).multiplyScalar(0.5)

  const dir = new THREE.Vector3().subVectors(toV, fromV)
  const up = new THREE.Vector3(0, 1, 0)
  const perp = new THREE.Vector3().crossVectors(dir, up).normalize()
  const offset = dir.length() * 0.3
  mid.add(perp.multiplyScalar(offset))

  const curve = new THREE.CatmullRomCurve3([fromV, mid, toV])
  return curve.getPoints(segments).map(p => [p.x, p.y, p.z] as [number, number, number])
}

/** A single connection line that pulses when a signal flows through it */
function ConnectionLine({ from, to, points }: { from: string; to: string; points: [number, number, number][] }) {
  const regions = useBrainStore((s) => s.regions)
  const signalFlows = useBrainStore((s) => s.signalFlows)
  const lineRef = useRef<any>(null)
  const pulseRef = useRef(0)
  const activeRef = useRef(false)

  const fromLevel = regions[from]?.level ?? 0
  const toLevel = regions[to]?.level ?? 0
  const bothActive = fromLevel > 0.1 && toLevel > 0.1

  // Check if there's a recent signal flow on this connection
  const hasSignal = signalFlows.some(
    (sf) => (sf.source === from && sf.target === to) || (sf.source === to && sf.target === from)
  )

  useFrame((_, delta) => {
    if (hasSignal || bothActive) {
      pulseRef.current = Math.min(1, pulseRef.current + delta * 3)
      activeRef.current = true
    } else {
      pulseRef.current = Math.max(0, pulseRef.current - delta * 0.8)
      if (pulseRef.current <= 0) activeRef.current = false
    }
  })

  const intensity = pulseRef.current
  const color = bothActive ? '#4ade80' : hasSignal ? '#60a5fa' : '#1e293b'
  const opacity = 0.06 + intensity * 0.5
  const lineWidth = 0.5 + intensity * 1.5

  return (
    <Line
      ref={lineRef}
      points={points}
      color={color}
      transparent
      opacity={opacity}
      lineWidth={lineWidth}
    />
  )
}

export function CurvedConnections() {
  const curves = useMemo(() => {
    return CONNECTIONS.map(([from, to]) => {
      const fromPos = POSITIONS[from]
      const toPos = POSITIONS[to]
      if (!fromPos || !toPos) return null
      return { from, to, points: computeCurvePoints(fromPos, toPos) }
    }).filter(Boolean) as { from: string; to: string; points: [number, number, number][] }[]
  }, [])

  return (
    <group>
      {curves.map(({ from, to, points }) => (
        <ConnectionLine key={`${from}-${to}`} from={from} to={to} points={points} />
      ))}
    </group>
  )
}

export { CONNECTIONS }
