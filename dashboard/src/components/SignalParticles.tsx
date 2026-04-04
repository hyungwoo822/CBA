import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useBrainStore } from '../stores/brainState'
import type { Particle } from '../stores/brainState'
import { POSITIONS, INPUT_POSITION } from '../constants/brainRegions'
import { computeCurvePoints } from './CurvedConnections'

const MAX_PARTICLES = 150
const tempObject = new THREE.Object3D()
const tempColor = new THREE.Color()

function getPosition(name: string): [number, number, number] {
  if (name === '_input') return INPUT_POSITION
  return POSITIONS[name] || [0, 0, 0]
}

const curveCache = new Map<string, THREE.CatmullRomCurve3>()

function getCurve(source: string, target: string): THREE.CatmullRomCurve3 {
  const key = `${source}->${target}`
  if (!curveCache.has(key)) {
    const from = getPosition(source)
    const to = getPosition(target)
    const points = computeCurvePoints(from, to, 32)
    curveCache.set(key, new THREE.CatmullRomCurve3(
      points.map(p => new THREE.Vector3(...p))
    ))
  }
  return curveCache.get(key)!
}

export function SignalParticles() {
  const meshRef = useRef<THREE.InstancedMesh>(null)
  const particlesRef = useRef<Particle[]>([])

  const storeParticles = useBrainStore((s) => s.particles)
  if (storeParticles !== particlesRef.current) {
    particlesRef.current = storeParticles
  }

  useFrame((_, delta) => {
    if (!meshRef.current) return
    const particles = particlesRef.current
    if (particles.length === 0) {
      meshRef.current.count = 0
      return
    }

    const updated: Particle[] = []
    let idx = 0
    const store = useBrainStore.getState()

    for (const p of particles) {
      if (p.delay > 0) {
        updated.push({ ...p, delay: p.delay - delta })
        continue
      }

      const newProgress = p.progress + delta * p.speed
      if (newProgress >= 1) {
        const region = store.regions[p.target]
        if (region) {
          store.setRegionActivation(p.target, Math.min(1, region.level + 0.15), 'high_activity')
        }
        continue
      }

      const curve = getCurve(p.source, p.target)
      const point = curve.getPointAt(Math.min(newProgress, 1))
      tempObject.position.copy(point)
      const scale = 0.5 + (1 - Math.abs(newProgress - 0.5) * 2) * 0.3
      tempObject.scale.setScalar(scale)
      tempObject.updateMatrix()
      meshRef.current.setMatrixAt(idx, tempObject.matrix)

      tempColor.set(p.color)
      meshRef.current.setColorAt(idx, tempColor)

      updated.push({ ...p, progress: newProgress })
      idx++
    }

    meshRef.current.count = idx
    meshRef.current.instanceMatrix.needsUpdate = true
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true
    }

    particlesRef.current = updated
    if (updated.length !== particles.length) {
      store.setParticles(updated)
    }
  })

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, MAX_PARTICLES]}>
      <sphereGeometry args={[0.5, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  )
}
