import { useRef, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { useBrainStore } from '../stores/brainState'
import { BrainModel } from './BrainModel'
import { CurvedConnections } from './CurvedConnections'
import { REGION_CONFIG } from '../constants/brainRegions'
import * as THREE from 'three'

const DECAY_RATE = 0.12

// Region-specific shapes conforming to brain anatomy
const REGION_SHAPES: Record<string, {
  scaleXYZ: [number, number, number]
  rotation: [number, number, number]
}> = {
  prefrontal_cortex: { scaleXYZ: [6, 4, 3], rotation: [0.3, 0, 0] },
  acc: { scaleXYZ: [2, 4, 1.5], rotation: [0.2, 0, 0] },
  amygdala: { scaleXYZ: [2, 2, 2.5], rotation: [0, 0.2, 0] },
  basal_ganglia: { scaleXYZ: [3, 2.5, 3], rotation: [0, 0, 0] },
  cerebellum: { scaleXYZ: [6, 3, 4], rotation: [0, 0, 0] },
  thalamus: { scaleXYZ: [3, 2, 2.5], rotation: [0, 0, 0] },
  hypothalamus: { scaleXYZ: [2, 1.5, 2], rotation: [0, 0, 0] },
  hippocampus: { scaleXYZ: [2, 1.5, 5], rotation: [0.1, 0.4, 0.2] },
  salience_network: { scaleXYZ: [2.5, 3, 2], rotation: [0, -0.2, 0] },
  visual_cortex: { scaleXYZ: [5, 3, 3], rotation: [-0.2, 0, 0] },
  auditory_cortex_l: { scaleXYZ: [2, 2, 2.5], rotation: [0, 0.3, 0] },
  auditory_cortex_r: { scaleXYZ: [2, 2, 2.5], rotation: [0, -0.3, 0] },
  wernicke: { scaleXYZ: [2.5, 1.5, 2], rotation: [0, 0.2, 0.1] },
  broca: { scaleXYZ: [2, 2, 1.5], rotation: [0, 0.2, 0] },
  brainstem: { scaleXYZ: [2, 4, 2.5], rotation: [0, 0, 0] },
  vta: { scaleXYZ: [1.5, 1, 1.5], rotation: [0, 0, 0] },
  corpus_callosum: { scaleXYZ: [5, 1.5, 3], rotation: [0, 0, 0] },
  angular_gyrus: { scaleXYZ: [2, 1.5, 2], rotation: [0, 0.3, 0.1] },
}

/** Create brain-tissue-like geometry with wrinkles for a region */
function createRegionGeometry(name: string): THREE.BufferGeometry {
  const isCerebellum = name === 'cerebellum'
  const segments = 24
  const geo = new THREE.SphereGeometry(1, segments, segments)
  const pos = geo.attributes.position
  const normal = geo.attributes.normal

  // Seed different wrinkle patterns per region
  const seed = Array.from(name).reduce((a, c) => a + c.charCodeAt(0), 0)

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i)
    const y = pos.getY(i)
    const z = pos.getZ(i)
    const nx = normal.getX(i)
    const ny = normal.getY(i)
    const nz = normal.getZ(i)

    let bump: number
    if (isCerebellum) {
      // Cerebellar folia — horizontal ridges (amplified)
      bump = Math.sin(y * 20) * 0.12 + Math.sin(y * 35 + x * 5) * 0.06
    } else {
      // Cortical sulci/gyri — brain wrinkle pattern (amplified)
      const f1 = Math.sin(x * 14 + seed * 0.1) * Math.cos(z * 11 + seed * 0.2) * 0.14
      const f2 = Math.sin(y * 18 + x * 7 + seed * 0.3) * 0.09
      const f3 = Math.cos(z * 22 + y * 13 + seed * 0.15) * 0.06
      bump = f1 + f2 + f3
    }

    pos.setXYZ(i, x + nx * bump, y + ny * bump, z + nz * bump)
  }

  geo.computeVertexNormals()
  return geo
}

function RegionNode({ name, config }: { name: string; config: typeof REGION_CONFIG[string] }) {
  const meshRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)
  const region = useBrainStore((s) => s.regions[name])
  const level = region?.level ?? 0
  const shape = REGION_SHAPES[name] || { scaleXYZ: [2, 2, 2], rotation: [0, 0, 0] }
  const pulsePhase = useRef(0)

  // Wrinkled geometry per region (memoized)
  const regionGeo = useMemo(() => createRegionGeometry(name), [name])

  const decayTimer = useRef(0)

  useFrame((_, delta) => {
    // Decay activation (throttled to avoid excessive store updates)
    decayTimer.current += delta
    if (level > 0.01 && decayTimer.current > 0.15) {
      decayTimer.current = 0
      const decayed = Math.max(0, level - DECAY_RATE * 0.15)
      if (Math.abs(decayed - level) > 0.01) {
        useBrainStore.getState().setRegionActivation(name, decayed, decayed > 0.1 ? 'active' : 'inactive')
      }
    }

    // Heartbeat pulse when active
    pulsePhase.current += delta * (3 + level * 8) // faster when more active
    // Double-beat pattern: quick thump-thump then pause
    const t = pulsePhase.current % (Math.PI * 2)
    const beat1 = Math.max(0, Math.sin(t * 2)) * 0.5
    const beat2 = Math.max(0, Math.sin(t * 2 - 1.2)) * 0.3
    const pulse = level > 0.05 ? 1.0 + (beat1 + beat2) * level * 0.12 : 1.0

    if (meshRef.current) {
      meshRef.current.scale.lerp(
        new THREE.Vector3(
          shape.scaleXYZ[0] * pulse,
          shape.scaleXYZ[1] * pulse,
          shape.scaleXYZ[2] * pulse,
        ),
        delta * 8, // faster lerp for snappy pulse
      )
    }
    if (glowRef.current) {
      const mat = glowRef.current.material as THREE.MeshBasicMaterial
      mat.opacity = level * 0.3 * pulse
    }
  })

  const isActive = level > 0.05
  const color = new THREE.Color(config.color)
  // Dim version of region color for inactive state
  const dimColor = new THREE.Color(config.color).multiplyScalar(0.25)

  // Inactive: dimmed region color (distinguishable). Active: full color.
  const matColor = isActive ? config.color : dimColor
  const matEmissive = isActive ? color : dimColor
  const emissiveIntensity = isActive ? 0.5 + level * 2.5 : 0.2
  const materialOpacity = isActive ? 0.85 : 0.3

  return (
    <group position={config.position} rotation={shape.rotation}>
      {/* Soft glow cloud — only when active */}
      <mesh ref={glowRef} scale={shape.scaleXYZ.map(s => s * 1.6) as unknown as THREE.Vector3}>
        <sphereGeometry args={[1, 10, 10]} />
        <meshBasicMaterial color={config.color} transparent opacity={0} depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
      {/* Brain-tissue region with wrinkles — clickable */}
      <mesh
        ref={meshRef}
        geometry={regionGeo}
        scale={shape.scaleXYZ}
        onClick={(e) => {
          e.stopPropagation()
          const store = useBrainStore.getState()
          store.setSelectedRegion(store.selectedRegion === name ? null : name)
        }}
        onPointerOver={() => { document.body.style.cursor = 'pointer' }}
        onPointerOut={() => { document.body.style.cursor = 'auto' }}
      >
        <meshStandardMaterial
          color={matColor}
          emissive={matEmissive}
          emissiveIntensity={emissiveIntensity}
          transparent
          opacity={materialOpacity}
          roughness={0.55}
          metalness={0.1}
          depthWrite={false}
        />
      </mesh>
    </group>
  )
}

/** Global brain heartbeat — subtle whole-brain pulse when any region is active */
function BrainPulse({ children }: { children: React.ReactNode }) {
  const groupRef = useRef<THREE.Group>(null)
  const regions = useBrainStore((s) => s.regions)
  const phaseRef = useRef(0)

  useFrame((_, delta) => {
    if (!groupRef.current) return

    // Average activation across all regions
    const values = Object.values(regions)
    const avgLevel = values.reduce((sum, r) => sum + r.level, 0) / values.length

    phaseRef.current += delta * 1.5
    const t = phaseRef.current % (Math.PI * 2)
    // Gentle breathing pulse
    const breathe = 1.0 + Math.sin(t) * avgLevel * 0.015

    groupRef.current.scale.setScalar(breathe)
  })

  return <group ref={groupRef}>{children}</group>
}

/** Projects all region 3D positions to 2D screen coordinates,
 *  accounting for the shared rotation group's transform. */
function RegionProjector({ rotGroupRef }: { rotGroupRef: React.RefObject<THREE.Group | null> }) {
  const { camera, size } = useThree()
  const timerRef = useRef(0)

  useFrame((_, delta) => {
    timerRef.current += delta
    if (timerRef.current < 0.1) return // throttle to ~10fps
    timerRef.current = 0

    const positions: Record<string, { x: number; y: number }> = {}
    const vec = new THREE.Vector3()
    for (const [name, config] of Object.entries(REGION_CONFIG)) {
      vec.set(...config.position)
      // Apply the rotation group's world matrix so 2D bubbles track the 3D rotation
      if (rotGroupRef.current) {
        vec.applyMatrix4(rotGroupRef.current.matrixWorld)
      }
      vec.project(camera)
      positions[name] = {
        x: (vec.x * 0.5 + 0.5) * size.width,
        y: (-vec.y * 0.5 + 0.5) * size.height + 48, // +48 for navbar
      }
    }
    useBrainStore.getState().setRegionScreenPositions(positions)
  })

  return null
}

export function BrainScene() {
  const rotGroupRef = useRef<THREE.Group>(null)

  useFrame((_, delta) => {
    if (rotGroupRef.current) {
      rotGroupRef.current.rotation.y += delta * 0.03
    }
  })

  return (
    <BrainPulse>
      <group ref={rotGroupRef}>
        <BrainModel />
        <CurvedConnections />
        {Object.entries(REGION_CONFIG).map(([name, config]) => (
          <RegionNode key={name} name={name} config={config} />
        ))}
      </group>
      <RegionProjector rotGroupRef={rotGroupRef} />
    </BrainPulse>
  )
}
