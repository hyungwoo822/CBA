import { useMemo } from 'react'
import * as THREE from 'three'

/**
 * Procedural anatomical brain model.
 * Translucent beige shell with visible sulci/gyri wrinkles.
 */

// Create brain-like bumpy sphere geometry with prominent wrinkles
function createHemisphereGeometry(): THREE.SphereGeometry {
  const geo = new THREE.SphereGeometry(1, 48, 48)
  const pos = geo.attributes.position
  const normal = geo.attributes.normal

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i)
    const y = pos.getY(i)
    const z = pos.getZ(i)

    // Sulci/gyri wrinkles — strongly amplified
    const freq1 = Math.sin(x * 12 + y * 8) * Math.cos(z * 10 + x * 6) * 0.12
    const freq2 = Math.sin(y * 15 + z * 12) * Math.cos(x * 9) * 0.08
    const freq3 = Math.sin(x * 20 + z * 18) * 0.06

    const medialFlatten = x > -0.1 ? Math.max(0, (x + 0.1) * 0.15) : 0

    const bump = freq1 + freq2 + freq3
    const nx = normal.getX(i)
    const ny = normal.getY(i)
    const nz = normal.getZ(i)

    pos.setXYZ(i, x + nx * bump - medialFlatten, y + ny * bump, z + nz * bump)
  }

  geo.computeVertexNormals()
  return geo
}

function createCerebellumGeometry(): THREE.SphereGeometry {
  const geo = new THREE.SphereGeometry(1, 32, 32)
  const pos = geo.attributes.position
  const normal = geo.attributes.normal

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i)
    const y = pos.getY(i)
    const z = pos.getZ(i)

    const folia = Math.sin(y * 25) * 0.09
    const nx = normal.getX(i)
    const ny = normal.getY(i)
    const nz = normal.getZ(i)

    pos.setXYZ(i, x + nx * folia, y + ny * folia, z + nz * folia)
  }

  geo.computeVertexNormals()
  return geo
}

export function BrainModel() {
  const leftHemiGeo = useMemo(() => createHemisphereGeometry(), [])
  const rightHemiGeo = useMemo(() => createHemisphereGeometry(), [])
  const cerebellumGeo = useMemo(() => createCerebellumGeometry(), [])

  // Beige/cream brain tissue color
  const brainColor = '#c8b89a'
  const brainEmissive = '#3a3020'

  return (
    <group>
      {/* Left hemisphere */}
      <mesh geometry={leftHemiGeo} position={[-5.5, 4, 1]} scale={[9, 13, 16]} rotation={[0.1, 0, -0.05]}>
        <meshStandardMaterial
          color={brainColor}
          emissive={brainEmissive}
          emissiveIntensity={0.3}
          transparent
          opacity={0.28}
          roughness={0.6}
          metalness={0.05}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* Right hemisphere */}
      <mesh geometry={rightHemiGeo} position={[5.5, 4, 1]} scale={[9, 13, 16]} rotation={[0.1, 0, 0.05]}>
        <meshStandardMaterial
          color={brainColor}
          emissive={brainEmissive}
          emissiveIntensity={0.3}
          transparent
          opacity={0.28}
          roughness={0.6}
          metalness={0.05}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* Longitudinal fissure */}
      <mesh position={[0, 8, 1]} rotation={[Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.5, 30]} />
        <meshBasicMaterial color="#040408" transparent opacity={0.5} side={THREE.DoubleSide} />
      </mesh>

      {/* Cerebellum */}
      <mesh geometry={cerebellumGeo} position={[0, -10, -8]} scale={[8, 5.5, 6]}>
        <meshStandardMaterial
          color={brainColor}
          emissive={brainEmissive}
          emissiveIntensity={0.25}
          transparent
          opacity={0.25}
          roughness={0.6}
          metalness={0.05}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* Brainstem */}
      <mesh position={[0, -16, -4]} rotation={[0.3, 0, 0]}>
        <cylinderGeometry args={[1.8, 1.2, 8, 16]} />
        <meshStandardMaterial
          color={brainColor}
          emissive={brainEmissive}
          emissiveIntensity={0.2}
          transparent
          opacity={0.22}
          roughness={0.6}
          metalness={0.05}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* Temporal lobes */}
      <mesh position={[-10, -3, 6]} scale={[5, 5, 8]} rotation={[0.2, 0.3, 0]}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshStandardMaterial
          color={brainColor}
          emissive={brainEmissive}
          emissiveIntensity={0.2}
          transparent
          opacity={0.22}
          roughness={0.6}
          metalness={0.05}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <mesh position={[10, -3, 6]} scale={[5, 5, 8]} rotation={[0.2, -0.3, 0]}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshStandardMaterial
          color={brainColor}
          emissive={brainEmissive}
          emissiveIntensity={0.2}
          transparent
          opacity={0.22}
          roughness={0.6}
          metalness={0.05}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  )
}
