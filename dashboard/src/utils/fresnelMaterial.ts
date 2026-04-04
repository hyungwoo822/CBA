import * as THREE from 'three'

/** Inject Fresnel rim-light into a meshStandardMaterial via onBeforeCompile */
export function applyFresnelRimLight(
  material: THREE.MeshStandardMaterial,
  rimColor = '#4a6090',
  rimPower = 2.0,
  rimStrength = 0.4
) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.rimColor = { value: new THREE.Color(rimColor) }
    shader.uniforms.rimPower = { value: rimPower }
    shader.uniforms.rimStrength = { value: rimStrength }

    shader.vertexShader = shader.vertexShader.replace(
      '#include <common>',
      `#include <common>
      varying vec3 vViewNormal;
      varying vec3 vViewDir;`
    )
    shader.vertexShader = shader.vertexShader.replace(
      '#include <worldpos_vertex>',
      `#include <worldpos_vertex>
      vViewNormal = normalize((modelMatrix * vec4(transformedNormal, 0.0)).xyz);
      vViewDir = normalize(cameraPosition - worldPosition.xyz);`
    )

    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <common>',
      `#include <common>
      uniform vec3 rimColor;
      uniform float rimPower;
      uniform float rimStrength;
      varying vec3 vViewNormal;
      varying vec3 vViewDir;`
    )
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <dithering_fragment>',
      `#include <dithering_fragment>
      float rimFactor = 1.0 - max(0.0, dot(vViewDir, vViewNormal));
      rimFactor = pow(rimFactor, rimPower) * rimStrength;
      gl_FragColor.rgb += rimColor * rimFactor;`
    )
  }
}
