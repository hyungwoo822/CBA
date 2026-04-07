# Dashboard UI Upgrade Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade brain dashboard with always-visible anatomical brain, unified input bar with audio orb, brain speech bubbles, region mini bubbles, and glassmorphism panel restyling.

**Architecture:** Modify existing React + Three.js + Zustand dashboard. Add Fresnel rim light shader to brain materials for wrinkle visibility. New DOM components for unified input bar, audio orb, and response bubbles. Three.js `Html` overlays for region mini bubbles. All animations via CSS transitions/keyframes.

**Tech Stack:** React 19, Three.js, @react-three/fiber, @react-three/drei (Html), Zustand, CSS transitions/keyframes

**Spec:** `docs/superpowers/specs/2026-03-17-dashboard-ui-upgrade-design.md`

---

## Chunk 1: Foundation — State, Fonts, CSS Variables

### Task 1: Add Google Fonts and CSS variables

**Files:**
- Modify: `dashboard/index.html`
- Modify: `dashboard/src/App.css`

- [ ] **Step 1: Add Google Fonts CDN to index.html**

In `dashboard/index.html`, add inside `<head>` before `</head>`:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400&display=swap" rel="stylesheet">
```

- [ ] **Step 2: Update CSS variables and add glassmorphism utilities**

Replace the `:root` block and `body` rule in `dashboard/src/App.css`:

```css
:root {
  --bg-primary: #050510;
  --bg-secondary: #0f1117;
  --bg-panel: rgba(8, 8, 20, 0.65);
  --bg-input: rgba(10, 10, 28, 0.8);
  --border: rgba(255, 255, 255, 0.04);
  --border-focus: rgba(59, 130, 246, 0.3);
  --text-primary: #e2e8f0;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --blue: #3b82f6;
  --green: #4ade80;
  --red: #ef4444;
  --orange: #f97316;
  --yellow: #fbbf24;
  --purple: #8b5cf6;
  --cyan: #06b6d4;
  --pink: #f43f5e;
  --panel-blur: blur(20px);
  --panel-radius: 16px;
  --panel-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  --font-ui: 'Inter', -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-ui);
  overflow: hidden;
}
```

- [ ] **Step 3: Add keyframe animations to App.css**

Append to `dashboard/src/App.css`:

```css
/* === Animations === */
@keyframes fadeIn {
  from { opacity: 0; transform: translateX(8px); }
  to { opacity: 1; transform: translateX(0); }
}
@keyframes fadeOut {
  from { opacity: 1; }
  to { opacity: 0; }
}
@keyframes pulseDot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
@keyframes waveform1 { 0%,100%{height:8px} 50%{height:20px} }
@keyframes waveform2 { 0%,100%{height:14px} 50%{height:6px} }
@keyframes waveform3 { 0%,100%{height:10px} 50%{height:24px} }
@keyframes waveform4 { 0%,100%{height:18px} 50%{height:8px} }
@keyframes waveform5 { 0%,100%{height:6px} 50%{height:16px} }
@keyframes waveform6 { 0%,100%{height:12px} 50%{height:22px} }
@keyframes waveform7 { 0%,100%{height:16px} 50%{height:10px} }
```

- [ ] **Step 4: Verify dev server runs**

Run: `cd dashboard && npm run dev`
Expected: Compiles without errors, page loads with new fonts applied.

- [ ] **Step 5: Commit**

```bash
git add dashboard/index.html dashboard/src/App.css
git commit -m "feat(dashboard): add Google Fonts and glassmorphism CSS variables"
```

---

### Task 2: Extend Zustand store with new state fields

**Files:**
- Modify: `dashboard/src/stores/brainState.ts`

- [ ] **Step 1: Add new fields to BrainState interface**

After the `particles: Particle[]` line (line 43), add:

```typescript
  lastResponse: string | null
  responseTimestamp: number
  isAudioMode: boolean
  audioState: 'idle' | 'listening' | 'processing' | 'done'
```

Note: Region descriptions use the static `REGION_DESCRIPTIONS` mapping from `brainRegions.ts` — no store field needed.

- [ ] **Step 2: Add new actions to BrainState interface**

After the `setParticles` action (line 52), add:

```typescript
  setLastResponse: (response: string) => void
  setAudioMode: (active: boolean) => void
  setAudioState: (state: BrainState['audioState']) => void
```

- [ ] **Step 3: Add initial values and action implementations**

After `particles: [],` (line 73), add initial values:

```typescript
  lastResponse: null,
  responseTimestamp: 0,
  isAudioMode: false,
  audioState: 'idle' as const,
```

After `setParticles` implementation (line 119), add:

```typescript
  setLastResponse: (response) => set({ lastResponse: response, responseTimestamp: Date.now() }),
  setAudioMode: (active) => set({ isAudioMode: active }),
  setAudioState: (audioState) => set({ audioState }),
```

- [ ] **Step 4: Verify build**

Run: `cd dashboard && npx tsc --noEmit`
Expected: No TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/stores/brainState.ts
git commit -m "feat(dashboard): extend Zustand store with response, audio, and region description state"
```

---

### Task 3: Add region description mapping to constants

**Files:**
- Modify: `dashboard/src/constants/brainRegions.ts`

- [ ] **Step 1: Add description mapping**

Append to `dashboard/src/constants/brainRegions.ts`:

```typescript
export const REGION_DESCRIPTIONS: Record<string, Record<string, string>> = {
  prefrontal_cortex: { active: '계획 수립 중', high_activity: '집중 분석 중' },
  acc: { active: '갈등 모니터링', high_activity: '오류 감지 중' },
  amygdala: { active: '감정 평가 중', high_activity: '위험 감지!' },
  basal_ganglia: { active: '행동 선택 중', high_activity: '행동 실행 중' },
  cerebellum: { active: '타이밍 조정 중', high_activity: '패턴 학습 중' },
  thalamus: { active: '신호 라우팅', high_activity: '입력 게이팅 중' },
  hypothalamus: { active: '항상성 조절', high_activity: '긴급 조절 중' },
  hippocampus: { active: '기억 인코딩', high_activity: '기억 통합 중' },
  salience_network: { active: '핵심 필터링', high_activity: '주의 전환 중' },
}
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/src/constants/brainRegions.ts
git commit -m "feat(dashboard): add region description mapping table"
```

---

## Chunk 2: Brain Visibility — Fresnel, Opacity, Always-visible Regions

### Task 4: Improve BrainModel shell visibility with Fresnel rim light

**Files:**
- Modify: `dashboard/src/components/BrainModel.tsx`

- [ ] **Step 1: Create shared Fresnel material utility**

Create `dashboard/src/utils/fresnelMaterial.ts` (create the `utils/` directory first):

```typescript
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
```

- [ ] **Step 2: Update BRAIN_MAT_PROPS for higher visibility**

Replace `BRAIN_MAT_PROPS`:

```typescript
const BRAIN_MAT_PROPS = {
  color: '#1e1e3e',
  transparent: true,
  opacity: 0.22,
  roughness: 0.6,
  metalness: 0.1,
  side: THREE.DoubleSide,
  depthWrite: false,
} as const
```

- [ ] **Step 3: Apply Fresnel to each mesh material using ref callbacks**

Replace the `BrainModel` component. Import from the shared utility:

```tsx
import { applyFresnelRimLight } from '../utils/fresnelMaterial'
```

Replace the entire `BrainModel` function:

```tsx
export function BrainModel() {
  const groupRef = useRef<THREE.Group>(null)

  const leftHemiGeo = useMemo(() => createHemisphereGeometry(), [])
  const rightHemiGeo = useMemo(() => createHemisphereGeometry(), [])
  const cerebellumGeo = useMemo(() => createCerebellumGeometry(), [])

  const matRef = (mat: THREE.MeshStandardMaterial | null) => {
    if (mat && !mat.userData._fresnelApplied) {
      applyFresnelRimLight(mat)
      mat.userData._fresnelApplied = true
      mat.needsUpdate = true
    }
  }

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.03
    }
  })

  return (
    <group ref={groupRef}>
      <mesh geometry={leftHemiGeo} position={[-5.5, 4, 1]} scale={[9, 13, 16]} rotation={[0.1, 0, -0.05]}>
        <meshStandardMaterial ref={matRef} {...BRAIN_MAT_PROPS} />
      </mesh>
      <mesh geometry={rightHemiGeo} position={[5.5, 4, 1]} scale={[9, 13, 16]} rotation={[0.1, 0, 0.05]}>
        <meshStandardMaterial ref={matRef} {...BRAIN_MAT_PROPS} />
      </mesh>
      <mesh position={[0, 8, 1]} rotation={[Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.5, 30]} />
        <meshBasicMaterial color="#060612" transparent opacity={0.4} side={THREE.DoubleSide} />
      </mesh>
      <mesh geometry={cerebellumGeo} position={[0, -10, -8]} scale={[8, 5.5, 6]}>
        <meshStandardMaterial ref={matRef} {...BRAIN_MAT_PROPS} opacity={0.24} />
      </mesh>
      <mesh position={[0, -16, -4]} rotation={[0.3, 0, 0]}>
        <cylinderGeometry args={[1.8, 1.2, 8, 16]} />
        <meshStandardMaterial ref={matRef} {...BRAIN_MAT_PROPS} opacity={0.18} />
      </mesh>
      <mesh position={[-10, -3, 6]} scale={[5, 5, 8]} rotation={[0.2, 0.3, 0]}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshStandardMaterial ref={matRef} {...BRAIN_MAT_PROPS} opacity={0.18} />
      </mesh>
      <mesh position={[10, -3, 6]} scale={[5, 5, 8]} rotation={[0.2, -0.3, 0]}>
        <sphereGeometry args={[1, 24, 24]} />
        <meshStandardMaterial ref={matRef} {...BRAIN_MAT_PROPS} opacity={0.18} />
      </mesh>
    </group>
  )
}
```

- [ ] **Step 4: Verify visually**

Run: `cd dashboard && npm run dev`
Expected: Brain shell is visibly brighter with blue-ish rim glow on edges. Wrinkles/sulci clearly visible even without active regions.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/utils/fresnelMaterial.ts dashboard/src/components/BrainModel.tsx
git commit -m "feat(dashboard): add Fresnel rim light and increase brain shell opacity"
```

---

### Task 5: Make inactive regions always visible with their color

**Files:**
- Modify: `dashboard/src/components/BrainScene.tsx`

- [ ] **Step 1: Update RegionNode material to keep color when inactive**

In `RegionNode`, replace the material logic (lines 110-131). Change the color/opacity/emissive calculations:

Replace:
```typescript
  const color = new THREE.Color(config.color)
  const emissiveIntensity = level > 0.05 ? level * 2.5 : 0.08
```

With:
```typescript
  const color = new THREE.Color(config.color)
  // Always show region color; brighter when active
  const baseEmissive = 0.15
  const emissiveIntensity = level > 0.05 ? baseEmissive + level * 2.5 : baseEmissive
  const materialOpacity = level > 0.05 ? 0.75 : 0.20
```

Then update the JSX material:
```tsx
        <meshStandardMaterial
          color={config.color}
          emissive={color}
          emissiveIntensity={emissiveIntensity}
          transparent
          opacity={materialOpacity}
          roughness={0.7}
          depthWrite={false}
        />
```

Note: `color` is now always `config.color` (never `#151520`), and `opacity` is `0.20` minimum.

- [ ] **Step 2: Apply Fresnel to region meshes**

The `applyFresnelRimLight` utility was already created in Task 4. Import it in `BrainScene.tsx`:

```typescript
import { applyFresnelRimLight } from '../utils/fresnelMaterial'
```

Add ref callback in `RegionNode`:

```typescript
  const matCallbackRef = (mat: THREE.MeshStandardMaterial | null) => {
    if (mat && !mat.userData._fresnelApplied) {
      applyFresnelRimLight(mat, config.color, 2.5, 0.3)
      mat.userData._fresnelApplied = true
      mat.needsUpdate = true
    }
  }
```

And add `ref={matCallbackRef}` to the `<meshStandardMaterial>`.

Also update `BrainModel.tsx` to import from the shared utility instead of defining its own.

- [ ] **Step 3: Verify visually**

Run: `cd dashboard && npm run dev`
Expected: All 9 regions visible at all times with their colors. Wrinkles clearly discernible. When a region activates, it smoothly brightens from the dim baseline.

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/components/BrainScene.tsx
git commit -m "feat(dashboard): always-visible regions with Fresnel rim light on wrinkles"
```

---

## Chunk 3: Panel Restyling — Glassmorphism

### Task 6: Restyle EventLog with glassmorphism

**Files:**
- Modify: `dashboard/src/components/EventLog.tsx`
- Modify: `dashboard/src/App.css`

- [ ] **Step 1: Replace EventLog CSS in App.css**

Replace the `.event-log`, `.event-item`, `.event-time`, `.event-tag` rules with:

```css
.event-log {
  position: absolute; top: 20px; left: 20px;
  background: var(--bg-panel); backdrop-filter: var(--panel-blur);
  -webkit-backdrop-filter: var(--panel-blur);
  border: 1px solid var(--border); border-radius: var(--panel-radius);
  padding: 16px; width: 280px; max-height: 280px;
  overflow-y: auto; font-size: 11px;
  box-shadow: var(--panel-shadow);
  z-index: 50;
}
.event-log-header {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 14px; padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
.event-log-header .dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--blue); box-shadow: 0 0 8px rgba(59,130,246,0.5);
}
.event-log-header span {
  font-size: 10px; font-weight: 500; letter-spacing: 1.5px;
  text-transform: uppercase; color: rgba(148,163,184,0.7);
}
.event-item {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.02);
}
.event-item:last-child { border-bottom: none; }
.event-dot {
  width: 5px; height: 5px; border-radius: 50%;
  margin-top: 5px; flex-shrink: 0;
}
.event-tag {
  font-family: var(--font-mono); font-size: 10px;
  color: rgba(148,163,184,0.6); line-height: 1.5;
}
.event-tag .label { font-weight: 500; margin-right: 6px; }
.event-tag .payload { color: rgba(148,163,184,0.4); font-size: 9px; }
```

- [ ] **Step 2: Update EventLog component**

Replace entire `EventLog.tsx`:

```tsx
import { useBrainStore } from '../stores/brainState'

const TAG_COLORS: Record<string, string> = {
  region_activation: '#3b82f6',
  network_switch: '#ef4444',
  routing_event: '#4ade80',
  memory_event: '#06b6d4',
  memory_flow: '#c084fc',
  neuromodulator: '#f97316',
  broadcast: '#fbbf24',
}

export function EventLog() {
  const events = useBrainStore((s) => s.events)
  const recent = events.slice(-7).reverse()

  return (
    <div className="event-log">
      <div className="event-log-header">
        <div className="dot" />
        <span>Event Stream</span>
      </div>
      {recent.map((evt, i) => (
        <div key={i} className="event-item">
          <div
            className="event-dot"
            style={{
              background: TAG_COLORS[evt.type] || '#94a3b8',
              boxShadow: `0 0 4px ${TAG_COLORS[evt.type] || '#94a3b8'}`,
            }}
          />
          <div className="event-tag">
            <span className="label" style={{ color: TAG_COLORS[evt.type] || '#94a3b8' }}>
              {evt.type.replace(/_/g, ' ').slice(0, 12)}
            </span>
            <span className="payload">{JSON.stringify(evt.payload).slice(0, 40)}</span>
          </div>
        </div>
      ))}
      {recent.length === 0 && <div style={{ color: '#475569', fontSize: 10 }}>Waiting for events...</div>}
    </div>
  )
}
```

- [ ] **Step 3: Verify visually, commit**

```bash
git add dashboard/src/components/EventLog.tsx dashboard/src/App.css
git commit -m "feat(dashboard): restyle EventLog with glassmorphism"
```

---

### Task 7: Restyle HUD with glassmorphism and gradient bars

**Files:**
- Modify: `dashboard/src/components/HUD.tsx`
- Modify: `dashboard/src/App.css`

- [ ] **Step 1: Replace HUD CSS in App.css**

Replace `.hud`, `.hud-title`, `.hud-row` rules with:

```css
.hud {
  position: absolute; top: 20px; right: 20px;
  background: var(--bg-panel); backdrop-filter: var(--panel-blur);
  -webkit-backdrop-filter: var(--panel-blur);
  border: 1px solid var(--border); border-radius: var(--panel-radius);
  padding: 16px; width: 200px;
  box-shadow: var(--panel-shadow);
  z-index: 50;
}
.hud .status-row {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 14px; padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
.hud .status-dot {
  width: 7px; height: 7px; border-radius: 50%;
  animation: pulseDot 2s ease-in-out infinite;
}
.hud .status-label { font-size: 11px; color: rgba(226,232,240,0.6); }
.hud .mode-badge {
  margin-left: auto; font-family: var(--font-mono);
  font-size: 9px; font-weight: 500; letter-spacing: 0.5px;
  padding: 3px 8px; border-radius: 6px;
}
.neuro-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.neuro-item { display: flex; flex-direction: column; gap: 4px; }
.neuro-label { font-size: 9px; color: rgba(148,163,184,0.5); letter-spacing: 0.3px; }
.neuro-bar-track { height: 3px; background: rgba(255,255,255,0.04); border-radius: 2px; overflow: hidden; }
.neuro-bar-fill { height: 100%; border-radius: 2px; transition: width 0.6s ease; }
.region-list {
  margin-top: 14px; padding-top: 10px;
  border-top: 1px solid var(--border);
}
.region-list-item { display: flex; align-items: center; gap: 6px; padding: 3px 0; }
.region-indicator { width: 4px; height: 4px; border-radius: 50%; transition: all 0.3s; }
.region-name { font-size: 10px; color: rgba(148,163,184,0.4); font-weight: 300; }
.region-name.active { color: rgba(226,232,240,0.8); font-weight: 400; }
```

- [ ] **Step 2: Rewrite HUD component**

Replace entire `HUD.tsx`:

```tsx
import { useBrainStore } from '../stores/brainState'
import { REGION_CONFIG } from '../constants/brainRegions'

const MODE_STYLES: Record<string, { label: string; bg: string; color: string; border: string }> = {
  executive_control: { label: 'ECN', bg: 'rgba(249,115,22,0.12)', color: '#f97316', border: 'rgba(249,115,22,0.15)' },
  default_mode: { label: 'DMN', bg: 'rgba(96,165,250,0.12)', color: '#60a5fa', border: 'rgba(96,165,250,0.15)' },
}

const NEURO_BARS = [
  { key: 'urgency' as const, label: 'Urgency', gradient: 'linear-gradient(90deg, #f97316, #fb923c)' },
  { key: 'learning_rate' as const, label: 'Learning', gradient: 'linear-gradient(90deg, #22c55e, #4ade80)' },
  { key: 'patience' as const, label: 'Patience', gradient: 'linear-gradient(90deg, #3b82f6, #60a5fa)' },
  { key: 'reward' as const, label: 'Reward', gradient: 'linear-gradient(90deg, #a855f7, #c084fc)' },
]

export function HUD() {
  const nm = useBrainStore((s) => s.neuromodulators)
  const mode = useBrainStore((s) => s.networkMode)
  const regions = useBrainStore((s) => s.regions)
  const connected = useBrainStore((s) => s.connected)

  const modeStyle = MODE_STYLES[mode] || { label: 'CRE', bg: 'rgba(139,92,246,0.12)', color: '#8b5cf6', border: 'rgba(139,92,246,0.15)' }

  return (
    <div className="hud">
      <div className="status-row">
        <div
          className="status-dot"
          style={{
            background: connected ? '#22c55e' : '#ef4444',
            boxShadow: connected ? '0 0 8px rgba(34,197,94,0.5)' : '0 0 8px rgba(239,68,68,0.5)',
          }}
        />
        <span className="status-label">{connected ? 'Connected' : 'Disconnected'}</span>
        <div
          className="mode-badge"
          style={{ background: modeStyle.bg, color: modeStyle.color, border: `1px solid ${modeStyle.border}` }}
        >
          {modeStyle.label}
        </div>
      </div>

      <div className="neuro-grid">
        {NEURO_BARS.map((bar) => (
          <div key={bar.key} className="neuro-item">
            <span className="neuro-label">{bar.label}</span>
            <div className="neuro-bar-track">
              <div
                className="neuro-bar-fill"
                style={{ width: `${Math.round(nm[bar.key] * 100)}%`, background: bar.gradient }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="region-list">
        {Object.entries(regions).map(([name, state]) => {
          const isActive = state.level > 0.1
          const cfg = REGION_CONFIG[name]
          return (
            <div key={name} className="region-list-item">
              <div
                className="region-indicator"
                style={{
                  background: isActive ? cfg?.color : 'rgba(100,100,100,0.2)',
                  boxShadow: isActive ? `0 0 4px ${cfg?.color}` : 'none',
                }}
              />
              <span className={`region-name${isActive ? ' active' : ''}`}>
                {name.replace(/_/g, ' ')}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Verify visually, commit**

Note: Do NOT remove `.hud-title` CSS yet — it's still used by `ChatInput.tsx` until Task 9 replaces it. Cleanup happens in Task 12.

```bash
git add dashboard/src/components/HUD.tsx dashboard/src/App.css
git commit -m "feat(dashboard): restyle HUD with glassmorphism and gradient bars"
```

---

### Task 8: Restyle MemoryFlowBar as slim pill

**Files:**
- Modify: `dashboard/src/components/MemoryFlowBar.tsx`
- Modify: `dashboard/src/App.css`

- [ ] **Step 1: Replace MemoryFlowBar CSS in App.css**

Replace `.memory-bar`, `.mem-stage`, `.mem-*` rules with:

```css
.memory-bar {
  position: absolute; bottom: 68px; left: 50%; transform: translateX(-50%);
  max-width: 540px; width: 90%;
  display: flex; align-items: center; gap: 4px;
  background: rgba(8,8,20,0.55); backdrop-filter: var(--panel-blur);
  -webkit-backdrop-filter: var(--panel-blur);
  border: 1px solid var(--border); border-radius: 12px;
  padding: 10px 20px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.2);
  z-index: 30;
}
.mem-stage {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 8px; flex: 1; justify-content: center;
}
.mem-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
.mem-label { font-size: 10px; font-weight: 400; letter-spacing: 0.2px; }
.mem-count { font-family: var(--font-mono); font-size: 9px; opacity: 0.5; }
.mem-arrow { color: rgba(148,163,184,0.15); font-size: 10px; }
.mem-sensory { background: rgba(148,163,184,0.05); }
.mem-sensory .mem-dot { background: rgba(148,163,184,0.4); }
.mem-sensory .mem-label { color: rgba(148,163,184,0.5); }
.mem-working { background: rgba(96,165,250,0.06); }
.mem-working .mem-dot { background: #60a5fa; box-shadow: 0 0 6px rgba(96,165,250,0.4); }
.mem-working .mem-label { color: rgba(96,165,250,0.7); }
.mem-staging { background: rgba(52,211,153,0.06); }
.mem-staging .mem-dot { background: #34d399; box-shadow: 0 0 6px rgba(52,211,153,0.4); }
.mem-staging .mem-label { color: rgba(52,211,153,0.7); }
.mem-semantic { background: rgba(192,132,252,0.06); }
.mem-semantic .mem-dot { background: #c084fc; box-shadow: 0 0 6px rgba(192,132,252,0.4); }
.mem-semantic .mem-label { color: rgba(192,132,252,0.7); }
```

- [ ] **Step 2: Rewrite MemoryFlowBar component**

Replace entire `MemoryFlowBar.tsx`:

```tsx
import { useBrainStore } from '../stores/brainState'

const STAGES = [
  { key: 'sensory' as const, label: 'Sensory', cls: 'mem-sensory' },
  { key: 'working' as const, label: 'Working', cls: 'mem-working' },
  { key: 'staging' as const, label: 'Staging', cls: 'mem-staging' },
  { key: 'semantic' as const, label: 'Semantic', cls: 'mem-semantic' },
]

export function MemoryFlowBar() {
  const mf = useBrainStore((s) => s.memoryFlow)

  return (
    <div className="memory-bar">
      {STAGES.flatMap((stage, i) => [
        i > 0 && <span key={`arrow-${i}`} className="mem-arrow">›</span>,
        <span key={stage.key} className={`mem-stage ${stage.cls}`}>
          <span className="mem-dot" />
          <span className="mem-label">{stage.label}</span>
          <span className="mem-count">
            {stage.key === 'working' ? `${mf[stage.key]}/4` : mf[stage.key]}
          </span>
        </span>,
      ])}
    </div>
  )
}
```

- [ ] **Step 3: Verify visually, commit**

```bash
git add dashboard/src/components/MemoryFlowBar.tsx dashboard/src/App.css
git commit -m "feat(dashboard): restyle MemoryFlowBar as slim centered pill"
```

---

## Chunk 4: Input UI — Unified Bar + Audio Orb

### Task 9: Rebuild ChatInput as unified bar with audio orb

**Files:**
- Create: `dashboard/src/components/AudioOrb.tsx`
- Modify: `dashboard/src/components/ChatInput.tsx`
- Modify: `dashboard/src/App.css`

- [ ] **Step 1: Create AudioOrb component**

Create `dashboard/src/components/AudioOrb.tsx`:

```tsx
import { useBrainStore } from '../stores/brainState'

export function AudioOrb() {
  const isAudioMode = useBrainStore((s) => s.isAudioMode)
  const audioState = useBrainStore((s) => s.audioState)
  const setAudioMode = useBrainStore((s) => s.setAudioMode)
  const setAudioState = useBrainStore((s) => s.setAudioState)

  const handleClick = () => {
    if (isAudioMode) {
      setAudioState('idle')
      setAudioMode(false)
    } else {
      setAudioMode(true)
      setAudioState('listening')
    }
  }

  if (!isAudioMode) return null

  return (
    <div className="audio-orb-container" onClick={handleClick}>
      <div className={`audio-orb ${audioState}`}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" y1="19" x2="12" y2="22"/>
        </svg>
      </div>
      <div className="audio-orb-label">
        {audioState === 'listening' ? 'Listening...' : audioState === 'processing' ? 'Processing...' : ''}
      </div>
      <div className="waveform">
        {[1,2,3,4,5,6,7].map((n) => (
          <div key={n} className="waveform-bar" style={{ animation: `waveform${n} ${0.4 + n * 0.1}s ease-in-out infinite` }} />
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Add AudioOrb and unified bar CSS to App.css**

Remove old `.chat-panel`, `.chat-response`, `.chat-form`, `.chat-input`, `.chat-send` rules. Replace with:

```css
/* === Unified Input Bar === */
.input-bar {
  position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
  max-width: 540px; width: 90%; z-index: 50;
}
.input-bar.dimmed { opacity: 0.4; pointer-events: none; }
.input-container {
  display: flex; align-items: center; gap: 8px;
  background: var(--bg-input); backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 28px; padding: 6px 6px 6px 22px;
  transition: border-color 0.3s, box-shadow 0.3s;
  box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.input-container:focus-within {
  border-color: var(--border-focus);
  box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 0 1px rgba(59,130,246,0.1);
}
.input-field {
  flex: 1; background: none; border: none; outline: none;
  color: var(--text-primary); font-family: var(--font-ui);
  font-size: 13px; font-weight: 300;
}
.input-field::placeholder { color: rgba(148,163,184,0.35); }
.btn-mic {
  width: 38px; height: 38px; border-radius: 50%;
  border: 1px solid rgba(255,255,255,0.15);
  background: rgba(255,255,255,0.08);
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; transition: all 0.3s; flex-shrink: 0;
}
.btn-mic:hover {
  background: rgba(255,255,255,0.15);
  border-color: rgba(255,255,255,0.25);
  box-shadow: 0 0 16px rgba(255,255,255,0.1);
}
.btn-mic svg { width: 16px; height: 16px; }
.btn-send {
  width: 38px; height: 38px; border-radius: 50%; border: none;
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; transition: all 0.3s; flex-shrink: 0;
  box-shadow: 0 2px 12px rgba(59,130,246,0.3);
}
.btn-send:hover { transform: scale(1.05); box-shadow: 0 4px 20px rgba(59,130,246,0.4); }
.btn-send:disabled { background: #334155; cursor: not-allowed; box-shadow: none; }
.btn-send svg { width: 16px; height: 16px; }

/* === Audio Orb === */
.audio-orb-container {
  position: absolute; bottom: 80px; left: 50%; transform: translateX(-50%);
  display: flex; flex-direction: column; align-items: center; gap: 8px;
  z-index: 100; cursor: pointer;
  animation: fadeIn 0.5s cubic-bezier(0.4,0,0.2,1);
}
.audio-orb {
  width: 64px; height: 64px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.4), rgba(255,255,255,0.1));
  border: 2px solid rgba(255,255,255,0.5);
  box-shadow: 0 0 50px rgba(255,255,255,0.2);
  display: flex; align-items: center; justify-content: center;
  transition: all 0.5s cubic-bezier(0.4,0,0.2,1);
}
.audio-orb:hover { box-shadow: 0 0 70px rgba(255,255,255,0.3); }
.audio-orb-label { font-size: 12px; color: rgba(255,255,255,0.7); font-weight: 300; }
.waveform { display: flex; gap: 3px; align-items: center; height: 24px; }
.waveform-bar {
  width: 3px; border-radius: 2px;
  background: rgba(255,255,255,0.6);
}
```

- [ ] **Step 3: Rewrite ChatInput component**

Replace entire `ChatInput.tsx`:

```tsx
import { useState } from 'react'
import { useBrainStore } from '../stores/brainState'
import { AudioOrb } from './AudioOrb'

export function ChatInput() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const isAudioMode = useBrainStore((s) => s.isAudioMode)
  const setAudioMode = useBrainStore((s) => s.setAudioMode)
  const setAudioState = useBrainStore((s) => s.setAudioState)
  const setLastResponse = useBrainStore((s) => s.setLastResponse)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!text.trim() || loading) return

    setLoading(true)
    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      const data = await res.json()
      setLastResponse(data.response || data.error || 'No response')
    } catch {
      setLastResponse('Connection error')
    }
    setLoading(false)
    setText('')
  }

  const handleMicClick = () => {
    setAudioMode(true)
    setAudioState('listening')
  }

  return (
    <>
      <AudioOrb />
      <div className={`input-bar${isAudioMode ? ' dimmed' : ''}`}>
        <form onSubmit={handleSubmit} className="input-container">
          <input
            className="input-field"
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={loading ? 'Processing...' : isAudioMode ? 'Audio mode active...' : 'Ask anything...'}
            disabled={loading || isAudioMode}
          />
          <button type="button" className="btn-mic" onClick={handleMicClick} disabled={isAudioMode}>
            <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" y1="19" x2="12" y2="22"/>
            </svg>
          </button>
          <button type="submit" className="btn-send" disabled={loading || isAudioMode || !text.trim()}>
            <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="19" x2="12" y2="5"/>
              <polyline points="5 12 12 5 19 12"/>
            </svg>
          </button>
        </form>
      </div>
    </>
  )
}
```

- [ ] **Step 4: Verify unified bar and audio orb visually**

Run: `cd dashboard && npm run dev`
Expected: Pill-shaped input bar at bottom. White mic button. Click mic → orb floats up with waveform. Click orb again → returns.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/components/AudioOrb.tsx dashboard/src/components/ChatInput.tsx dashboard/src/App.css
git commit -m "feat(dashboard): unified input bar with float-up audio orb"
```

---

## Chunk 5: Response Bubbles — Brain Speech + Region Mini Bubbles

### Task 10: Create BrainResponseBubble

**Files:**
- Create: `dashboard/src/components/BrainResponseBubble.tsx`
- Modify: `dashboard/src/components/BrainScene.tsx`
- Modify: `dashboard/src/App.css`

- [ ] **Step 1: Add CSS for brain response bubble**

Append to `App.css`:

```css
/* === Brain Response Bubble === */
.brain-response-bubble {
  background: rgba(8,8,20,0.85); backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(59,130,246,0.15);
  border-radius: 14px; padding: 12px 16px;
  max-width: 220px; pointer-events: none;
  box-shadow: 0 8px 32px rgba(0,0,0,0.25);
  animation: fadeIn 0.3s ease;
}
.brain-response-bubble.fading { animation: fadeOut 0.5s ease forwards; }
.brain-response-bubble .br-indicator {
  display: flex; align-items: center; gap: 5px; margin-bottom: 6px;
}
.brain-response-bubble .br-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--blue); box-shadow: 0 0 6px rgba(59,130,246,0.5);
  animation: pulseDot 1.5s ease-in-out infinite;
}
.brain-response-bubble .br-label { font-size: 9px; color: rgba(148,163,184,0.5); }
.brain-response-bubble .br-text {
  font-size: 12px; color: rgba(226,232,240,0.85);
  line-height: 1.6; font-weight: 300;
}
```

- [ ] **Step 2: Create BrainResponseBubble component**

Create `dashboard/src/components/BrainResponseBubble.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { Html } from '@react-three/drei'
import { useBrainStore } from '../stores/brainState'

export function BrainResponseBubble() {
  const lastResponse = useBrainStore((s) => s.lastResponse)
  const responseTimestamp = useBrainStore((s) => s.responseTimestamp)
  const [visible, setVisible] = useState(false)
  const [fading, setFading] = useState(false)

  useEffect(() => {
    if (!lastResponse) return
    setVisible(true)
    setFading(false)

    const fadeTimer = setTimeout(() => setFading(true), 3000)
    const hideTimer = setTimeout(() => setVisible(false), 3500)

    return () => {
      clearTimeout(fadeTimer)
      clearTimeout(hideTimer)
    }
  }, [lastResponse, responseTimestamp])

  if (!visible || !lastResponse) return null

  return (
    <Html position={[18, 14, 10]} center={false} style={{ pointerEvents: 'none' }}>
      <div className={`brain-response-bubble${fading ? ' fading' : ''}`}>
        <div className="br-indicator">
          <div className="br-dot" />
          <span className="br-label">Brain</span>
        </div>
        <div className="br-text">{lastResponse}</div>
      </div>
    </Html>
  )
}
```

- [ ] **Step 3: Add BrainResponseBubble to BrainScene**

In `BrainScene.tsx`, add import and render inside `<BrainPulse>`:

```tsx
import { BrainResponseBubble } from './BrainResponseBubble'
```

Add inside the `BrainScene` return, after `SignalParticles`:

```tsx
      <BrainResponseBubble />
```

- [ ] **Step 4: Verify, commit**

```bash
git add dashboard/src/components/BrainResponseBubble.tsx dashboard/src/components/BrainScene.tsx dashboard/src/App.css
git commit -m "feat(dashboard): brain speech bubble with 3s fade-out"
```

---

### Task 11: Create RegionBubble mini tooltips

**Files:**
- Create: `dashboard/src/components/RegionBubble.tsx`
- Modify: `dashboard/src/components/BrainScene.tsx`
- Modify: `dashboard/src/App.css`

- [ ] **Step 1: Add CSS for region bubbles**

Append to `App.css`:

```css
/* === Region Mini Bubbles === */
.region-bubble {
  background: rgba(8,8,20,0.88); backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid transparent;
  border-radius: 10px; padding: 6px 10px;
  max-width: 140px; pointer-events: none;
  box-shadow: 0 4px 16px rgba(0,0,0,0.3);
  animation: fadeIn 0.4s ease;
}
.region-bubble .rb-name {
  font-size: 8px; font-weight: 500; letter-spacing: 0.5px;
  text-transform: uppercase; margin-bottom: 3px;
}
.region-bubble .rb-text {
  font-size: 10px; color: rgba(203,213,225,0.8);
  line-height: 1.4; font-weight: 300;
}
```

- [ ] **Step 2: Create RegionBubble component**

Create `dashboard/src/components/RegionBubble.tsx`:

```tsx
import { Html } from '@react-three/drei'
import { useBrainStore } from '../stores/brainState'
import { REGION_CONFIG, REGION_DESCRIPTIONS } from '../constants/brainRegions'

export function RegionBubble({ name }: { name: string }) {
  const region = useBrainStore((s) => s.regions[name])
  const config = REGION_CONFIG[name]
  if (!region || region.level <= 0.1 || !config) return null

  const descriptions = REGION_DESCRIPTIONS[name]
  const text = descriptions?.[region.mode] || descriptions?.active || ''
  if (!text) return null

  // Offset the bubble to the right/left of the region
  const offsetX = config.position[0] >= 0 ? 6 : -6
  const pos: [number, number, number] = [
    config.position[0] + offsetX,
    config.position[1] + 2,
    config.position[2],
  ]

  return (
    <Html position={pos} center={false} style={{ pointerEvents: 'none' }}>
      <div className="region-bubble" style={{ borderColor: `${config.color}40` }}>
        <div className="rb-name" style={{ color: config.color }}>{name.replace(/_/g, ' ')}</div>
        <div className="rb-text">{text}</div>
      </div>
    </Html>
  )
}
```

- [ ] **Step 3: Add RegionBubbles to BrainScene**

In `BrainScene.tsx`, add import:

```tsx
import { RegionBubble } from './RegionBubble'
```

In the `BrainScene` return, add after the `RegionNode` map:

```tsx
      {Object.keys(REGION_CONFIG).map((name) => (
        <RegionBubble key={`bubble-${name}`} name={name} />
      ))}
```

- [ ] **Step 4: Verify, commit**

Note: `RegionBubble` reads `region.mode` from the store and looks up `REGION_DESCRIPTIONS` constant directly — no useWebSocket changes needed.

```bash
git add dashboard/src/components/RegionBubble.tsx dashboard/src/components/BrainScene.tsx dashboard/src/App.css
git commit -m "feat(dashboard): region mini bubbles showing processing info"
```

---

## Chunk 6: Final Polish

### Task 12: Clean up App.css — remove orphaned old rules

**Files:**
- Modify: `dashboard/src/App.css`

- [ ] **Step 1: Remove all old rules that were replaced**

Delete these rules (all consumers have been rewritten by now):
- `.hud-title` (was used by EventLog, HUD, ChatInput — all rewritten)
- `.hud-row`, `.hud-row .label`, `.hud-row .value` (replaced by HUD grid)
- `.mode-dmn`, `.mode-ecn`, `.mode-creative` (replaced by inline mode-badge styles)
- `.mem-hippo` (renamed to `.mem-staging`)
- `.chat-panel`, `.chat-response`, `.chat-form`, `.chat-input`, `.chat-send` (replaced by `.input-bar` etc.)

- [ ] **Step 2: Verify no CSS errors, commit**

```bash
git add dashboard/src/App.css
git commit -m "chore(dashboard): clean up orphaned CSS rules"
```

---

### Task 13: Visual verification and final adjustments

**Files:**
- All modified files

- [ ] **Step 1: Run dev server and verify all features**

Run: `cd dashboard && npm run dev`

Checklist:
- [ ] Brain shell always visible with Fresnel rim glow on wrinkles
- [ ] All 9 regions visible in dim state with their colors
- [ ] Regions brighten smoothly when activated
- [ ] EventLog: glassmorphism, color dots, max 7 items
- [ ] HUD: gradient bars, mode badge, region list
- [ ] MemoryFlowBar: slim centered pill aligned with input bar
- [ ] Input bar: pill shape, white mic button, blue send button
- [ ] Audio orb: float-up on mic click, waveform bars, click to dismiss
- [ ] Brain speech bubble: appears on API response, fades after 3s
- [ ] Region mini bubbles: appear next to active regions with Korean text

- [ ] **Step 2: Fix any visual issues found**

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(dashboard): complete UI upgrade — brain visibility, input bar, bubbles, glassmorphism"
```
