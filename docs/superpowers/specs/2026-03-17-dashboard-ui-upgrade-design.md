# Dashboard UI Upgrade Design

## Overview

Brain-agent 대시보드 UI 업그레이드. 뇌 시각화의 가시성 개선, 텍스트+오디오 통합 입력 UI 추가, 패널 리스타일링.

## Problem

1. 비활성 상태에서 뇌가 거의 보이지 않음 (opacity 0.08, color #151520)
2. 해부학적 요소(주름/sulci/gyri)가 비활성 시 안 보임
3. 입력 UI가 단순 텍스트만 지원, 오디오 입력 미지원
4. 기존 패널(HUD, EventLog, MemoryFlowBar)이 시각적으로 투박함

## Design Decisions

| 항목 | 결정 |
|------|------|
| 비활성 region 표현 | 고유 색상 opacity 0.15~0.25, 주름 항상 보임 |
| 입력 UI 레이아웃 | Unified Bar (하단 중앙 pill shape) |
| 오디오 모드 전환 | Float-up (오브가 뇌 방향으로 떠오름) |
| 응답 표시 | Brain Speech Bubble (뇌 옆 말풍선, 3초 fade out) |
| Region 처리 정보 | Mini Bubbles (활성 region 옆 미니 말풍선) |
| 기존 패널 | 전부 유지 + glassmorphism 리스타일링 |
| 마이크 버튼 색상 | 흰색 |

---

## 1. Brain Visibility

### 1.1 비활성 Region

현재 `RegionNode`에서 `level <= 0.05`일 때:
- color: `#151520` (거의 검정)
- opacity: `0.08`
- emissiveIntensity: `0.08`

변경:
- color: 각 region 고유 색상 유지
- opacity: `0.20` (최소)
- emissiveIntensity: `0.15` (최소)
- 주름 텍스처가 항상 인지 가능

### 1.2 주름(Sulci/Gyri) 강조

현재 `createRegionGeometry`에서 procedural wrinkle을 생성하지만, 비활성 시 너무 어두워서 안 보임.

방법:
- **Fresnel rim light**: 기존 `meshStandardMaterial`의 `onBeforeCompile`로 Fresnel 셰이더 주입. 가장자리를 밝게 해서 주름 입체감 강조
- **Emissive map**: 주름 골 부분에 미세한 emissive를 더해서 형태 인지
- BrainModel 외곽 shell opacity: `0.12` → `0.22`로 증가
- 외곽 shell에도 동일한 Fresnel rim light 적용

### 1.3 활성 Region (기존 유지 + 개선)

- 고유 색상으로 밝기 증가 (level에 비례)
- Glow cloud opacity: `level * 0.3 * pulse` (기존 유지)
- Heartbeat double-beat pulse (기존 유지)
- 비활성→활성 전환 시 색상이 자연스럽게 밝아지는 것 (이미 dim glow 상태이므로 더 자연스러움)

---

## 2. Input UI

### 2.1 Unified Bar (메인 입력)

위치: 화면 하단 중앙, `bottom: 20px`
크기: `max-width: 540px; width: 90%`, pill shape (`border-radius: 28px`)

구성:
```
[ placeholder text ... ] [🎤 mic] [↑ send]
```

스타일:
- background: `rgba(10, 10, 28, 0.8)`
- `backdrop-filter: blur(24px)`
- border: `1px solid rgba(255, 255, 255, 0.06)`
- focus 시 border: `rgba(59, 130, 246, 0.3)` + subtle glow
- 마이크 버튼: 흰색 stroke, `rgba(255, 255, 255, 0.08)` 배경, `rgba(255, 255, 255, 0.15)` 테두리
- 전송 버튼: blue gradient (`#3b82f6` → `#2563eb`), 흰색 arrow

### 2.2 Audio Mode (Float-up Orb)

트리거: 마이크 버튼 클릭

**오브는 DOM 요소로 유지** (Canvas 진입 없음). CSS transitions로 `position: absolute` + `transform` 애니메이션.

상태 머신:
```
idle → listening → processing → done → idle
```
- `idle`: 마이크 버튼이 입력바 안에 있음
- `listening`: 오브가 떠올라 있고 waveform 표시 중
- `processing`: 음성 인식 처리 중 (spinner)
- `done`: 인식 텍스트가 입력바에 채워지고 오브 복귀

전환 애니메이션 (CSS transition, `transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1)`):
1. 마이크 버튼이 바에서 분리
2. 오브가 뇌 방향(위쪽)으로 떠오름 (뇌와 입력바 사이 중간 지점)
3. 오브 주변에 glow 확장 (`box-shadow: 0 0 50px`)
4. 오브 아래에 waveform visualizer 표시 (장식 애니메이션, CSS keyframes로 bar 높이 변화)
5. 입력바는 dimmed (`opacity: 0.4`), placeholder: "Audio mode active..."

오브 스타일:
- 크기: `64px × 64px`, 원형
- background: `radial-gradient(circle, rgba(255,255,255,0.4), rgba(255,255,255,0.1))`
- border: `2px solid rgba(255, 255, 255, 0.5)`
- glow: `0 0 50px rgba(255, 255, 255, 0.2)`
- "Listening..." 텍스트 아래에 표시

Waveform visualizer:
- 7개 세로 bar, 높이가 CSS keyframes로 랜덤하게 변화
- 흰색, `opacity: 0.5~0.8`
- 실제 오디오 분석 아님 (장식용)

종료:
- 오브 다시 클릭 → `idle`로 즉시 복귀 (인식 취소)
- 음성 인식 완료 → `processing` → `done` → `idle`
- 5초 무음 시 자동 종료
- 오브가 다시 입력바로 돌아오며 축소
- 인식된 텍스트가 입력바에 채워짐

에러 처리:
- `no-speech` / `audio-capture` 에러: 오브가 붉은색으로 잠깐 flash 후 `idle`로 복귀

### 2.3 구현 참고

- 오디오 입력은 Web Speech API (`SpeechRecognition`) 또는 외부 STT 연동
- 현재는 UI 껍데기만 구현, 실제 STT 연동은 별도 작업
- Web Speech API는 Chromium 기반 브라우저에서만 지원 (개발자 대시보드이므로 허용)
- WebSocket으로 `{ type: "user_input", text: string, source: "text" | "audio" }` 전송

---

## 3. Response Display

### 3.1 Brain Speech Bubble

위치: 뇌 3D 모델 옆 (HTML overlay, `@react-three/drei`의 `Html` 컴포넌트)

트리거: API 응답 수신 시

동작:
1. fade-in 애니메이션 (0.3s, translateX 살짝)
2. 3초간 표시
3. fade-out (0.5s)
4. 새 응답 오면 이전 것 즉시 교체

스타일:
- background: `rgba(8, 8, 20, 0.85)`
- `backdrop-filter: blur(16px)`
- border: `1px solid rgba(59, 130, 246, 0.15)`
- border-radius: `14px`
- max-width: `220px`
- 내부: 파란 dot 인디케이터 + "Brain" 라벨 + 응답 텍스트

### 3.2 Region Mini Bubbles

위치: 각 활성 region의 3D 위치에서 오프셋 (HTML overlay)

트리거: region의 `level > 0.1`이고 처리 정보가 있을 때

내용: region 이름 + 현재 처리 중인 짧은 텍스트

스타일:
- background: `rgba(8, 8, 20, 0.88)`
- `backdrop-filter: blur(12px)`
- border: region 고유 색상 (`1px solid rgba(color, 0.25)`)
- border-radius: `10px`
- padding: `6px 10px`
- max-width: `140px`
- region 이름: 고유 색상, 8px uppercase
- 처리 텍스트: `rgba(203, 213, 225, 0.8)`, 10px

데이터 소스: 프론트엔드 매핑 테이블 (백엔드 변경 불필요)

기본 매핑:
| Region | active | high_activity |
|--------|--------|---------------|
| prefrontal_cortex | 계획 수립 중 | 집중 분석 중 |
| acc | 갈등 모니터링 | 오류 감지 중 |
| amygdala | 감정 평가 중 | 위험 감지! |
| basal_ganglia | 행동 선택 중 | 행동 실행 중 |
| cerebellum | 타이밍 조정 중 | 패턴 학습 중 |
| thalamus | 신호 라우팅 | 입력 게이팅 중 |
| hypothalamus | 항상성 조절 | 긴급 조절 중 |
| hippocampus | 기억 인코딩 | 기억 통합 중 |
| salience_network | 핵심 필터링 | 주의 전환 중 |

향후 WebSocket `region_activation`에 `description` 필드가 추가되면 매핑 대신 실시간 텍스트 사용 가능.

---

## 4. Panel Restyling

### 4.1 공통 스타일

모든 패널에 적용:
- background: `rgba(8, 8, 20, 0.65)`
- `backdrop-filter: blur(20px)`
- border: `1px solid rgba(255, 255, 255, 0.04)`
- border-radius: `16px`
- box-shadow: `0 8px 32px rgba(0, 0, 0, 0.3)`
- 폰트: Inter (UI), JetBrains Mono (데이터/코드)

### 4.2 EventLog (좌상단)

- 헤더: dot 인디케이터 + "EVENT STREAM" (uppercase, letter-spacing)
- 각 이벤트: 컬러 dot (기존 색상 유지) + monospace 텍스트
- 구분선: `1px solid rgba(255, 255, 255, 0.02)` (매우 미세)
- 최대 5~7개 표시 (현재 15 → 축소하여 컴팩트하게)

### 4.3 HUD (우상단)

- 상태: green dot (pulse 애니메이션) + "Connected" + mode badge (pill shape)
- Neuromodulator: 2×2 grid, 각각 라벨 + gradient bar (숫자 대신 시각적 바)
  - Urgency: orange gradient
  - Learning: green gradient
  - Patience: blue gradient
  - Reward: purple gradient
- Region list: dot 인디케이터 + 이름, 활성 시 밝게

### 4.4 MemoryFlowBar (하단, 입력바 위)

- 위치: 입력바 바로 위 (`bottom: 68px`), 중앙 정렬
- 크기: 입력바와 동일한 `max-width: 540px; width: 90%` (시각적 정렬)
- 형태: 슬림한 pill, 가로 한 줄
- 각 stage: 컬러 dot + 이름 + count
- 화살표: `›` (미세한 색상)
- 전체 높이: ~36px (현재보다 훨씬 작음)

---

## 5. Technical Implementation

### 5.1 수정 파일

| 파일 | 변경 내용 |
|------|----------|
| `BrainScene.tsx` | RegionNode에 Html overlay 추가 (mini bubbles) |
| `BrainModel.tsx` | 외곽 shell opacity 증가, Fresnel rim light 추가 |
| `RegionNode` (in BrainScene) | 비활성 시 고유색상+opacity 유지, emissive 최소값 상향 |
| `ChatInput.tsx` | Unified Bar 리디자인 + Audio mode + Float-up orb |
| `HUD.tsx` | Glassmorphism + gradient bars + region list 리스타일 |
| `EventLog.tsx` | Glassmorphism + 컴팩트 디자인 |
| `MemoryFlowBar.tsx` | Slim pill 디자인 |
| `App.css` | 공통 glassmorphism 변수, Inter/JetBrains Mono 폰트 |
| `brainState.ts` | 새 필드 추가: `lastResponse`, `isAudioMode`, `audioState`, `regionDescriptions` |
| `useWebSocket.ts` | region_activation에 description 파싱 추가 |

### 5.2 새 컴포넌트

| 컴포넌트 | 역할 |
|----------|------|
| `BrainResponseBubble.tsx` | 뇌 옆 말풍선 (fade in/out) |
| `RegionBubble.tsx` | 활성 region 미니 말풍선 (Html overlay) |
| `AudioOrb.tsx` | Float-up 오디오 오브 + waveform |

### 5.3 State 스키마 추가 (brainState.ts)

```typescript
// 기존 필드에 추가
lastResponse: string | null           // Brain speech bubble 텍스트
responseTimestamp: number              // 응답 시간 (fade timer용)
isAudioMode: boolean                   // 오디오 모드 활성 여부
audioState: 'idle' | 'listening' | 'processing' | 'done'
regionDescriptions: Record<string, string>  // region별 처리 텍스트
```

응답 데이터 흐름: `ChatInput` → `POST /api/process` → 응답을 Zustand store의 `lastResponse`에 저장 → `BrainResponseBubble`이 store 구독하여 표시. DOM과 Canvas 양쪽에서 접근 가능.

### 5.4 Z-index 레이어링

```
z-index: 100  — AudioOrb (float-up 시)
z-index: 50   — Input bar, panels
z-index: 30   — MemoryFlowBar
z-index: 10   — Brain response bubble
z-index: auto — Canvas (Three.js)
```

### 5.5 애니메이션 전략

모든 DOM 애니메이션은 CSS transitions + CSS keyframes로 구현 (추가 의존성 없음):
- Float-up orb: `transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1)`
- Bubble fade: `@keyframes fadeIn`, `@keyframes fadeOut`
- Waveform bars: `@keyframes waveform` (height 변화)
- 3D 내부 애니메이션: 기존 `useFrame` 방식 유지

### 5.6 의존성

- `@react-three/drei`의 `Html` 컴포넌트 (이미 설치됨)
- Google Fonts: `index.html`에 CDN `<link>` 추가 (Inter, JetBrains Mono)
- Web Speech API (브라우저 내장, Chromium 전용)

---

## 6. Non-Goals

- 실제 STT/음성인식 백엔드 연동 (UI 껍데기만)
- 채팅 히스토리 저장/표시
- 모바일 반응형 레이아웃
- 기존 3D brain geometry 변경 (주름 생성 로직은 유지, 렌더링 파라미터만 조정)
