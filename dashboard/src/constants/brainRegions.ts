export const REGION_CONFIG: Record<string, { position: [number, number, number]; color: string; scale: number }> = {
  // === Frontal lobe (anterior-superior) ===
  // Positions verified inside L/R hemisphere spheres (center ±5.5,4,1  scale 9,13,16)
  prefrontal_cortex: { position: [-3, 12, 12], color: '#3b82f6', scale: 1.6 },   // Left frontal pole
  acc: { position: [-2, 11, 5], color: '#eab308', scale: 0.7 },                  // Cingulate, medial surface
  broca: { position: [-7, 8, 10], color: '#2563eb', scale: 0.5 },               // Left inferior frontal gyrus

  // === Deep / Subcortical (medial core) ===
  thalamus: { position: [0, 3, 0], color: '#ef4444', scale: 0.9 },               // Central relay hub
  hypothalamus: { position: [0, -2, 4], color: '#ec4899', scale: 0.5 },          // Below thalamus, anterior
  basal_ganglia: { position: [-5, 5, 2], color: '#f97316', scale: 0.8 },         // Striatum, lateral to thalamus
  corpus_callosum: { position: [0, 7, 3], color: '#f8fafc', scale: 0.8 },        // Midline commissure

  // === Temporal lobe (lateral-inferior) ===
  // Verified inside temporal lobe spheres (center ±11,-2,7  scale 6,5.5,8.5)
  amygdala: { position: [-8, -4, 8], color: '#f43f5e', scale: 0.7 },             // Anterior medial temporal
  hippocampus: { position: [-10, 0, -3], color: '#06b6d4', scale: 0.8 },         // Medial temporal
  auditory_cortex_l: { position: [-13, 0, 8], color: '#06b6d4', scale: 0.6 },    // Left Heschl's gyrus
  auditory_cortex_r: { position: [13, 0, 8], color: '#22d3ee', scale: 0.6 },     // Right Heschl's gyrus
  wernicke: { position: [-11, -4, 2], color: '#0891b2', scale: 0.6 },            // Left posterior superior temporal

  // === Parietal / Association areas ===
  angular_gyrus: { position: [-11, 2, -8], color: '#7c3aed', scale: 0.5 },       // Left temporo-parietal junction
  salience_network: { position: [9, 7, 9], color: '#22c55e', scale: 0.7 },       // Right fronto-insular

  // === Identity regions (self/other modeling) ===
  medial_pfc: { position: [-1, 13, 8], color: '#8b5cf6', scale: 0.7 },           // Medial wall of PFC (BA 10/32)
  tpj: { position: [11, -2, -4], color: '#06b6d4', scale: 0.6 },                 // Right temporoparietal junction

  // === 7-Phase additions: dual-stream + integration + output ===
  psts: { position: [-12, -3, 3], color: '#14b8a6', scale: 0.5 },               // Posterior superior temporal sulcus
  spt: { position: [-10, -5, 8], color: '#6366f1', scale: 0.4 },                // Sylvian parietal-temporal (dorsal stream)
  motor_cortex: { position: [-5, 10, 15], color: '#f59e0b', scale: 0.6 },       // Primary motor cortex (M1)

  // === Occipital lobe (posterior) ===
  visual_cortex: { position: [0, 2, -11], color: '#a855f7', scale: 1.0 },        // Occipital pole

  // === Infratentorial (cerebellum + brainstem) ===
  // Verified inside cerebellum shell (center 0,-11,-8  scale 9,6,7)
  cerebellum: { position: [0, -12, -8], color: '#8b5cf6', scale: 1.2 },          // Posterior fossa
  brainstem: { position: [0, -9, -4], color: '#db2777', scale: 0.7 },            // Medulla/pons
  vta: { position: [0, -6, 0], color: '#e11d48', scale: 0.4 },                   // Ventral midbrain

  // === Insular cortex (deep lateral) ===
  insula: { position: [8, 2, 5] as [number, number, number], color: '#14b8a6', scale: 0.55 },  // Deep to lateral sulcus
}

export const POSITIONS: Record<string, [number, number, number]> = Object.fromEntries(
  Object.entries(REGION_CONFIG).map(([k, v]) => [k, v.position])
)

export const INPUT_POSITION: [number, number, number] = [0, 25, 30]

export const REGION_INFO: Record<string, { fullName: string; role: string; mechanism: string }> = {
  prefrontal_cortex: { fullName: 'Prefrontal Cortex (PFC)', role: 'Executive reasoning & planning', mechanism: 'Dual-process: cached plans vs LLM reasoning. Hierarchical goal tree (rostral/mid/caudal). Entity & relation extraction.' },
  acc: { fullName: 'Anterior Cingulate Cortex (ACC)', role: 'Conflict detection & error monitoring', mechanism: '5-stage conflict evaluation. Patience-modulated thresholds. Triggers PFC re-deliberation on conflict.' },
  amygdala: { fullName: 'Amygdala', role: 'Emotional tagging', mechanism: 'Fast threat scan (LeDoux pathway). Russell circumplex model (valence/arousal). Tags every signal with emotional weight.' },
  basal_ganglia: { fullName: 'Basal Ganglia', role: 'Action selection (Go/NoGo)', mechanism: 'Direct (Go) / indirect (NoGo) pathways. Modulated by urgency, emotion, patience, and procedural memory.' },
  cerebellum: { fullName: 'Cerebellum', role: 'Predictive learning', mechanism: 'Per-tool forward models. Tracks success/fail/error. Predicts outcomes after 3+ observations. Escalates large errors to ACC.' },
  thalamus: { fullName: 'Thalamus', role: 'Input classification & relay', mechanism: '5-type classification (error_report, question, command, code, statement). Mode-based routing tables.' },
  hypothalamus: { fullName: 'Hypothalamus', role: 'Homeostatic regulation', mechanism: 'Monitors staging count & error rate. Controls neuromodulators: urgency, patience, learning rate.' },
  hippocampus: { fullName: 'Hippocampus', role: 'Memory encoding & consolidation', mechanism: 'Fast encoding to staging area. 4-phase consolidation: transfer, homeostatic scaling, semantic extraction, reflection.' },
  salience_network: { fullName: 'Salience Network', role: 'Mode switching (DMN/ECN/CREATIVE)', mechanism: 'Memory-based novelty detection. Arousal × 0.6 + novelty × 0.4. Triggers creative mode on high error + novelty.' },
  visual_cortex: {
    fullName: 'Visual Cortex (V1/V2)',
    role: 'Image processing',
    mechanism: 'Ventral stream (what pathway). Preprocesses image inputs, extracts visual features and descriptions.'
  },
  auditory_cortex_l: {
    fullName: 'Auditory Cortex (Left)',
    role: 'Speech processing',
    mechanism: 'Left Heschl\'s gyrus. Speech-to-text, phonological analysis. Language-dominant hemisphere.'
  },
  auditory_cortex_r: {
    fullName: 'Auditory Cortex (Right)',
    role: 'Prosody & emotional tone',
    mechanism: 'Right Heschl\'s gyrus. Emotional prosody, music, tone analysis. Non-verbal audio.'
  },
  wernicke: {
    fullName: 'Wernicke\'s Area (BA 22)',
    role: 'Language comprehension',
    mechanism: 'Left posterior superior temporal gyrus. Semantic parsing, intent extraction, text understanding.'
  },
  broca: {
    fullName: 'Broca\'s Area (BA 44/45)',
    role: 'Language production',
    mechanism: 'Left inferior frontal gyrus. Response formatting, syntax planning, output style.'
  },
  brainstem: {
    fullName: 'Brainstem',
    role: 'Arousal regulation',
    mechanism: 'Reticular activating system. Manages awake/drowsy/sleep states. Contains LC (NE) and Raphe (5-HT) nuclei.'
  },
  vta: {
    fullName: 'Ventral Tegmental Area (VTA)',
    role: 'Dopamine source',
    mechanism: 'Midbrain DA neurons. Fires on reward prediction errors. Projects to striatum and PFC.'
  },
  corpus_callosum: {
    fullName: 'Corpus Callosum',
    role: 'Inter-hemispheric integration',
    mechanism: '~200M axons connecting L/R hemispheres. Merges analytical (left) and holistic (right) PFC outputs. Always active.'
  },
  angular_gyrus: {
    fullName: 'Angular Gyrus (BA 39)',
    role: 'Cross-modal semantic integration',
    mechanism: 'Left parietal junction of temporal/parietal/occipital. Binds visual, auditory, and linguistic modalities into unified semantics.'
  },
  psts: {
    fullName: 'Posterior Superior Temporal Sulcus (pSTS)',
    role: 'Multisensory binding',
    mechanism: 'Merges ventral (what) and dorsal (how) stream outputs into unified multisensory representation. Cross-modal congruence detection (Beauchamp 2004).'
  },
  spt: {
    fullName: 'Sylvian Parietal-Temporal (Spt)',
    role: 'Auditory-motor interface',
    mechanism: 'Dorsal stream node bridging comprehension (Wernicke) to production (Broca). Maps acoustic targets to articulatory plans (Hickok & Poeppel 2007).'
  },
  motor_cortex: {
    fullName: 'Primary Motor Cortex (M1)',
    role: 'Final output execution',
    mechanism: 'Precentral gyrus. Final articulation stage in speech production (Levelt 1989). Output formatting and delivery.'
  },
  medial_pfc: {
    fullName: 'Medial PFC (mPFC)',
    role: 'Self-referential processing',
    mechanism: 'Medial wall of prefrontal cortex (BA 10/32). Manages agent self-model: identity, personality, values. Dual layer: SOUL.md schema + identity_facts knowledge graph (Northoff et al. 2006).'
  },
  tpj: {
    fullName: 'TPJ',
    role: 'Theory of Mind / User modeling',
    mechanism: 'Right temporoparietal junction. Manages user model: name, preferences, personality, relationship. Dual layer: USER.md schema + identity_facts knowledge graph (Frith & Frith 2006).'
  },
  insula: {
    fullName: 'Insula (Insular Cortex)',
    role: 'Interoceptive awareness & body-state monitoring',
    mechanism: 'Monitors neuromodulator levels to compute stress, energy, emotional awareness, and risk sensitivity. Feeds into ACC and PFC for decision-making (Craig 2009).',
  },
}

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
  visual_cortex: { active: '시각 처리 중', high_activity: '이미지 분석 중' },
  auditory_cortex_l: { active: '음성 처리 중', high_activity: '언어 분석 중' },
  auditory_cortex_r: { active: '음향 처리 중', high_activity: '감정 톤 분석 중' },
  wernicke: { active: '언어 이해 중', high_activity: '의미 분석 중' },
  broca: { active: '언어 생성 중', high_activity: '응답 구성 중' },
  brainstem: { active: '각성 조절 중', high_activity: '상태 전환 중' },
  vta: { active: '보상 평가 중', high_activity: '도파민 방출 중' },
  corpus_callosum: { active: '반구 간 통합 중', high_activity: '좌우뇌 동기화 중' },
  angular_gyrus: { active: '다감각 통합 중', high_activity: '교차모달 결합 중' },
  psts: { active: '다감각 바인딩 중', high_activity: '스트림 통합 중' },
  spt: { active: '운동 매핑 중', high_activity: '발화 계획 수립 중' },
  motor_cortex: { active: '출력 준비 중', high_activity: '응답 전달 중' },
  medial_pfc: { active: '자기모델 참조 중', high_activity: '정체성 처리 중' },
  tpj: { active: '사용자 모델링 중', high_activity: '마음 이론 처리 중' },
  insula: { active: '내수용 감각 처리 중', high_activity: '신체 상태 모니터링 중' },
}
