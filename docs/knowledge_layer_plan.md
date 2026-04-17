# Knowledge Layer Plan — Phase 0~7

**Date:** 2026-04-16
**Branch:** saju (local only)
**Status:** Phase 4 complete - moving to Phase 5

---

## 1. 목적

CBA의 기존 6-layer memory store (sensory/working/staging/episodic/semantic/procedural) 위에
**workspace-aware knowledge graph layer** 를 구축한다.

**최종 목표:** 유저가 multimodal (text/image/audio/PDF) 로 비즈니스 로직 정보를 제공하면,
이를 합리적 로직으로 저장·관리하여 코딩 에이전트가 MCP tool call 로 질의할 수 있게 한다.

**핵심 원칙:**
- **정보 무손실** — 원본은 항상 보존, 요약은 보조 레이어
- **논리적 비약에 질문** — ambiguity/contradiction 감지 시 유저에게 반드시 되물음
- **neuroscience 근거 유지** — 모든 변경/추가는 논문 근거 보유
- **broad 도메인 대응** — 도메인 어휘는 declared/extensible, 하드코딩 금지

---

## 2. 아키텍처 결정 (11개, 합의 완료)

| # | 결정 | 선택 | 근거 |
|---|---|---|---|
| 1 | 새 store 위치 | `brain_agent/memory/` sibling | Region이 아닌 memory 중앙집중 |
| 2 | Workspace 모델 | Multi-workspace + cross-reference | edge는 한 workspace 소유, target은 다른 workspace 가능 (`target_workspace_id`) |
| 3 | Ontology 등록 | Hybrid + 4-tier confidence (PROVISIONAL→STABLE→CANONICAL→USER_GROUND_TRUTH) | LLM self-confidence bias 방지, N회 재등장 시 STABLE 자동 승격 (C1) |
| 4 | Seed | Universal base + optional templates | 7 node types + 10 relation types 공통 base |
| 5 | identity_facts | Personal workspace 의 코드 adapter | 기존 테이블 유지, 양방향 우회 |
| 6 | 질문 발동 | Severity-tiered (minor=silent, moderate=append, severe=block) | ACC conflict monitoring 확장 |
| 7 | Raw vault | Hybrid (<10MB 복제, >=10MB pointer+SHA256) | 정보 무손실 원칙 |
| 8 | MCP | 스키마만 대비, tool 구현 보류 | 차후 별도 plan |
| 9 | Workspace 선택 | Session default + LLM override ask | Wernicke comprehension 에서 workspace hint 감지 |
| 10 | Decay | Workspace default + type-level override | none/slow/normal per workspace, type이 override |
| 11 | 추출 파이프라인 | Multi-stage adaptive (6단계) | Triage→Extract→**Temporal Resolve**→Validate→Severity→Broca (C3) |

---

## 3. 전체 데이터 흐름

```
User Input (text/image/audio/PDF)
  │
  ├─[Raw Vault]── ingest() → sources 테이블 + vault/<sha256> 파일 보존
  │
  ├─[Stage 1: Triage]── workspace 라우팅 + input 분류
  │   ├─ session default workspace 확인 (workspace_session 테이블)
  │   ├─ LLM/규칙으로 override 판단 → ask user if ambiguous
  │   └─ output: {target_workspace_id, input_kind, severity_hint, skip_stages}
  │
  ├─[Stage 2: Extract]── workspace ontology 제약 내 structured extraction
  │   ├─ system prompt: ontology_store.get_node_types(ws) + get_relation_types(ws)
  │   ├─ structured output → JSON schema validation
  │   ├─ output: {nodes[], edges[], new_type_proposals[], narrative_chunk}
  │   └─ new_type_proposals → ontology_store.propose_*()
  │
  ├─[Stage 2.5: Temporal Resolve]── "이전 fact 의 update" vs "진짜 contradiction" 판단 (C3)
  │   ├─ same (subject, relation) with different target 감지
  │   ├─ temporal markers 확인 (now/지금/현재/이제/예전에/이전에/used to …)
  │   ├─ markers 있음 → supersede: old.valid_to=now, new.valid_from=now
  │   ├─ markers 모호 → LLM classify_temporal 단일 call 로 판단
  │   └─ 진짜 contradiction 만 Stage 3 로 전달 (false-positive block 방지)
  │
  ├─[Stage 3: Validate]── 기존 facts 와 대조
  │   ├─ contradiction detection → contradictions_store.detect()
  │   ├─ missing premise detection → open_questions_store.add_question()
  │   └─ output: {validated_facts[], contradictions[], open_questions[]}
  │
  ├─[Stage 4: Severity Branch]
  │   ├─ severe (모순+핵심전제) → response_mode='block', 질문만 반환
  │   ├─ moderate (AMBIGUOUS 다수) → response_mode='append', 질문 덧붙임
  │   └─ minor/none → response_mode='normal'
  │
  ├─[Stage 5: Broca Refine]── personal workspace만, 기존 PSC refine 재활용
  │
  └─[Persist]── validated facts → **hippocampal_staging only** (C5: semantic/episodic 직접 적재 금지)
                ConsolidationEngine 이 staging → semantic/episodic 승격 담당
                (N회 재확인 + PROVISIONAL→STABLE 승격 조건 충족 시)
```

---

## 4. Cross-cutting Concerns (전 Phase 공통)

### 4.1 MemoryManager 통합

모든 새 store 는 `MemoryManager` 의 attribute 로 등록되어 lifecycle 관리됨.

**`brain_agent/memory/manager.py` 수정 사항:**

```python
class MemoryManager:
    def __init__(self, db_dir, embed_fn, ...):
        # 기존 store 유지
        self.sensory = SensoryBuffer()
        self.working = WorkingMemory(...)
        self.staging = HippocampalStaging(...)
        self.episodic = EpisodicStore(...)
        self.semantic = SemanticStore(...)
        self.procedural = ProceduralStore(...)
        self.brain_state = BrainStateStore(...)
        # 새 store 추가
        self.workspace = WorkspaceStore(db_path=os.path.join(db_dir, "workspaces.db"))
        self.ontology = OntologyStore(db_path=os.path.join(db_dir, "ontology.db"))
        self.raw_vault = RawVault(db_path=os.path.join(db_dir, "sources.db"),
                                   vault_dir=os.path.join(db_dir, "vault"))
        self.contradictions = ContradictionsStore(db_path=os.path.join(db_dir, "contradictions.db"))
        self.open_questions = OpenQuestionsStore(db_path=os.path.join(db_dir, "open_questions.db"))

    async def initialize(self):
        # 기존 초기화
        await self.staging.initialize()
        await self.episodic.initialize()
        await self.semantic.initialize()
        await self.procedural.initialize()
        await self.brain_state.initialize()
        # 새 store 초기화 (순서 중요: workspace → ontology → seed → 나머지)
        await self.workspace.initialize()       # personal workspace 자동 생성
        await self.ontology.initialize()         # 테이블 생성
        personal_ws = await self.workspace.get_or_create_personal()
        await self.ontology.seed_universal(personal_ws["id"])  # universal seed
        await self.raw_vault.initialize()
        await self.contradictions.initialize()
        await self.open_questions.initialize()

    async def close(self):
        # 기존 + 새 store 모두 close
        for store in [self.staging, self.episodic, self.semantic, self.procedural,
                       self.brain_state, self.workspace, self.ontology,
                       self.raw_vault, self.contradictions, self.open_questions]:
            await store.close()
```

### 4.2 BrainAgent 통합

`brain_agent/agent.py` 수정:
- `BrainAgent.__init__` 에서 `MemoryManager` 생성 시 변경 사항 없음 (manager 가 내부 처리)
- `ProcessingPipeline` 생성 시 `memory.workspace`, `memory.ontology` 등 접근 가능 (memory facade 통해)
- `BrainAgent.process()` 에서 `session_id` 를 workspace selection 에 전달

### 4.3 SessionManager 와 Workspace lifecycle

```
session_start(session_id)
  → workspace_store.get_session_workspace(session_id)
  → null 이면: 마지막 session 의 workspace 상속, 또는 personal fallback
  → pipeline 에 current_workspace_id 전달

session 중:
  → Stage 1 Triage 가 workspace override 감지 시
  → pipeline → PipelineResult.workspace_ask = "이 내용은 X workspace 로 보이는데 전환할까요?"
  → 유저 확인 → workspace_store.set_session_workspace(session_id, new_ws_id)

session_end(session_id)
  → workspace_session 레코드 유지 (다음 session 의 default 로 사용)
```

### 4.4 Backward Compatibility 전략

**원칙:** workspace_id 를 인자로 받지 않는 기존 코드 경로는 자동으로 `personal` workspace 로 동작.

구체적:
- `semantic_store.add_relationship()` — workspace_id 인자 추가, default='personal'
- `semantic_store.get_relationships()` — workspace_id 인자 추가, default=None (None=전체, 기존 동작)
- `semantic_store.search()` — workspace_id 인자 추가, default=None
- `episodic_store.save()` — workspace_id 인자 추가, default='personal'
- `episodic_store.get_recent()` — workspace_id 인자 추가, default=None
- `procedural_store.save()` — workspace_id 인자 추가, default='personal'
- `procedural_store.match()` — workspace_id 인자 추가, default=None

**모든 기존 호출자는 인자 안 넘기면 기존과 동일 동작.** 새 호출자만 workspace_id 명시.

### 4.5 데이터 마이그레이션

**스크립트:** `brain_agent/migrations/001_workspace_columns.py`

```
1. DB 백업: data_dir/*.db → data_dir/backup_YYYYMMDD/
2. workspace.db 생성 + personal workspace 삽입
3. ontology.db 생성 + universal seed 삽입
4. 기존 테이블에 ALTER TABLE ADD COLUMN:
   - knowledge_graph: workspace_id, target_workspace_id, source_ref, valid_from, valid_to, superseded_by, type_id
   - episodes: workspace_id, source_id, event_type, actor
   - procedures: workspace_id, trigger_embedding, applicable_scope
5. 기존 데이터 UPDATE SET workspace_id = 'personal'
6. INDEX 생성 (4.7 참조)
7. 무결성 검증: COUNT(*) before == COUNT(*) after
8. 롤백: backup 에서 복원
```

마이그레이션은 `MemoryManager.initialize()` 에서 자동 감지 + 실행. `brain_agent/migrations/` 디렉터리에 번호 순서로 관리. `brain_state.db` 에 `schema_version` 테이블 추가해서 적용된 migration 추적.

### 4.6 에러 처리 매트릭스

| 상황 | 처리 | 결과 |
|---|---|---|
| 존재하지 않는 workspace_id 참조 | ValueError 발생 | 호출자가 처리 |
| ontology proposal 이 미승인 상태에서 참조 | generic type (Concept/Entity) 로 fallback | 경고 로그 |
| raw vault 파일 누락 (pointer 모드) | `verify_integrity()` 실패, `integrity_valid=False` 마킹 | 유저에게 경고, 추출은 extracted_text 로 진행 |
| workspace 삭제 중 extraction 진행 | workspace_id FK 검증 → extraction 롤백 | 에러 반환 |
| 동일 SHA256 파일 재업로드 | dedup — 기존 source 반환 | 정상 (중복 방지) |
| Stage 2 에서 LLM 이 ontology 위반 출력 | JSON schema validation 실패 → retry 1회 → 실패 시 generic fallback | 경고 + generic type 적용 |
| Stage 3 에서 contradiction 의 양측 모두 EXTRACTED | severity='severe' 자동 | block-and-ask |

### 4.7 인덱싱 전략

모든 workspace-partitioned 테이블에 workspace_id 인덱스 추가:

```sql
CREATE INDEX idx_kg_workspace ON knowledge_graph(workspace_id);
CREATE INDEX idx_kg_workspace_source ON knowledge_graph(workspace_id, source_node);
CREATE INDEX idx_kg_workspace_target ON knowledge_graph(workspace_id, target_node);
CREATE INDEX idx_kg_target_workspace ON knowledge_graph(target_workspace_id);  -- M3: cross-ref reverse query
CREATE INDEX idx_kg_never_decay ON knowledge_graph(workspace_id, never_decay);  -- S8
CREATE INDEX idx_episodes_workspace ON episodes(workspace_id);
CREATE INDEX idx_episodes_workspace_interaction ON episodes(workspace_id, last_interaction);
CREATE INDEX idx_episodes_never_decay ON episodes(workspace_id, never_decay);   -- S8
CREATE INDEX idx_procedures_workspace ON procedures(workspace_id);
CREATE INDEX idx_contradictions_workspace ON contradictions(workspace_id, status);
CREATE INDEX idx_open_questions_workspace ON open_questions(workspace_id, severity);
CREATE INDEX idx_node_types_workspace ON node_types(workspace_id);
CREATE INDEX idx_node_types_confidence ON node_types(workspace_id, confidence);  -- C1
CREATE INDEX idx_relation_types_workspace ON relation_types(workspace_id);
CREATE INDEX idx_relation_types_confidence ON relation_types(workspace_id, confidence);  -- C1
CREATE INDEX idx_sources_workspace ON sources(workspace_id);
CREATE INDEX idx_sources_sha256 ON sources(sha256);
```

### 4.8 동시성

- SQLite WAL 모드 사용 (기존 설정 유지, `PRAGMA journal_mode=WAL`)
- 쓰기 직렬화: workspace 단위 write lock (Python `asyncio.Lock` per workspace_id)
- `pending_ontology_proposals` 에 UNIQUE(workspace_id, kind, proposed_name) 제약 → 중복 제안 방지
- `workspace_session` 쓰기는 session 단위로 serialize (같은 session 이 동시 write 안 함)

### 4.9 테스트 인프라

**`tests/conftest.py` 수정:**

```python
@pytest.fixture
async def memory_manager(tmp_path):
    """All tests get a MemoryManager with personal workspace auto-created."""
    mm = MemoryManager(db_dir=str(tmp_path), embed_fn=mock_embed, ...)
    await mm.initialize()  # personal workspace + universal seed 자동
    yield mm
    await mm.close()

@pytest.fixture
async def personal_workspace_id(memory_manager):
    ws = await memory_manager.workspace.get_or_create_personal()
    return ws["id"]
```

기존 621+ 테스트:
- 대부분 변경 불필요 (backward compat 보장, 기존 인터페이스 유지)
- `semantic_store` 직접 접근 테스트만 workspace_id default 동작 확인 추가
- 새 기능 테스트는 phase 별로 추가 (각 phase 의 테스트 섹션 참조)

### 4.10 CLI 확장

`brain_agent/cli/commands.py` 에 추가:

```
brain-agent workspace list              # 모든 workspace 목록
brain-agent workspace create <name>     # 새 workspace 생성
brain-agent workspace use <name>        # 현재 session 의 default workspace 전환
brain-agent workspace describe <name>   # workspace 상세 (ontology, stats)
brain-agent workspace delete <name>     # 삭제 (personal 불가)

brain-agent ontology list <workspace>         # 등록된 node_types + relation_types
brain-agent ontology pending <workspace>      # 미승인 proposal 목록
brain-agent ontology approve <proposal_id>    # 승인
brain-agent ontology reject <proposal_id>     # 거절

brain-agent questions list [workspace]        # 미답변 open questions
brain-agent questions answer <question_id> <answer>  # 답변
brain-agent contradictions list [workspace]   # 미해결 contradictions
```

### 4.11 Config 확장

`brain_agent/config/schema.py` 에 추가:

```python
class WorkspaceConfig(BaseModel):
    default_decay_policy: str = "normal"   # none | slow | normal
    vault_size_threshold_mb: int = 10
    vault_dir: str = "vault"

class ExtractionConfig(BaseModel):
    triage_model: str = "auto"     # haiku 급 작은 모델 or 규칙
    extract_model: str = "auto"    # 메인 모델
    max_retry: int = 1
    enable_severity_block: bool = True
```

### 4.12 Dashboard/WebSocket 변경

`brain_agent/dashboard/emitter.py` 에 추가 이벤트:

```python
# 새 이벤트 타입
"workspace_changed"         # {workspace_id, workspace_name}
"clarification_requested"   # {questions: [...], severity, workspace_id}
"contradiction_detected"    # {subject, value_a, value_b, severity}
"ontology_proposal"         # {kind, name, confidence, workspace_id}
```

### 4.13 Expression Mode 통합

`brain_agent/regions/prefrontal.py` 의 EXPRESSION_MODE_INSTRUCTION 수정:
- 현재: "You answer ONLY from facts stored in your memory"
- 변경: workspace-aware retrieval → "You answer ONLY from facts in the CURRENT WORKSPACE"
- open_questions 연동: fact 가 없으면 → open_questions_store.add_question() 호출 + "그건 아직 모르겠어" 응답
- contradictions 연동: 모순이 있는 fact 참조 시 → "이 부분은 두 가지 정보가 충돌해서 확인이 필요해" 응답

### 4.14 graph_analysis.py Workspace 대응

`brain_agent/memory/graph_analysis.py` 의 모든 함수 (`cluster_graph`, `hub_concepts`, `surprising_connections`, `graph_diff`, `assembly_coactivation`) 가 받는 NetworkX 그래프를 workspace 필터링된 것으로:
- `semantic_store.export_as_networkx(workspace_id=None)` → workspace_id 지정 시 해당 workspace 만 export
- cross-workspace edge 포함 여부: `include_cross_refs=True` 옵션

### 4.15 Retrieval Engine Workspace 대응

`brain_agent/memory/manager.py:retrieve()` 수정:
- workspace_id 인자 추가 (default=None → 전체, 기존 동작)
- semantic search: workspace 필터 적용
- episodic search: workspace 필터 적용
- spread_activation: workspace 내부 + cross-ref 옵션

### 4.16 Hippocampal Staging Workspace 대응

`brain_agent/memory/hippocampal_staging.py` 의 `staging_memories` 테이블:
```sql
ALTER TABLE staging_memories ADD COLUMN workspace_id TEXT DEFAULT 'personal';
```
- `encode()` 에 workspace_id 인자 추가, default='personal'
- `get_unconsolidated()` 에 workspace_id 필터 추가 (default=None → 전체)
- pattern separation 도 workspace 범위 내에서만 비교

### 4.17 Forgetting Engine + Dreaming Engine

**ForgettingEngine:** workspace decay policy 와 연동
- `apply_interference()`: workspace decay_policy='none' 이면 간섭 적용 안 함
- `retention()`: workspace decay_policy 에 따라 decay_rate 조절

**DreamingEngine:** **all-workspaces cross-domain integration** (S5 수정)
- REM sleep 의 cross-domain binding (Stickgold 2005, Walker & Stickgold 2006) 보존 — workspace 별 격리 금지
- `dream_cycle()`: 전체 memory pool 에서 dream. workspace_id 는 origin tracking 용
- 발생한 insight (novel edge/association) 는 "연관된 workspace 들 모두"에 cross-ref 로 등록
- business workspace 도 dream 필요 (다른 프로젝트에서의 유사 패턴 발견)

### 4.18 Retrieval-time Contradiction Monitoring (S1)

ACC 의 continuous conflict monitoring (Botvinick 2001) 을 extraction 시점뿐 아니라 **retrieval 시점**에도 적용:

- `MemoryManager.retrieve(..., workspace_id=None)` 이 결과 set 반환 전에 후처리
- 결과 subject 들에 대해 `contradictions_store.get_for_subject_batch(workspace_id, subject_ids)` 호출
- 결과 dict 에 `contradictions: list[dict]` 필드 포함 — retrieval 결과 중 모순 있는 subject 에 flag
- Expression mode (4.13) 는 이 flag 를 참고해 "이 부분은 두 가지 정보 충돌"로 응답 분기

### 4.19 Retrieval-as-Reconstruction (S2)

Bartlett (1932) schema theory: recall 은 verbatim 재생이 아닌 schema-guided reconstruction.
Plan 이 storage 완결성에 집중한 만큼 retrieval 쪽에서 명시적 mechanism 필요:

- `RetrievalResult` 에 추가 필드:
  - `gaps: list[dict]` — 검색된 fact 들의 missing required property (ontology schema 기반)
  - `inference_fill: list[dict]` — LLM 이 빈칸을 채우려 시도한 것 (명시적 표시, 실제 fact 와 구분)
- LLM 이 응답 생성 시 gaps 를 silent 하게 채우지 말고 `open_questions_store.add_question()` 으로 raise
- Expression mode 는 inference_fill 이 있을 때 "정확한 기록은 없고 추정한 부분" 명시

### 4.20 Source Type Distinction (S3)

Johnson, Hashtroudi & Lindsay (1993) source monitoring framework: source confusion 은 주요 memory error.
`sources.kind` 는 media type (pdf/image/audio) 이지 epistemic type 이 아님.

**추가 컬럼** (Phase 1 migration 에 포함):
- `knowledge_graph.epistemic_source TEXT DEFAULT 'asserted'`
  - `asserted` — 유저가 직접 주장
  - `cited` — 유저가 제3자를 인용
  - `inferred` — LLM 이 다른 fact 에서 유도
  - `observed` — 문서/이미지/센서 기록

Extraction Stage 2 prompt 에 epistemic_source 판정 포함 (각 edge 마다 명시).

### 4.21 Edge-level Importance Weighting (S7)

기존 CBA 의 event-level emotional boost (consolidation.py `ach_transfer_factor`) 를 workspace-level policy 로 단순화하면 amygdala (LeDoux 1996) 의 per-event modulation 해상도 손실.

**추가 컬럼** (Phase 1):
- `knowledge_graph.importance_score REAL DEFAULT 0.5` (0~1)

Stage 2 extraction 에서 LLM 이 edge 마다 importance 판정 (system prompt 에 기준 명시: 유저가 강조/반복/emotional language 사용 시 ↑).
Phase 6 decay 는 `decay_factor = workspace_policy × (1 - importance_score × 0.5)` 로 반영 — importance 높으면 decay 감속.

### 4.22 Information Never-Decay Flag (S8)

project_intent.md 의 "정보 손실 = 기능 회귀" 원칙과 plan 의 decay 구조 간 잠재 충돌.
Workspace routing 실수로 business 정보가 personal 에 들어가 decay 되는 사고 방지:

**추가 컬럼** (Phase 1):
- `knowledge_graph.never_decay INTEGER DEFAULT 0`
- `episodes.never_decay INTEGER DEFAULT 0`

Stage 2 extraction 에서 LLM 이 "이건 business logic / spec / requirement 다" signal 감지 시 `never_decay=1`.
ForgettingEngine/ConsolidationEngine 은 `never_decay=1` row 를 skip.

### 4.23 Multi-store Persistence Atomicity (M2)

`_persist()` 가 raw_vault / staging / contradictions / open_questions / ontology 5개 store 에 write — partial failure 시 state 꼬임.

- SQLite multi-DB transaction 은 복잡하므로 **logical transaction**:
  1. raw_vault 먼저 commit (원본 보존 최우선)
  2. 이후 store 들은 `source_ref = source["id"]` 로 연결
  3. 각 store write 실패 시 해당 row `status='partial'` 마킹, 후속 retry 가능
  4. `source_ref` 가 존재하는 source 를 가리키는지 무결성 체크 job (일간)

### 4.24 Input Multi-label (M6)

대화는 복합 intent — "오 맞다 (confirmation), 근데 수정하자 (correction + request)".
TriageResult.input_kind 를 단일 str → `list[str]` 로:
- 여러 kind 동시 가능
- skip_stages 는 kinds 의 union 으로 계산 (가장 보수적 선택)

### 4.25 Wernicke Role Annotation (M1)

Plan 5-4 에서 `workspace_hint` 를 Wernicke comprehension 결과에 추가. 하지만 Wernicke 는 phonological/semantic parsing 담당이지 pragmatic/discourse 는 TPJ/dlPFC.

**Mitigation (주석 수준):**
- `wernicke.py` 에서 `workspace_hint` 추출은 Wernicke 본연 기능이 아닌 "dlPFC-style pragmatic analysis" 를 Wernicke region 에 편의상 첨부한 것임을 comment 로 명시
- 향후 pragmatic 전담 region 분리 시 해당 기능을 이동 (별도 plan)

---

## 5. Phase 별 상세

### Phase 0 — Foundation

**목표:** workspace 추상화 + ontology 등록 + universal seed
**산출물:** 5 파일 (3 신규 + 2 수정)

#### 0-1. `brain_agent/memory/workspace_store.py` (신규)

**스키마:**
```sql
CREATE TABLE workspaces (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    decay_policy TEXT DEFAULT 'normal',   -- none | slow | normal
    template_id TEXT,
    template_version TEXT,                 -- e.g., "1.0" — Phase 7 upgrade 시 diff 기준 (M8)
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE workspace_session (
    session_id TEXT PRIMARY KEY,
    current_workspace_id TEXT NOT NULL,
    set_at TEXT
);
```

**API:**
```python
class WorkspaceStore:
    def __init__(self, db_path: str): ...
    async def initialize(self) -> None:
        """DB 생성 + personal workspace 자동 생성."""
    async def close(self) -> None: ...

    # CRUD
    async def create_workspace(self, name: str, description: str = "",
                                decay_policy: str = "normal",
                                template_id: str | None = None) -> dict:
        """workspace 생성. name UNIQUE 위반 시 ValueError."""
    async def get_workspace(self, name_or_id: str) -> dict | None:
        """name 또는 id 로 조회."""
    async def list_workspaces(self) -> list[dict]: ...
    async def update_workspace(self, ws_id: str, **kwargs) -> None:
        """decay_policy, description 등 수정."""
    async def delete_workspace(self, ws_id: str) -> None:
        """personal workspace 삭제 시도 시 ValueError."""

    # Personal
    async def get_or_create_personal(self) -> dict:
        """id='personal', name='Personal Knowledge', decay_policy='normal'."""

    # Session binding
    async def set_session_workspace(self, session_id: str, workspace_id: str) -> None: ...
    async def get_session_workspace(self, session_id: str) -> str | None:
        """null 이면 personal fallback."""
    async def get_last_workspace(self) -> str:
        """가장 최근 session 의 workspace. 없으면 'personal'."""
```

**에러 처리:**
- `create_workspace`: name UNIQUE 위반 → ValueError("Workspace '{name}' already exists")
- `delete_workspace`: personal 삭제 → ValueError("Cannot delete personal workspace")
- `set_session_workspace`: 존재하지 않는 workspace_id → ValueError

**테스트:**
- create → get → list → update → delete 기본 CRUD
- personal 자동 생성 (initialize 후 get_or_create_personal)
- personal 삭제 시도 → ValueError
- session binding: set → get → workspace 전환
- get_last_workspace: 빈 DB → 'personal' fallback

#### 0-2. `brain_agent/memory/ontology_store.py` (신규)

**스키마:**
```sql
CREATE TABLE node_types (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    parent_type_id TEXT,
    schema TEXT DEFAULT '{}',              -- JSON: {"props": [...], "required": [...]}
    decay_override TEXT,                   -- null = workspace default 사용
    confidence TEXT DEFAULT 'PROVISIONAL', -- PROVISIONAL | STABLE | CANONICAL | USER_GROUND_TRUTH (C1)
    occurrence_count INTEGER DEFAULT 1,    -- 재등장 시 ++, N 도달 시 STABLE 자동 승격 (C1)
    source_snippet TEXT,                    -- 원본 발화 audit trail, confidence 판정 근거 (C1)
    source_id TEXT DEFAULT 'seed',          -- seed | llm:extractor | user
    created_at TEXT,
    promoted_at TEXT,                       -- 마지막 confidence 승격 시점
    UNIQUE(workspace_id, name)
);

CREATE TABLE relation_types (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    domain_type_id TEXT,                   -- 허용 source node type id (null = any)
    range_type_id TEXT,                    -- 허용 target node type id (null = any)
    transitive INTEGER DEFAULT 0,
    symmetric INTEGER DEFAULT 0,
    inverse_of TEXT,                       -- relation_types.id
    confidence TEXT DEFAULT 'PROVISIONAL', -- 4-tier (C1)
    occurrence_count INTEGER DEFAULT 1,
    source_snippet TEXT,
    source_id TEXT DEFAULT 'seed',
    created_at TEXT,
    promoted_at TEXT,
    UNIQUE(workspace_id, name)
);

CREATE TABLE pending_ontology_proposals (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    kind TEXT NOT NULL,                   -- node_type | relation_type
    proposed_name TEXT NOT NULL,
    definition TEXT DEFAULT '{}',         -- JSON
    proposed_by TEXT DEFAULT 'llm:extractor',
    confidence TEXT,
    source_input TEXT,                    -- 원본 발화 snippet
    proposed_at TEXT,
    approved_by TEXT,
    approved_at TEXT,
    status TEXT DEFAULT 'pending',        -- pending | approved | rejected
    UNIQUE(workspace_id, kind, proposed_name)
);
```

**Universal ontology 가시성:**
- universal seed 는 `workspace_id='__universal__'` 특수 namespace 로 저장
- `get_node_types(workspace_id)`: workspace 고유 + `__universal__` 합산 반환
- `resolve_node_type(workspace_id, name)`: workspace 우선 → `__universal__` fallback

**is_a 상속 규칙:**
- transitive: A is_a B, B is_a C → A is_a C
- non-reflexive: A is_a A 금지
- DAG 강제: cycle 감지 시 거부
- `resolve_parent_chain(type_id)` → [type, parent, grandparent, ...]

**API:**
```python
class OntologyStore:
    def __init__(self, db_path: str): ...
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...

    # Seed
    async def seed_universal(self, workspace_id: str = "__universal__") -> None:
        """universal base ontology 삽입 (이미 존재하면 skip)."""

    # Registration (EXTRACTED auto-register)
    async def register_node_type(self, workspace_id: str, name: str,
                                  parent_name: str | None = None,
                                  schema: dict | None = None,
                                  decay_override: str | None = None,
                                  source_id: str = "llm:extractor") -> dict:
        """EXTRACTED confidence 의 새 type 자동 등록. 이미 존재하면 기존 반환."""
    async def register_relation_type(self, workspace_id: str, name: str,
                                      domain_type: str | None = None,
                                      range_type: str | None = None,
                                      transitive: bool = False,
                                      symmetric: bool = False,
                                      source_id: str = "llm:extractor") -> dict: ...

    # Proposal (INFERRED/AMBIGUOUS → approval queue)
    async def propose_node_type(self, workspace_id: str, name: str,
                                 definition: dict, confidence: str,
                                 source_input: str = "") -> dict:
        """pending_ontology_proposals 에 삽입. UNIQUE 위반 시 기존 proposal 반환."""
    async def propose_relation_type(self, workspace_id: str, name: str,
                                     definition: dict, confidence: str,
                                     source_input: str = "") -> dict: ...

    # Approval
    async def approve_proposal(self, proposal_id: str, approved_by: str = "user") -> dict:
        """proposal → 정식 node_type/relation_type 으로 이관. status='approved'."""
    async def reject_proposal(self, proposal_id: str) -> None:
        """status='rejected'."""
    async def list_pending(self, workspace_id: str) -> list[dict]: ...

    # Query
    async def get_node_types(self, workspace_id: str) -> list[dict]:
        """workspace 고유 + __universal__ 합산."""
    async def get_relation_types(self, workspace_id: str) -> list[dict]:
        """workspace 고유 + __universal__ 합산."""
    async def resolve_node_type(self, workspace_id: str, name: str) -> dict | None:
        """workspace 우선 → __universal__ fallback."""
    async def resolve_relation_type(self, workspace_id: str, name: str) -> dict | None: ...
    async def resolve_parent_chain(self, type_id: str) -> list[dict]:
        """is_a 체인 순회. cycle 감지 시 ValueError."""
    async def validate_node_properties(self, type_id: str, properties: dict) -> tuple[bool, list[str]]:
        """type schema 에 대해 properties 검증. (valid, errors)."""

    # Confidence lifecycle (C1)
    async def increment_occurrence(self, type_id: str) -> dict:
        """재등장 시 occurrence_count++. 임계치 (default N=3) 도달 시 PROVISIONAL→STABLE 자동 승격.
        반환: {confidence, occurrence_count, promoted: bool}."""
    async def promote_confidence(self, type_id: str, to_level: str,
                                   promoted_by: str = "system") -> None:
        """강제 승격. PROVISIONAL→STABLE (auto), STABLE→CANONICAL (user approve),
        CANONICAL→USER_GROUND_TRUTH (user explicit). 역방향 금지."""
    async def resolve_type_or_fallback(self, workspace_id: str, name: str,
                                         min_confidence: str = "PROVISIONAL") -> dict:
        """특정 confidence 이상의 type 만 resolve. 이하면 generic Concept/Entity fallback.
        Severity 판정이 PROVISIONAL 을 ground truth 로 취급하지 않도록 강제."""
```

**Confidence tier semantics (C1):**
- `PROVISIONAL`: LLM 이 추출한 신규 type, 1회 등장. **Severity 판정 시 INFERRED 동급 취급** (ground truth 아님).
- `STABLE`: N회 재등장 자동 승격. Severity 계산에 정상 반영.
- `CANONICAL`: 유저가 approve proposal 또는 `brain-agent ontology approve` 실행.
- `USER_GROUND_TRUTH`: 유저가 직접 declare (CLI 또는 explicit chat). 절대 decay 안 됨.

**Promotion 임계값:**
- `ExtractionConfig.promotion_threshold_n: int = 3` (config 로 조정 가능)
- occurrence_count >= threshold → PROVISIONAL→STABLE 자동 승격
- STABLE→CANONICAL 는 자동 승격 없음 (user action 필수)

**에러 처리:**
- `register_node_type`: parent 가 존재하지 않으면 ValueError
- `approve_proposal`: proposal status != 'pending' 이면 ValueError
- `resolve_parent_chain`: cycle 감지 → ValueError("Cycle detected in type hierarchy")
- `seed_universal`: 이미 seed 완료 → skip (idempotent). seed type 은 confidence='CANONICAL' 로 삽입.
- `promote_confidence`: 역방향 승격 시도 → ValueError

**테스트:**
- universal seed: 7 node types + 10 relation types 삽입 확인
- register: 새 type 자동 등록 + 중복 시 기존 반환
- propose → approve → 정식 등록 full flow
- propose → reject flow
- resolve: workspace 우선, __universal__ fallback
- parent chain: Entity → Person → custom type 순회
- parent chain cycle 감지
- property validation: required 필드 누락 → (False, ["missing: happened_at"])
- UNIQUE 제약: 같은 proposal 재제출 → 기존 반환

#### 0-3. `brain_agent/memory/ontology_seed.py` (신규)

```python
UNIVERSAL_NODE_TYPES = [
    {"name": "Entity",    "parent": None,     "schema": {},
     "description": "Root type for tangible things"},
    {"name": "Person",    "parent": "Entity",  "schema": {},
     "description": "A human actor or stakeholder"},
    {"name": "Artifact",  "parent": "Entity",  "schema": {},
     "description": "A created object (code, document, tool)"},
    {"name": "Event",     "parent": None,
     "schema": {"props": ["happened_at", "actor"], "required": ["happened_at"]},
     "description": "A temporal occurrence"},
    {"name": "Concept",   "parent": None,       "schema": {},
     "description": "An abstract idea or domain term"},
    {"name": "Statement", "parent": None,
     "schema": {"props": ["asserter", "confidence"], "required": []},
     "description": "A claim, decision, or assertion"},
    {"name": "Source",    "parent": None,
     "schema": {"props": ["uri", "sha256", "kind"], "required": []},
     "description": "Provenance pointer to original input"},
]

UNIVERSAL_RELATION_TYPES = [
    {"name": "is_a",          "transitive": True,  "symmetric": False,
     "description": "Type-subtype hierarchy"},
    {"name": "part_of",       "transitive": True,  "symmetric": False,
     "inverse_of": "has_part", "description": "Mereological containment"},
    {"name": "has_part",      "transitive": True,  "symmetric": False,
     "inverse_of": "part_of"},
    {"name": "refers_to",     "transitive": False, "symmetric": False,
     "description": "Generic reference/pointer"},
    {"name": "happened_at",   "transitive": False, "symmetric": False,
     "domain": "Event", "description": "Temporal anchoring"},
    {"name": "said_by",       "transitive": False, "symmetric": False,
     "domain": "Statement", "description": "Attribution of a claim"},
    {"name": "contradicts",   "transitive": False, "symmetric": True,
     "description": "Two statements that cannot both be true"},
    {"name": "supersedes",    "transitive": True,  "symmetric": False,
     "inverse_of": "superseded_by",
     "description": "Newer version replaces older"},
    {"name": "superseded_by", "transitive": True,  "symmetric": False,
     "inverse_of": "supersedes"},
    {"name": "derived_from",  "transitive": True,  "symmetric": False,
     "description": "Causal or logical derivation"},
]
```

#### 0-4. `brain_agent/memory/manager.py` 수정

Section 4.1 에 명시된 대로 새 store attribute + initialize/close 추가.

#### 0-5. 테스트

`tests/memory/test_workspace_store.py`:
- CRUD full cycle
- personal 자동 생성 + 삭제 방지
- session workspace binding + fallback

`tests/memory/test_ontology_store.py`:
- seed 삽입 + idempotent 재삽입 (seed 는 confidence='CANONICAL' 로 삽입됨 확인)
- register + propose + approve/reject (register 신규는 confidence='PROVISIONAL', occurrence_count=1, source_snippet 보존)
- hybrid 정책 검증 (PROVISIONAL 부터 시작, N회 재등장 시 STABLE 자동 승격)
- __universal__ 가시성 (모든 workspace 에서 보임)
- parent chain 순회 + cycle 감지
- property validation
- **C1 confidence lifecycle:**
  - increment_occurrence N-1 회 → PROVISIONAL 유지
  - increment_occurrence N 회째 → STABLE 자동 승격 + promoted_at 기록
  - promote_confidence: STABLE→CANONICAL, CANONICAL→USER_GROUND_TRUTH 정상 동작
  - promote_confidence 역방향 (CANONICAL→PROVISIONAL) → ValueError
  - resolve_type_or_fallback(min_confidence='STABLE'): PROVISIONAL 은 generic Concept fallback
  - source_snippet 보존 및 조회

`tests/memory/test_ontology_seed.py`:
- seed 상수 무결성 (7 types, 10 relations)
- parent 참조 유효성 (Person.parent == "Entity" 존재)
- inverse 쌍 일치 (part_of ↔ has_part)

---

### Phase 1 — Raw Vault + Schema Enrichment

**목표:** 원본 보존 + 기존 store 에 workspace/provenance 컬럼 추가
**산출물:** 2 신규 + 4 수정 + migration script

#### 1-1. `brain_agent/memory/raw_vault.py` (신규)

**스키마:**
```sql
CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    workspace_id TEXT,
    kind TEXT NOT NULL,                 -- user_utterance | pdf | image | audio | url | file | meeting | inference
    uri TEXT,                           -- original path or URL
    sha256 TEXT,
    vault_path TEXT,                    -- data_dir/vault/<sha256> (< 10MB) or null
    mime_type TEXT,
    original_filename TEXT,
    extracted_text TEXT,
    byte_size INTEGER,
    integrity_valid INTEGER DEFAULT 1,  -- 0 = 파일 누락/손상 감지됨
    last_verified TEXT,
    ingested_at TEXT
);

CREATE INDEX idx_sources_workspace ON sources(workspace_id);
CREATE INDEX idx_sources_sha256 ON sources(sha256);
```

**API:**
```python
class RawVault:
    VAULT_SIZE_THRESHOLD = 10 * 1024 * 1024  # 10MB

    async def ingest(self, workspace_id: str, kind: str,
                      data: bytes | None = None, path: str | None = None,
                      mime: str = "", filename: str = "",
                      extracted_text: str = "") -> dict:
        """원본 ingest. SHA256 dedup: 동일 hash 존재 시 기존 source 반환."""
    async def get_source(self, source_id: str) -> dict | None: ...
    async def get_raw_bytes(self, source_id: str) -> bytes | None:
        """vault 에서 원본 바이트 로드. pointer 모드면 원본 경로에서 읽기 시도."""
    async def verify_integrity(self, source_id: str) -> bool:
        """SHA256 재계산 대조. 실패 시 integrity_valid=0 마킹."""
    async def list_sources(self, workspace_id: str) -> list[dict]: ...
    async def find_by_sha256(self, sha256: str) -> dict | None: ...
```

#### 1-2. Schema migration

`brain_agent/migrations/001_workspace_columns.py`:

**knowledge_graph:**
```sql
ALTER TABLE knowledge_graph ADD COLUMN workspace_id TEXT DEFAULT 'personal';
ALTER TABLE knowledge_graph ADD COLUMN target_workspace_id TEXT;   -- cross-reference
ALTER TABLE knowledge_graph ADD COLUMN source_ref TEXT;             -- sources.id
ALTER TABLE knowledge_graph ADD COLUMN valid_from TEXT;
ALTER TABLE knowledge_graph ADD COLUMN valid_to TEXT;               -- null = current
ALTER TABLE knowledge_graph ADD COLUMN superseded_by TEXT;
ALTER TABLE knowledge_graph ADD COLUMN type_id TEXT;                -- node_types.id
ALTER TABLE knowledge_graph ADD COLUMN epistemic_source TEXT DEFAULT 'asserted';  -- asserted|cited|inferred|observed (S3)
ALTER TABLE knowledge_graph ADD COLUMN importance_score REAL DEFAULT 0.5;         -- 0~1, Stage 2 LLM 판정 (S7)
ALTER TABLE knowledge_graph ADD COLUMN never_decay INTEGER DEFAULT 0;              -- 1=decay/forgetting skip (S8)

CREATE INDEX idx_kg_workspace ON knowledge_graph(workspace_id);
CREATE INDEX idx_kg_workspace_source ON knowledge_graph(workspace_id, source_node);
CREATE INDEX idx_kg_workspace_target ON knowledge_graph(workspace_id, target_node);
CREATE INDEX idx_kg_target_workspace ON knowledge_graph(target_workspace_id);       -- M3: cross-ref reverse query
CREATE INDEX idx_kg_never_decay ON knowledge_graph(workspace_id, never_decay);      -- S8: decay skip filter
UPDATE knowledge_graph SET workspace_id = 'personal' WHERE workspace_id IS NULL;
```

**episodes:**
```sql
ALTER TABLE episodes ADD COLUMN workspace_id TEXT DEFAULT 'personal';
ALTER TABLE episodes ADD COLUMN source_id TEXT;        -- sources.id
ALTER TABLE episodes ADD COLUMN event_type TEXT DEFAULT 'conversation_turn';
ALTER TABLE episodes ADD COLUMN actor TEXT;
ALTER TABLE episodes ADD COLUMN importance_score REAL DEFAULT 0.5;   -- S7
ALTER TABLE episodes ADD COLUMN never_decay INTEGER DEFAULT 0;        -- S8

CREATE INDEX idx_episodes_workspace ON episodes(workspace_id);
CREATE INDEX idx_episodes_workspace_interaction ON episodes(workspace_id, last_interaction);
CREATE INDEX idx_episodes_never_decay ON episodes(workspace_id, never_decay);
UPDATE episodes SET workspace_id = 'personal' WHERE workspace_id IS NULL;
```

**procedures:**
```sql
ALTER TABLE procedures ADD COLUMN workspace_id TEXT DEFAULT 'personal';
ALTER TABLE procedures ADD COLUMN trigger_embedding TEXT;
ALTER TABLE procedures ADD COLUMN applicable_scope TEXT DEFAULT '{}';
ALTER TABLE procedures ADD COLUMN source_id TEXT;

CREATE INDEX idx_procedures_workspace ON procedures(workspace_id);
UPDATE procedures SET workspace_id = 'personal' WHERE workspace_id IS NULL;
```

**staging_memories:**
```sql
ALTER TABLE staging_memories ADD COLUMN workspace_id TEXT DEFAULT 'personal';
```

**ChromaDB:**
- 기존 `semantic_memory` collection 유지
- metadata 에 `workspace_id` 필드 추가 (신규 document 부터)
- search 시 `where={"workspace_id": ws_id}` 필터 지원
- 기존 document 은 metadata 에 `workspace_id` 없음 → `personal` 으로 간주 (코드 레벨 fallback)

#### 1-3. 기존 store API 확장 (backward compatible)

**semantic_store.py:**
- `add_relationship(..., workspace_id='personal', target_workspace_id=None, source_ref=None, valid_from=None)`
- `get_relationships(node, workspace_id=None)` — None=전체
- `search(query, top_k, workspace_id=None)` — ChromaDB metadata filter
- `add(content, ..., workspace_id='personal')` — ChromaDB metadata 에 workspace_id 추가
- `export_as_networkx(workspace_id=None, include_cross_refs=True)` — 필터링

**episodic_store.py:**
- `save(..., workspace_id='personal', source_id=None, event_type='conversation_turn', actor=None)`
- `get_recent(limit, workspace_id=None)` — None=전체

**procedural_store.py:**
- `save(..., workspace_id='personal', trigger_embedding=None)`
- `match(input_text, workspace_id=None)` — None=전체. trigger_embedding 있으면 semantic match 우선.

**hippocampal_staging.py:**
- `encode(..., workspace_id='personal')`
- `get_unconsolidated(workspace_id=None)`

#### 1-4. 테스트

- raw vault: text/image/large file ingest, SHA256 dedup, integrity check, pointer 모드 파일 누락 감지
- migration: 기존 DB 에 ALTER 적용 후 데이터 무결성
- backward compat: 기존 코드 (workspace_id 안 넘김) 동작 불변
- cross-workspace edge: workspace A → B 참조 생성 + 조회
- ChromaDB workspace filter: workspace_id 로 검색 범위 제한

---

### Phase 2 — Contradiction & Open Question Stores

**Status:** Phase 2 complete - stores ready for Phase 3 / Phase 5 wiring

**목표:** severity-tier 발동의 저장 측
**산출물:** 2 신규

#### 2-1. `brain_agent/memory/contradictions_store.py` (신규)

**스키마:**
```sql
CREATE TABLE contradictions (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    subject_node TEXT NOT NULL,
    key_or_relation TEXT NOT NULL,
    value_a TEXT NOT NULL,
    value_a_source TEXT,                -- sources.id
    value_a_confidence TEXT,
    value_b TEXT NOT NULL,
    value_b_source TEXT,
    value_b_confidence TEXT,
    severity TEXT DEFAULT 'moderate',    -- minor | moderate | severe
    status TEXT DEFAULT 'open',          -- open | resolved | dismissed
    detected_at TEXT,
    resolved_at TEXT,
    resolved_by TEXT,
    resolution TEXT,
    resolution_confidence TEXT
);

CREATE INDEX idx_contradictions_workspace ON contradictions(workspace_id, status);
```

**Severity 산정 알고리즘:**
```
severity = 'minor'   if both confidences contain INFERRED and subject is not in core_node_set
severity = 'moderate' if one side is EXTRACTED, other is INFERRED
severity = 'severe'  if both sides are EXTRACTED
                      OR subject is in workspace's core_node_set (high-degree hub nodes)
                      OR key_or_relation is a 'supersedes' or 'contradicts' relation
```

**API:**
```python
class ContradictionsStore:
    async def detect(self, workspace_id: str, subject: str, key_or_relation: str,
                      value_a: str, value_b: str,
                      value_a_source: str = "", value_b_source: str = "",
                      value_a_confidence: str = "INFERRED",
                      value_b_confidence: str = "INFERRED") -> dict:
        """contradiction 등록 + severity 자동 산정."""
    async def resolve(self, contradiction_id: str, resolution: str,
                       resolved_by: str = "user",
                       resolution_confidence: str = "EXTRACTED") -> None: ...
    async def dismiss(self, contradiction_id: str) -> None: ...
    async def list_open(self, workspace_id: str) -> list[dict]: ...
    async def list_by_severity(self, workspace_id: str, severity: str) -> list[dict]: ...
    async def get_for_subject(self, workspace_id: str, subject: str) -> list[dict]: ...
    async def get_for_subject_batch(self, workspace_id: str,
                                      subject_ids: list[str]) -> dict[str, list[dict]]:
        """(S1) Retrieval 결과 set 의 subject 들에 대해 open contradictions 일괄 조회.
        Phase 5 pipeline 이 `manager.retrieve()` 결과의 subject 들을 모아서 단일 query 로 호출.
        반환: {subject_id: [contradiction_dict, ...]}. subject_id 가 누락된 경우 빈 list."""
```

#### 2-2. `brain_agent/memory/open_questions_store.py` (신규)

**스키마:**
```sql
CREATE TABLE open_questions (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    question TEXT NOT NULL,
    raised_by TEXT NOT NULL,             -- ambiguity_detector | unknown_fact | contradiction | user
    context_node TEXT,
    context_input TEXT,                  -- 원본 발화 snippet
    severity TEXT DEFAULT 'moderate',    -- minor | moderate | severe
    blocking INTEGER DEFAULT 0,          -- 1 = response_mode='block'
    asked_at TEXT,
    answered_at TEXT,
    answer TEXT,
    answer_source TEXT                   -- sources.id
);

CREATE INDEX idx_open_questions_workspace ON open_questions(workspace_id, severity);
CREATE INDEX idx_open_questions_blocking ON open_questions(workspace_id, blocking);
```

**blocking 규칙:**
```
blocking = 1  if severity == 'severe'
blocking = 0  otherwise
```

**API:**
```python
class OpenQuestionsStore:
    async def add_question(self, workspace_id: str, question: str,
                            raised_by: str, severity: str = "moderate",
                            context_node: str = "", context_input: str = "") -> dict:
        """question 등록. severity='severe' → blocking=1 자동."""
    async def answer_question(self, question_id: str, answer: str,
                               answer_source: str = "") -> None: ...
    async def list_unanswered(self, workspace_id: str) -> list[dict]: ...
    async def list_blocking(self, workspace_id: str) -> list[dict]:
        """blocking=1 AND answered_at IS NULL."""
    async def list_by_severity(self, workspace_id: str, severity: str) -> list[dict]: ...
    async def count_blocking(self, workspace_id: str) -> int: ...
```

#### 2-3. 테스트

- contradiction CRUD + severity 자동 산정
- resolve + dismiss flow
- severity 산정: EXTRACTED+EXTRACTED → severe, INFERRED+INFERRED → minor
- **get_for_subject_batch: 다수 subject 일괄 조회 (retrieval 후처리용, S1)**
- open question: add + answer + list_blocking
- blocking 자동 설정: severity='severe' → blocking=1
- count_blocking

---

### Phase 3 — Multi-Stage Adaptive Extractor

**목표:** PSC 분해, workspace-aware structured extraction
**산출물:** `brain_agent/extraction/` 모듈

#### 3-0. ExtractionResult 타입 정의

```python
@dataclass
class ExtractionResult:
    workspace_id: str
    source_id: str                           # raw_vault source
    nodes: list[dict]                        # [{type, label, properties, confidence}]
    edges: list[dict]                        # [{source, relation, target, confidence, target_workspace_id}]
    contradictions: list[dict]               # [{subject, key, value_a, value_b, severity}]
    open_questions: list[dict]               # [{question, raised_by, severity}]
    new_type_proposals: list[dict]           # [{kind, name, definition, confidence}]
    narrative_chunk: str                      # 원문 보존
    response_text: str                        # Broca 정제 결과 (personal only)
    response_mode: str = "normal"             # normal | append | block
    clarification_questions: list[str] = field(default_factory=list)
```

#### 3-1. Stage 1 — Triage (`brain_agent/extraction/triage.py`)

**입력:** raw text + session_id + current workspace_id
**출력:**
```python
@dataclass
class TriageResult:
    target_workspace_id: str      # 확정된 workspace
    input_kinds: list[str]        # (M6) multi-label: [greeting|spec_drop|question|correction|confirmation|...]
    severity_hint: str            # none | minor | moderate | severe
    skip_stages: list[int]        # e.g., [2,3] for greetings. 여러 kind 의 union — 가장 보수적 선택
    workspace_ask: str | None     # "이 내용은 X workspace 로 보이는데?" (null=확실)
```

**로직:**
1. `workspace_store.get_session_workspace(session_id)` → current_ws
2. 입력 분류 — **multi-label** (M6, 규칙 기반 우선, 필요시 작은 모델):
   - pattern matching 으로 여러 label 동시 수집 — 예: "오 맞다 (confirmation), 근데 수정하자 (correction + request)"
   - labels 후보: `greeting | farewell | confirmation | question | correction | spec_drop | request`
   - skip_stages 는 kinds union 에 대해 가장 보수적으로 선택:
     - 모든 kind 가 skip 허용 시에만 skip (예: `{greeting}` 만 있으면 skip_stages=[2,3])
     - correction/spec_drop/request 이 하나라도 있으면 skip_stages=[]
   - 빈 list 방지: 어떤 pattern 도 매칭 안 되면 `["spec_drop"]` default
3. Workspace override 감지:
   - 텍스트에 다른 workspace 이름 직접 언급 → override 후보
   - LLM comprehension hint (Phase 5 에서 Wernicke 가 제공) → confidence > 0.8 이면 override
   - override 시 `workspace_ask` 채워서 유저에게 확인

#### 3-2. Stage 2 — Workspace-Aware Extract (`brain_agent/extraction/extractor.py`)

**입력:** text + TriageResult + workspace ontology
**출력:** nodes[], edges[], new_type_proposals[], narrative_chunk

**System prompt 구성:**
```
You are a knowledge extractor. Given user input, extract structured facts.

## Available Node Types (use ONLY these, or propose new ones)
{ontology_store.get_node_types(workspace_id) → formatted list}

## Available Relation Types (use ONLY these, or propose new ones)
{ontology_store.get_relation_types(workspace_id) → formatted list}

## Output format (JSON)
{JSON schema definition}

## Rules
- Use existing types when possible
- If a concept doesn't fit any type, add to "new_type_proposals" with confidence
- Confidence: EXTRACTED (explicitly stated), INFERRED (logically derived), AMBIGUOUS (uncertain)
- All labels in English lowercase
- For cross-workspace references, specify target_workspace_id
- **epistemic_source (S3)** per edge: asserted (user claims) | cited (user quotes someone) | inferred (logical derivation) | observed (from document/image/sensor)
- **importance_score (S7)** per edge, 0~1: higher when user emphasizes ("중요한 건", "절대", repeated), emotional language, or explicit priority. Default 0.5.
- **never_decay (S8)** per edge, 0|1: set 1 when edge is business logic / spec / requirement / decision that must not be forgotten. Default 0.
```

**JSON schema validation:**
1. LLM 출력 파싱
2. 각 node 의 type 이 workspace ontology 에 존재하는지 확인
3. 각 edge 의 relation 이 ontology 에 존재하는지 확인
4. 위반 시: retry 1회 (구체적 에러 메시지 포함) → 재실패 시 generic type (Concept/refers_to) fallback
5. node properties 를 `ontology_store.validate_node_properties()` 로 검증

**new_type_proposals 처리:**
- confidence=EXTRACTED → `ontology_store.register_node_type()` 즉시 등록
- confidence=INFERRED/AMBIGUOUS → `ontology_store.propose_node_type()` 큐

#### 3-2.5. Stage 2.5 — Temporal Resolution (`brain_agent/extraction/temporal_resolver.py`) (신규, C3)

**목적:** 유저의 **상태 변화** ("예전엔 X 지금은 Y") 가 Stage 3 Validator 에서 severe contradiction 으로 오판되어 auto-block 되는 사고 방지. Conway (2005) self-memory system 의 time-indexed fact 개념 반영.

**입력:** Stage 2 output (nodes[], edges[]) + narrative_chunk + workspace_id
**출력:**
```python
@dataclass
class TemporalResolveResult:
    update_ops: list[dict]          # 기존 edge supersede/reinforce 작업
    new_edges: list[dict]            # Stage 3 로 전달할 신규 edge
    reinforced_edges: list[dict]    # 동일 fact 재확인 (PROVISIONAL→STABLE 승격 신호)
```

**알고리즘:**
```python
TEMPORAL_MARKERS_CURRENT = ["now", "지금", "현재", "이제", "오늘", "요즘", "currently"]
TEMPORAL_MARKERS_PAST    = ["예전에", "이전에", "옛날엔", "before", "used to", "previously", "작년", "지난달"]

for edge in extracted.edges:
    existing = await semantic_store.get_relationships(edge["source"], workspace_id)
    same_sr = [ex for ex in existing if ex["relation"] == edge["relation"] and ex["valid_to"] is None]

    if not same_sr:
        result.new_edges.append(edge)
        continue

    ex = same_sr[0]

    # Case A: 동일 target 재확인 → reinforce (occurrence_count++)
    if ex["target"] == edge["target"]:
        result.reinforced_edges.append(ex)
        continue

    # Case B: Different target — temporal update vs contradiction 판단
    has_current = any(m in narrative_chunk.lower() for m in TEMPORAL_MARKERS_CURRENT)
    has_past    = any(m in narrative_chunk.lower() for m in TEMPORAL_MARKERS_PAST)

    if has_current or has_past:
        # 명시적 temporal update
        result.update_ops.append({
            "type": "supersede",
            "edge_id": ex["id"],
            "valid_to": now_iso(),
        })
        result.new_edges.append({**edge, "valid_from": now_iso()})
        continue

    # Case C: 모호 — LLM 단일 call 로 판단
    judgment = await llm.classify_temporal(
        old_fact=ex, new_fact=edge, context=narrative_chunk,
    )  # returns "update" | "contradiction" | "ambiguous"

    if judgment == "update":
        result.update_ops.append({"type": "supersede", "edge_id": ex["id"], "valid_to": now_iso()})
        result.new_edges.append({**edge, "valid_from": now_iso()})
    elif judgment == "contradiction":
        # Stage 3 로 넘겨서 severity 계산
        result.new_edges.append(edge)
    else:  # ambiguous
        # Stage 3 로 보내되 hint 첨부 → open_question 으로 이어짐
        edge["temporal_ambiguous"] = True
        result.new_edges.append(edge)
```

**LLM classify_temporal prompt 요약:**
```
Given old_fact and new_fact with same subject+relation but different target,
classify: Is the user expressing a state change (update), or do they hold both
claims simultaneously (contradiction), or is it unclear (ambiguous)?
Context: {narrative_chunk}
Return single word: update | contradiction | ambiguous
```

**Stage 3 와의 관계:**
- `reinforced_edges` → Stage 3 skip, 바로 _persist 단계에서 `ontology.increment_occurrence()` 호출
- `update_ops` → Stage 3 skip, _persist 에서 기존 edge `valid_to` 설정 + 새 edge 삽입
- `new_edges` → Stage 3 로 전달 (진짜 contradiction 만 severity 계산)

**테스트:**
- 명시적 past marker: "이전엔 Python 이었는데 지금은 Go" → supersede + new edge
- 동일 target 재확인: "나는 여전히 Python 쓴다" → reinforced (occurrence_count 증가 신호)
- Markers 없음 + LLM=update 판정: 1회 LLM call, 결과 supersede
- Markers 없음 + LLM=contradiction: Stage 3 로 전달
- Markers 없음 + LLM=ambiguous: temporal_ambiguous=True 로 태깅, open_question 유도
- 기존 valid_to 설정된 edge 는 same_sr 에서 제외 (이미 superseded)
- Stage 2 output 없는 경우 (skip_stages=[2]) → Stage 2.5 도 skip

#### 3-3. Stage 3 — Validate (`brain_agent/extraction/validator.py`)

**입력:** Stage 2 output + 기존 workspace facts
**출력:** validated_facts[], contradictions[], open_questions[]

**Contradiction detection 알고리즘:**
```python
for each extracted edge (subject, relation, target):
    existing = semantic_store.get_relationships(subject, workspace_id)
    for ex in existing:
        if ex.relation == relation and ex.target != target:
            # Same subject+relation, different target
            # Check temporal: if ex.valid_to is set → already superseded, skip
            # Check alias: fuzzy match target vs ex.target → skip if alias
            # Otherwise: contradiction
            severity = compute_severity(
                new_confidence=edge.confidence,
                existing_confidence=ex.confidence,
                subject_is_hub=is_hub_node(subject, workspace_id),
            )
            contradictions.append({...})
```

**Missing premise detection:**
```python
for each extracted node:
    if node.type has required properties (from ontology schema):
        for required_prop in schema["required"]:
            if required_prop not in node.properties:
                open_questions.append({
                    "question": f"{node.label}의 {required_prop}이(가) 명시되지 않았습니다. 어떤 값인가요?",
                    "raised_by": "ambiguity_detector",
                    "severity": "moderate",
                })
```

**주의: 과도한 질문 방지**
- 한 추출에서 생성되는 open_questions 는 최대 3개로 제한
- severity='severe' 우선, 나머지는 silent queue

**Pattern separation (S4) — 유사 Event 오판 방지:**

Dentate gyrus pattern separation (Yassa & Stark 2011) 을 event node 병합 방지에 반영:
```python
for node in extracted.nodes:
    if node.type == "Event" or is_subtype_of(node.type, "Event"):
        # 기존 Event node 중 happened_at 이 fuzzy match 하는 것 조회
        candidates = await semantic_store.find_events_near(
            workspace_id, node.properties.get("happened_at"),
            window_hours=24,
        )
        if candidates:
            for c in candidates:
                label_sim = fuzzy_ratio(node.label, c["label"])
                if label_sim > 0.75 and c["id"] != node.id:
                    # 유사 event — merge 인지 distinct 인지 확인 필요
                    open_questions.append({
                        "question": f"'{node.label}' ({node.properties['happened_at']}) 는 "
                                     f"'{c['label']}' ({c['happened_at']}) 와 같은 사건인가요, "
                                     f"다른 사건인가요?",
                        "raised_by": "pattern_separation",
                        "severity": "moderate",
                        "context_node": node.label,
                    })
                    break  # 가장 유사한 후보 1건만 질문 (과도한 질문 방지)
```

**Feeling-of-Knowing integration (S6) — pre-retrieval metamemory:**

Hart (1965) FOK 는 pre-retrieval phenomenon — "알 것 같다/모를 것 같다" 를 검색 시도 *전에* 판단.
현재 설계의 open_questions_store 는 post-hoc detection 이므로 retrieval 쪽에서 pre-retrieval FOK 구현 (Phase 5 상세). Validator 단계에서는 **pre-retrieval signal** 을 생성:
```python
# Extraction 시 "유저가 묻고 있는데 우리 KG 에 관련 fact 없음" 감지
if "question" in triage.input_kinds:
    hits = await semantic_store.search(text, workspace_id, top_k=5)
    if not hits or max(h["similarity"] for h in hits) < 0.3:
        open_questions.append({
            "question": f"'{text[:80]}' 에 대한 정보가 없습니다. 어떤 input 을 주시면 답할 수 있을까요?",
            "raised_by": "fok_pre_retrieval",
            "severity": "moderate",
        })
```

#### 3-4. Stage 4 — Severity Branch (`brain_agent/extraction/severity.py`)

**입력:** contradictions[] + open_questions[]
**출력:** response_mode + clarification_questions[]

```python
def compute_response_mode(contradictions, open_questions):
    has_severe = any(c["severity"] == "severe" for c in contradictions)
    has_blocking_q = any(q["severity"] == "severe" for q in open_questions)

    if has_severe or has_blocking_q:
        return "block"
    elif contradictions or any(q["severity"] == "moderate" for q in open_questions):
        return "append"
    else:
        return "normal"
```

**Stage 5 transition:**
- `block` → Stage 5 skip. ExtractionResult.response_text = "" (비어있음). clarification_questions 만 채움.
- `append` → Stage 5 실행 (personal only). clarification_questions 도 채움.
- `normal` → Stage 5 실행 (personal only). clarification_questions 비어있음.

#### 3-5. Stage 5 — Broca Refine (`brain_agent/extraction/refiner.py`)

- personal workspace 전용
- 기존 PSC 의 `refined_response` 부분 재활용
- 입력: agent response text + language
- 출력: polished text
- business workspace → skip (response_text = "")

#### 3-6. Orchestrator (`brain_agent/extraction/orchestrator.py`)

```python
class ExtractionOrchestrator:
    def __init__(self, memory: MemoryManager, llm_provider, config: ExtractionConfig): ...

    async def extract(self, text: str, image: bytes | None = None,
                       audio: bytes | None = None,
                       session_id: str = "",
                       comprehension: dict | None = None) -> ExtractionResult:
        # 0. Raw vault ingest
        source = await self.memory.raw_vault.ingest(
            workspace_id=triage.target_workspace_id,
            kind="user_utterance", data=text.encode(), ...
        )

        # 1. Triage
        triage = await self._triage(text, session_id)

        # 2. Extract (skip if triage says so)
        if 2 not in triage.skip_stages:
            extracted = await self._extract(text, triage.target_workspace_id)
        else:
            extracted = empty_extraction()

        # 2.5. Temporal Resolution (C3) — update vs contradiction 판단
        if 2 not in triage.skip_stages and extracted.edges:
            temporal = await self._temporal_resolve(
                extracted, text, triage.target_workspace_id,
            )
        else:
            temporal = empty_temporal()

        # 3. Validate (skip if triage says so) — temporal.new_edges 만 validation 대상
        if 3 not in triage.skip_stages:
            validated = await self._validate(
                extracted._replace(edges=temporal.new_edges),
                triage.target_workspace_id,
            )
        else:
            validated = empty_validation()

        # 4. Severity branch
        severity_result = self._severity_branch(validated)

        # 5. Broca refine (personal only, not blocked)
        response_text = ""
        if severity_result.response_mode != "block":
            ws = await self.memory.workspace.get_workspace(triage.target_workspace_id)
            if ws and ws["name"] == "Personal Knowledge":
                response_text = await self._refine(...)

        # 6. PERSIST — C5: staging only, ConsolidationEngine 이 승격 담당
        await self._persist(triage, extracted, validated, temporal, source)

        return ExtractionResult(
            workspace_id=triage.target_workspace_id,
            source_id=source["id"],
            nodes=extracted.nodes,
            edges=extracted.edges,
            contradictions=validated.contradictions,
            open_questions=validated.open_questions,
            new_type_proposals=extracted.new_type_proposals,
            narrative_chunk=text,
            response_text=response_text,
            response_mode=severity_result.response_mode,
            clarification_questions=severity_result.questions,
        )

    async def _persist(self, triage, extracted, validated, temporal, source):
        """추출 결과를 적절한 store 에 적재. (C5) semantic/episodic 직접 적재 금지 — staging 만 경유.
        ConsolidationEngine 이 N회 재확인 + PROVISIONAL→STABLE 승격 조건 달성 시 semantic/episodic 으로 승격."""
        ws_id = triage.target_workspace_id

        # 1. raw_vault 먼저 commit (4.23 logical transaction: 원본 보존 최우선)
        #    source 는 이미 orchestrator 진입 시 ingest 됨 (이 함수는 그 뒤)

        # 2. Reinforced edges (Stage 2.5 결과) → ontology occurrence 증가 + staging reinforce
        for ex_edge in temporal.reinforced_edges:
            if ex_edge.get("type_id"):
                await self.memory.ontology.increment_occurrence(ex_edge["type_id"])
            # Staging 의 기존 trace 도 strength boost (consolidation 이 활용)
            await self.memory.staging.reinforce(
                workspace_id=ws_id, edge_id=ex_edge["id"], source_id=source["id"],
            )

        # 3. Update ops (Stage 2.5) → 기존 edge valid_to 설정
        for op in temporal.update_ops:
            if op["type"] == "supersede":
                await self.memory.semantic.mark_superseded(
                    edge_id=op["edge_id"], valid_to=op["valid_to"],
                )  # valid_to 만 UPDATE — new fact 는 아래 staging 으로

        # 4. New edges → **hippocampal_staging only** (C5: semantic 직접 write 금지)
        for edge in temporal.new_edges:
            await self.memory.staging.encode_edge(
                workspace_id=ws_id,
                source_node=edge["source"],
                relation=edge["relation"],
                target_node=edge["target"],
                target_workspace_id=edge.get("target_workspace_id"),
                source_ref=source["id"],
                confidence=edge["confidence"],             # PROVISIONAL 주로
                epistemic_source=edge.get("epistemic_source", "asserted"),  # S3
                importance_score=edge.get("importance_score", 0.5),         # S7
                never_decay=edge.get("never_decay", 0),                      # S8
                valid_from=now_iso(),
                temporal_ambiguous=edge.get("temporal_ambiguous", False),
            )

        # 5. Contradictions → contradictions_store (별도 store, staging 경유 안 함)
        for c in validated.contradictions:
            await self.memory.contradictions.detect(
                workspace_id=ws_id, subject=c["subject"],
                key_or_relation=c["key"], value_a=c["value_a"],
                value_b=c["value_b"], ...
            )

        # 6. Open questions → open_questions_store
        for q in validated.open_questions:
            await self.memory.open_questions.add_question(
                workspace_id=ws_id, question=q["question"],
                raised_by=q["raised_by"], severity=q["severity"],
                context_input=q.get("context_input", ""),
            )

        # 7. New type proposals → ontology_store (C1: PROVISIONAL 로 register, INFERRED/AMBIGUOUS 는 propose)
        for p in extracted.new_type_proposals:
            # register 는 자동으로 confidence='PROVISIONAL' + occurrence_count=1 + source_snippet 기록
            if p["confidence"] == "EXTRACTED":
                await self.memory.ontology.register_node_type(
                    ws_id, p["name"], source_snippet=p.get("source_snippet", ""),
                )
            else:
                await self.memory.ontology.propose_node_type(
                    ws_id, p["name"], definition=p["definition"],
                    confidence=p["confidence"], source_input=p.get("source_snippet", ""),
                )

        # 8. Narrative chunk → hippocampal_staging (C5: episodic 직접 적재 금지)
        await self.memory.staging.encode(
            content=extracted.narrative_chunk,
            entities={"nodes": [n["label"] for n in extracted.nodes]},
            workspace_id=ws_id,
            source_id=source["id"],
            event_type=triage.input_kinds[0] if triage.input_kinds else "unknown",
            importance_score=extracted.importance_score_avg,   # S7: edges 평균
            never_decay=any(e.get("never_decay") for e in temporal.new_edges),  # S8
        )
```

**ConsolidationEngine 승격 조건 (C5 + C1 연동):**
- staging → semantic/episodic 승격은 `ConsolidationEngine.consolidate()` 에서:
  1. staging trace 의 cumulative strength 가 threshold 이상 (기존 로직)
  2. 관련 node/relation types 중 최소 하나가 STABLE 이상 (C1)
  3. `never_decay=1` 또는 `importance_score > 0.7` 이면 threshold 완화
- Single-extraction fact 는 staging 에만 존재 → 재등장 없으면 forget 가능 (PROVISIONAL 특성 반영)

#### 3-7. 테스트

- Triage: greeting → skip, spec_drop → full, question → partial
- **Triage multi-label (M6): "오 맞다, 근데 수정하자" → input_kinds=[confirmation, correction] + skip_stages=[]**
- Triage: workspace override 감지 → workspace_ask 반환
- Extract: ontology 내 type → 정상 추출
- Extract: ontology 외 type → new_type_proposals
- Extract: JSON schema validation fail → retry → fallback
- **Extract: edge 에 epistemic_source/importance_score/never_decay 포함 (S3/S7/S8)**
- **Temporal Resolve (Stage 2.5, C3): 명시적 past marker → supersede**
- **Temporal Resolve: 동일 target 재확인 → reinforced_edges (occurrence_count 증가 신호)**
- **Temporal Resolve: markers 없음 + LLM=update → supersede**
- **Temporal Resolve: markers 없음 + LLM=contradiction → Stage 3 로 전달 (severe auto-block 회피)**
- Validate: 모순 감지 (same subject+relation, different target 이면서 temporal=False)
- Validate: missing required property → open_question
- Validate: 과도한 질문 방지 (max 3)
- **Validate pattern separation (S4): 유사 Event + happened_at 근접 → merge/distinct 질문**
- **Validate FOK pre-retrieval (S6): question input + KG hit=0 → fok_pre_retrieval question**
- Severity: severe → block, moderate → append, none → normal
- Orchestrator: full end-to-end (mock LLM)
- **Orchestrator: _persist 는 staging 만 write, semantic/episodic 직접 write 없음 (C5)**
- Orchestrator: block mode — response_text 비어있고 questions 만 반환

---

### Phase 4 — Personal Workspace Adapter

**Status:** Phase 4 complete - personal workspace adapter live, identity_facts callers untouched

**목표:** 기존 identity_facts 를 personal workspace 인터페이스로 감싸기
**산출물:** `brain_agent/memory/personal_adapter.py`

#### 4-1. Adapter 구현

```python
class PersonalAdapter:
    """identity_facts 테이블과 workspace node 인터페이스 간 양방향 어댑터."""

    def __init__(self, workspace_store, ontology_store, semantic_store): ...

    # 기존 호출자용 (backward compat)
    async def get_user_facts(self) -> list[dict]:
        """= semantic_store.get_identity_facts("user_model"). 기존 형태 그대로."""
    async def get_self_facts(self) -> list[dict]:
        """= semantic_store.get_identity_facts("self_model")."""
    async def add_user_fact(self, key, value, confidence=1.0) -> None:
        """= semantic_store.add_identity_fact("user_model", key, value, ...)."""

    # 새 인터페이스용 (workspace node 형태)
    async def render_as_nodes(self) -> list[dict]:
        """identity_facts → workspace node 변환.
        매핑:
        - fact_type='user_model' → type='Person', label='user'
        - fact_type='self_model' → type='Person', label='agent'
        - identity_facts.key → node property key
        - identity_facts.value → node property value
        - identity_facts.confidence → edge confidence
        """

    async def write_from_nodes(self, nodes: list[dict]) -> None:
        """workspace node → identity_facts 역변환.
        'user' label 의 Person node → user_model facts
        'agent' label 의 Person node → self_model facts
        """
```

#### 4-2. 기존 코드 경로 보존

수정 불필요 파일 (adapter 가 중개):
- `narrative_consolidation.py` — `semantic_store.get_identity_facts()` 직접 호출 유지
- `manager.py:render_user_context()` — 기존 로직 유지
- `pipeline.py` — mPFC/TPJ 에 identity 주입하는 기존 경로 유지
- `prefrontal.py` — PFC system prompt 의 USER MODEL 섹션 유지

#### 4-3. 테스트

- get_user_facts() → 기존 identity_facts 데이터 그대로 반환
- render_as_nodes() → Person type node 형태로 변환
- write_from_nodes() → identity_facts 에 역저장
- round-trip: identity_facts → render_as_nodes → write_from_nodes → identity_facts 무결성
- 기존 narrative_consolidation 회귀 테스트

---

### Phase 5 — Pipeline Integration

**목표:** 기존 7-phase pipeline 에 새 knowledge layer 연결
**산출물:** `pipeline.py`, `agent.py` 수정

#### 5-1. ProcessingPipeline 수정

```python
class ProcessingPipeline:
    def __init__(self, memory, llm_provider, emitter, ...):
        # 기존 초기화 유지
        # 추가:
        self.extraction_orchestrator = ExtractionOrchestrator(
            memory=memory, llm_provider=llm_provider,
            config=ExtractionConfig()
        )
```

#### 5-2. process_request() 흐름 수정

```python
async def process_request(self, text, image=None, audio=None, trace_run=None,
                           interaction_mode="question"):
    # 기존 Phase 1-3 유지 (Sensory → Dual Stream → Integration)

    # === 여기서 분기 ===
    # 기존 PSC 호출 대신 ExtractionOrchestrator 사용
    extraction_result = await self.extraction_orchestrator.extract(
        text=text, image=image, audio=audio,
        session_id=self._session_id,
        comprehension=comprehension,  # Wernicke 결과
    )

    # Severity 기반 응답 모드
    if extraction_result.response_mode == "block":
        # 답변 생산 안 함, 질문만 반환
        return PipelineResult(
            response="",
            response_mode="block",
            clarification_questions=extraction_result.clarification_questions,
            # ... 기타 필드
        )

    # 기존 PFC → ACC → BG → Cerebellum → Broca 흐름 유지
    # ...

    # Append mode: 정상 응답 + 질문 추가
    if extraction_result.response_mode == "append":
        result.clarification_questions = extraction_result.clarification_questions

    return result
```

#### 5-3. PipelineResult 확장

```python
@dataclass
class PipelineResult:
    response: str = ""
    actions_taken: list[dict] = field(default_factory=list)
    network_mode: str = ""
    signals_processed: int = 0
    memories_retrieved: list[dict] = field(default_factory=list)
    memory_encoded: bool = False
    from_cache: bool = False
    # 새 필드 (기존 호출자는 무시 가능)
    response_mode: str = "normal"          # normal | append | block
    clarification_questions: list[str] = field(default_factory=list)
    workspace_id: str = ""
    workspace_ask: str | None = None        # workspace 전환 확인 질문
    # S1: retrieval 후처리에서 감지된 모순
    retrieval_contradictions: list[dict] = field(default_factory=list)
    # S2: retrieval-as-reconstruction 명시
    retrieval_gaps: list[dict] = field(default_factory=list)           # missing required props
    retrieval_inference_fill: list[dict] = field(default_factory=list)  # LLM 이 추정한 부분
```

#### 5-3.5. Retrieval Post-Processing (S1 + S2)

**위치:** `MemoryManager.retrieve()` 또는 Pipeline 의 retrieval phase 직후.

```python
async def retrieve_with_contradictions(self, query, workspace_id=None, top_k=10):
    # 1. 기존 retrieve (semantic + episodic + spread_activation)
    result = await self.retrieve(query, workspace_id=workspace_id, top_k=top_k)

    # 2. S1: 결과 subject 들에 대해 open contradictions 일괄 조회
    subject_ids = list({r["subject"] for r in result.get("edges", []) if r.get("subject")})
    if subject_ids and workspace_id:
        contra_map = await self.contradictions.get_for_subject_batch(workspace_id, subject_ids)
        result["contradictions"] = [
            c for sublist in contra_map.values() for c in sublist
        ]

    # 3. S2: gaps detection (retrieval-as-reconstruction)
    result["gaps"] = []
    for node in result.get("nodes", []):
        type_id = node.get("type_id")
        if type_id:
            type_def = await self.ontology.resolve_node_type_by_id(type_id)
            required = type_def.get("schema", {}).get("required", [])
            for req in required:
                if req not in node.get("properties", {}):
                    result["gaps"].append({
                        "node": node["label"],
                        "missing": req,
                        "type": type_def["name"],
                    })

    # 4. inference_fill 은 LLM 응답 생성 단계에서 tracking (prefrontal 에서 채움)
    return result
```

**Expression Mode (4.13) 와의 연동:**
- `retrieval_contradictions` 비어있지 않으면 응답에 "이 부분은 두 가지 정보가 충돌" 명시
- `retrieval_gaps` 비어있지 않으면 silent 하게 채우지 말고 `open_questions_store.add_question()` 호출
- `retrieval_inference_fill` 은 prefrontal system prompt 에 "다음은 추정한 부분: ..." 명시하도록 지시

#### 5-4. LLM Override Detection (Wernicke 연동)

```python
# wernicke.py 의 comprehension 결과에 workspace_hint 추가
# M1 NOTE: workspace_hint 는 엄밀히 Wernicke (phonological/semantic parsing) 본연 기능이 아니라
# pragmatic/discourse analysis (TPJ/dlPFC 역할) 를 편의상 Wernicke region 에 첨부한 것.
# 향후 pragmatic 전담 region 분리 시 해당 기능을 이동. 코드에 주석 명시.
comprehension = {
    "intent": "...",
    "complexity": "...",
    "keywords": [...],
    "language": "...",
    "workspace_hint": "billing-service",     # 새 필드 (pragmatic, Wernicke 에 편의상 attach)
    "workspace_hint_confidence": 0.85,       # 새 필드
}
```

이 hint 를 Stage 1 Triage 가 받아서 workspace override 판단.

#### 5-5. Broca Region 수정

```python
# broca.py 수정:
# response_mode='block' 일 때 질문 포맷팅
async def format_response(self, ..., response_mode="normal",
                           clarification_questions=None):
    if response_mode == "block" and clarification_questions:
        return self._format_questions(clarification_questions)
    # 기존 응답 포맷팅
    ...
```

#### 5-6. 테스트

- workspace 선택이 pipeline 전체에 전파
- PSC 대신 orchestrator 호출 확인
- block mode: response="" + questions only
- append mode: response + questions
- normal mode: 기존 동작 그대로
- workspace_ask: LLM 감지 → 유저에게 전달
- **retrieve_with_contradictions (S1): retrieval 결과에 open contradictions 포함**
- **retrieve_with_contradictions (S2): gaps 자동 감지 — missing required property → gaps list**
- **Expression mode: retrieval_contradictions 존재 시 응답에 언급**
- 기존 personal 대화 회귀 테스트 (가장 중요)

---

### Phase 6 — Decay Policy

**목표:** workspace/type 별 차등 decay
**산출물:** `consolidation.py`, `forgetting.py` 수정

#### 6-1. ConsolidationEngine 수정

```python
async def consolidate(self) -> ConsolidationResult:
    # Phase 1: transfer 시 workspace decay 확인
    for mem in memories:
        # S8: never_decay=1 → transfer 하되 strength 유지
        if mem.get("never_decay"):
            strength = mem["strength"]
            # Staging → semantic 승격은 여전히 occurrence 조건 검사
            continue_to_promote(mem, strength)
            continue

        ws = await self._get_workspace(mem.get("workspace_id", "personal"))
        decay_policy = ws["decay_policy"]  # none | slow | normal

        # Type-level override
        entity_types = await self._resolve_entity_types(mem["entities"])
        for et in entity_types:
            if et.get("decay_override"):
                decay_policy = et["decay_override"]
                break  # 가장 restrictive 한 type 우선

        # S7: importance_score 반영 — 높을수록 decay 감속
        importance = mem.get("importance_score", 0.5)
        importance_factor = 1.0 - importance * 0.5   # importance=1 → 0.5x decay, importance=0 → 1.0x

        if decay_policy == "none":
            strength = mem["strength"]
        elif decay_policy == "slow":
            strength = mem["strength"] * (1.0 - (1.0 - 0.99) * importance_factor)
        else:  # normal
            base_factor = ach_transfer_factor  # 기존 emotional boost 로직
            adjusted_factor = 1.0 - (1.0 - base_factor) * importance_factor
            strength = mem["strength"] * adjusted_factor

    # Phase 2: homeostatic scaling — workspace 별 분기
    for ep in all_episodes:
        # S8: never_decay=1 row 는 skip
        if ep.get("never_decay"):
            continue

        ws_id = ep.get("workspace_id", "personal")
        decay_policy = await self._get_decay_policy(ws_id, ep.get("entities", {}))
        if decay_policy == "none":
            continue  # 감퇴 면제
        elif decay_policy == "slow":
            factor = 0.99
            threshold = 0.01
        else:
            factor = HOMEOSTATIC_FACTOR  # 0.95
            threshold = PRUNING_THRESHOLD  # 0.05

        # S7: importance_score 반영
        importance = ep.get("importance_score", 0.5)
        adjusted_factor = 1.0 - (1.0 - factor) * (1.0 - importance * 0.5)
        new_strength = ep["strength"] * adjusted_factor
        # ...
```

#### 6-2. Semantic Edge Decay

```python
# semantic_store.py 수정
async def decay_edge_weights(self, factor=0.95, workspace_id=None):
    # S8: never_decay=1 edge 는 WHERE 절에서 제외
    # S7: importance_score 반영 — factor 를 edge 별로 조정
    if workspace_id:
        await self._graph_db.execute("""
            UPDATE knowledge_graph
            SET weight = weight * (1.0 - (1.0 - ?) * (1.0 - importance_score * 0.5))
            WHERE workspace_id = ? AND never_decay = 0
        """, (factor, workspace_id))
    else:
        # workspace 별 decay_policy 에 따라 분기, never_decay=0 만
        ...

async def prune_weak_edges(self, min_weight=0.1, workspace_id=None):
    # none 정책 workspace 는 pruning 제외
    # S8: never_decay=1 edge 는 min_weight 무관 pruning 제외
    ...
```

#### 6-3. Forgetting Engine 수정

```python
# forgetting.py 수정
def apply_interference(self, strength, similarity, decay_policy="normal",
                         never_decay=False, importance_score=0.5):
    # S8: never_decay=1 → 간섭 면제
    if never_decay:
        return strength
    if decay_policy == "none":
        return strength  # 간섭 면제
    # S7: importance 높을수록 간섭 감속
    importance_factor = 1.0 - importance_score * 0.5
    # ... 기존 로직 × importance_factor
```

#### 6-4. 테스트

- none workspace → consolidation 후 strength 불변
- slow workspace → 0.99 factor 적용
- normal workspace → 기존 0.95 factor
- type override: workspace=normal 이지만 type=Requirement(decay=none) → 면제
- 혼합 consolidation: 3 workspace 동시 (none + slow + normal)
- edge decay: workspace='personal' 만 decay, business workspace 는 유지
- forgetting interference: none workspace → 간섭 면제
- **never_decay=1 edge (S8): workspace=personal 이어도 decay/forgetting/pruning 전부 skip**
- **importance_score 반영 (S7): importance=1.0 → decay factor 가 절반 완화**
- **personal workspace 에 잘못 들어간 business fact (never_decay=1): decay 안 됨 (project_intent 보호)**

---

### Phase 7 — Domain Templates

**목표:** 사전 정의 ontology template + loader
**산출물:** `brain_agent/memory/templates/` 디렉터리

#### 7-1. Template 구조

`brain_agent/memory/templates/software_project.py`:
```python
TEMPLATE = {
    "name": "software-project",
    "version": "1.0",
    "description": "Software engineering project knowledge",
    "decay_policy": "none",
    "node_types": [
        {"name": "Requirement", "parent": "Statement",
         "schema": {"props": ["priority", "status", "acceptance_criteria"], "required": []}},
        {"name": "Decision",    "parent": "Statement",
         "schema": {"props": ["date", "rationale", "alternatives", "decided_by"], "required": ["rationale"]}},
        {"name": "Module",      "parent": "Artifact",
         "schema": {"props": ["path", "language", "version"], "required": []}},
        {"name": "Interface",   "parent": "Artifact",
         "schema": {"props": ["protocol", "spec_url", "auth_method"], "required": []}},
        {"name": "Constraint",  "parent": "Statement",
         "schema": {"props": ["hard_or_soft", "metric", "threshold"], "required": []}},
        {"name": "Risk",        "parent": "Concept",
         "schema": {"props": ["likelihood", "impact", "mitigation"], "required": []}},
        {"name": "NonGoal",     "parent": "Statement", "schema": {}},
        {"name": "Workflow",    "parent": "Event",
         "schema": {"props": ["trigger", "steps", "output"], "required": []}},
        {"name": "DomainTerm",  "parent": "Concept",
         "schema": {"props": ["definition", "examples"], "required": ["definition"]}},
        {"name": "DataModel",   "parent": "Artifact",
         "schema": {"props": ["fields", "constraints", "storage"], "required": []}},
    ],
    "relation_types": [
        {"name": "depends_on",         "transitive": True},
        {"name": "implements",         "domain": "Module",     "range": "Requirement"},
        {"name": "constrains",         "domain": "Constraint"},
        {"name": "blocks",             "symmetric": False},
        {"name": "trades_off_against", "symmetric": True},
        {"name": "belongs_to",         "range": "Module"},
        {"name": "conflicts_with",     "symmetric": True},
        {"name": "mitigates",          "domain": "Artifact",   "range": "Risk"},
        {"name": "exposes",            "domain": "Module",     "range": "Interface"},
        {"name": "stores",             "domain": "Module",     "range": "DataModel"},
    ],
}
```

`brain_agent/memory/templates/research_notes.py`:
```python
TEMPLATE = {
    "name": "research-notes",
    "version": "1.0",
    "description": "Research and academic knowledge",
    "decay_policy": "none",
    "node_types": [
        {"name": "Hypothesis",     "parent": "Statement", "schema": {"props": ["testable", "status"]}},
        {"name": "Experiment",     "parent": "Event",     "schema": {"props": ["method", "sample_size"]}},
        {"name": "Result",         "parent": "Statement", "schema": {"props": ["p_value", "effect_size"]}},
        {"name": "Citation",       "parent": "Source",    "schema": {"props": ["authors", "year", "doi"]}},
        {"name": "Finding",        "parent": "Statement", "schema": {"props": ["replicable"]}},
        {"name": "Methodology",    "parent": "Concept",   "schema": {"props": ["strengths", "limitations"]}},
    ],
    "relation_types": [
        {"name": "tests",           "domain": "Experiment", "range": "Hypothesis"},
        {"name": "supports",        "domain": "Result",     "range": "Hypothesis"},
        {"name": "refutes",         "domain": "Result",     "range": "Hypothesis"},
        {"name": "cites",           "symmetric": False},
        {"name": "replicates",      "domain": "Experiment", "range": "Experiment"},
        {"name": "extends",         "symmetric": False},
    ],
}
```

`brain_agent/memory/templates/personal_knowledge.py`:
```python
TEMPLATE = {
    "name": "personal-knowledge",
    "version": "1.0",
    "description": "Personal life knowledge and preferences",
    "decay_policy": "normal",
    "node_types": [
        {"name": "Preference", "parent": "Concept",   "schema": {"props": ["strength", "since"]}},
        {"name": "Habit",      "parent": "Event",     "schema": {"props": ["frequency", "trigger"]}},
        {"name": "Belief",     "parent": "Statement", "schema": {"props": ["certainty"]}},
        {"name": "Memory",     "parent": "Event",     "schema": {"props": ["when", "where", "who"]}},
        {"name": "Goal",       "parent": "Statement", "schema": {"props": ["deadline", "progress"]}},
    ],
    "relation_types": [
        {"name": "prefers_over",  "symmetric": False},
        {"name": "causes",        "transitive": True},
        {"name": "reminds_of",    "symmetric": True},
        {"name": "wants",         "domain": "Person", "range": "Goal"},
        {"name": "knows",         "symmetric": True},
    ],
}
```

#### 7-2. Template Loader

```python
# ontology_store.py 에 추가
async def apply_template(self, workspace_id: str, template_name: str) -> None:
    """template 의 type/relation 을 workspace 에 적용.
    1. universal seed 존재 확인 → 없으면 seed_universal() 먼저
    2. template 의 node_types 삽입 (parent 참조 해석)
    3. template 의 relation_types 삽입 (domain/range 참조 해석)
    4. workspace.template_id = template_name 으로 업데이트
    """

async def _resolve_parent_for_template(self, workspace_id, parent_name):
    """template 의 parent 참조를 resolve. universal 또는 동 template 내 다른 type."""
```

**Template composition (복수 적용):**
- 같은 이름의 type 이 이미 존재하면 → schema 를 merge (props 합집합, required 합집합)
- 같은 이름의 relation 이 이미 존재하면 → domain/range 가 더 넓은 쪽 우선
- 충돌 시 나중 template 이 이김 (last-writer-wins)

**Template versioning (M8):**
- `workspaces.template_id` = "software-project" (name only)
- `workspaces.template_version` = "1.0" (별도 컬럼, Phase 0 workspaces 스키마 참조)
- template 파일은 `VERSION = "1.0"` 상수 export — loader 가 apply 시 이 값을 workspaces.template_version 에 기록

**Upgrade API:**
```python
async def upgrade_template(self, workspace_id: str, target_version: str,
                             dry_run: bool = False) -> dict:
    """Template 을 현재 버전 → target_version 으로 diff 적용.

    단계:
    1. 현재 workspace 의 template_version 확인
    2. 두 버전 template 모듈 import 후 diff 계산:
       - added_node_types / added_relation_types → register (신규만, 기존 type 충돌 시 skip + warn)
       - removed_*_types → soft delete (이미 사용 중인 type 은 삭제 금지, deprecated 마킹만)
       - modified_schema → property 추가는 OK, required 추가는 기존 node 검증 후 open_question 큐로
    3. dry_run=True 면 diff 만 반환, 실제 적용 안 함
    4. workspace.template_version 업데이트

    반환: {added: [...], removed: [...], modified: [...], warnings: [...]}
    """

async def downgrade_template(self, workspace_id: str, target_version: str) -> dict:
    """다운그레이드 금지. ValueError 발생 (데이터 손실 위험)."""
```

**Upgrade 정책:**
- major version bump (1.0 → 2.0): 유저 명시적 승인 필수 (CLI `brain-agent ontology upgrade <ws> <ver> --confirm`)
- minor/patch (1.0 → 1.1): added_* 는 자동 적용, removed_*/modified_required 는 dry_run 결과 유저에게 제시 후 승인

#### 7-3. 테스트

- software-project template 적용 → 10 node types + 10 relation types 확인
- parent chain: Requirement → Statement → universal 순회
- 두 workspace 에 다른 template → 격리
- template composition: 두 template 적용 후 type 합산
- template 이 없는 workspace 에 적용 → universal auto-seed
- **template versioning (M8):**
  - apply_template 시 workspaces.template_version 자동 기록
  - upgrade_template dry_run: diff 만 반환, DB 변경 없음
  - upgrade_template minor (1.0→1.1): added_types 자동 적용
  - upgrade_template 시 사용 중인 type 삭제 시도 → soft delete (deprecated 마킹)
  - upgrade_template major (1.0→2.0): --confirm 없으면 ValueError
  - downgrade_template: 무조건 ValueError

---

### Phase 8 — Visualization & Human-in-the-Loop

**목표:** CLI 없이도 curation 휴먼 루프(질문 응답 / 모순 해결 / ontology proposal 승인)와 coding agent export 미리보기를 대시보드에서 수행한다. 기존 FastAPI + WebSocket + React + Zustand 스택을 재사용.

**산출물:**
- 백엔드: `brain_agent/dashboard/server.py` 엔드포인트 확장 + `emitter.py` 신규 이벤트 7종
- 프론트엔드: `dashboard/src/components/` 에 신규 컴포넌트 7종 + 기존 3종 확장
- 통합: ExtractionOrchestrator 가 curation 이벤트 emit

**전제 원칙:**
- **LLM provider 비의존**: 모든 모델 설정 필드는 litellm 호환 identifier 문자열. default `"auto"` → `LLMProvider.get_default_model()` 으로 fallback. UI model selector 는 `/api/llm/providers` 응답을 그대로 렌더 — 특정 vendor 하드코딩 금지.
- **per-call-site 독립 설정**: triage / extract / temporal classify / refine 각각 독립 model 설정. 전역 default 는 각 필드 `"auto"` 로 일괄 fallback.
- **Backward compat**: 기존 `KnowledgeGraphPanel`, `MemoryPanel` 은 workspace_id 없이도 동작 (기존 전체 graph 표시 = 기존 동작).

#### 8-1. REST 엔드포인트 (`brain_agent/dashboard/server.py` 확장)

**Workspace:**
```
GET    /api/workspaces                       list_workspaces()
POST   /api/workspaces                       {name, description, decay_policy, template?}
GET    /api/workspaces/{ws_id}               get_workspace() + stats(node_count, edge_count, pending_count)
PATCH  /api/workspaces/{ws_id}               update_workspace()
DELETE /api/workspaces/{ws_id}               delete_workspace() (personal 거부)
GET    /api/workspaces/current?session_id=   get_session_workspace()
PUT    /api/workspaces/current               {session_id, workspace_id} → set_session_workspace()
```

**Ontology:**
```
GET    /api/ontology/{ws_id}/types           node_types + relation_types + confidence tiers
GET    /api/ontology/{ws_id}/proposals       list_pending(workspace_id=ws_id)
POST   /api/ontology/proposals/{id}/approve  {approved_by?}
POST   /api/ontology/proposals/{id}/reject
```

**Questions / Contradictions:**
```
GET    /api/questions/{ws_id}?severity=      list_unanswered(workspace_id, severity filter)
POST   /api/questions/{q_id}/answer          {answer, answer_source?}
GET    /api/contradictions/{ws_id}           list_open(workspace_id)
POST   /api/contradictions/{c_id}/resolve    {resolution, resolved_by, resolution_confidence}
                                              → semantic_store.mark_superseded(선택 안 된 edge, valid_to=now)
POST   /api/contradictions/{c_id}/dismiss
```

**Raw Vault:**
```
GET    /api/sources/{source_id}              메타 (workspace_id, kind, uri, sha256, integrity_valid, ingested_at)
GET    /api/sources/{source_id}/raw          바이너리 (integrity_valid=0 시 503 + warning)
GET    /api/sources/{source_id}/text         extracted_text 만
```

**KG (workspace 필터 확장):**
```
GET    /api/memory/knowledge-graph?workspace_id=&include_cross_refs=true
       → semantic_store.export_as_networkx(workspace_id, include_cross_refs)
GET    /api/memory/timeline?workspace_id=&subject=
       → (subject, relation) 별 valid_from/valid_to/superseded_by 체인
```

**Export Preview (D):**
```
POST   /api/export/preview
Body:  {workspace_id, filters: {never_decay_only?, min_importance?, min_confidence?, include_raw_vault?}}
Response: §8-6 JSON shape
```

**Model Inventory:**
```
GET    /api/llm/providers
Response: {
  "default_model": "<LLMProvider.get_default_model()>",
  "available": [
    {"id": "claude-sonnet-4-6", "vendor": "anthropic", "available": true},
    {"id": "gpt-4o-mini", "vendor": "openai", "available": true},
    {"id": "ollama/llama3", "vendor": "ollama", "available": false, "reason": "ollama not reachable"},
    ...
  ]
}
```
litellm 의 `litellm.model_list` + provider 환경 변수 resolve 를 통해 실제 사용 가능 여부 반환.

#### 8-2. WebSocket 이벤트 (`emitter.py` 확장)

기존 `knowledge_update`, `memory_event` 패턴 준수. 추가 이벤트 7종:
```python
"workspace_changed"         # {workspace_id, workspace_name, session_id}
"clarification_requested"   # {question_id, question, severity, workspace_id, context_input, raised_by}
"contradiction_detected"    # {contradiction_id, subject, value_a, value_b, severity, workspace_id}
"ontology_proposal"         # {proposal_id, kind, proposed_name, confidence, workspace_id, source_snippet}
"question_answered"         # {question_id, workspace_id}
"contradiction_resolved"    # {contradiction_id, resolution, workspace_id}
"proposal_decided"          # {proposal_id, status, workspace_id}
```

Emit 지점:
- `WorkspaceStore.set_session_workspace()` → `workspace_changed`
- `ExtractionOrchestrator._persist()`:
  - 신규 contradiction 등록 시 → `contradiction_detected`
  - 신규 open_question 등록 시 → `clarification_requested`
  - 신규 proposal 등록 시 → `ontology_proposal`
- Curation API 가 answer/resolve/approve/reject 처리 후 → `question_answered` / `contradiction_resolved` / `proposal_decided`

#### 8-3. Config 확장 (`brain_agent/config/schema.py`)

§4.11 의 `ExtractionConfig` 확장:
```python
class ExtractionConfig(BaseModel):
    # per-call-site 독립 설정, default "auto" → LLMProvider.get_default_model()
    triage_model: str = "auto"
    extract_model: str = "auto"
    temporal_classify_model: str = "auto"   # Stage 2.5 LLM classify
    refine_model: str = "auto"              # Stage 5 Broca
    max_retry: int = 1
    enable_severity_block: bool = True
    promotion_threshold_n: int = 3          # C1 PROVISIONAL→STABLE
```

구현 규약:
- 각 stage 모듈 (`triage.py`, `extractor.py`, `temporal_resolver.py`, `refiner.py`) 은 `llm_provider: LLMProvider` 를 인자로 받음.
- stage 내부에서 `model_name = self.config.<stage>_model; if model_name == "auto": model_name = llm_provider.get_default_model()` → `await llm_provider.chat(model=model_name, ...)`.
- Vendor 특정 import (e.g., `anthropic`, `openai`) 금지. litellm 만 경유.

#### 8-4. Frontend 신규 컴포넌트 (`dashboard/src/`)

**State 확장 (`stores/brainState.ts`):**
```typescript
interface BrainState {
  // 기존 필드 유지
  currentWorkspace: {id: string, name: string} | null
  workspaces: Workspace[]
  openQuestions: OpenQuestion[]
  contradictions: Contradiction[]
  ontologyProposals: Proposal[]
  availableModels: ModelInfo[]
  // WS 이벤트 핸들러 (useWebSocket.ts) 가 위 배열을 incremental update
}
```

**신규 컴포넌트 (`components/`):**
- `WorkspaceSelector.tsx` — HUD 헤더 dropdown. current + all workspaces. 전환 시 PUT /api/workspaces/current.
- `CurationInbox.tsx` — 3-tab 컨테이너 (Questions / Contradictions / Proposals). Tab 별 badge count (severity="severe" 는 빨강).
  - `QuestionCard.tsx` — severity icon + question text + context_input snippet + answer textarea + file attach (raw vault source 링크 생성).
  - `ContradictionCard.tsx` — 2-column diff (value_a vs value_b). 각 쪽에 source_snippet, epistemic_source 뱃지. 4개 resolution 버튼: [A 선택] [B 선택] [둘 다 맞음 (temporal split)] [Dismiss].
  - `ProposalCard.tsx` — kind 뱃지(node_type/relation_type) + proposed_name + definition JSON + source_snippet + [Approve] [Reject].
- `RawVaultPanel.tsx` — 우측 drawer. source_id 받아서 /api/sources/{id} + /api/sources/{id}/text 로드. kind=image/pdf 면 썸네일.
- `TimelineView.tsx` — subject 받아서 chronological lane. 각 edge 가 (valid_from → valid_to) bar. supersede 화살표.
- `ExportPreviewModal.tsx` — D 전용. workspace selector + filter checkboxes + JSON preview (monaco readonly) + Copy / Download `.json`.
- `ModelSelector.tsx` — 설정 drawer. `/api/llm/providers` 호출 후 4개 독립 dropdown (triage/extract/temporal/refine). available=false 는 비활성 상태로 표시하되 선택 가능 (runtime 실패 가능성 명시).

**기존 컴포넌트 수정:**
- `KnowledgeGraphPanel.tsx` — prop `workspaceId?: string` 추가. `/api/memory/knowledge-graph?workspace_id=...&include_cross_refs=...` 호출. Cross-ref edge 는 점선(dashArray) + 다른 색상. Hover 시 RawVaultPanel open (node.source_ref 기반).
- `KnowledgeGraphModal.tsx` — WorkspaceSelector 통합 + "Show cross-refs" toggle + "Importance overlay" toggle + "Never-decay highlight" toggle.
- `HUD.tsx` — 상단에 current workspace badge + unresolved questions count badge (클릭 시 CurationInbox 오픈).
- `MemoryPanel.tsx` — workspace filter 전파.

**신규 hooks (`hooks/`):**
- `useWorkspace.ts` — current workspace read/write + session persistence + workspace_changed WS 구독.
- `useCurationInbox.ts` — clarification_requested / contradiction_detected / ontology_proposal WS 구독 + optimistic answer/resolve mutation.

#### 8-5. UX 플로우

**Clarification block 발생 시 (채널 이중화):**
```
1. Pipeline 이 response_mode='block' 로 응답
   → 채팅창: BrainResponseBubble 변형 (질문 카드 + answer 입력)
2. 동시에 WS "clarification_requested" 발송 → CurationInbox Questions 탭에도 등록
3. 유저가 어느 쪽에서든 답변
   → POST /api/questions/{id}/answer
   → 백엔드가 WS "question_answered" emit
   → 양쪽 UI 에서 해당 카드 제거 (state 동기화)
4. Pipeline 은 다음 turn 에서 open_questions_store 답변 반영
```

**Contradiction 발생 시:**
```
1. Stage 3 감지 → "contradiction_detected" emit
2. 채팅은 severity 에 따라 block/append/normal
3. Inbox Contradictions 탭에 즉시 등장 (badge +1)
4. 유저가 ContradictionCard 에서 [A 선택]
   → POST /api/contradictions/{id}/resolve
   → semantic_store.mark_superseded(B.edge_id, valid_to=now)
   → contradictions_store.resolve() status='resolved'
   → WS "contradiction_resolved" emit
```

**Ontology proposal 처리:**
```
1. Stage 2 가 INFERRED/AMBIGUOUS type proposal 생성 → "ontology_proposal" WS emit
2. CurationInbox Proposals 탭 badge +1
3. 유저가 ProposalCard [Approve]
   → POST /api/ontology/proposals/{id}/approve
   → ontology_store.approve_proposal() → node_types/relation_types 로 이관 (confidence='CANONICAL')
   → WS "proposal_decided" emit
```

#### 8-6. Export Preview JSON Shape (MCP 응답과 동일)

```json
{
  "workspace": {
    "id": "billing-service",
    "name": "Billing Service",
    "template": "software-project/1.0",
    "decay_policy": "none"
  },
  "ontology": {
    "node_types": [],
    "relation_types": []
  },
  "facts": [
    {
      "subject": "order-api",
      "relation": "implements",
      "target": "idempotent-payment",
      "confidence": "STABLE",
      "importance_score": 0.8,
      "never_decay": true,
      "epistemic_source": "asserted",
      "source": {"id": "src_abc", "kind": "user_utterance", "snippet": "..."},
      "valid_from": "2026-03-01T...",
      "valid_to": null
    }
  ],
  "open_questions": [],
  "unresolved_contradictions": [],
  "raw_vault_refs": []
}
```

**중요:** 이 JSON shape 은 향후 MCP tool 응답 payload 와 동일 — MCP 구현 시 재사용, 추가 설계 불필요.

**Filter 매트릭스:**
| Filter | 기본값 | 효과 |
|---|---|---|
| `never_decay_only` | false | true → `never_decay=1` facts 만 |
| `min_importance` | 0.0 | edges[importance_score >= min_importance] |
| `min_confidence` | "PROVISIONAL" | 해당 tier 이상 type 만 |
| `include_raw_vault` | false | true → raw_vault_refs 채움 + facts[i].source.content_inline 포함 가능 |

#### 8-7. 테스트

**Backend:**
- `tests/dashboard/test_workspace_api.py` — 7개 workspace endpoint CRUD + session binding
- `tests/dashboard/test_ontology_api.py` — proposal list/approve/reject flow + WS emit
- `tests/dashboard/test_curation_api.py` — questions/contradictions full flow + optimistic UI 시나리오 (answer 이후 WS emit)
- `tests/dashboard/test_export_preview.py` — filter 매트릭스 (4개 filter 교차) + JSON shape 검증
- `tests/dashboard/test_source_api.py` — raw vault 바이너리 + integrity_valid=0 시 503
- `tests/dashboard/test_llm_provider_listing.py` — litellm `model_list` resolve + 환경 변수 누락 시 `available=false`

**Frontend (Playwright/Vitest, dashboard 기존 패턴):**
- WorkspaceSelector: 전환 시 current 업데이트 + WS event 수신
- CurationInbox: 3 탭 badge 카운트 + WS incremental update
- ContradictionCard: 4개 resolution 버튼 → 올바른 API 호출
- ExportPreviewModal: filter 토글 시 preview refetch + Copy/Download 동작
- ModelSelector: /api/llm/providers 응답 렌더 + available=false 비활성 상태 표시
- KnowledgeGraphPanel: workspaceId prop 변경 시 refetch + cross-ref 점선 스타일

**회귀:**
- 기존 KnowledgeGraphPanel 이 workspaceId 없이도 동작 (전체 graph)
- 기존 621+ 백엔드 테스트 통과 유지

#### 8-8. MVP 슬라이싱 (구현 순서)

| Slice | Phase 의존 | Frontend | Backend |
|---|---|---|---|
| **MVP-v1** | 0, 1, 4 | WorkspaceSelector, KG workspace filter | /api/workspaces/*, /api/memory/knowledge-graph?workspace_id= |
| **MVP-v2** | 2, 3 (stage 1~2.5) | CurationInbox (3 탭), Q/C/P cards | /api/questions/*, /api/contradictions/*, /api/ontology/proposals/*, WS 이벤트 4종 |
| **MVP-v3** | 3 (완), 5, 6 | RawVaultPanel, TimelineView, ExportPreviewModal, ModelSelector | /api/sources/*, /api/export/preview, /api/llm/providers |

Phase 7 (templates) 는 Phase 8 와 **독립** — workspace 생성 시 template 선택 UI 는 Phase 8-1 `POST /api/workspaces` body 에 `template?` 필드로 이미 포함. 실제 template 목록 노출은 Phase 7 산출물이 있어야 의미 있음.

---

## 6. 의존성 그래프

```
Phase 0 (workspace + ontology + seed)
  │
  ├── Phase 1 (raw vault + schema migration)
  │     │
  │     ├── Phase 3 (extractor) ←── Phase 2 (contradictions + questions)
  │     │     │
  │     │     └── Phase 5 (pipeline integration) ←── Phase 4 (personal adapter)
  │     │           │
  │     │           └── Phase 8 (visualization & human-in-the-loop)
  │     │                 — MVP-v1: needs 0, 1, 4
  │     │                 — MVP-v2: needs 2, 3 (stages 1~2.5)
  │     │                 — MVP-v3: needs 3 (full), 5, 6
  │     │
  │     └── Phase 6 (decay policy)
  │
  └── Phase 7 (templates)   → Phase 8 template selector UI 가 consume
```

**Critical path:** 0 → 1 → 2 → 3 → 5 → 8 (MVP-v2+)
**Parallel candidates:** Phase 4 (Phase 1 이후 독립), Phase 6 (Phase 1 이후 독립), Phase 7 (Phase 0 이후 독립)
**Phase 8 slicing:** MVP-v1 은 Phase 4 완료 후 바로 시작 가능. MVP-v2/v3 는 Phase 3/5 종속.

---

## 7. Neuroscience 추가 참고문헌

기존 CBA references 에 추가:
- **Johnson, Hashtroudi & Lindsay (1993)** — Source Memory Framework → raw vault provenance
- **Moscovitch & Nadel (1997)** — Multiple Trace Theory → append-only versioning
- **Hart (1965)** — Feeling of Knowing → open_questions_store
- **Brown & McNeill (1966)** — Tip-of-the-tongue phenomenon → expression mode gap detection
- **Bartlett (1932)** — Schema Theory → workspace = schema frame
- **van Kesteren et al. (2012)** — Schema-dependent encoding → workspace-aware extraction
- **Ashby & Maddox (2011)** — Category learning → ontology type hierarchy

---

## 8. 비고

- MCP tool 구현은 전 Phase 완료 후 별도 plan. **Phase 8-6 export JSON shape 을 MCP 응답 shape 으로 재사용** — MCP 구현 시 추가 설계 불필요.
- saju 브랜치에서 작업, push 금지 (로컬 전용)
- 기존 테스트 621+ 깨지지 않도록 각 Phase 에서 회귀 테스트 포함
- 모든 새 store 는 `PRAGMA journal_mode=WAL` + aiosqlite 사용 (기존 패턴 준수)
- Phase 0 구현 시작 후, 각 Phase 완료 시 code review agent 로 검토
- **LLM provider 비의존 (Phase 8-3)**: 모든 새 LLM 호출 지점(triage/extract/temporal_classify/refine) 은 `brain_agent/providers/base.py::LLMProvider` 경유. 특정 vendor import 금지. Config 는 문자열 identifier, default `"auto"` → `get_default_model()` fallback. 이 원칙은 Phase 3 구현 시부터 준수 (Phase 8 은 이 원칙의 UI 노출 지점일 뿐).

---

## 9. Review Revisions (2026-04-16, pre-implementation review)

구현 직전 neuroscience/cognitive science 관점 심층 리뷰에서 제기된 21개 권고 중 합의 결정을 유지하는 범위 내에서 반영. 근거 논문·판단·미반영 risk 의 트레이서빌리티 보존용.

### 9.1 반영된 변경사항 — Critical (plan architecture 통합)

| # | 반영 위치 | 근거 |
|---|---|---|
| **C1** 4-tier confidence (PROVISIONAL→STABLE→CANONICAL→USER_GROUND_TRUTH) + occurrence_count + source_snippet | §2 결정표 #3, §5 Phase 0 ontology schema/API, §5 Phase 0 tests | Kadavath et al. (2022) LLM self-calibration miscalibration; Loftus misinformation effect — high-confidence false memory; Dunning-Kruger 유추 |
| **C3** Stage 2.5 Temporal Resolve (update vs contradiction 사전 판단) | §3 데이터 흐름, §2 결정표 #11, §5 Phase 3-2.5 신설 | Conway (2005) self-memory system — time-indexed autobiographical facts; temporal context as encoding dimension |
| **C5** _persist 는 hippocampal_staging 만 경유, semantic/episodic 직접 write 금지 | §3 Persist 주석, §5 Phase 3-6 orchestrator._persist() | Squire (1992) multiple memory systems; McClelland et al. (1995) Complementary Learning Systems — hippocampus fast, neocortex slow/repeated |

### 9.2 반영된 변경사항 — Serious (Phase 별 분산)

| # | 반영 위치 | 근거 |
|---|---|---|
| **S1** Retrieval-time contradiction monitoring (`get_for_subject_batch`) | §4.18, §5 Phase 2 API, §5 Phase 5-3.5 | Botvinick (2001) ACC — conflict monitoring is continuous, not encoding-only |
| **S2** Retrieval-as-reconstruction (gaps, inference_fill) | §4.19, §5 Phase 5-3 PipelineResult, §5 Phase 5-3.5 | Bartlett (1932) schema theory — recall is reconstruction, not replay |
| **S3** Epistemic source distinction (asserted/cited/inferred/observed) | §4.20, §5 Phase 1 knowledge_graph column, §5 Phase 3-2 Stage 2 prompt | Johnson, Hashtroudi & Lindsay (1993) source monitoring framework |
| **S4** Pattern separation for similar Events | §5 Phase 3-3 Validator | Yassa & Stark (2011) dentate gyrus pattern separation |
| **S5** Dream engine all-workspaces (not per-workspace) | §4.17 수정 | Stickgold (2005), Walker & Stickgold (2006) REM cross-domain integration |
| **S6** FOK pre-retrieval (metamemory gap detection) | §5 Phase 3-3 Validator FOK section | Hart (1965) feeling-of-knowing; Brown & McNeill (1966) TOT |
| **S7** Edge-level importance_score in decay formulas | §4.21, §5 Phase 1 column, §5 Phase 3-2 prompt, §5 Phase 6 | LeDoux (1996) amygdala event-level modulation; limit of workspace-level policy |
| **S8** never_decay bit to protect business logic | §4.22, §5 Phase 1 column, §5 Phase 6 | project_intent.md "정보 손실 = 기능 회귀" 원칙 보호 |

### 9.3 반영된 변경사항 — Minor (Phase 별 분산)

| # | 반영 위치 |
|---|---|
| **M1** Wernicke workspace_hint 역할 정합성 주석 | §4.25, §5 Phase 5-4 comment |
| **M2** Multi-store persistence logical transaction | §4.23 |
| **M3** `target_workspace_id` 인덱스 | §4.7 index 리스트, §5 Phase 1 migration |
| **M4** __universal__ namespace hierarchy (future) | §4.25 note 수준 (당장은 현재 구조 유지) |
| **M6** Triage input_kinds multi-label | §4.24, §5 Phase 3-1 TriageResult |
| **M7** Implicit/priming memory hook (future) | (미반영, 별도 plan — 현 구조 design hook 만) |
| **M8** Template versioning (upgrade_template API) | §5 Phase 0 workspaces 컬럼, §5 Phase 7-2 |

### 9.4 미반영 — 합의 결정 유지 (modelling risk, 모니터링 포인트)

아래 3개는 유저가 이전 brainstorming 세션에서 다른 옵션을 명시적으로 거절하고 채택한 결정이므로, 리뷰 권고가 논리적으로 더 타당하더라도 **현 결정을 유지**. 각 결정에 대해 구현 후 모니터링 지표를 설정하여 실제 사용 데이터가 리뷰 권고와 합치하면 재검토.

**C2 — Workspace hard boundary (결정 2 유지: edge single-owner + cross-reference)**
- 근거 논문 재검토: Bartlett (1932), van Kesteren et al. (2012), Tse et al. (2007) 은 schema co-activation / cross-domain binding 을 기술. 리뷰 권고는 edge relevance tags (multi-owner) 였음.
- 유지 이유: 유저가 이미 "option 3: Multi-workspace + cross-reference" 를 다른 후보 (global merge, single-owner no cross-ref) 대비 명시적 선택.
- **Risk:** "billing-service workspace 에서 배운 SQL idempotency 패턴" 이 "order-service workspace" 에서 invisible. Cross-ref 수동 선언이 실제 운용에서 누락되면 cross-domain knowledge siloing.
- **모니터링:** 운용 1주차 이후 주간 지표 — (a) cross-reference edge 비율 / 전체 edge, (b) 유저가 "다른 workspace 에 있는 fact 찾는데 안 나옴" 류 불만 발생 건수. (a) < 5% 이거나 (b) 발생 시 relevance tags 로 재검토.

**C4 — Severity='severe' auto-block (결정 6 유지: severe → block-and-ask)**
- 근거 재검토: Botvinick (2001) ACC 는 conflict *signal* 만 담당, block 은 dlPFC 영역. 리뷰 권고는 default `enable_severity_block=False`, block 조건을 core-identity 모순만으로 제한.
- 유지 이유: 유저가 이미 severity-tiered 3-level 발동 구조를 브레인스토밍에서 확정. "논리적 비약엔 질문" (feedback_info_handling.md) 원칙과도 부합 — block 은 그 극단 표현.
- **Risk:** False-positive severe 판정이 반복되면 UX degradation (답변 대신 질문 연발). C1 + C3 수정이 false-positive 를 충분히 줄일 것으로 가정하지만 미확인.
- **모니터링:** 운용 1주차 지표 — block rate (전체 response 대비 block mode 비율). **> 5% 시 재검토**. 또한 block 후 유저가 질문에 답하지 않고 다른 주제로 전환하는 비율 (block 무시율). > 30% 시 재검토.

**M5 — Ontology approval queue 무기한 pending (결정 3 유지: 수동 approve only)**
- 유지 이유: 결정 3 의 hybrid 정책은 "INFERRED/AMBIGUOUS 는 유저 판단 필수" 가 핵심. Auto-approve escape valve 는 이 의도를 훼손.
- 덧붙여 **C1 반영으로 EXTRACTED auto-register 대신 PROVISIONAL 로 시작하는 새 안전장치 추가됨** — 이미 auto-registration 위험은 C1 으로 부분 mitigation 됨.
- **Risk:** INFERRED/AMBIGUOUS pending queue 가 stale 상태로 누적 → 해당 concept 이 extraction 시 generic Concept/refers_to 로 fallback 되어 expressiveness 손실.
- **모니터링:** pending_ontology_proposals 의 oldest-pending-age 및 size. 2주 이상 pending 이 20건 이상이면 CLI notification — "처리 필요한 proposal 20+ 건 있음".

### 9.5 구현 전 우선 반영 체크리스트

| 항목 | 반영 Phase | 확인 |
|---|---|---|
| C1 4-tier confidence schema + API | Phase 0 | §5 Phase 0-2 ✓ |
| C1 promote_confidence / increment_occurrence | Phase 0 | §5 Phase 0-2 ✓ |
| C1 source_snippet 필수 보존 | Phase 0 | §5 Phase 0-2 ✓ |
| C3 Stage 2.5 Temporal Resolve | Phase 3 | §5 Phase 3-2.5 ✓ |
| C5 _persist staging-only | Phase 3 | §5 Phase 3-6 ✓ |
| S3 epistemic_source column | Phase 1 | §5 Phase 1-2 ✓ |
| S7 importance_score column | Phase 1 | §5 Phase 1-2 ✓ |
| S8 never_decay column | Phase 1 | §5 Phase 1-2 ✓ |
| M3 cross-ref reverse index | Phase 1 | §5 Phase 1-2 ✓ |
| M8 template_version column | Phase 0 | §5 Phase 0-1 ✓ |

### 9.6 추가 인용 논문

§7 (Neuroscience 추가 참고문헌) 에 이미 포함된 것 외 리뷰에서 새로 추가:
- **Kadavath et al. (2022)** "Language Models (Mostly) Know What They Know" — LLM confidence self-assessment miscalibration 근거 (C1)
- **McClelland, McNaughton & O'Reilly (1995)** "Why there are complementary learning systems..." — CLS, staging bypass 금지 근거 (C5)
- **Conway (2005)** "Memory and the Self" — time-indexed autobiographical facts (C3)
- **Yassa & Stark (2011)** "Pattern separation in the hippocampus" — DG pattern separation (S4)
- **LeDoux (1996)** "The Emotional Brain" — amygdala event-level modulation (S7)
- **Squire (1992)** "Memory and the hippocampus" — multiple memory systems (C5)
- **Stickgold (2005)**, **Walker & Stickgold (2006)** — REM cross-domain integration (S5)
