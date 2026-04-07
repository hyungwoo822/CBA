# Multimodal Input Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable image, PDF/document, and audio (mic) inputs to flow through anatomically correct neural pathways with real LLM processing (vision, STT), configurable per-modality models.

**Architecture:** Server accepts multipart/form-data with files. File type determines processing: images go through V1 with vision LLM, documents get text-extracted and merge into text, audio goes through A1 with Whisper STT. Config allows per-modality model override.

**Tech Stack:** Python (litellm, pypdf, python-docx), TypeScript/React (FormData, MediaRecorder API)

---

### Task 1: Add ModalityConfig to config schema

**Files:**
- Modify: `brain_agent/config/schema.py`

- [ ] **Step 1: Add ModalityConfig class**

```python
# Add after DashboardConfig class (line 43)
class ModalityConfig(BaseModel):
    """Per-modality LLM model configuration.

    Each sensory modality can use a different model optimized for that input type.
    Falls back to agent.model if not specified.
    """
    vision_model: str = ""   # V1: vision LLM (e.g. "openai/gpt-4o"). Empty = use agent.model
    stt_model: str = ""      # A1: speech-to-text (e.g. "whisper-1"). Empty = use agent.model
    text_model: str = ""     # Wernicke/PFC/Broca. Empty = use agent.model
```

Add to BrainAgentConfig:
```python
modality: ModalityConfig = Field(default_factory=ModalityConfig)
```

- [ ] **Step 2: Commit**

```bash
git add brain_agent/config/schema.py
git commit -m "feat(config): add ModalityConfig for per-modality LLM models"
```

---

### Task 2: Server accepts multipart/form-data with file classification

**Files:**
- Modify: `brain_agent/dashboard/server.py`

- [ ] **Step 1: Add text extraction utilities**

Add at the top of server.py (after imports):

```python
import tempfile

def extract_text_from_file(filename: str, content: bytes) -> str | None:
    """Extract text from PDF, DOCX, or TXT files."""
    lower = filename.lower()
    if lower.endswith(".txt"):
        return content.decode("utf-8", errors="replace")
    if lower.endswith(".pdf"):
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return None
    if lower.endswith((".docx", ".doc")):
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return None
    return None
```

- [ ] **Step 2: Replace /api/process endpoint**

Replace the existing `@app.post("/api/process")` endpoint. Remove the `ProcessRequest` Pydantic model. New endpoint accepts `multipart/form-data`:

```python
from fastapi import File, UploadFile, Form

@app.post("/api/process")
async def process_message(
    text: str = Form(default=""),
    files: list[UploadFile] = File(default=[]),
):
    agent_inst: BrainAgent | None = _state["agent"]
    if agent_inst is None:
        return {"error": "agent not initialized"}

    image_bytes: bytes | None = None
    audio_bytes: bytes | None = None
    doc_texts: list[str] = []

    for f in files:
        content = await f.read()
        ct = (f.content_type or "").lower()
        name = f.filename or ""

        if ct.startswith("image/"):
            image_bytes = content
        elif ct.startswith("audio/") or name.endswith((".webm", ".wav", ".mp3", ".ogg")):
            audio_bytes = content
        else:
            # Try text extraction (PDF, DOCX, TXT)
            extracted = extract_text_from_file(name, content)
            if extracted:
                doc_texts.append(f"[Document: {name}]\n{extracted}")

    # Merge document text with user text
    full_text = text
    if doc_texts:
        full_text = text + "\n\n" + "\n\n".join(doc_texts) if text else "\n\n".join(doc_texts)

    if not full_text.strip() and not image_bytes and not audio_bytes:
        return {"error": "empty message"}

    result = await agent_inst.process(full_text, image=image_bytes, audio=audio_bytes)
    pipeline = agent_inst.pipeline

    await _emitter.region_activation("hippocampus", 0.7, "active")
    stats = await agent_inst.memory.stats()
    await _emitter.memory_flow(
        stats["sensory"], stats["working"],
        stats["staging"], stats["episodic"], stats["semantic"],
    )
    await _emitter.neuromodulator_update(**pipeline.neuromodulators.snapshot())

    return {
        "response": result.response,
        "network_mode": result.network_mode,
        "signals_processed": result.signals_processed,
        "actions": result.actions_taken,
    }
```

- [ ] **Step 3: Update agent.process() to accept image/audio**

In `brain_agent/agent.py`, update the `process` method signature:

```python
async def process(self, text: str, image: bytes | None = None, audio: bytes | None = None) -> PipelineResult:
    # ... existing session logic ...
    result = await self.pipeline.process_request(text, image=image, audio=audio)
    return result
```

- [ ] **Step 4: Commit**

```bash
git add brain_agent/dashboard/server.py brain_agent/agent.py
git commit -m "feat(server): multipart file upload with image/audio/document classification"
```

---

### Task 3: V1 Visual Cortex — Vision LLM call

**Files:**
- Modify: `brain_agent/regions/visual_cortex.py`

- [ ] **Step 1: Add vision LLM to V1**

Rewrite visual_cortex.py to call a vision-capable LLM when image data is present:

```python
"""Visual Cortex — Image input processing via Vision LLM.
Brain mapping: Occipital lobe (V1→V2→ventral stream).
AI function: Calls vision-capable LLM to extract scene description,
objects, spatial layout, and text/OCR from images.

References:
  - Hubel & Wiesel (1959): V1 feature extraction
  - Ungerleider & Mishkin (1982): Ventral "what" stream
"""
from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from brain_agent.regions.base import BrainRegion, Vec3, Lobe, Hemisphere
from brain_agent.core.signals import Signal, SignalType

if TYPE_CHECKING:
    from brain_agent.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_VISION_SYSTEM_PROMPT = """\
You are V1 (primary visual cortex) in a brain-inspired AI system.
Analyze the image and return a JSON object:
{
  "description": "detailed scene description",
  "objects": ["object1", "object2"],
  "text_content": "any text/writing visible in the image",
  "spatial_layout": "brief spatial description (foreground, background, etc.)",
  "emotional_tone": "neutral|positive|negative|threatening|calming"
}
Return ONLY valid JSON."""


class VisualCortex(BrainRegion):
    """Primary visual cortex — processes image inputs via vision LLM."""

    def __init__(self, llm_provider: LLMProvider | None = None, vision_model: str = ""):
        super().__init__(
            name="visual_cortex",
            position=Vec3(0, -40, -10),
            lobe=Lobe.OCCIPITAL,
            hemisphere=Hemisphere.BILATERAL,
            llm_provider=llm_provider,
        )
        self._vision_model = vision_model

    async def process(self, signal: Signal) -> Signal | None:
        image_data = signal.payload.get("image_data")
        if image_data is None:
            return signal

        signal.payload["modality"] = "visual"

        if self.llm_provider is not None:
            features = await self._analyze_with_vision_llm(image_data)
        else:
            features = self._basic_features(image_data)

        signal.payload["visual_features"] = features
        if features.get("description") and not signal.payload.get("text"):
            signal.payload["text"] = features["description"]

        self.emit_activation(0.8 if self.llm_provider else 0.5)
        return signal

    async def _analyze_with_vision_llm(self, image_data: bytes | str) -> dict:
        """Call vision LLM to analyze image (Hubel & Wiesel 1959: feature extraction)."""
        try:
            if isinstance(image_data, (bytes, bytearray)):
                b64 = base64.b64encode(image_data).decode()
                image_url = f"data:image/jpeg;base64,{b64}"
            else:
                image_url = image_data

            messages = [
                {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Analyze this image."},
                ]},
            ]

            response = await self.llm_provider.chat(
                messages,
                model=self._vision_model or None,
                max_tokens=500,
                temperature=0.1,
            )

            if response.content:
                import json
                text = response.content.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    text = "\n".join(lines).strip()
                return json.loads(text)
        except Exception as e:
            logger.warning("V1 vision LLM failed: %s", e)

        return self._basic_features(image_data)

    @staticmethod
    def _basic_features(image_data) -> dict:
        features = {}
        if isinstance(image_data, (bytes, bytearray)):
            features["size_bytes"] = len(image_data)
            features["description"] = f"Image ({len(image_data)} bytes)"
        return features
```

- [ ] **Step 2: Wire vision LLM provider in pipeline.py**

In pipeline `__init__`, update VisualCortex initialization:

```python
self.visual_cortex = VisualCortex(
    llm_provider=llm_provider,
    vision_model=getattr(getattr(memory, '_config', None), 'modality', None)
                 and memory._config.modality.vision_model or "",
)
```

Actually, simpler — pass from the pipeline's config. Since pipeline doesn't have config directly, pass the model name. In `__init__`:

```python
self.visual_cortex = VisualCortex(llm_provider=llm_provider)
```

The vision_model can be set via the VisualCortex after initialization, or the provider's default model will be used (litellm handles vision-capable models).

- [ ] **Step 3: Commit**

```bash
git add brain_agent/regions/visual_cortex.py brain_agent/pipeline.py
git commit -m "feat(V1): vision LLM for image analysis in visual cortex"
```

---

### Task 4: A1 Auditory Cortex — Whisper STT

**Files:**
- Modify: `brain_agent/regions/auditory_cortex.py`

- [ ] **Step 1: Add STT to A1 Left**

Rewrite AuditoryCortexLeft to call Whisper when audio_data is raw bytes (no transcript yet):

```python
class AuditoryCortexLeft(BrainRegion):
    """Left auditory cortex — speech processing via STT.

    When raw audio bytes are present and no transcript exists,
    calls Whisper STT to transcribe speech to text.
    """

    def __init__(self, llm_provider: LLMProvider | None = None):
        super().__init__(
            name="auditory_cortex_left",
            position=Vec3(-35, -10, 10),
            lobe=Lobe.TEMPORAL,
            hemisphere=Hemisphere.LEFT,
            llm_provider=llm_provider,
        )

    async def process(self, signal: Signal) -> Signal | None:
        audio_data = signal.payload.get("audio_data")
        if audio_data is None:
            return signal

        transcript = signal.payload.get("transcript", "")

        if not transcript and isinstance(audio_data, (bytes, bytearray)) and self.llm_provider:
            transcript = await self._transcribe(audio_data)

        if transcript:
            signal.payload["text"] = transcript
            signal.payload["transcript"] = transcript

        signal.payload["modality"] = "audio"
        self.emit_activation(0.7 if transcript else 0.4)
        return signal

    async def _transcribe(self, audio_data: bytes) -> str:
        """Call Whisper STT (A1 speech processing)."""
        try:
            import litellm
            import tempfile, os

            # Write to temp file (Whisper API needs a file)
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name

            try:
                response = await litellm.atranscription(
                    model="whisper-1",
                    file=open(temp_path, "rb"),
                )
                return response.text or ""
            finally:
                os.unlink(temp_path)
        except Exception as e:
            logger.warning("A1 STT failed: %s", e)
            return ""
```

Add `import logging` and `logger = logging.getLogger(__name__)` at top.

Add `llm_provider` parameter to `AuditoryCortexRight.__init__` as well (for future prosody analysis), but keep its processing logic the same for now.

- [ ] **Step 2: Wire in pipeline.py**

Update pipeline `__init__` to pass llm_provider to auditory cortex:
```python
self.auditory_cortex_l = AuditoryCortexLeft(llm_provider=llm_provider)
self.auditory_cortex_r = AuditoryCortexRight()
```

- [ ] **Step 3: Commit**

```bash
git add brain_agent/regions/auditory_cortex.py brain_agent/pipeline.py
git commit -m "feat(A1): Whisper STT for speech-to-text in auditory cortex"
```

---

### Task 5: Dashboard — FormData upload + mic recording

**Files:**
- Modify: `dashboard/src/stores/brainState.ts`
- Modify: `dashboard/src/components/MemoryPanel.tsx`

- [ ] **Step 1: Update submitChat to use FormData**

In `brainState.ts`, replace the `submitChat` function's fetch call:

```typescript
submitChat: async () => {
    const s = useBrainStore.getState()
    const text = s.chatInputText.trim()
    if ((!text && s.attachedFiles.length === 0) || s.chatLoading) return

    const msgFiles = s.attachedFiles.map((f) => ({ name: f.name, type: f.type, url: f.url }))
    s.addChatMessage({ role: 'user', text, files: msgFiles.length > 0 ? msgFiles : undefined, ts: Date.now() })

    set({ chatLoading: true, chatInputText: '' })
    useBrainStore.getState().clearThinkingSteps()

    try {
      // Build FormData with text + attached files
      const fd = new FormData()
      fd.append('text', text)

      // Convert blob URLs back to files
      for (const af of s.attachedFiles) {
        try {
          const resp = await fetch(af.url)
          const blob = await resp.blob()
          fd.append('files', blob, af.name)
        } catch { /* skip failed file */ }
      }

      const res = await fetch('/api/process', {
        method: 'POST',
        body: fd,  // No Content-Type header — browser sets multipart boundary
      })
      const data = await res.json()
      const responseText = data.response || data.error || 'No response'
      useBrainStore.getState().setLastResponse(responseText)
      const steps = [...useBrainStore.getState().thinkingSteps]
      useBrainStore.getState().addChatMessage({ role: 'brain', text: responseText, thinkingSteps: steps.length > 0 ? steps : undefined, ts: Date.now() })
    } catch {
      useBrainStore.getState().setLastResponse('Connection error')
      useBrainStore.getState().addChatMessage({ role: 'brain', text: 'Connection error', ts: Date.now() })
    }
    set({ chatLoading: false })
    useBrainStore.getState().clearAttachedFiles()
},
```

- [ ] **Step 2: Add mic recording handler**

Add a `submitAudio` action to brainState that records from mic, creates a blob, and sends via FormData:

```typescript
submitAudio: async (audioBlob: Blob) => {
    const s = useBrainStore.getState()
    if (s.chatLoading) return

    s.addChatMessage({ role: 'user', text: '🎤 (voice message)', ts: Date.now() })
    set({ chatLoading: true, isAudioMode: false, audioState: 'processing' })
    useBrainStore.getState().clearThinkingSteps()

    try {
      const fd = new FormData()
      fd.append('text', '')
      fd.append('files', audioBlob, 'recording.webm')

      const res = await fetch('/api/process', { method: 'POST', body: fd })
      const data = await res.json()
      const responseText = data.response || data.error || 'No response'
      useBrainStore.getState().setLastResponse(responseText)
      const steps = [...useBrainStore.getState().thinkingSteps]
      useBrainStore.getState().addChatMessage({ role: 'brain', text: responseText, thinkingSteps: steps.length > 0 ? steps : undefined, ts: Date.now() })
    } catch {
      useBrainStore.getState().setLastResponse('Connection error')
      useBrainStore.getState().addChatMessage({ role: 'brain', text: 'Connection error', ts: Date.now() })
    }
    set({ chatLoading: false, audioState: 'idle' })
},
```

- [ ] **Step 3: Wire AudioOrb to use MediaRecorder**

In `dashboard/src/components/AudioOrb.tsx`, add MediaRecorder logic:
- On mic button click → start MediaRecorder
- On stop → get blob → call `submitAudio(blob)`
- Visual feedback during recording (existing orb animation)

Read AudioOrb.tsx first to understand its current structure before modifying.

- [ ] **Step 4: Build dashboard**

```bash
cd dashboard && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/stores/brainState.ts dashboard/src/components/MemoryPanel.tsx dashboard/src/components/AudioOrb.tsx
git commit -m "feat(dashboard): FormData file upload + mic recording for multimodal input"
```

---

### Task 6: Install dependencies + integration test

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add PDF/DOCX dependencies**

```bash
uv add pypdf python-docx
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/core/ tests/regions/ -q --tb=short
```

- [ ] **Step 3: Final commit and push**

```bash
git add -A
git commit -m "feat(multimodal): image vision, audio STT, PDF/doc text extraction

Complete multimodal input pipeline:
- V1 Visual Cortex: vision LLM for image analysis
- A1 Auditory Cortex: Whisper STT for speech-to-text
- Server: multipart/form-data with file type classification
- PDF/DOCX: text extraction merged into text input
- Dashboard: FormData upload + MediaRecorder mic recording
- Config: ModalityConfig for per-modality model override"

git push origin main
```
