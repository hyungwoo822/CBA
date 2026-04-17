"""WebSocket event server for the brain-agent dashboard.
Streams real-time brain region activation events to connected clients.
Spec ref: Section 7.3 WebSocket Events."""
from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from dotenv import load_dotenv

load_dotenv()
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile
import aiosqlite


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


_chat_db_path: str | None = None

async def _get_chat_db() -> aiosqlite.Connection:
    """Get or create chat history DB connection."""
    global _chat_db_path
    if not _chat_db_path:
        _chat_db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "chat_history.db",
        )
    db = await aiosqlite.connect(_chat_db_path)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            files TEXT DEFAULT '[]',
            thinking_steps TEXT DEFAULT '[]',
            ts REAL NOT NULL
        )
    """)
    await db.commit()
    return db


EVENT_BUFFER_SIZE = 1000
MAX_FPS = 60
FRAME_INTERVAL = 1.0 / MAX_FPS


@dataclass
class DashboardEvent:
    event_type: str
    payload: dict[str, Any]
    timestamp: float = 0.0

    def to_json(self) -> str:
        return json.dumps({"type": self.event_type, "payload": self.payload, "ts": self.timestamp})


class EventBus:
    """Collects events from the pipeline and broadcasts to WebSocket clients."""

    def __init__(self, buffer_size: int = EVENT_BUFFER_SIZE):
        self._buffer: deque[DashboardEvent] = deque(maxlen=buffer_size)
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        import time
        evt = DashboardEvent(event_type=event_type, payload=payload, timestamp=time.time())
        self._buffer.append(evt)
        # Broadcast to all connected clients
        disconnected = []
        msg = evt.to_json()
        for ws in self._clients:
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self._clients.remove(ws)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)
        # Send recent buffer to new client
        for evt in self._buffer:
            try:
                await ws.send_text(evt.to_json())
            except Exception:
                break

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def get_recent(self, n: int = 50) -> list[dict]:
        return [{"type": e.event_type, "payload": e.payload, "ts": e.timestamp}
                for e in list(self._buffer)[-n:]]


# Global event bus instance
event_bus = EventBus()


def create_app(static_dir: str | None = None, agent: "BrainAgent | None" = None) -> FastAPI:
    """Create the FastAPI app for the dashboard.

    Parameters
    ----------
    static_dir : str | None
        Path to the React build directory to serve as static files.
    agent : BrainAgent | None
        An existing BrainAgent instance. If *None* one is created on startup
        with ``use_mock_embeddings=True``.
    """
    from brain_agent.agent import BrainAgent
    from brain_agent.dashboard.emitter import DashboardEmitter

    _state: dict[str, Any] = {"agent": agent, "owns_agent": agent is None}
    _emitter = DashboardEmitter()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        # Startup
        if _state["agent"] is None:
            # Auto-detect API key from any supported provider
            _PROVIDER_ENV_KEYS = [
                ("OPENAI_API_KEY", "openai"),
                ("ANTHROPIC_API_KEY", "anthropic"),
                ("GEMINI_API_KEY", "gemini"),
                ("GOOGLE_API_KEY", "gemini"),
                ("XAI_API_KEY", "xai"),
            ]
            api_key = None
            detected_provider = None
            for env_var, provider_name in _PROVIDER_ENV_KEYS:
                val = os.environ.get(env_var, "")
                if val:
                    if api_key is None:
                        api_key = val
                        detected_provider = provider_name
                    print(f"[brain-agent] {env_var}: detected (len={len(val)})")

            model = os.environ.get("BRAIN_AGENT_MODEL", None)
            print(f"[brain-agent] Provider: {detected_provider or 'none'}, Model: {model or 'default'}")

            _state["agent"] = BrainAgent(
                use_mock_embeddings=True,
                api_key=api_key,
                model=model,
            )
            await _state["agent"].initialize()
            print(f"[brain-agent] LLM provider: {_state['agent']._llm_provider}")
        # Wire emitter into pipeline
        _state["agent"].pipeline._emitter = _emitter

        # Start channel adapters with shared agent
        from brain_agent.channels.manager import ChannelManager
        channel_mgr = ChannelManager()
        await channel_mgr.start_all(_state["agent"])
        _state["channels"] = channel_mgr
        active = channel_mgr.status()
        if active:
            print(f"[brain-agent] Channels: {', '.join(c['name'] for c in active)}")
        else:
            print("[brain-agent] No channel tokens configured")

        yield

        # Shutdown channels first, then agent
        if _state.get("channels"):
            await _state["channels"].stop_all()
        if _state["agent"] and _state["owns_agent"]:
            await _state["agent"].close()

    app = FastAPI(title="Brain Agent Dashboard", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from brain_agent.dashboard.routers import (
        curation as curation_router,
        export as export_router,
        kg as kg_router,
        llm as llm_router,
        ontology as ontology_router,
        sources as sources_router,
        timeline as timeline_router,
        workspaces as workspaces_router,
    )

    app.include_router(workspaces_router.build_router(_state))
    app.include_router(kg_router.build_router(_state))
    app.include_router(ontology_router.build_router(_state))
    app.include_router(curation_router.build_router(_state))
    app.include_router(sources_router.build_router(_state))
    app.include_router(timeline_router.build_router(_state))
    app.include_router(export_router.build_router(_state))
    app.include_router(llm_router.build_router(_state))

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await event_bus.connect(websocket)
        # Send current brain state immediately on connect so dashboard
        # doesn't show stale 0.5 defaults after refresh
        agent_inst: BrainAgent | None = _state["agent"]
        if agent_inst and agent_inst._initialized:
            import time
            pipeline = agent_inst.pipeline
            nm = pipeline.neuromodulators.snapshot()
            init_event = json.dumps({
                "type": "neuromodulator",
                "payload": nm,
                "ts": time.time(),
            })
            try:
                await websocket.send_text(init_event)
            except Exception:
                pass
            # Also send current network mode
            mode_event = json.dumps({
                "type": "network_switch",
                "payload": {
                    "from": "",
                    "to": pipeline.network_ctrl.current_mode.value,
                    "trigger": "init",
                },
                "ts": time.time(),
            })
            try:
                await websocket.send_text(mode_event)
            except Exception:
                pass
        try:
            while True:
                data = await websocket.receive_text()
        except WebSocketDisconnect:
            await event_bus.disconnect(websocket)

    @app.get("/api/status")
    async def get_status():
        return {
            "status": "running",
            "clients": event_bus.client_count,
            "events_buffered": len(event_bus._buffer),
        }

    @app.get("/api/channels")
    async def get_channels():
        """List all connected channels with broadcast status."""
        channel_mgr = _state.get("channels")
        if not channel_mgr:
            return []
        return channel_mgr.status()

    @app.put("/api/channels/{name}/broadcast")
    async def set_channel_broadcast(name: str, body: dict):
        """Toggle broadcast for a channel. Body: {"enabled": true/false}"""
        channel_mgr = _state.get("channels")
        if not channel_mgr:
            return {"error": "no channel manager"}
        enabled = body.get("enabled", True)
        channel_mgr.set_broadcast(name, enabled)
        return {"status": "ok", "channel": name, "broadcast": enabled}

    @app.get("/api/events/recent")
    async def get_recent_events(n: int = 50):
        return event_bus.get_recent(n)

    @app.get("/api/state")
    async def get_brain_state():
        """Return current brain state (neuromodulators, mode, region activations).

        Called by dashboard on mount/refresh to restore UI without waiting
        for the next processing cycle.
        """
        agent_inst: BrainAgent | None = _state["agent"]
        if not agent_inst or not agent_inst._initialized:
            return {"error": "agent not initialized"}
        pipeline = agent_inst.pipeline
        return {
            "neuromodulators": pipeline.neuromodulators.snapshot(),
            "network_mode": pipeline.network_ctrl.current_mode.value,
            "regions": {
                r.name: r.activation_level for r in pipeline._all_regions()
            },
        }

    @app.get("/api/neuromodulators/history")
    async def get_neuromodulator_history(limit: int = 100):
        """Return neuromodulator time series for dashboard graphs."""
        agent_inst: BrainAgent | None = _state["agent"]
        if not agent_inst or not agent_inst._initialized:
            return {"history": []}
        try:
            history = await agent_inst.memory.brain_state.get_neuromodulator_history(limit)
            return {"history": history}
        except Exception as e:
            return {"history": [], "error": str(e)}

    @app.post("/api/process")
    async def process_message(
        text: str = Form(default=""),
        mode: str = Form(default="question"),
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
                extracted = extract_text_from_file(name, content)
                if extracted:
                    doc_texts.append(f"[Document: {name}]\n{extracted}")

        full_text = text
        if doc_texts:
            full_text = text + "\n\n" + "\n\n".join(doc_texts) if text else "\n\n".join(doc_texts)

        if not full_text.strip() and not image_bytes and not audio_bytes:
            return {"error": "empty message"}

        result = await agent_inst.process(full_text, image=image_bytes, audio=audio_bytes, interaction_mode=mode)
        pipeline = agent_inst.pipeline

        # Broadcast response to enabled channels
        channel_mgr = _state.get("channels")
        if channel_mgr:
            await channel_mgr.broadcast_response(result.response)

        await _emitter.region_activation("hippocampus", 0.7, "active")
        stats = await agent_inst.memory.stats()
        await _emitter.memory_flow(
            stats["sensory"], stats["working"],
            stats["staging"], stats["episodic"], stats["semantic"],
            stats.get("procedural", 0),
        )
        await _emitter.neuromodulator_update(**pipeline.neuromodulators.snapshot())

        return {
            "response": result.response,
            "network_mode": result.network_mode,
            "signals_processed": result.signals_processed,
            "actions": result.actions_taken,
        }

    @app.get("/api/memory/stats")
    async def get_memory_stats():
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"error": "agent not initialized"}
        stats = await agent_inst.memory.stats()
        return stats

    @app.get("/api/memory/working")
    async def get_working_memory():
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"error": "agent not initialized"}
        wm = agent_inst.memory.working
        items = []
        for slot_name, slot_items in wm._slots.items():
            for item in slot_items:
                items.append({
                    "content": item.content,
                    "slot": item.slot,
                    "reference_count": item.reference_count,
                })
        return {
            "items": items,
            "capacity": sum(wm._capacities.values()),
            "capacities": dict(wm._capacities),
        }

    @app.get("/api/memory/episodic")
    async def get_episodic_memory(limit: int = 50):
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"error": "agent not initialized"}
        episodes = await agent_inst.memory.episodic.get_recent(limit=limit)
        return {"episodes": [
            {
                "id": ep["id"],
                "timestamp": ep.get("timestamp", ""),
                "content": ep.get("content", ""),
                "strength": ep.get("strength", 1.0),
                "access_count": ep.get("access_count", 0),
                "emotional_tag": ep.get("emotional_tag", {}),
                "entities": ep.get("entities", {}),
            }
            for ep in episodes
        ]}

    @app.get("/api/memory/semantic")
    async def get_semantic_memory(limit: int = 50):
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"error": "agent not initialized"}
        store = agent_inst.memory.semantic
        relations = []
        try:
            if store._graph_db:
                async with store._graph_db.execute(
                    "SELECT source_node, relation, target_node, weight, "
                    "category, occurrence_count, origin "
                    "FROM knowledge_graph ORDER BY weight DESC LIMIT ?",
                    (limit,),
                ) as cursor:
                    rows = await cursor.fetchall()
                for row in rows:
                    relations.append({
                        "source": row[0],
                        "relation": row[1],
                        "target": row[2],
                        "weight": row[3],
                        "category": row[4] if len(row) > 4 else "GENERAL",
                        "occurrence_count": row[5] if len(row) > 5 else 1,
                        "origin": row[6] if len(row) > 6 else "unknown",
                    })
        except Exception:
            pass
        # Also include vector store count
        vector_count = 0
        try:
            vector_count = await store.count()
        except Exception:
            pass
        return {"relations": relations, "vector_count": vector_count}

    @app.get("/api/memory/procedural")
    async def get_procedural_memory(limit: int = 50):
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"error": "agent not initialized"}
        store = agent_inst.memory.procedural
        procedures = []
        try:
            procs = await store.get_all()
            for p in procs:
                procedures.append({
                    "id": p["id"],
                    "trigger_pattern": p["trigger_pattern"],
                    "strategy": p.get("strategy", ""),
                    "action_sequence": p["action_sequence"],
                    "success_rate": p["success_rate"],
                    "execution_count": p["execution_count"],
                    "stage": p["stage"],
                })
        except Exception:
            pass
        return {"procedures": procedures}

    @app.get("/api/memory/staging")
    async def get_staging_memory(limit: int = 50):
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"error": "agent not initialized"}
        items = await agent_inst.memory.staging.get_unconsolidated()
        return {"items": [
            {
                "id": item["id"],
                "timestamp": item.get("timestamp", ""),
                "content": item.get("content", ""),
                "strength": item.get("strength", 1.0),
                "emotional_tag": item.get("emotional_tag", {}),
                "consolidated": item.get("consolidated", False),
                "source_modality": item.get("source_modality", "text"),
                "entities": item.get("entities", {}),
            }
            for item in items[:limit]
        ]}

    @app.get("/api/memory/semantic/documents")
    async def get_semantic_documents(limit: int = 50):
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"error": "agent not initialized"}
        store = agent_inst.memory.semantic
        documents = []
        try:
            if store._collection:
                results = store._collection.peek(limit=limit)
                if results and results.get("ids"):
                    for i, doc_id in enumerate(results["ids"]):
                        documents.append({
                            "id": doc_id,
                            "content": results["documents"][i] if results.get("documents") else "",
                            "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                        })
        except Exception:
            pass
        return {"documents": documents}

    @app.get("/api/memory/hyperedges")
    async def get_hyperedges():
        """Return all cell assemblies (hyperedges)."""
        agent_inst = _state["agent"]
        if not agent_inst or not agent_inst.memory:
            return {"hyperedges": []}
        try:
            edges = await agent_inst.memory.semantic.get_hyperedges()
            return {"hyperedges": edges}
        except Exception:
            return {"hyperedges": []}

    @app.get("/api/memory/search")
    async def search_memory(q: str = "", top_k: int = 10):
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"error": "agent not initialized"}
        if not q.strip():
            return {"results": []}
        results = await agent_inst.memory.retrieve(query=q, top_k=top_k)
        return {"results": [
            {
                "id": r.get("id", ""),
                "content": r.get("content", ""),
                "source": r.get("source", ""),
                "score": round(r.get("score", 0), 4),
                "relevance": round(r.get("relevance", 0), 4),
            }
            for r in results
        ]}

    @app.post("/api/memory/reset")
    async def reset_memory():
        """Clear all memory stores (staging, episodic, procedural, semantic)."""
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"error": "agent not initialized"}
        try:
            await agent_inst.memory.close()
            import shutil, os
            data_dir = agent_inst._data_dir
            for name in ("staging.db", "episodic.db", "procedural.db", "graph.db"):
                path = os.path.join(data_dir, name)
                if os.path.exists(path):
                    os.remove(path)
            chroma_dir = os.path.join(data_dir, "chroma")
            if os.path.isdir(chroma_dir):
                shutil.rmtree(chroma_dir)
            await agent_inst.memory.initialize()
            agent_inst.memory.working.clear()
            return {"status": "ok", "message": "All memory stores cleared"}
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/chat/history")
    async def get_chat_history(limit: int = 200):
        """Load chat history from DB — shared across all browsers."""
        try:
            db = await _get_chat_db()
            async with db.execute(
                "SELECT role, text, files, thinking_steps, ts FROM chat_messages ORDER BY ts DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
            await db.close()
            messages = []
            for row in reversed(rows):  # Reverse to get chronological order
                msg = {"role": row[0], "text": row[1], "ts": row[4]}
                files = json.loads(row[2]) if row[2] and row[2] != '[]' else None
                steps = json.loads(row[3]) if row[3] and row[3] != '[]' else None
                if files:
                    msg["files"] = files
                if steps:
                    msg["thinkingSteps"] = steps
                messages.append(msg)
            return {"messages": messages}
        except Exception as e:
            return {"messages": [], "error": str(e)}

    @app.post("/api/chat/save")
    async def save_chat_message(body: dict):
        """Save a single chat message to DB."""
        try:
            db = await _get_chat_db()
            await db.execute(
                "INSERT INTO chat_messages (role, text, files, thinking_steps, ts) VALUES (?, ?, ?, ?, ?)",
                (
                    body.get("role", "user"),
                    body.get("text", ""),
                    json.dumps(body.get("files", []), ensure_ascii=False),
                    json.dumps(body.get("thinkingSteps", []), ensure_ascii=False),
                    body.get("ts", 0),
                ),
            )
            await db.commit()
            await db.close()
            return {"status": "ok"}
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/chat/clear")
    async def clear_chat_history():
        """Clear all chat history."""
        try:
            db = await _get_chat_db()
            await db.execute("DELETE FROM chat_messages")
            await db.commit()
            await db.close()
            return {"status": "ok"}
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/profile")
    async def get_profile():
        """Read user profile from identity_facts + SOUL.md."""
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"error": "agent not initialized"}

        result = {}

        # User profile: rendered from identity_facts (single source of truth)
        try:
            result["USER.md"] = await agent_inst.memory.render_user_context()
        except Exception:
            result["USER.md"] = ""

        # SOUL.md: still a file (agent self-schema)
        data_dir = agent_inst._data_dir
        soul_path = os.path.join(data_dir, "SOUL.md")
        if os.path.isfile(soul_path):
            with open(soul_path, "r", encoding="utf-8") as f:
                result["SOUL.md"] = f.read()
        else:
            result["SOUL.md"] = ""

        # Also return raw identity_facts for dashboard display
        try:
            identity = await agent_inst.memory.retrieve_identity()
            result["identity_facts"] = identity
        except Exception:
            result["identity_facts"] = {"self_model": [], "user_model": []}

        return result

    @app.put("/api/profile")
    async def update_profile(body: dict):
        """Update SOUL.md and/or identity_facts from request body."""
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"error": "agent not initialized"}
        updated = []

        # SOUL.md: still a file
        if "SOUL.md" in body and isinstance(body["SOUL.md"], str):
            data_dir = agent_inst._data_dir
            os.makedirs(data_dir, exist_ok=True)
            path = os.path.join(data_dir, "SOUL.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(body["SOUL.md"])
            updated.append("SOUL.md")

        # identity_facts: update via semantic store
        if "identity_facts" in body and isinstance(body["identity_facts"], dict):
            store = agent_inst.memory.semantic
            for fact in body["identity_facts"].get("user_model", []):
                if fact.get("key") and fact.get("value"):
                    await store.add_identity_fact(
                        "user_model", fact["key"], fact["value"],
                        source="dashboard", confidence=1.0,
                    )
            for fact in body["identity_facts"].get("self_model", []):
                if fact.get("key") and fact.get("value"):
                    await store.add_identity_fact(
                        "self_model", fact["key"], fact["value"],
                        source="dashboard", confidence=1.0,
                    )
            updated.append("identity_facts")

        return {"status": "ok", "updated": updated}

    @app.get("/api/cloning-score")
    async def get_cloning_score():
        """Get the user-agent knowledge graph similarity score."""
        agent_inst = _state["agent"]
        if not agent_inst:
            return {"cloning_score": 0.0, "error": "agent not initialized"}
        try:
            score = await agent_inst.memory.semantic.compute_cloning_score()
            return score
        except Exception as e:
            return {"cloning_score": 0.0, "error": str(e)}

    # Serve static files (React build) if directory exists.
    # IMPORTANT: This must be LAST — mounted sub-apps match before routes,
    # so we use a catch-all GET route instead to avoid blocking /api and /ws.
    if static_dir and os.path.isdir(static_dir):
        from fastapi.responses import FileResponse, Response
        import mimetypes

        @app.get("/{full_path:path}")
        async def serve_static(full_path: str):
            # Try exact file first
            file_path = os.path.join(static_dir, full_path)
            if os.path.isfile(file_path):
                content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
                return FileResponse(file_path, media_type=content_type)
            # Fallback to index.html (SPA routing)
            index = os.path.join(static_dir, "index.html")
            if os.path.isfile(index):
                return FileResponse(index, media_type="text/html")
            return Response(status_code=404)

    return app


def run_server(port: int = 3000, static_dir: str | None = None):
    """Run the dashboard server."""
    import uvicorn
    app = create_app(static_dir=static_dir)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
