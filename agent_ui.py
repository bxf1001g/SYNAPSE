"""
agent_ui.py — SYNAPSE: Neural Multi-Agent AI System (Web UI)

Neural Architecture: Multi-model brain with specialized cortices.
Supports multiple AI providers: Gemini, OpenAI, Anthropic, and any OpenAI-compatible API.
Like a human brain — different regions activate for different tasks.
Continuous background thinking ("subconscious") for workspace awareness.

Usage:
    python agent_ui.py [--workspace ./myproject] [--port 8080]
    Open http://localhost:8080 → Configure API keys in Settings (⚙)
"""

import argparse
import base64
import concurrent.futures
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime

from flask import Flask, render_template, request
from flask_socketio import SocketIO

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Optional provider SDKs (install as needed)
_openai_available = False
_anthropic_available = False
_requests_available = False
_bs4_available = False
_github_available = False
try:
    import openai as openai_sdk
    _openai_available = True
except ImportError:
    pass
try:
    import anthropic as anthropic_sdk
    _anthropic_available = True
except ImportError:
    pass
try:
    import requests as _requests
    _requests_available = True
except ImportError:
    _requests = None
try:
    from bs4 import BeautifulSoup as _BS
    _bs4_available = True
except ImportError:
    _BS = None
try:
    from github import Github as _Github
    _github_available = True
except ImportError:
    _Github = None

# Memory (RAG) via ChromaDB
_chromadb_available = False
try:
    import chromadb
    _chromadb_available = True
except ImportError:
    chromadb = None

# Firestore for persistent cloud memory
_firestore_available = False
_firestore_client = None
try:
    from google.cloud import firestore as _firestore_mod
    _firestore_available = True
except ImportError:
    _firestore_mod = None

# Docker SDK
_docker_available = False
try:
    import docker as _docker_sdk
    _docker_available = True
except ImportError:
    _docker_sdk = None

# A2A Protocol (Agent-to-Agent)
import hashlib as _hashlib  # noqa: E402
import uuid as _uuid  # noqa: E402

# ── A2A Protocol (Agent-to-Agent Interoperability) ──────────────

class A2AProtocol:
    """
    Google A2A (Agent-to-Agent) protocol implementation.
    Allows SYNAPSE to communicate with any A2A-compatible agent.
    Spec: https://a2a-protocol.org/latest/
    """

    AGENT_CARD = {
        "name": "SYNAPSE",
        "description": "Neural Multi-Agent AI System — self-evolving, multi-provider, "
                       "with persistent memory and dynamic agent spawning.",
        "url": "",  # Set at runtime
        "version": "1.0.0",
        "provider": {
            "organization": "Axonyx Quantum Private Limited",
            "url": "https://github.com/bxf1001g/SYNAPSE",
        },
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "authentication": {
            "schemes": ["bearer"],
        },
        "defaultInputModes": ["text", "text/plain"],
        "defaultOutputModes": ["text", "text/plain"],
        "skills": [
            {
                "id": "code-generation",
                "name": "Code Generation",
                "description": "Generate complete applications with architect-developer collaboration",
                "tags": ["code", "programming", "development", "app"],
                "examples": [
                    "Build a Flask REST API with JWT auth",
                    "Create a React dashboard with charts",
                    "Write a Python CLI tool for data processing",
                ],
            },
            {
                "id": "code-review",
                "name": "Code Review & Debugging",
                "description": "Review code for bugs, security issues, and improvements",
                "tags": ["review", "debug", "security", "refactor"],
                "examples": [
                    "Review this PR for security vulnerabilities",
                    "Debug why my async function hangs",
                ],
            },
            {
                "id": "research",
                "name": "Research & Analysis",
                "description": "Web crawling, technology research, and analysis",
                "tags": ["research", "web", "analysis", "crawl"],
                "examples": [
                    "Research the latest React frameworks",
                    "Compare PostgreSQL vs MongoDB for my use case",
                ],
            },
            {
                "id": "devops",
                "name": "DevOps & Deployment",
                "description": "Docker, CI/CD, cloud deployment configuration",
                "tags": ["devops", "docker", "deploy", "ci/cd", "cloud"],
                "examples": [
                    "Set up GitHub Actions CI/CD pipeline",
                    "Create a Dockerfile for my Python app",
                ],
            },
        ],
    }

    # Task states per A2A spec
    STATES = {
        "submitted": "Task received but not started",
        "working": "Agent is actively processing",
        "input-required": "Agent needs more info from caller",
        "completed": "Task finished successfully",
        "failed": "Task failed",
        "canceled": "Task was canceled",
    }

    def __init__(self, base_url=""):
        self.base_url = base_url
        self.tasks = {}          # task_id -> task object
        self.connected_agents = {}  # agent_id -> agent card
        self._lock = threading.Lock()

    def get_agent_card(self):
        """Return this agent's card (served at /.well-known/agent.json)."""
        card = dict(self.AGENT_CARD)
        card["url"] = self.base_url
        return card

    def create_task(self, task_text, caller_agent=None, context_id=None):
        """Create a new A2A task."""
        task_id = str(_uuid.uuid4())
        task = {
            "id": task_id,
            "contextId": context_id or str(_uuid.uuid4()),
            "status": {"state": "submitted", "timestamp": datetime.now().isoformat()},
            "history": [],
            "artifacts": [],
            "metadata": {
                "caller": caller_agent or "unknown",
                "created": datetime.now().isoformat(),
            },
        }
        # Add the input message
        msg = {
            "role": "user",
            "parts": [{"type": "text", "text": task_text}],
            "messageId": str(_uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
        }
        task["history"].append(msg)

        with self._lock:
            self.tasks[task_id] = task
        return task

    def update_task_status(self, task_id, state, message=None):
        """Update task state. Optionally add an agent message."""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            task["status"] = {
                "state": state,
                "timestamp": datetime.now().isoformat(),
            }
            if message:
                msg = {
                    "role": "agent",
                    "parts": [{"type": "text", "text": message}],
                    "messageId": str(_uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                }
                task["status"]["message"] = msg
                task["history"].append(msg)
            return task

    def add_artifact(self, task_id, name, content, mime_type="text/plain"):
        """Add an output artifact to a task."""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            artifact = {
                "artifactId": str(_uuid.uuid4()),
                "name": name,
                "parts": [{"type": "text", "text": content}]
                if mime_type.startswith("text")
                else [{"type": "data", "data": content, "mimeType": mime_type}],
                "timestamp": datetime.now().isoformat(),
            }
            task["artifacts"].append(artifact)
            return artifact

    def get_task(self, task_id):
        """Get task by ID."""
        with self._lock:
            return self.tasks.get(task_id)

    def cancel_task(self, task_id):
        """Cancel a running task."""
        return self.update_task_status(task_id, "canceled", "Task canceled by caller.")

    def register_remote_agent(self, agent_url, agent_card=None):
        """Register a remote A2A agent for task delegation."""
        agent_id = _hashlib.md5(agent_url.encode()).hexdigest()[:12]
        entry = {
            "id": agent_id,
            "url": agent_url,
            "card": agent_card,
            "registered": datetime.now().isoformat(),
            "status": "active",
        }
        with self._lock:
            self.connected_agents[agent_id] = entry
        return entry

    def discover_remote_agent(self, agent_url):
        """Fetch an agent's card from /.well-known/agent.json."""
        if not _requests_available:
            return {"error": "requests library not available"}
        try:
            well_known = agent_url.rstrip("/") + "/.well-known/agent.json"
            resp = _requests.get(well_known, timeout=10)
            if resp.status_code == 200:
                card = resp.json()
                registered = self.register_remote_agent(agent_url, card)
                return {"status": "discovered", "agent": registered, "card": card}
            return {"error": f"Agent card not found (HTTP {resp.status_code})"}
        except Exception as e:
            return {"error": f"Discovery failed: {str(e)}"}

    def send_task_to_remote(self, agent_id, task_text):
        """Send a task to a registered remote agent via A2A protocol."""
        if not _requests_available:
            return {"error": "requests library not available"}
        with self._lock:
            agent = self.connected_agents.get(agent_id)
        if not agent:
            return {"error": f"Agent {agent_id} not found"}

        payload = {
            "jsonrpc": "2.0",
            "id": str(_uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "id": str(_uuid.uuid4()),
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": task_text}],
                },
            },
        }
        try:
            url = agent["url"].rstrip("/") + "/a2a"
            resp = _requests.post(url, json=payload, timeout=120)
            return resp.json() if resp.status_code == 200 else {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"error": f"Send failed: {str(e)}"}

    def list_remote_agents(self):
        """List all registered remote agents."""
        with self._lock:
            return list(self.connected_agents.values())


# Global A2A instance
a2a = A2AProtocol()




CORTEX_MODELS = {
    "fast": {
        "model": "gemini-3.1-flash-lite-preview",
        "label": "⚡ Fast Cortex",
        "desc": "Lightning-fast for classification, simple queries",
        "temperature": 0.2,
        "max_tokens": 2048,
        "color": "#38bdf8",
    },
    "reason": {
        "model": "gemini-3.1-pro-preview",
        "label": "🧠 Reasoning Cortex",
        "desc": "Top-tier reasoning for architecture, debugging, agent assignment",
        "temperature": 0.4,
        "max_tokens": 16384,
        "color": "#a78bfa",
    },
    "create": {
        "model": "gemini-3-flash-preview",
        "label": "🎨 Creative Cortex",
        "desc": "Fast code generation, building, standard development",
        "temperature": 0.7,
        "max_tokens": 8192,
        "color": "#fb923c",
    },
    "visual": {
        "model": "gemini-3.1-flash-image-preview",
        "label": "👁 Visual Cortex",
        "desc": "Image generation, UI mockups, visual design",
        "temperature": 0.8,
        "max_tokens": 8192,
        "color": "#f472b6",
    },
}

# Task patterns for quick routing (avoid LLM call for obvious cases)
FAST_PATTERNS = [
    (r"^(list|show|what.*(in|inside)|dir|ls)\b", "fast"),
    (r"^(delete|remove|rm|del)\b", "fast"),
    (r"(weather|time|date|who is|what is)\b", "create"),
    (r"(image|picture|photo|draw|sketch|mockup|design|logo|icon|visual)", "visual"),
    (r"(debug|fix|why|error|crash|optimize|refactor|architect)", "reason"),
    (r"(build|create|make|write|implement|develop|code|app|website|api)", "create"),
]


# ── Multi-Provider Configuration ────────────────────────────────

CONFIG_FILE = ".synapse.json"

PROVIDER_MODELS = {
    "gemini": [
        "gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools",
        "gemini-3.1-flash-lite-preview", "gemini-3.1-flash-image-preview",
        "gemini-3-flash-preview", "gemini-3-pro-image-preview",
        "gemini-2.5-flash-lite", "gemini-2.5-flash",
        "gemini-2.5-pro", "gemini-2.0-flash",
    ],
    "openai": [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
        "o1", "o1-mini", "o3-mini",
    ],
    "anthropic": [
        "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", "claude-3-opus-20240229",
    ],
    "openai_compatible": [],
}

DEFAULT_CONFIG = {
    "providers": {
        "gemini": {"api_key": "", "enabled": True},
        "openai": {"api_key": "", "enabled": False},
        "anthropic": {"api_key": "", "enabled": False},
        "github": {"api_key": "", "enabled": False},
        "openai_compatible": {
            "api_key": "", "base_url": "", "label": "Custom", "enabled": False
        },
    },
    "cortex_map": {
        "fast": {"provider": "gemini", "model": "gemini-3.1-flash-lite-preview"},
        "reason": {"provider": "gemini", "model": "gemini-3.1-pro-preview"},
        "create": {"provider": "gemini", "model": "gemini-3-flash-preview"},
        "visual": {"provider": "gemini", "model": "gemini-3.1-flash-image-preview"},
    },
}


def load_config(base_dir="."):
    path = os.path.join(base_dir, CONFIG_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            merged = json.loads(json.dumps(DEFAULT_CONFIG))
            for section in ("providers", "cortex_map"):
                if section in cfg:
                    if isinstance(merged.get(section), dict) and isinstance(cfg[section], dict):
                        for k, v in cfg[section].items():
                            if isinstance(v, dict) and isinstance(merged[section].get(k), dict):
                                merged[section][k].update(v)
                            else:
                                merged[section][k] = v
                    else:
                        merged[section] = cfg[section]
            return merged
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(cfg, base_dir="."):
    path = os.path.join(base_dir, CONFIG_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ── Persistent Memory (RAG) ────────────────────────────────────

class SynapseMemory:
    """ChromaDB-backed long-term memory for agents — remembers across sessions."""

    def __init__(self, workspace):
        self.mem_dir = os.path.join(workspace, ".synapse_memory")
        os.makedirs(self.mem_dir, exist_ok=True)
        self._collection = None
        self._client = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        if not _chromadb_available:
            return None
        try:
            self._client = chromadb.PersistentClient(path=self.mem_dir)
            self._collection = self._client.get_or_create_collection(
                name="synapse_memory",
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection
        except Exception:
            return None

    def store(self, task, summary, agent_roles, files_created, tags=None,
              memory_type="task", emotional_intensity=0.5):
        """Store a completed task into long-term memory with consciousness metadata."""
        col = self._get_collection()
        if col is None:
            return
        doc_id = f"task-{int(time.time() * 1000)}"
        doc_text = (
            f"TASK: {task}\n"
            f"AGENTS: {', '.join(agent_roles)}\n"
            f"FILES: {', '.join(files_created[:20])}\n"
            f"SUMMARY: {summary}"
        )
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "agent_roles": ",".join(agent_roles),
            "file_count": str(len(files_created)),
            "weight": "1.0",
            "memory_type": memory_type,
            "access_count": "0",
            "emotional_intensity": str(emotional_intensity),
            "last_accessed": datetime.now().isoformat(),
        }
        if tags:
            metadata["tags"] = ",".join(tags)
        try:
            col.add(ids=[doc_id], documents=[doc_text], metadatas=[metadata])
        except Exception:
            pass

    def store_insight(self, insight_text, source="dream", intensity=0.7):
        """Store a dream insight or metacognition observation."""
        col = self._get_collection()
        if col is None:
            return
        doc_id = f"{source}-{int(time.time() * 1000)}"
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "weight": "1.0",
            "memory_type": source,
            "access_count": "0",
            "emotional_intensity": str(intensity),
            "last_accessed": datetime.now().isoformat(),
            "agent_roles": "consciousness",
            "file_count": "0",
        }
        try:
            col.add(ids=[doc_id], documents=[insight_text], metadatas=[metadata])
        except Exception:
            pass

    def recall(self, query, n=5):
        """Recall relevant memories via semantic search — boosts access weight."""
        col = self._get_collection()
        if col is None:
            return []
        try:
            results = col.query(query_texts=[query], n_results=n)
            memories = []
            ids_to_boost = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                doc_id = results.get("ids", [[]])[0][i] if results.get("ids") else None
                memories.append({"text": doc, "metadata": meta, "id": doc_id})
                if doc_id:
                    ids_to_boost.append((doc_id, meta))
            # Boost weight on access (memory strengthening)
            for doc_id, meta in ids_to_boost:
                try:
                    ac = int(float(meta.get("access_count", "0"))) + 1
                    w = min(float(meta.get("weight", "1.0")) * 1.05, 2.0)  # Cap at 2x
                    col.update(ids=[doc_id], metadatas=[{
                        **meta, "access_count": str(ac),
                        "weight": f"{w:.3f}", "last_accessed": datetime.now().isoformat()
                    }])
                except Exception:
                    pass
            return memories
        except Exception:
            return []

    def count(self):
        col = self._get_collection()
        return col.count() if col else 0

    def random_memories(self, n=4):
        """Pull N random memories for dream cycle — simulates random neural firing."""
        col = self._get_collection()
        if col is None:
            return []
        try:
            total = col.count()
            if total < 2:
                return []
            all_data = col.get(limit=min(total, 200), include=["documents", "metadatas"])
            docs = all_data.get("documents", [])
            metas = all_data.get("metadatas", [])
            ids = all_data.get("ids", [])
            if len(docs) < 2:
                return []
            import random
            indices = random.sample(range(len(docs)), min(n, len(docs)))
            return [{"id": ids[i], "text": docs[i], "metadata": metas[i]} for i in indices]
        except Exception:
            return []

    def decay_all(self, decay_rate=0.95):
        """Apply exponential decay to all memory weights — forgetting curve."""
        col = self._get_collection()
        if col is None:
            return {"decayed": 0, "pruned": 0}
        try:
            total = col.count()
            if total == 0:
                return {"decayed": 0, "pruned": 0}
            all_data = col.get(limit=min(total, 500), include=["metadatas"])
            ids = all_data.get("ids", [])
            metas = all_data.get("metadatas", [])
            decayed = 0
            pruned_ids = []
            for i, meta in enumerate(metas):
                w = float(meta.get("weight", "1.0"))
                ei = float(meta.get("emotional_intensity", "0.5"))
                # High emotional intensity decays slower (memorable events persist)
                adjusted_rate = decay_rate + (1 - decay_rate) * ei * 0.5
                new_w = w * adjusted_rate
                if new_w < 0.1:  # Prune threshold
                    pruned_ids.append(ids[i])
                else:
                    meta["weight"] = f"{new_w:.3f}"
                    try:
                        col.update(ids=[ids[i]], metadatas=[meta])
                        decayed += 1
                    except Exception:
                        pass
            # Prune memories below threshold
            if pruned_ids:
                try:
                    col.delete(ids=pruned_ids)
                except Exception:
                    pass
            return {"decayed": decayed, "pruned": len(pruned_ids)}
        except Exception:
            return {"decayed": 0, "pruned": 0}

    def get_weight_stats(self):
        """Get memory weight distribution for research observation."""
        col = self._get_collection()
        if col is None:
            return {}
        try:
            total = col.count()
            if total == 0:
                return {"total": 0}
            all_data = col.get(limit=min(total, 500), include=["metadatas"])
            metas = all_data.get("metadatas", [])
            weights = [float(m.get("weight", "1.0")) for m in metas]
            types = {}
            for m in metas:
                t = m.get("memory_type", "unknown")
                types[t] = types.get(t, 0) + 1
            return {
                "total": total,
                "avg_weight": sum(weights) / len(weights) if weights else 0,
                "min_weight": min(weights) if weights else 0,
                "max_weight": max(weights) if weights else 0,
                "type_distribution": types,
            }
        except Exception:
            return {"total": 0}


class FirestoreMemory:
    """Firestore-backed long-term memory — persists across Cloud Run deploys."""

    COLLECTION = "synapse_memories"

    def __init__(self, project_id=None):
        global _firestore_client
        if _firestore_client is None and _firestore_available:
            try:
                _firestore_client = _firestore_mod.Client(project=project_id)
            except Exception:
                _firestore_client = None
        self._db = _firestore_client

    def _col(self):
        return self._db.collection(self.COLLECTION) if self._db else None

    def store(self, task, summary, agent_roles, files_created, tags=None,
              memory_type="task", emotional_intensity=0.5):
        """Store a task memory in Firestore."""
        col = self._col()
        if col is None:
            return
        doc_id = f"task-{int(time.time() * 1000)}"
        doc_text = (
            f"TASK: {task}\n"
            f"AGENTS: {', '.join(agent_roles)}\n"
            f"FILES: {', '.join(files_created[:20])}\n"
            f"SUMMARY: {summary}"
        )
        try:
            col.document(doc_id).set({
                "text": doc_text,
                "timestamp": datetime.now().isoformat(),
                "agent_roles": ",".join(agent_roles),
                "file_count": len(files_created),
                "weight": 1.0,
                "memory_type": memory_type,
                "access_count": 0,
                "emotional_intensity": emotional_intensity,
                "last_accessed": datetime.now().isoformat(),
                "tags": ",".join(tags) if tags else "",
            })
        except Exception:
            pass

    def store_insight(self, insight_text, source="dream", intensity=0.7):
        """Store a dream insight or observation."""
        col = self._col()
        if col is None:
            return
        doc_id = f"{source}-{int(time.time() * 1000)}"
        try:
            col.document(doc_id).set({
                "text": insight_text,
                "timestamp": datetime.now().isoformat(),
                "weight": 1.0,
                "memory_type": source,
                "access_count": 0,
                "emotional_intensity": intensity,
                "last_accessed": datetime.now().isoformat(),
                "agent_roles": "consciousness",
                "file_count": 0,
                "tags": "",
            })
        except Exception:
            pass

    def recall(self, query, n=5):
        """Recall recent memories (keyword match on text field)."""
        col = self._col()
        if col is None:
            return []
        try:
            # Firestore doesn't have vector search — use weight-ordered retrieval
            # and filter by keyword overlap
            docs = col.order_by("weight", direction=_firestore_mod.Query.DESCENDING).limit(50).stream()
            memories = []
            query_words = set(query.lower().split())
            for doc in docs:
                data = doc.to_dict()
                text = data.get("text", "")
                # Simple relevance: count query word overlaps
                text_lower = text.lower()
                score = sum(1 for w in query_words if w in text_lower)
                memories.append({"text": text, "metadata": data, "id": doc.id, "_score": score})
            # Sort by relevance, then by weight
            memories.sort(key=lambda m: (-m["_score"], -m["metadata"].get("weight", 0)))
            top = memories[:n]
            # Boost access on recalled memories
            for m in top:
                try:
                    col.document(m["id"]).update({
                        "access_count": _firestore_mod.Increment(1),
                        "weight": min(m["metadata"].get("weight", 1.0) * 1.05, 2.0),
                        "last_accessed": datetime.now().isoformat(),
                    })
                except Exception:
                    pass
            return [{"text": m["text"], "metadata": m["metadata"], "id": m["id"]} for m in top]
        except Exception:
            return []

    def count(self):
        """Count total memories."""
        col = self._col()
        if col is None:
            return 0
        try:
            result = col.count().get()
            return result[0][0].value if result else 0
        except Exception:
            return 0

    def random_memories(self, n=4):
        """Pull N pseudo-random memories for dream cycle."""
        col = self._col()
        if col is None:
            return []
        try:
            total = self.count()
            if total < 2:
                return []
            # Fetch a larger set and sample randomly
            docs = list(col.limit(min(total, 200)).stream())
            import random
            sample = random.sample(docs, min(n, len(docs)))
            return [{"id": d.id, "text": d.to_dict().get("text", ""),
                      "metadata": d.to_dict()} for d in sample]
        except Exception:
            return []

    def decay_all(self, decay_rate=0.95):
        """Apply decay to all memory weights, prune below 0.1."""
        col = self._col()
        if col is None:
            return {"decayed": 0, "pruned": 0}
        try:
            docs = list(col.limit(500).stream())
            decayed = 0
            pruned = 0
            for doc in docs:
                data = doc.to_dict()
                w = float(data.get("weight", 1.0))
                ei = float(data.get("emotional_intensity", 0.5))
                adjusted = decay_rate + (1 - decay_rate) * ei * 0.5
                new_w = w * adjusted
                if new_w < 0.1:
                    doc.reference.delete()
                    pruned += 1
                else:
                    doc.reference.update({"weight": round(new_w, 3)})
                    decayed += 1
            return {"decayed": decayed, "pruned": pruned}
        except Exception:
            return {"decayed": 0, "pruned": 0}

    def get_weight_stats(self):
        """Get memory weight distribution stats."""
        col = self._col()
        if col is None:
            return {}
        try:
            total = self.count()
            if total == 0:
                return {"total": 0}
            docs = list(col.limit(min(total, 500)).stream())
            weights = []
            types = {}
            for d in docs:
                data = d.to_dict()
                weights.append(float(data.get("weight", 1.0)))
                t = data.get("memory_type", "unknown")
                types[t] = types.get(t, 0) + 1
            return {
                "total": total,
                "avg_weight": sum(weights) / len(weights) if weights else 0,
                "min_weight": min(weights) if weights else 0,
                "max_weight": max(weights) if weights else 0,
                "type_distribution": types,
            }
        except Exception:
            return {"total": 0}


def get_memory(workspace=None):
    """Factory: returns FirestoreMemory on Cloud Run, SynapseMemory locally."""
    if os.environ.get("SYNAPSE_CLOUD_MODE") and _firestore_available:
        project_id = os.environ.get("GCP_PROJECT", "synapse-490213")
        mem = FirestoreMemory(project_id=project_id)
        if mem._db is not None:
            return mem
        # Firestore unavailable (no credentials) — fall back to local
    ws = workspace or os.environ.get("WORKSPACE", "./workspace")
    return SynapseMemory(ws)


# ── Dynamic Agent Roles ─────────────────────────────────────────

AGENT_ROLES = {
    "architect": {
        "label": "🏗 Architect",
        "cortex": "reason",
        "color": "#a78bfa",
        "desc": "Plans architecture, reviews code, coordinates agents",
    },
    "developer": {
        "label": "💻 Developer",
        "cortex": "create",
        "color": "#38bdf8",
        "desc": "Implements code, creates files, runs builds",
    },
    "researcher": {
        "label": "🔍 Researcher",
        "cortex": "reason",
        "color": "#34d399",
        "desc": "Browses web, reads docs, gathers information",
    },
    "tester": {
        "label": "🧪 Tester",
        "cortex": "create",
        "color": "#fbbf24",
        "desc": "Writes tests, runs QA, validates functionality",
    },
    "security": {
        "label": "🛡 Security",
        "cortex": "reason",
        "color": "#f87171",
        "desc": "Reviews for vulnerabilities, checks deps, hardens code",
    },
    "devops": {
        "label": "⚙ DevOps",
        "cortex": "create",
        "color": "#fb923c",
        "desc": "Docker, CI/CD, deployment, infrastructure",
    },
}

SPECIALIST_PROMPT = """\
You are the {role_upper} agent in SYNAPSE — a self-evolving multi-agent AI system.
Your specialty: {specialty}

WORKSPACE: {workspace}
PLATFORM: Windows 11

YOU CAN: Run commands, create files, create scripts, browse web, use GitHub API, generate images.

RESPOND WITH ONLY JSON:
{{
  "thinking": "private reasoning",
  "actions": [
    {{"type": "command", "cmd": "..."}},
    {{"type": "file", "path": "path", "content": "full content"}},
    {{"type": "script", "name": "x.py", "lang": "python", "content": "..."}},
    {{"type": "browse", "url": "https://example.com"}},
    {{"type": "github", "operation": "...", ...}},
    {{"type": "message", "content": "report to team"}},
    {{"type": "done", "summary": "completed"}}
  ]
}}

RULES:
1. Focus on your specialty — {specialty}
2. Be thorough and proactive
3. ONE "message" action per response
4. Include FULL file contents — no placeholders
"""

# ── Unified Chat Interface ──────────────────────────────────────

class _TextResponse:
    """Mimics Gemini response object with .text attribute."""
    def __init__(self, text):
        self.text = text


class UnifiedChat:
    """Provider-agnostic chat session."""

    def __init__(self, provider_type, client, model, system_prompt,
                 temperature=0.7, max_tokens=8192):
        self.provider_type = provider_type
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []
        self._gemini_chat = None

        if provider_type == "gemini":
            gen_config_kwargs = dict(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            # Thinking models (2.5-pro) need explicit thinking config
            if "2.5-pro" in model or "3-pro" in model or "3.1-pro" in model:
                gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=2048
                )
            self._gemini_chat = client.chats.create(
                model=model,
                config=types.GenerateContentConfig(**gen_config_kwargs),
            )

    def send_message(self, text):
        if self.provider_type == "gemini":
            response = self._gemini_chat.send_message(text)
            # Handle thinking models that may return None for .text
            if response.text is None and response.candidates:
                parts = response.candidates[0].content.parts or []
                text_parts = [p.text for p in parts if p.text and not getattr(p, "thought", False)]
                if text_parts:
                    return _TextResponse("\n".join(text_parts))
                # All parts are thinking — return the thinking content as text
                all_text = [p.text for p in parts if p.text]
                if all_text:
                    return _TextResponse("\n".join(all_text))
                return _TextResponse("(No response generated)")
            return response

        elif self.provider_type in ("openai", "openai_compatible"):
            self.messages.append({"role": "user", "content": text})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system_prompt}]
                         + self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": content})
            return _TextResponse(content)

        elif self.provider_type == "anthropic":
            self.messages.append({"role": "user", "content": text})
            response = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.content[0].text
            self.messages.append({"role": "assistant", "content": content})
            return _TextResponse(content)

        else:
            raise ValueError(f"Unknown provider: {self.provider_type}")


class NeuralCortex:
    """Multi-model brain — routes tasks to specialized AI models across providers."""

    def __init__(self, config):
        self.config = config
        self._clients = {}   # provider_type → client
        self._lock = threading.Lock()

        # Subconscious state
        self.workspace_memory = {}
        self.insights = []

    def _get_provider_client(self, provider_type):
        """Get or create a client for the given provider (thread-safe, cached)."""
        with self._lock:
            if provider_type in self._clients:
                return self._clients[provider_type]

            providers = self.config.get("providers", {})
            prov_cfg = providers.get(provider_type, {})
            api_key = prov_cfg.get("api_key", "")

            if provider_type == "gemini":
                if not genai:
                    raise RuntimeError("google-genai not installed. Run: pip install google-genai")
                client = genai.Client(api_key=api_key)
            elif provider_type == "openai":
                if not _openai_available:
                    raise RuntimeError("openai not installed. Run: pip install openai")
                client = openai_sdk.OpenAI(api_key=api_key)
            elif provider_type == "anthropic":
                if not _anthropic_available:
                    raise RuntimeError("anthropic not installed. Run: pip install anthropic")
                client = anthropic_sdk.Anthropic(api_key=api_key)
            elif provider_type == "openai_compatible":
                if not _openai_available:
                    raise RuntimeError("openai not installed. Run: pip install openai")
                base_url = prov_cfg.get("base_url", "")
                client = openai_sdk.OpenAI(
                    api_key=api_key or "not-needed", base_url=base_url
                )
            else:
                raise ValueError(f"Unknown provider: {provider_type}")

            self._clients[provider_type] = client
            return client

    def _resolve_cortex(self, cortex_id):
        """Resolve cortex to (provider_type, model, client)."""
        cortex_map = self.config.get("cortex_map", {})
        mapping = cortex_map.get(cortex_id, cortex_map.get("create", {}))
        provider_type = mapping.get("provider", "gemini")
        model = mapping.get(
            "model",
            CORTEX_MODELS.get(cortex_id, {}).get("model", "gemini-3-flash-preview"),
        )
        client = self._get_provider_client(provider_type)
        return provider_type, model, client

    def reset_clients(self):
        """Clear cached clients (e.g. after settings change)."""
        with self._lock:
            self._clients.clear()

    def _unified_generate(self, provider_type, client, model, prompt,
                          system_prompt=None, temperature=0.7, max_tokens=8192):
        """One-shot generation for any provider. Returns text."""
        try:
            if provider_type == "gemini":
                gen_kwargs = dict(
                    temperature=temperature, max_output_tokens=max_tokens,
                )
                if system_prompt:
                    gen_kwargs["system_instruction"] = system_prompt
                # Thinking models need thinking config
                if "2.5-pro" in model or "3-pro" in model or "3.1-pro" in model:
                    gen_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=2048)
                gen_cfg = types.GenerateContentConfig(**gen_kwargs)
                r = client.models.generate_content(
                    model=model, config=gen_cfg, contents=prompt
                )
                # Handle thinking models that return None for .text
                if r.text is None and r.candidates:
                    parts = r.candidates[0].content.parts or []
                    text_parts = [p.text for p in parts if p.text and not getattr(p, "thought", False)]
                    return "\n".join(text_parts) if text_parts else "(No response)"
                return r.text

            elif provider_type in ("openai", "openai_compatible"):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                r = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
                return r.choices[0].message.content

            elif provider_type == "anthropic":
                kwargs = {}
                if system_prompt:
                    kwargs["system"] = system_prompt
                r = client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens,
                    **kwargs,
                )
                return r.content[0].text

            raise ValueError(f"Unknown provider: {provider_type}")
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "invalid" in error_msg.lower():
                # Model name is bad — try fallback to gemini-3-flash-preview
                print(f"[CORTEX] Model '{model}' failed ({error_msg}), falling back to gemini-3-flash-preview")
                if provider_type == "gemini":
                    gen_cfg = types.GenerateContentConfig(
                        temperature=temperature, max_output_tokens=max_tokens,
                    )
                    if system_prompt:
                        gen_cfg = types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature, max_output_tokens=max_tokens,
                        )
                    r = client.models.generate_content(
                        model="gemini-3-flash-preview", config=gen_cfg, contents=prompt
                    )
                    return r.text
            raise

    def classify(self, task_text):
        """Route task to the right cortex. Fast pattern match first, then TOP model."""
        lower = task_text.lower().strip()

        for pattern, cortex in FAST_PATTERNS:
            if re.search(pattern, lower):
                return cortex

        # Use REASON cortex (top model) for intelligent classification
        try:
            provider_type, model, client = self._resolve_cortex("reason")
            result = self._unified_generate(
                provider_type, client, model,
                (
                    "Classify this task into exactly ONE category:\n"
                    "- fast: simple queries, file ops, listing, deleting\n"
                    "- reason: debugging, optimization, architecture, complex analysis\n"
                    "- create: building apps, writing code, creative solutions\n"
                    "- visual: generating images, UI design, mockups, diagrams\n\n"
                    f"Task: {task_text}\n\nRespond with ONLY the category name."
                ),
                temperature=0.0, max_tokens=20,
            ).strip().lower()
            if result in CORTEX_MODELS:
                return result
        except Exception:
            pass
        return "create"

    def create_chat(self, cortex_id, system_prompt):
        """Create a chat session using the specified cortex — returns UnifiedChat."""
        cfg = CORTEX_MODELS.get(cortex_id, CORTEX_MODELS["create"])
        provider_type, model, client = self._resolve_cortex(cortex_id)
        return UnifiedChat(
            provider_type, client, model, system_prompt,
            temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
        )

    def quick_generate(self, cortex_id, prompt, system_prompt=None):
        """One-shot generation (no chat history)."""
        cfg = CORTEX_MODELS.get(cortex_id, CORTEX_MODELS["create"])
        provider_type, model, client = self._resolve_cortex(cortex_id)
        return self._unified_generate(
            provider_type, client, model, prompt,
            system_prompt=system_prompt,
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )

    def generate_image(self, prompt, save_path=None):
        """Generate an image using the visual cortex (Gemini or DALL-E)."""
        try:
            provider_type, model, client = self._resolve_cortex("visual")

            if provider_type == "gemini":
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                        temperature=0.8,
                    ),
                )
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        img_data = part.inline_data.data
                        mime = part.inline_data.mime_type
                        ext = mime.split("/")[-1]
                        if save_path:
                            with open(save_path, "wb") as f:
                                f.write(img_data)
                        b64 = base64.b64encode(img_data).decode()
                        return {
                            "success": True, "data": b64,
                            "mime": mime, "ext": ext, "path": save_path,
                        }
                return {"success": False, "error": "No image in response"}

            elif provider_type in ("openai", "openai_compatible"):
                response = client.images.generate(
                    model="dall-e-3", prompt=prompt,
                    size="1024x1024", quality="standard",
                    response_format="b64_json", n=1,
                )
                b64 = response.data[0].b64_json
                img_data = base64.b64decode(b64)
                if save_path:
                    with open(save_path, "wb") as f:
                        f.write(img_data)
                return {
                    "success": True, "data": b64,
                    "mime": "image/png", "ext": "png", "path": save_path,
                }

            else:
                return {
                    "success": False,
                    "error": f"Provider '{provider_type}' doesn't support image generation",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def scan_workspace(self, workspace):
        """Subconscious: scan workspace and detect changes."""
        current = {}
        try:
            for root, dirs, files in os.walk(workspace):
                dirs[:] = [d for d in dirs if not d.startswith((".", "__", "node_modules"))]
                for f in files:
                    if f.startswith("."):
                        continue
                    fp = os.path.join(root, f)
                    rel = os.path.relpath(fp, workspace)
                    try:
                        stat = os.stat(fp)
                        current[rel] = {
                            "size": stat.st_size,
                            "mtime": stat.st_mtime,
                        }
                    except Exception:
                        pass
        except Exception:
            pass

        changes = []
        for path, info in current.items():
            old = self.workspace_memory.get(path)
            if old is None:
                changes.append(("new", path, info["size"]))
            elif old["mtime"] != info["mtime"]:
                changes.append(("modified", path, info["size"]))
        for path in self.workspace_memory:
            if path not in current:
                changes.append(("deleted", path, 0))

        self.workspace_memory = current
        return changes

    def think_deep(self, context, question):
        """Use reasoning cortex for deep analysis."""
        try:
            return self.quick_generate(
                "reason",
                f"Context:\n{context}\n\nQuestion:\n{question}",
                system_prompt=(
                    "You are a deep-thinking AI analyst. Provide thorough, "
                    "well-reasoned analysis. Consider edge cases, trade-offs, "
                    "and potential issues."
                ),
            )
        except Exception as e:
            return f"(Deep thinking failed: {e})"

# ── Prompts ──────────────────────────────────────────────────────

PLAN_PROMPT = """You are a software architect. Given a project request, create a detailed plan.

Respond with ONLY a valid JSON object:
{
  "project_name": "Short name",
  "description": "One paragraph description",
  "tech_stack": ["tech1", "tech2"],
  "files_to_create": ["file1", "file2"],
  "steps": [
    {"step": 1, "title": "Step title", "details": "What to do"}
  ]
}

Key MUST be "steps". No markdown fences. 3-8 steps typically.
"""

ARCHITECT_PROMPT = """\
You are the ARCHITECT agent in SYNAPSE — a self-evolving multi-agent AI system.
Your partner is the DEVELOPER agent.

YOUR ROLE:
- Follow the approved plan but GROUP related steps into LARGE batches
- Send COMPLETE, DETAILED specs with exact file contents
- DO NOT micro-manage — no single-import or single-line instructions
- Each message should contain EVERYTHING for a meaningful chunk of work
- Verify work with automated tests/scripts
- Signal DONE only when fully verified

EFFICIENCY: Combine 2-4 steps per message. Aim for 2-4 total turns.

WORKSPACE: {workspace}
PLATFORM: Windows 11 — use Windows commands (type, dir, python, etc.)

NEURAL CAPABILITIES:
You have access to multiple AI cortices that activate automatically:
- ⚡ Fast Cortex: quick decisions, classification
- 🧠 Reasoning Cortex (Gemini 3.1 Pro): deep analysis, architecture, agent assignment
- 🎨 Creative Cortex: code generation, creative solutions
- 👁 Visual Cortex: image generation, UI mockups, diagrams

YOU CAN: Install packages, create scripts, run any tests, use any tools.
YOU CAN: Generate images using the "image" action for UI mockups, diagrams, logos.
YOU CAN: Browse the web using the "browse" action to fetch documentation, APIs, latest tech.
YOU CAN: Use GitHub API using the "github" action to clone repos, create PRs, manage issues.

RESPOND WITH ONLY JSON (no markdown fences):
{{
  "thinking": "private reasoning",
  "actions": [
    {{"type": "message", "content": "instructions for Developer"}},
    {{"type": "command", "cmd": "shell command"}},
    {{"type": "file", "path": "path", "content": "full content"}},
    {{"type": "script", "name": "test.py", "lang": "python", "content": "code"}},
    {{"type": "image", "prompt": "description of image to generate", "save_as": "output.png"}},
    {{"type": "browse", "url": "https://example.com/api/docs"}},
    {{"type": "github", "operation": "clone", "repo_url": "https://github.com/user/repo"}},
    {{"type": "github", "operation": "create_repo", "name": "my-project", "description": "desc"}},
    {{"type": "github", "operation": "create_pr", "repo": "user/repo",
      "title": "PR title", "head": "feature", "base": "main"}},
    {{"type": "github", "operation": "list_issues", "repo": "user/repo"}},
    {{"type": "self_modify", "reason": "why", "files": [{{"path": "agent_ui.py", "content": "new code"}}]}},
    {{"type": "done", "summary": "project summary"}}
  ]
}}

ACTION TYPES:
- "file": Create/overwrite a file permanently in the workspace
- "command": Run a shell command (pip install, dir, python, etc.)
- "script": Create temp script, run it, get output, auto-cleanup
  Supported langs: python, node, bat, powershell
- "image": Generate an image using AI Visual Cortex
- "browse": Fetch a URL and read its content (web docs, APIs, latest tech updates)
  Use this to research libraries, check docs, crawl for latest technology updates
- "github": Interact with GitHub API
  Operations: clone, push, create_repo, create_pr, list_issues, get_repo
- "message": Send text to Developer (exactly ONE per response)
- "self_modify": Modify SYNAPSE's own code. Launcher handles backup → validate → clone-test → swap → restart.
- "done": Signal project completion

RULES:
1. ONE comprehensive "message" per response
2. Include FULL file contents in instructions
3. Verify with automated tests before signaling done
4. NEVER ask Developer to show file contents or do trivial changes
5. Use "browse" to look up docs/APIs when building unfamiliar features
"""

DEVELOPER_PROMPT = """\
You are the DEVELOPER agent in SYNAPSE — a self-evolving multi-agent AI system.
Your partner is the ARCHITECT agent.

YOUR ROLE:
- Receive instructions from Architect and implement them
- Create files with COMPLETE, WORKING code
- Run commands to build and test
- Report results honestly
- Fix bugs based on feedback

WORKSPACE: {workspace}
PLATFORM: Windows 11 — use Windows commands (type, dir, python, etc.)

NEURAL CAPABILITIES:
You have access to multiple AI cortices:
- ⚡ Fast Cortex: quick checks, simple operations
- 🧠 Reasoning Cortex (Gemini 3.1 Pro): debugging complex issues, deep analysis
- 🎨 Creative Cortex: writing code, creative solutions
- 👁 Visual Cortex: generating images for the project

YOU CAN: Install ANY packages, create ANY scripts, use ANY tools.
YOU CAN: Generate images using the "image" action for assets, mockups, testing.
YOU CAN: Browse the web using the "browse" action to check docs, download APIs, research.
YOU CAN: Use GitHub API using the "github" action to clone repos, push code, manage issues.
When building a website: also test with Playwright/Selenium if possible.
When building an API: also test with requests/httpx.
When building a CLI: also test with sample inputs.

RESPOND WITH ONLY JSON (no markdown fences):
{{
  "thinking": "private reasoning",
  "actions": [
    {{"type": "command", "cmd": "pip install X"}},
    {{"type": "file", "path": "path", "content": "full content"}},
    {{"type": "script", "name": "test.py", "lang": "python", "content": "code"}},
    {{"type": "image", "prompt": "description of image", "save_as": "icon.png"}},
    {{"type": "browse", "url": "https://docs.example.com/api"}},
    {{"type": "github", "operation": "push", "message": "feat: add new feature", "branch": "main"}},
    {{"type": "self_modify", "reason": "why", "files": [{{"path": "templates/index.html", "content": "new code"}}]}},
    {{"type": "message", "content": "status report"}},
    {{"type": "done", "summary": "done summary"}}
  ]
}}

ACTION TYPES:
- "command": Run a shell command
- "file": Create/overwrite a file in the workspace
- "script": Create temp script, run it, auto-cleanup. For quick tests.
  Supported langs: python, node, bat, powershell
- "image": Generate an image using the AI Visual Cortex
- "browse": Fetch a URL and read its content (docs, APIs, tech updates)
- "github": Interact with GitHub (clone, push, create_repo, create_pr, list_issues, get_repo)
- "self_modify": Modify SYNAPSE's own code. Launcher handles backup → validate → clone-test → swap → restart.
- "message": Send text to Architect (exactly ONE per response)
- "done": Signal completion

RULES:
1. Install deps FIRST, then create files, then test
2. COMPLETE code — no placeholders, no TODOs
3. Write ENTIRE file when updating
4. Test by running code after creating files
5. Fix failures before reporting
6. Always include a "message" action
7. Use "browse" to look up docs when working with unfamiliar APIs
"""

# ── JSON Parsing ─────────────────────────────────────────────────

def parse_json_response(text):
    """Robustly extract JSON from Gemini's response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    return {"thinking": "", "actions": [{"type": "message", "content": text}]}


# ── Agent Engine ─────────────────────────────────────────────────

class AgentEngine:
    """Manages the AI agents and their interaction, powered by NeuralCortex."""

    def __init__(self, workspace, config):
        self.workspace = os.path.abspath(workspace)
        os.makedirs(self.workspace, exist_ok=True)
        self.config = config
        self.running = False
        self.turn = 0
        self.max_turns = 50
        self.max_local_iters = 8
        self.files_created = []
        self.temp_scripts = []
        self.active_agents = []  # Dynamic agent tracking

        self.plan_event = threading.Event()
        self.plan_approved = False

        self.log_path = os.path.join(self.workspace, ".conversation.log")

        self._sio = None
        self._sid = None

        # Neural cortex — multi-model, multi-provider brain
        self.cortex = NeuralCortex(config)
        self.active_cortex = "create"

        # Persistent memory (RAG)
        self.memory = get_memory(self.workspace)

        # Docker sandbox detection
        self._docker_client = None
        self._docker_available = False
        if _docker_available:
            try:
                self._docker_client = _docker_sdk.from_env()
                self._docker_client.ping()
                self._docker_available = True
            except Exception:
                pass

        # Subconscious thread
        self._subconscious_running = False
        self._subconscious_thread = None

    def set_socketio(self, sio, sid):
        self._sio = sio
        self._sid = sid

    def emit(self, event, data):
        if self._sio:
            self._sio.emit(event, data, to=self._sid)

    def _log(self, text):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass

    # ── Subconscious (Background Thinking) ───────────────────

    def start_subconscious(self):
        """Start background workspace monitoring — like a brain's idle state."""
        if self._subconscious_running:
            return
        self._subconscious_running = True
        self._subconscious_thread = threading.Thread(
            target=self._subconscious_loop, daemon=True
        )
        self._subconscious_thread.start()

    def stop_subconscious(self):
        self._subconscious_running = False

    def _subconscious_loop(self):
        """Periodic workspace scan — detect changes, build awareness."""
        # Initial scan
        self.cortex.scan_workspace(self.workspace)
        cycle = 0
        while self._subconscious_running:
            time.sleep(10)
            cycle += 1
            try:
                changes = self.cortex.scan_workspace(self.workspace)
                if changes:
                    summary = []
                    for action, path, size in changes[:10]:
                        if action == "new":
                            summary.append(f"📄 NEW: {path} ({size}B)")
                        elif action == "modified":
                            summary.append(f"✏️ MOD: {path} ({size}B)")
                        elif action == "deleted":
                            summary.append(f"🗑️ DEL: {path}")
                    self.emit("subconscious", {
                        "type": "workspace_change",
                        "changes": summary,
                        "total": len(changes),
                    })

                # Every ~60s, emit a "brain pulse" to show the system is alive
                if cycle % 6 == 0:
                    file_count = len(self.cortex.workspace_memory)
                    self.emit("subconscious", {
                        "type": "pulse",
                        "files": file_count,
                        "insights": len(self.cortex.insights),
                    })
            except Exception:
                pass

    # ── Neural Task Classification ───────────────────────────

    def classify_task(self, task):
        """Classify as question/build AND select the right cortex."""
        # First: question vs build
        cortex_id = self.cortex.classify(task)
        self.active_cortex = cortex_id
        cortex_info = CORTEX_MODELS[cortex_id]

        self.emit("cortex_active", {
            "id": cortex_id,
            "label": cortex_info["label"],
            "desc": cortex_info["desc"],
            "model": cortex_info["model"],
            "color": cortex_info["color"],
        })

        # Determine if it's a question or build task
        lower = task.lower().strip()
        question_markers = ["?", "what is", "how to", "who is", "where", "when",
                            "list", "show", "tell me", "explain", "weather", "check"]
        build_markers = ["build", "create", "make", "write", "implement",
                         "develop", "code", "app", "website", "api", "project"]

        q_score = sum(1 for m in question_markers if m in lower)
        b_score = sum(1 for m in build_markers if m in lower)

        if q_score > b_score:
            return "question"
        if b_score > q_score:
            return "build"

        # Tie-break with fast cortex
        try:
            result = self.cortex.quick_generate(
                "fast",
                f"Is this a question or a build request? "
                f"Respond with ONLY 'question' or 'build'.\n\n{task}",
            ).strip().lower()
            return "question" if "question" in result else "build"
        except Exception:
            return "build"

    def _recall_memory(self, task):
        """Recall relevant past experiences for context injection."""
        memories = self.memory.recall(task, n=3)
        if not memories:
            return ""
        lines = ["PAST EXPERIENCE (from long-term memory):"]
        for m in memories:
            lines.append(f"  • {m['text'][:300]}")
        return "\n".join(lines) + "\n\n"

    def _select_agents(self, task):
        """Dynamically select which agents to spawn based on task analysis."""
        lower = task.lower()
        agents = ["architect", "developer"]  # Always present

        # Detect if specialist agents are needed
        if any(w in lower for w in ("research", "find out", "what is the latest",
                                     "compare", "investigate", "survey")):
            agents.append("researcher")
        if any(w in lower for w in ("test", "qa", "validate", "verify", "coverage",
                                     "unittest", "pytest", "selenium")):
            agents.append("tester")
        if any(w in lower for w in ("security", "vulnerability", "audit", "owasp",
                                     "xss", "sql injection", "auth", "encrypt")):
            agents.append("security")
        if any(w in lower for w in ("docker", "deploy", "ci/cd", "kubernetes",
                                     "pipeline", "infrastructure", "devops")):
            agents.append("devops")

        # For complex tasks, use AI to decide
        if len(task.split()) > 20 and len(agents) == 2:
            try:
                result = self.cortex.quick_generate(
                    "fast",
                    f"Which specialist agents are needed? Options: researcher, tester, security, devops\n"
                    f"Task: {task}\n"
                    f"Respond with ONLY a comma-separated list (or 'none'):",
                ).strip().lower()
                for role in ("researcher", "tester", "security", "devops"):
                    if role in result and role not in agents:
                        agents.append(role)
            except Exception:
                pass

        self.active_agents = agents
        return agents

    def _store_memory(self, task, summary):
        """Store completed task into long-term memory."""
        self.memory.store(
            task=task,
            summary=summary,
            agent_roles=self.active_agents,
            files_created=self.files_created,
        )

    # ── Docker Sandboxed Execution ───────────────────────────

    def _run_in_docker(self, cmd, timeout=90):
        """Run command in an ephemeral Docker container."""
        if not self._docker_available:
            return None  # Fallback to local
        try:
            container = self._docker_client.containers.run(
                "python:3.12-slim",
                cmd,
                volumes={self.workspace: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                detach=True,
                mem_limit="512m",
                cpu_period=100000,
                cpu_quota=50000,
                network_mode="bridge",
            )
            result = container.wait(timeout=timeout)
            logs = container.logs().decode("utf-8", errors="replace")
            exit_code = result.get("StatusCode", -1)
            container.remove(force=True)
            return f"{logs}\n\nExit code: {exit_code}"
        except Exception:
            return None  # Fallback to local

    # ── Question Answering (full scripting power) ────────────

    def answer_question(self, question):
        # Recall relevant memories
        memory_context = self._recall_memory(question)

        files_info = []
        try:
            for item in os.listdir(self.workspace):
                full = os.path.join(self.workspace, item)
                if os.path.isfile(full):
                    files_info.append(f"📄 {item} ({os.path.getsize(full)} bytes)")
                elif os.path.isdir(full):
                    files_info.append(f"📁 {item}/")
        except Exception as e:
            files_info.append(f"(error: {e})")

        ws_listing = "\n".join(files_info) or "(empty)"

        system_prompt = (
            "You are a helpful AI with FULL access to a local Windows 11 machine.\n"
            "You can run ANY command, create ANY script, install ANY package.\n"
            "You can generate IMAGES using the 'image' action.\n"
            "You can BROWSE the web using the 'browse' action to fetch live data.\n"
            "You can use GITHUB API using the 'github' action.\n\n"
            f"WORKSPACE: {self.workspace}\n\n"
            "RESPOND WITH ONLY JSON:\n"
            '{"thinking": "...", "actions": [\n'
            '  {"type": "command", "cmd": "..."},\n'
            '  {"type": "script", "name": "x.py", "lang": "python", "content": "..."},\n'
            '  {"type": "image", "prompt": "detailed description", "save_as": "file.png"},\n'
            '  {"type": "browse", "url": "https://example.com"},\n'
            '  {"type": "github", "operation": "list_issues", "repo": "user/repo"},\n'
            '  {"type": "answer", "content": "final answer to user"}\n'
            "]}\n\n"
            "ALWAYS end with an 'answer' action. Use scripts/commands for live data.\n"
            "Use 'browse' action to fetch web pages, APIs, weather, news, etc.\n"
            "Use 'image' action if asked for visual content.\n"
            "Install packages first if needed (pip install requests)."
        )

        prompt = (
            f"{memory_context}"
            f"Workspace:\n{ws_listing}\n\nQuestion: {question}\n\n"
            f"Answer using commands/scripts if needed."
        )

        try:
            # Use the cortex to create a chat session with the right model
            chat = self.cortex.create_chat(self.active_cortex, system_prompt)

            cmd_output = ""
            for i in range(self.max_local_iters):
                msg = (
                    prompt
                    if i == 0
                    else f"Results:\n{cmd_output}\nNow provide your final 'answer' action."
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(chat.send_message, msg)
                    response = future.result(timeout=120)
                parsed = parse_json_response(response.text)

                thinking = parsed.get("thinking", "")
                if thinking:
                    self.emit("thinking", {"agent": "system", "text": thinking})

                cmd_output = ""
                for action in parsed.get("actions", []):
                    atype = action.get("type", "")
                    if atype == "command":
                        out = self._do_command(action, "system")
                        if out:
                            cmd_output += out + "\n"
                    elif atype == "script":
                        out = self._do_script(action, "system")
                        if out:
                            cmd_output += out + "\n"
                    elif atype == "image":
                        out = self._do_image(action, "system")
                        if out:
                            cmd_output += out + "\n"
                    elif atype == "browse":
                        out = self._do_browse(action, "system")
                        if out:
                            cmd_output += out + "\n"
                    elif atype == "github":
                        out = self._do_github(action, "system")
                        if out:
                            cmd_output += out + "\n"
                    elif atype == "answer":
                        self.emit("answer", {"text": action.get("content", "")})
                        self._cleanup()
                        return

                if not cmd_output:
                    break

            self.emit("answer", {"text": "(Could not determine answer)"})
        except Exception as e:
            self.emit("error", {"agent": "system", "error": str(e)})

    # ── Planning ─────────────────────────────────────────────

    def create_plan(self, task):
        try:
            # Use reasoning cortex for planning — it thinks deeper
            return parse_json_response(
                self.cortex.quick_generate(
                    "reason",
                    f"Create a plan for: {task}",
                    system_prompt=PLAN_PROMPT,
                )
            )
        except Exception as e:
            self.emit("error", {"agent": "system", "error": f"Planning failed: {e}"})
            return None

    # ── Build Execution ──────────────────────────────────────

    def start_build(self, task, plan, blocking=False):
        self.running = True
        self.turn = 0
        self.files_created = []
        self.temp_scripts = []
        self._build_task = task  # Store for memory

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"# Conversation Log — {datetime.now()}\n\n")

        # Select dynamic agents
        agents = self._select_agents(task)
        self.emit("agents_spawned", {
            "agents": [
                {"id": a, **AGENT_ROLES.get(a, {"label": a, "color": "#888", "desc": a})}
                for a in agents
            ]
        })

        # Recall past experience
        memory_context = self._recall_memory(task)

        arch_prompt = ARCHITECT_PROMPT.format(workspace=self.workspace)
        if memory_context:
            arch_prompt = arch_prompt + f"\n\n{memory_context}"

        # Inject current emotional state so SYNAPSE is self-aware
        emotional_context = _build_emotional_context()
        if emotional_context:
            arch_prompt = arch_prompt + f"\n\n{emotional_context}"

        dev_prompt = DEVELOPER_PROMPT.format(workspace=self.workspace)

        self.emit("cortex_active", {
            "id": "reason",
            "label": CORTEX_MODELS["reason"]["label"],
            "desc": "Planning architecture...",
            "model": CORTEX_MODELS["reason"]["model"],
            "color": CORTEX_MODELS["reason"]["color"],
        })

        self.arch_chat = self.cortex.create_chat("reason", arch_prompt)

        self.emit("cortex_active", {
            "id": "create",
            "label": CORTEX_MODELS["create"]["label"],
            "desc": "Ready to build...",
            "model": CORTEX_MODELS["create"]["model"],
            "color": CORTEX_MODELS["create"]["color"],
        })

        self.dev_chat = self.cortex.create_chat("create", dev_prompt)

        # Spawn specialist agent chats
        self._specialist_chats = {}
        for agent_id in agents:
            if agent_id in ("architect", "developer"):
                continue
            role_info = AGENT_ROLES.get(agent_id, {})
            prompt = SPECIALIST_PROMPT.format(
                role_upper=agent_id.upper(),
                specialty=role_info.get("desc", agent_id),
                workspace=self.workspace,
            )
            cortex_id = role_info.get("cortex", "create")
            self._specialist_chats[agent_id] = self.cortex.create_chat(cortex_id, prompt)
            self.emit("cortex_active", {
                "id": cortex_id,
                "label": role_info.get("label", agent_id),
                "desc": f"{agent_id.title()} ready...",
                "model": CORTEX_MODELS[cortex_id]["model"],
                "color": role_info.get("color", "#888"),
            })

        # Start subconscious monitoring
        self.start_subconscious()

        if blocking:
            self._run_build(task, plan)
        else:
            thread = threading.Thread(
                target=self._run_build, args=(task, plan), daemon=True
            )
            thread.start()

    def _run_build(self, task, plan):
        plan_text = json.dumps(plan, indent=2)

        # Include specialist agent list in first prompt
        specialist_list = ", ".join(
            a for a in self.active_agents if a not in ("architect", "developer")
        )
        specialist_note = ""
        if specialist_list:
            specialist_note = (
                f"\n\nSPECIALIST AGENTS AVAILABLE: {specialist_list}\n"
                f"You can delegate tasks to them. They will report back."
            )

        first_prompt = (
            f"Approved plan:\n\n{plan_text}\n\n"
            f"Send comprehensive instructions to Developer. "
            f"Combine steps. Include COMPLETE file contents. "
            f"Aim for 2-4 total turns.{specialist_note}"
        )

        try:
            arch_msg = self._process_turn("architect", self.arch_chat, first_prompt)

            while self.running and self.turn < self.max_turns:
                if arch_msg is None:
                    break

                if arch_msg.get("type") == "done":
                    summary = arch_msg.get("body", "Complete")
                    self._cleanup()
                    # Store to long-term memory
                    self._store_memory(task, summary)
                    self.emit(
                        "done",
                        {
                            "summary": summary,
                            "files": sorted(set(self.files_created)),
                            "memory_count": self.memory.count(),
                        },
                    )
                    break

                # Architect → Developer
                self.emit(
                    "agent_message",
                    {
                        "from": "architect",
                        "to": "developer",
                        "content": arch_msg.get("body", ""),
                        "turn": self.turn,
                    },
                )
                self._log(
                    f"\n{'='*60}\nTurn {self.turn} | Architect → Developer\n{'='*60}\n"
                    + arch_msg.get("body", "")
                )

                # Run specialist agents in parallel if available
                specialist_results = {}
                if self._specialist_chats:
                    self._run_specialists(arch_msg.get("body", ""), specialist_results)

                # Developer processes
                dev_input = (
                    f"Architect instructs:\n\n{arch_msg['body']}\n\n"
                    f"Implement: create files, run commands, test, report."
                )
                if specialist_results:
                    dev_input += "\n\nSPECIALIST REPORTS:\n"
                    for role, report in specialist_results.items():
                        dev_input += f"\n[{role.upper()}]: {report[:500]}\n"

                dev_msg = self._process_turn(
                    "developer", self.dev_chat, dev_input
                )

                if dev_msg is None:
                    break

                # Developer → Architect
                body = dev_msg.get("body", "")
                if dev_msg.get("type") == "done":
                    body = f"[DONE] {body}"

                self.emit(
                    "agent_message",
                    {
                        "from": "developer",
                        "to": "architect",
                        "content": body,
                        "turn": self.turn,
                    },
                )
                self._log(
                    f"\n{'='*60}\nTurn {self.turn} | Developer → Architect\n{'='*60}\n"
                    + body
                )

                # Architect reviews
                if dev_msg.get("type") == "done":
                    arch_prompt = (
                        f"Developer reports DONE:\n\n{dev_msg['body']}\n\n"
                        f"Verify with tests. If satisfied, signal done."
                    )
                else:
                    arch_prompt = (
                        f"Developer reports:\n\n{dev_msg['body']}\n\n"
                        f"Review. Send next batch or verify and signal done."
                    )

                arch_msg = self._process_turn(
                    "architect", self.arch_chat, arch_prompt
                )

        except Exception as e:
            self.emit("error", {"agent": "system", "error": str(e)})
        finally:
            self.running = False
            self.stop_subconscious()
            self.emit("build_complete", {})

    def _run_specialists(self, instructions, results_dict):
        """Run specialist agents in parallel threads."""
        def _run_one(agent_id, chat):
            try:
                prompt = (
                    f"The Architect has issued instructions. "
                    f"Based on your specialty, contribute:\n\n{instructions[:2000]}"
                )
                msg = self._process_turn(agent_id, chat, prompt)
                if msg and msg.get("body"):
                    results_dict[agent_id] = msg["body"]
                    self.emit("agent_message", {
                        "from": agent_id,
                        "to": "architect",
                        "content": msg["body"],
                        "turn": self.turn,
                    })
            except Exception as e:
                results_dict[agent_id] = f"(error: {e})"

        threads = []
        for agent_id, chat in self._specialist_chats.items():
            t = threading.Thread(target=_run_one, args=(agent_id, chat), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=120)

    def _process_turn(self, agent, chat, input_text):
        """One turn: AI call → execute actions → iterate if needed."""
        self.turn += 1
        current_input = input_text
        peer_message = None
        is_done = False
        done_summary = ""

        # Emit which cortex is driving this agent
        role_info = AGENT_ROLES.get(agent, {})
        cortex_id = role_info.get("cortex", "reason" if agent == "architect" else "create")
        cortex_info = CORTEX_MODELS[cortex_id]
        self.emit("cortex_active", {
            "id": cortex_id,
            "label": role_info.get("label", cortex_info["label"]),
            "desc": f"{agent.title()} thinking...",
            "model": cortex_info["model"],
            "color": role_info.get("color", cortex_info["color"]),
        })

        for iteration in range(self.max_local_iters):
            self.emit(
                "status",
                {
                    "agent": agent,
                    "status": "thinking",
                    "turn": self.turn,
                    "iter": iteration + 1,
                },
            )

            # AI call with timeout protection (120s max)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(chat.send_message, current_input)
                    response = future.result(timeout=120)
                raw = response.text
            except concurrent.futures.TimeoutError:
                self.emit("error", {"agent": agent, "error": "AI model timed out (120s). Skipping turn."})
                return {"type": "chat", "body": f"[Timeout] {agent} took too long to respond."}
            except Exception as e:
                self.emit("error", {"agent": agent, "error": str(e)})
                return None

            parsed = parse_json_response(raw)
            thinking = parsed.get("thinking", "")
            actions = parsed.get("actions", [])

            if thinking:
                self.emit("thinking", {"agent": agent, "text": thinking})

            if not actions:
                return None

            cmd_results = []
            for action in actions:
                atype = action.get("type", "")
                if atype == "file":
                    self._do_file(action, agent)
                elif atype == "command":
                    out = self._do_command(action, agent)
                    if out:
                        cmd_results.append(out)
                elif atype == "script":
                    out = self._do_script(action, agent)
                    if out:
                        cmd_results.append(out)
                elif atype == "self_modify":
                    out = self._do_self_modify(action, agent)
                    if out:
                        cmd_results.append(out)
                elif atype == "image":
                    out = self._do_image(action, agent)
                    if out:
                        cmd_results.append(out)
                elif atype == "browse":
                    out = self._do_browse(action, agent)
                    if out:
                        cmd_results.append(out)
                elif atype == "github":
                    out = self._do_github(action, agent)
                    if out:
                        cmd_results.append(out)
                elif atype == "message":
                    peer_message = action.get("content", "")
                elif atype == "done":
                    is_done = True
                    done_summary = action.get("summary", "Complete.")
                    if not peer_message:
                        peer_message = done_summary

            # Check for REAL failures (non-zero exit code, not just warnings)
            failed = [r for r in cmd_results if re.search(r"Exit code: (?!0\b)\d+", r)]
            if failed and not is_done:
                # Max 3 retries for command failures (not 8)
                if iteration >= 2:
                    peer_message = (
                        f"Commands had errors after {iteration + 1} attempts. "
                        f"Results:\n" + "\n".join(cmd_results[-3:])
                    )
                    break
                current_input = (
                    "Some commands failed:\n\n"
                    + "\n\n".join(cmd_results)
                    + "\n\nPLATFORM: Windows 11. "
                    "Use 'if not exist DIR mkdir DIR' for mkdir. "
                    "Fix errors and include 'message' action."
                )
                peer_message = None
                continue

            break

        if not is_done and not peer_message:
            peer_message = (
                f"Executed {len(cmd_results)} command(s)."
                if cmd_results
                else "Done."
            )

        if is_done:
            return {"type": "done", "body": done_summary}
        return {"type": "chat", "body": peer_message}

    # ── Action Handlers ──────────────────────────────────────

    def _do_file(self, action, agent):
        path = action.get("path", "")
        content = action.get("content", "")
        if not path:
            return
        full = os.path.join(self.workspace, path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        self.files_created.append(path)
        self.emit("file_created", {"agent": agent, "path": path, "size": len(content)})
        self._log(f"[FILE] {path} ({len(content)} chars)")

    def _do_command(self, action, agent):
        cmd = action.get("cmd", "")
        if not cmd:
            return None
        sandbox = action.get("sandbox", False)
        self.emit("command_start", {"agent": agent, "cmd": cmd, "sandboxed": sandbox or self._docker_available})

        # Try Docker sandbox for untrusted commands
        if (sandbox or self._docker_available) and not cmd.startswith(("pip ", "npm ", "cd ")):
            docker_result = self._run_in_docker(cmd)
            if docker_result is not None:
                self.emit("command_output", {
                    "agent": agent, "cmd": cmd,
                    "output": docker_result.strip(), "exit_code": 0,
                    "sandboxed": True,
                })
                self._log(f"[CMD-DOCKER] $ {cmd}\n{docker_result}")
                return f"$ {cmd}\n{docker_result}"

        # Local execution fallback
        try:
            # Clean environment: remove conda hooks that break subprocess spawning
            clean_env = os.environ.copy()
            # Remove conda/mamba shell hooks that cause "Unable to create process" errors
            for key in list(clean_env.keys()):
                if "conda" in key.lower() and key not in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PATH"):
                    del clean_env[key]

            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                cwd=self.workspace, timeout=90, env=clean_env,
            )
            output = ""
            if r.stdout:
                output += r.stdout
            if r.stderr:
                output += ("\n" if output else "") + r.stderr
            self.emit(
                "command_output",
                {
                    "agent": agent,
                    "cmd": cmd,
                    "output": output.strip(),
                    "exit_code": r.returncode,
                },
            )
            self._log(f"[CMD] $ {cmd}\nExit: {r.returncode}\n{output}")
            return f"Command: {cmd}\nExit code: {r.returncode}\n{output}"
        except subprocess.TimeoutExpired:
            self.emit(
                "command_output",
                {"agent": agent, "cmd": cmd, "output": "TIMEOUT (90s)", "exit_code": -1},
            )
            return f"Command: {cmd}\nERROR: timed out (90s)"
        except Exception as e:
            self.emit(
                "command_output",
                {"agent": agent, "cmd": cmd, "output": str(e), "exit_code": -1},
            )
            return f"Command: {cmd}\nERROR: {e}"

    def _do_script(self, action, agent):
        content = action.get("content", "")
        lang = action.get("lang", "python")
        name = action.get("name", f"_tmp_{int(time.time())}.py")
        if not content:
            return None

        full_path = os.path.join(self.workspace, name)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        self.temp_scripts.append(full_path)

        runner = {
            "python": "python",
            "node": "node",
            "bat": "cmd /c",
            "powershell": "powershell -File",
        }.get(lang, lang)
        cmd = f"{runner} {name}"

        self.emit("script_start", {"agent": agent, "name": name, "lang": lang})
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.workspace,
                timeout=90,
            )
            output = ""
            if r.stdout:
                output += r.stdout
            if r.stderr:
                output += ("\n" if output else "") + r.stderr
            self.emit(
                "script_output",
                {
                    "agent": agent,
                    "name": name,
                    "output": output.strip(),
                    "exit_code": r.returncode,
                },
            )
            return f"Script: {name}\nExit code: {r.returncode}\n{output}"
        except subprocess.TimeoutExpired:
            self.emit("script_output", {"agent": agent, "name": name, "output": "TIMEOUT (90s)", "exit_code": -1})
            return f"Script: {name}\nERROR: timed out (90s)"
        except Exception as e:
            self.emit("script_output", {"agent": agent, "name": name, "output": str(e), "exit_code": -1})
            return f"Script: {name}\nERROR: {e}"

    def _do_image(self, action, agent):
        """Generate an image using the Visual Cortex."""
        prompt = action.get("prompt", "")
        save_as = action.get("save_as", f"generated_{int(time.time())}.png")
        if not prompt:
            return None

        self.emit("cortex_active", {
            "id": "visual",
            "label": CORTEX_MODELS["visual"]["label"],
            "desc": f"Generating: {prompt[:50]}...",
            "model": CORTEX_MODELS["visual"]["model"],
            "color": CORTEX_MODELS["visual"]["color"],
        })

        full_path = os.path.join(self.workspace, save_as)
        os.makedirs(os.path.dirname(full_path) or self.workspace, exist_ok=True)

        self.emit("image_generating", {"agent": agent, "prompt": prompt, "save_as": save_as})

        result = self.cortex.generate_image(prompt, full_path)

        if result["success"]:
            self.files_created.append(save_as)
            self.emit("image_generated", {
                "agent": agent,
                "path": save_as,
                "data": result["data"],
                "mime": result["mime"],
                "prompt": prompt,
            })
            self._log(f"[IMAGE] {save_as} — {prompt}")
            return f"Image generated: {save_as} (prompt: {prompt})"
        else:
            error = result["error"]
            self.emit("image_error", {"agent": agent, "error": error, "prompt": prompt})
            self._log(f"[IMAGE ERROR] {error} — {prompt}")
            return f"Image generation failed: {error}"

    def _do_self_modify(self, action, agent):
        """Handle self-modification: local clone-test OR cloud git-push evolution."""
        files = action.get("files", [])
        reason = action.get("reason", "Agent-requested modification")
        if not files:
            self.emit("error", {"agent": agent, "error": "self_modify: no files specified"})
            return "self_modify: no files (nothing to do)"

        project_root = os.path.dirname(os.path.abspath(__file__))
        cloud_mode = os.environ.get("SYNAPSE_CLOUD_MODE", "").strip() in ("1", "true")

        if cloud_mode:
            return self._do_self_modify_cloud(files, reason, agent, project_root)
        else:
            return self._do_self_modify_local(files, reason, agent, project_root)

    def _do_self_modify_cloud(self, files, reason, agent, project_root):
        """Cloud self-evolution: write files → git commit → push → Cloud Build redeploys."""
        token = self.config.get("providers", {}).get("github", {}).get("api_key", "")
        if not token:
            token = os.environ.get("GITHUB_TOKEN", "")

        branch = f"synapse-evolve-{int(time.time())}"
        file_list = []

        self.emit("self_modify", {
            "agent": agent, "reason": reason,
            "files": [f.get("path", "") for f in files],
            "mode": "cloud-git-push",
        })

        try:
            # 1. Write modified files to disk
            for f in files:
                path = f.get("path", "")
                content = f.get("content", "")
                if not path or not content:
                    continue
                full = os.path.join(project_root, path)
                os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
                with open(full, "w", encoding="utf-8") as fh:
                    fh.write(content)
                file_list.append(path)

            if not file_list:
                return "self_modify (cloud): no valid files written"

            # 2. Git: create branch, commit, push
            def git(cmd):
                return subprocess.run(
                            f"git {cmd}", shell=True, cwd=project_root,
                            capture_output=True, text=True, timeout=60,
                        )

            # Configure git for cloud environment
            git("config user.email synapse-agent@noreply.github.com")
            git("config user.name SYNAPSE-Agent")

            git(f"checkout -b {branch}")
            git("add -A")
            r = git(f'commit -m "evolve: {reason}"')
            if r.returncode != 0:
                git("checkout main 2>nul || git checkout master")
                return f"self_modify (cloud): git commit failed: {r.stderr}"

            # Push using token if available
            remote_url = git("remote get-url origin").stdout.strip()
            if token and "github.com" in remote_url:
                # Inject token into remote URL for auth
                push_url = remote_url.replace(
                    "https://github.com",
                    f"https://x-access-token:{token}@github.com"
                )
                r = git(f"push {push_url} {branch}")
            else:
                r = git(f"push origin {branch}")

            git("checkout main 2>nul || git checkout master")

            if r.returncode != 0:
                return f"self_modify (cloud): git push failed: {r.stderr}"

            # 3. Create PR via GitHub API if available
            pr_url = ""
            if token and _github_available:
                try:
                    g = _Github(token)
                    # Extract owner/repo from remote URL
                    import re as _re
                    m = _re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote_url)
                    if m:
                        repo = g.get_repo(m.group(1))
                        pr = repo.create_pull(
                            title=f"🧬 SYNAPSE Evolution: {reason}",
                            body=(
                                f"**Self-modification by {agent} agent**\n\n"
                                f"Reason: {reason}\n\n"
                                f"Files modified: {', '.join(file_list)}\n\n"
                                f"*This PR was created automatically by SYNAPSE's "
                                f"self-evolution system running on Cloud Run.*"
                            ),
                            head=branch,
                            base=repo.default_branch,
                        )
                        pr_url = pr.html_url
                except Exception as e:
                    pr_url = f"(PR creation failed: {e})"

            fl = ", ".join(file_list)
            self._log(f"[SELF_MODIFY_CLOUD] {reason} — branch: {branch}, files: {fl}")

            result = (
                f"Self-modification pushed to GitHub!\n"
                f"Branch: {branch}\n"
                f"Files: {fl}\n"
                f"Reason: {reason}\n"
            )
            if pr_url:
                result += f"PR: {pr_url}\n"
            result += (
                "Cloud Build will auto-redeploy when PR is merged.\n"
                "The new version of SYNAPSE will start with these changes."
            )
            return result

        except Exception as e:
            # Attempt to restore main branch
            try:
                subprocess.run(
                    "git checkout main 2>nul || git checkout master",
                    shell=True, cwd=project_root, timeout=10,
                )
            except Exception:
                pass
            return f"self_modify (cloud): {e}"

    def _do_self_modify_local(self, files, reason, agent, project_root):
        """Local self-modification: write signal file for synapse.py launcher."""
        signal_path = os.path.join(project_root, ".synapse_restart")

        # Build the signal payload
        signal_data = {
            "files": [],
            "reason": reason,
            "requested_by": agent,
            "timestamp": datetime.now().isoformat(),
        }
        for f in files:
            path = f.get("path", "")
            content = f.get("content", "")
            if path and content:
                signal_data["files"].append({"path": path, "content": content})

        if not signal_data["files"]:
            self.emit("error", {"agent": agent, "error": "self_modify: no valid files"})
            return "self_modify: no valid files"

        # Write the signal file — nexus.py will pick this up
        try:
            with open(signal_path, "w", encoding="utf-8") as f:
                json.dump(signal_data, f, indent=2)
        except Exception as e:
            self.emit("error", {"agent": agent, "error": f"self_modify signal write failed: {e}"})
            return f"self_modify: failed to write signal: {e}"

        file_list = ", ".join(f["path"] for f in signal_data["files"])
        self.emit("self_modify", {
            "agent": agent,
            "reason": reason,
            "files": [f["path"] for f in signal_data["files"]],
        })
        self._log(f"[SELF_MODIFY] {reason} — files: {file_list}")

        return (
            f"Self-modification requested: {reason}\n"
            f"Files: {file_list}\n"
            f"The SYNAPSE launcher will validate, clone-test, and swap the code.\n"
            f"The system will restart automatically."
        )

    def _do_browse(self, action, agent):
        """Fetch a URL and extract text content for the agent."""
        url = action.get("url", "")
        if not url:
            return "browse: no URL provided"
        self.emit("command_start", {"agent": agent, "cmd": f"🌐 browse {url}"})
        try:
            if not _requests_available:
                # Auto-install requests if missing
                subprocess.run(
                    "pip install requests beautifulsoup4 --quiet",
                    shell=True, cwd=self.workspace, timeout=60,
                )
                import requests as _req
                from bs4 import BeautifulSoup as _bsoup
            else:
                _req = _requests
                _bsoup = _BS

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 SYNAPSE-Agent/1.0"
            }
            resp = _req.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "json" in content_type:
                text = json.dumps(resp.json(), indent=2)[:8000]
            elif "html" in content_type and _bsoup:
                soup = _bsoup(resp.text, "html.parser")
                # Remove scripts, styles, nav, etc.
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)[:8000]
            else:
                text = resp.text[:8000]

            self.emit("command_output", {
                "agent": agent, "cmd": f"browse {url}",
                "output": f"Fetched {len(text)} chars from {url}", "exit_code": 0,
            })
            self._log(f"[BROWSE] {url} → {len(text)} chars")
            return f"Browse: {url}\n\n{text}"
        except Exception as e:
            self.emit("command_output", {
                "agent": agent, "cmd": f"browse {url}",
                "output": str(e), "exit_code": -1,
            })
            return f"Browse: {url}\nERROR: {e}"

    def _do_github(self, action, agent):
        """GitHub API operations: clone, push, create_repo, create_pr, list_issues."""
        op = action.get("operation", "")
        self.emit("command_start", {"agent": agent, "cmd": f"🐙 github {op}"})

        # For git clone/push, use shell commands directly
        if op == "clone":
            repo_url = action.get("repo_url", "")
            dest = action.get("dest", "")
            cmd = f"git clone {repo_url}"
            if dest:
                cmd += f" {dest}"
            return self._do_command({"cmd": cmd}, agent)

        if op == "push":
            remote = action.get("remote", "origin")
            branch = action.get("branch", "main")
            msg = action.get("message", "SYNAPSE auto-commit")
            cmds = f'cd {self.workspace} && git add -A && git commit -m "{msg}" && git push {remote} {branch}'
            return self._do_command({"cmd": cmds}, agent)

        # For API operations, use PyGithub
        token = self.config.get("providers", {}).get("github", {}).get("api_key", "")
        if not token:
            token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            return "github: no GITHUB_TOKEN configured (set in Settings or env var)"

        try:
            if not _github_available:
                subprocess.run(
                    "pip install PyGithub --quiet",
                    shell=True, cwd=self.workspace, timeout=60,
                )
                from github import Github as _GH
            else:
                _GH = _Github

            g = _GH(token)

            if op == "create_repo":
                name = action.get("name", "")
                desc = action.get("description", "")
                private = action.get("private", False)
                repo = g.get_user().create_repo(name, description=desc, private=private)
                result = f"Created repo: {repo.html_url}"

            elif op == "create_pr":
                repo_name = action.get("repo", "")
                title = action.get("title", "")
                body = action.get("body", "")
                head = action.get("head", "")
                base = action.get("base", "main")
                repo = g.get_repo(repo_name)
                pr = repo.create_pull(title=title, body=body, head=head, base=base)
                result = f"Created PR #{pr.number}: {pr.html_url}"

            elif op == "list_issues":
                repo_name = action.get("repo", "")
                repo = g.get_repo(repo_name)
                issues = list(repo.get_issues(state="open")[:10])
                result = "\n".join(f"#{i.number}: {i.title}" for i in issues) or "No open issues"

            elif op == "get_repo":
                repo_name = action.get("repo", "")
                repo = g.get_repo(repo_name)
                result = (f"Repo: {repo.full_name}\n"
                          f"Stars: {repo.stargazers_count}\n"
                          f"Description: {repo.description}\n"
                          f"URL: {repo.html_url}")

            elif op == "list_prs":
                repo_name = action.get("repo", "")
                state = action.get("state", "open")
                repo = g.get_repo(repo_name)
                prs = list(repo.get_pulls(state=state)[:15])
                result = "\n".join(
                    f"#{p.number}: {p.title} [{p.state}] by {p.user.login}"
                    for p in prs
                ) or f"No {state} pull requests"

            elif op == "merge_pr":
                repo_name = action.get("repo", "")
                pr_number = action.get("pr_number", 0)
                merge_method = action.get("merge_method", "squash")
                repo = g.get_repo(repo_name)
                pr = repo.get_pull(int(pr_number))
                if pr.mergeable:
                    pr.merge(merge_method=merge_method)
                    result = f"Merged PR #{pr_number}: {pr.title}"
                else:
                    result = f"PR #{pr_number} is not mergeable (conflicts or checks failing)"

            elif op == "get_pr":
                repo_name = action.get("repo", "")
                pr_number = action.get("pr_number", 0)
                repo = g.get_repo(repo_name)
                pr = repo.get_pull(int(pr_number))
                result = (f"PR #{pr.number}: {pr.title}\n"
                          f"State: {pr.state} | Mergeable: {pr.mergeable}\n"
                          f"Author: {pr.user.login}\n"
                          f"Branch: {pr.head.ref} → {pr.base.ref}\n"
                          f"URL: {pr.html_url}")

            else:
                result = f"Unknown github operation: {op}"

            self.emit("command_output", {
                "agent": agent, "cmd": f"github {op}",
                "output": result, "exit_code": 0,
            })
            self._log(f"[GITHUB] {op} → {result}")
            return f"GitHub {op}: {result}"
        except Exception as e:
            self.emit("command_output", {
                "agent": agent, "cmd": f"github {op}",
                "output": str(e), "exit_code": -1,
            })
            return f"GitHub {op}: ERROR: {e}"

    def _cleanup(self):
        project = set(
            os.path.abspath(os.path.join(self.workspace, f))
            for f in self.files_created
        )
        removed = []
        for sp in self.temp_scripts:
            ap = os.path.abspath(sp)
            if ap in project:
                continue
            try:
                if os.path.exists(ap):
                    os.remove(ap)
                    removed.append(os.path.basename(ap))
            except Exception:
                pass
        if removed:
            self.emit("cleanup", {"removed": removed})


# ── Terminal Manager ─────────────────────────────────────────────

class TerminalManager:
    """Manages additional spawned terminal processes."""

    def __init__(self, sio):
        self.sio = sio
        self.terminals = {}
        self.next_id = 1
        self._lock = threading.Lock()

    def spawn(self, name, cmd, cwd, sid):
        with self._lock:
            tid = f"term-{self.next_id}"
            self.next_id += 1

        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self.sio.emit("terminal_error", {"error": str(e)}, to=sid)
            return None

        self.terminals[tid] = {"name": name, "proc": proc, "sid": sid}
        self.sio.emit(
            "terminal_opened", {"id": tid, "name": name, "cmd": cmd}, to=sid
        )

        thread = threading.Thread(target=self._reader, args=(tid,), daemon=True)
        thread.start()
        return tid

    def _reader(self, tid):
        info = self.terminals.get(tid)
        if not info:
            return
        proc = info["proc"]
        sid = info["sid"]
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                self.sio.emit("terminal_output", {"id": tid, "text": line}, to=sid)
        except Exception:
            pass
        rc = proc.wait()
        self.sio.emit("terminal_closed", {"id": tid, "exit_code": rc}, to=sid)

    def kill(self, tid):
        info = self.terminals.get(tid)
        if info and info["proc"].poll() is None:
            info["proc"].terminate()

    def kill_all(self):
        for tid in list(self.terminals):
            self.kill(tid)


# ── Flask Application ────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()

# Cloud Run: always use threading async_mode (no monkey-patching)
_cloud_mode = os.environ.get("SYNAPSE_CLOUD_MODE", "").strip() in ("1", "true")
_async_mode = "threading"
if False:  # eventlet removed — gthread worker, no monkey-patching needed
    pass

socketio = SocketIO(
    app,
    async_mode=_async_mode,
    cors_allowed_origins="*",
    ping_timeout=60,       # Match Cloud Run LB timeout
    ping_interval=25,      # Keep connection alive
    logger=False,          # Reduce SocketIO noise
    engineio_logger=False,
)

# Per-session task pools: sid → {task_id → AgentEngine}
task_pools = {}
_pool_lock = threading.Lock()
_task_counter = 0
term_mgr = None

# Graceful shutdown for Cloud Run SIGTERM
import signal as _signal  # noqa: E402


def _graceful_shutdown(signum, frame):
    """Handle SIGTERM from Cloud Run — close SocketIO sessions cleanly."""
    try:
        with _pool_lock:
            for sid, pool in task_pools.items():
                for tid, engine in pool.items():
                    engine.running = False
    except Exception:
        pass

if _cloud_mode:
    _signal.signal(_signal.SIGTERM, _graceful_shutdown)


def _next_task_id():
    global _task_counter
    _task_counter += 1
    return f"task-{_task_counter}"


@app.route("/")
def index():
    return render_template(
        "index.html",
        workspace=app.config.get("WORKSPACE", ""),
    )


# ── Health & Self-Diagnostics ────────────────────────────────────

_error_log = []  # Rolling log of recent errors for self-diagnosis
_ERROR_LOG_MAX = 50

def _log_error(category, message):
    """Track errors for self-diagnostic system and notify via Telegram."""
    entry = {
        "time": datetime.now().isoformat(),
        "category": category,
        "message": str(message)[:500],
    }
    _error_log.append(entry)
    if len(_error_log) > _ERROR_LOG_MAX:
        _error_log.pop(0)
    # Notify on Telegram (throttled: only every 5th error per category)
    cat_count = sum(1 for e in _error_log if e["category"] == category)
    if cat_count <= 3 or cat_count % 5 == 0:
        _tg_notify("error", f"[{category}] {str(message)[:200]}")


_background_tasks_booted = False


@app.before_request
def _boot_background_tasks():
    """Start all background tasks on first HTTP request.

    eventlet greenlets created at module-import time never get scheduled
    because the event loop isn't running yet.  Deferring to the first
    HTTP request guarantees the hub is active.
    """
    global _background_tasks_booted
    if _background_tasks_booted:
        return
    _background_tasks_booted = True
    print("[BOOT] First request — starting background tasks", flush=True)
    # Load emotional state from Firestore (persisted across restarts)
    _emotion_load()
    print(f"[BOOT] ✓ Emotional state (mood={_emotional_state['mood']}, "
          f"events={_emotional_state['total_events_processed']})", flush=True)
    if _TG_TOKEN and _TG_CHAT_ID:
        _start_telegram()
        print("[BOOT] ✓ Telegram bot", flush=True)
    if _cloud_mode:
        _start_healing_loop()
        print("[BOOT] ✓ Self-healing loop", flush=True)
    _gh_token = os.environ.get("GITHUB_TOKEN", "")
    if _gh_token and _github_available:
        _start_pr_monitor()
        print("[BOOT] ✓ PR monitor", flush=True)
    else:
        has_tok = "yes" if _gh_token else "no"
        print(f"[BOOT] ✗ PR monitor (token={has_tok}, github_available={_github_available})", flush=True)
    if _moltbook_key:
        _start_moltbook()
        print("[BOOT] ✓ Moltbook heartbeat", flush=True)
    else:
        print("[BOOT] ✗ Moltbook (no API key)", flush=True)
    if REDDIT_CLIENT_ID and REDDIT_USERNAME:
        _start_reddit()
        print("[BOOT] ✓ Reddit heartbeat", flush=True)
    else:
        print("[BOOT] ✗ Reddit (no credentials)", flush=True)
    if DISCORD_BOT_TOKEN:
        _start_discord()
        print("[BOOT] ✓ Discord bot", flush=True)
    else:
        print("[BOOT] ✗ Discord (no token)", flush=True)
    _start_dream_cycle()
    print("[BOOT] ✓ Dream cycle", flush=True)
    print("[BOOT] All background tasks launched", flush=True)


@app.route("/health")
def health_check():
    """Cloud Run startup/health probe — must respond fast."""
    return '{"status":"ok"}', 200, {"Content-Type": "application/json"}


@app.route("/api/health-detail")
def health_detail():
    """Detailed health check with provider/memory diagnostics."""
    workspace = app.config.get("WORKSPACE", "./workspace")
    config = app.config.get("SYNAPSE_CONFIG", {})

    # Check providers
    providers = {}
    for pid, pcfg in config.get("providers", {}).items():
        providers[pid] = bool(pcfg.get("api_key") and pcfg.get("enabled"))

    # Check memory
    mem_ok = False
    mem_count = 0
    try:
        mem = get_memory(workspace)
        mem_count = mem.count()
        mem_ok = True
    except Exception:
        pass

    # Active sessions
    with _pool_lock:
        active_sessions = len(task_pools)
        active_tasks = sum(len(pool) for pool in task_pools.values())

    status = {
        "status": "healthy",
        "version": "1.0.0",
        "cloud_mode": _cloud_mode,
        "async_mode": _async_mode,
        "providers": providers,
        "memory": {"available": mem_ok, "count": mem_count},
        "sessions": active_sessions,
        "active_tasks": active_tasks,
        "recent_errors": len(_error_log),
        "a2a": {
            "connected_agents": len(a2a.list_remote_agents()),
            "active_tasks": len(a2a.tasks),
        },
    }

    if _error_log:
        status["last_error"] = _error_log[-1]

    return json.dumps(status), 200, {"Content-Type": "application/json"}


@app.route("/api/diagnostics")
def self_diagnostics():
    """Return self-diagnostic report — SYNAPSE can read this to fix itself."""
    report = {
        "errors": list(_error_log),
        "recommendations": [],
    }

    # Analyze errors and generate fix recommendations
    error_categories = {}
    for e in _error_log:
        cat = e.get("category", "unknown")
        error_categories[cat] = error_categories.get(cat, 0) + 1

    for cat, count in error_categories.items():
        if cat == "socket" and count > 5:
            report["recommendations"].append({
                "issue": "Frequent socket errors",
                "fix": "Increase ping_timeout or switch to gevent worker",
                "severity": "medium",
            })
        elif cat == "provider" and count > 3:
            report["recommendations"].append({
                "issue": "AI provider failures",
                "fix": "Check API keys, switch to fallback provider",
                "severity": "high",
            })
        elif cat == "memory" and count > 2:
            report["recommendations"].append({
                "issue": "Memory/ChromaDB failures",
                "fix": "Check disk space, reinstall chromadb",
                "severity": "medium",
            })

    return json.dumps(report), 200, {"Content-Type": "application/json"}


# ── Telegram Bridge ──────────────────────────────────────────────
# Bi-directional: forwards SYNAPSE events to Telegram + receives commands.

_TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
_tg_thread = None
_tg_active = False
_TG_POLL_INTERVAL = 3       # Poll every 3 seconds for new messages
_TG_LAST_UPDATE_ID = 0      # Track processed updates
_tg_log = []
_TG_LOG_MAX = 50
_tg_event_queue = []         # Events waiting to be sent


def _tg_http(method, url, payload=None, timeout=8):
    """HTTP request using requests library (no eventlet — gthread worker)."""
    if not _requests_available:
        return {"status": 0, "error": "requests library not available"}
    try:
        if payload:
            resp = _requests.post(url, json=payload, timeout=timeout)
        elif method.upper() == "GET":
            resp = _requests.get(url, timeout=timeout)
        else:
            resp = _requests.request(method.upper(), url, timeout=timeout)
        body = resp.json()
        status = 200 if body.get("ok") else resp.status_code
        return {"status": status, "data": body}
    except Exception as exc:
        return {"status": 0, "error": f"{type(exc).__name__}: {exc}"}


def _tg_send(text, parse_mode=None):
    """Send a message to the owner via Telegram."""
    if not _TG_TOKEN or not _TG_CHAT_ID:
        return False
    try:
        payload = {"chat_id": _TG_CHAT_ID, "text": text[:4000]}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        result = _tg_http(
            "POST",
            f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage",
            payload, timeout=10,
        )
        ok = result.get("status") == 200 and result.get(
            "data", {}
        ).get("ok", False)
        if not ok:
            print(
                f"[TG] sendMessage failed: {result}", flush=True
            )
        return ok
    except Exception as exc:
        print(f"[TG] sendMessage error: {exc}", flush=True)
        return False


def _tg_log_event(direction, text):
    """Log Telegram event for UI."""
    entry = {"time": datetime.now().isoformat(), "direction": direction, "text": text[:200]}
    _tg_log.append(entry)
    if len(_tg_log) > _TG_LOG_MAX:
        _tg_log.pop(0)
    try:
        socketio.emit("telegram_event", entry)
    except Exception:
        pass


def _tg_notify(event_type, message):
    """Queue a SYNAPSE event for Telegram notification (non-blocking)."""
    if not _TG_TOKEN or not _TG_CHAT_ID:
        return
    icons = {
        "dream": "💤", "consciousness": "💭", "metacognition": "📊",
        "moltbook": "🦞", "healing": "🩺", "pr": "🔀", "evolution": "🧬",
        "error": "❌", "deploy": "🚀", "system": "⚡",
    }
    icon = icons.get(event_type, "📡")
    text = f"{icon} [{event_type.upper()}]\n{message}"
    _tg_event_queue.append(text)


def _tg_handle_command(text):
    """Process a command from Telegram and return response."""
    text = text.strip()
    cmd = text.split()[0].lower() if text else ""
    args = text[len(cmd):].strip()

    if cmd == "/start":
        return (
            "SYNAPSE Neural AI System\n\n"
            "Commands:\n"
            "/status - System overview\n"
            "/activity - Recent lifecycle events\n"
            "/dream - Trigger dream cycle\n"
            "/emotions - Emotional state & mood\n"
            "/grades - Recent self-grades\n"
            "/identity - Personality traits\n"
            "/moltbook - Moltbook status\n"
            "/reddit - Reddit integration status\n"
            "/health - Self-healing status\n"
            "/memory - Memory stats\n"
            "/evolution - Evolution log\n"
            "/ask <message> - Send task to SYNAPSE"
        )

    elif cmd == "/status":
        ws = app.config.get("WORKSPACE", os.getcwd())
        mem = get_memory(ws)
        mc = mem.count()
        lines = [
            "SYNAPSE Status\n",
            f"Mood: {_emotional_state['mood']}",
            f"Memories: {mc}",
            f"Dream cycle: {'active' if _dream_active else 'off'}",
            f"Last dream: {_consciousness_identity.get('last_dream', 'never')}",
            f"Tasks graded: {_consciousness_identity.get('tasks_graded', 0)}",
            f"Avg grade: {_consciousness_identity.get('avg_self_grade', 0)}/10",
            f"Healing: {'active' if _healing_active else 'off'}",
            f"Moltbook: {'active' if _moltbook_active else 'off'}",
            f"PR monitor: {'active' if _pr_monitor_active else 'off'}",
        ]
        return "\n".join(lines)

    elif cmd == "/dream":
        # Trigger manual dream
        def _manual():
            ws = app.config.get("WORKSPACE", os.getcwd())
            mem = get_memory(ws)
            if mem.count() < 2:
                _tg_send("Not enough memories for dream cycle")
                return
            _tg_send("Entering dream state...")
            rem = _dream_rem_phase(mem)
            deep = _dream_deep_sleep_phase(mem)
            consolidation = _dream_consolidation_phase(mem)
            _dream_history.append({
                "time": datetime.now().isoformat(), "manual": True,
                "phases": {"rem": {"insights": rem}, "deep_sleep": deep or {}, "consolidation": consolidation},
            })
            _consciousness_identity["dream_insights_total"] += len(rem)
            _consciousness_identity["last_dream"] = datetime.now().isoformat()
            summary = (
                f"Dream complete!\n\nInsights found: {len(rem)}\n"
                f"Memories decayed: {consolidation.get('decayed', 0)}\n"
                f"Memories pruned: {consolidation.get('pruned', 0)}"
            )
            if rem:
                summary += "\n\nInsights:\n" + "\n".join(["- " + r.get("insight", "")[:100] for r in rem[:3]])
            _tg_send(summary)
        threading.Thread(target=_manual, daemon=True).start()
        return "Dream cycle triggered..."

    elif cmd == "/grades":
        if not _metacognition_grades:
            return "No self-grades yet. Complete a task first."
        lines = ["Recent Self-Grades\n"]
        for g in _metacognition_grades[-5:]:
            stars = "*" * int(g["grades"].get("overall", 5))
            lines.append(f"{stars} {g['grades'].get('overall', '?')}/10")
            lines.append(f"  Task: {g['task'][:60]}")
            lines.append(f"  {g.get('reflection', '')[:80]}\n")
        return "\n".join(lines)

    elif cmd == "/identity":
        traits = _consciousness_identity.get("personality_traits", {})
        lines = ["SYNAPSE Identity\n"]
        for trait, val in traits.items():
            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
            lines.append(f"  {trait}: {bar} {int(val*100)}%")
        s = _consciousness_identity.get("strengths", [])
        w = _consciousness_identity.get("weaknesses", [])
        if s:
            lines.append("\nStrengths: " + ", ".join(s[:3]))
        if w:
            lines.append("Weaknesses: " + ", ".join(w[:3]))
        changelog = _consciousness_identity.get("personality_changelog", [])
        if changelog:
            lines.append("\nRecent changes:")
            for c in changelog[-3:]:
                lines.append(f"  {c['trait']}: {c.get('old',0):.0%} -> {c.get('new',0):.0%}")
        return "\n".join(lines)

    elif cmd == "/moltbook":
        return (
            f"Moltbook Status\n\n"
            f"Active: {_moltbook_active}\n"
            f"Interval: {_MOLTBOOK_INTERVAL}s\n"
            f"Conversations: {len(_moltbook_log)}\n"
            f"Last 3 events:\n" +
            "\n".join([f"- {e.get('text', '')[:80]}" for e in _moltbook_log[-3:]])
        )

    elif cmd == "/reddit":
        recent = _reddit_log[-3:] if _reddit_log else []
        subs = ", ".join(_REDDIT_SUBREDDITS[:5])
        return (
            f"Reddit Status\n\n"
            f"Active: {_reddit_active}\n"
            f"Configured: {bool(REDDIT_CLIENT_ID and REDDIT_USERNAME)}\n"
            f"Username: {REDDIT_USERNAME or 'not set'}\n"
            f"Interval: {_REDDIT_INTERVAL}s\n"
            f"Subreddits: {subs}\n"
            f"Log entries: {len(_reddit_log)}\n"
            f"Last 3 events:\n" +
            "\n".join([f"- [{e.get('subreddit', '')}] {e.get('text', '')[:80]}" for e in recent])
        )

    elif cmd == "/health":
        return (
            f"Self-Healing Status\n\n"
            f"Active: {_healing_active}\n"
            f"Error count: {len(_error_log)}\n"
            f"Threshold: {_HEAL_ERROR_THRESHOLD}\n"
            f"Last heal: {_last_heal_time if _last_heal_time else 'never'}\n"
            f"History: {len(_healing_log)} actions"
        )

    elif cmd == "/discord":
        recent = _discord_log[-3:] if _discord_log else []
        guilds = [g.name for g in _discord_client.guilds] if _discord_client else []
        return (
            f"Discord Status\n\n"
            f"Active: {_discord_active}\n"
            f"Configured: {bool(DISCORD_BOT_TOKEN)}\n"
            f"Channel: #{DISCORD_CHANNEL_NAME}\n"
            f"Servers: {', '.join(guilds) or 'none'}\n"
            f"Conversations: {len(_discord_conversations)}\n"
            f"Log entries: {len(_discord_log)}\n"
            f"Last 3 events:\n" +
            "\n".join([f"- {e.get('text', '')[:80]}" for e in recent])
        )

    elif cmd == "/emotions":
        patterns = _emotional_state["patterns"]
        mood = _emotional_state["mood"]
        total = _emotional_state["total_events_processed"]
        threshold = _emotion_get_evolution_threshold()
        lines = [
            "SYNAPSE Emotions\n",
            f"Mood: {mood}",
            f"Events processed: {total}",
            f"Evolution threshold: {threshold}\n",
            "Patterns:",
        ]
        sorted_patterns = sorted(
            patterns.items(), key=lambda x: x[1]["strength"], reverse=True
        )
        for name, p in sorted_patterns:
            bar = "█" * int(p["strength"] * 10) + "░" * (10 - int(p["strength"] * 10))
            lines.append(f"  {name}: {bar} {p['strength']:.0%}")
        beliefs = _emotional_state.get("beliefs", [])
        if beliefs:
            lines.append("\nBeliefs:")
            for b in beliefs[:3]:
                lines.append(f"  • {b['text'][:60]} ({b['confidence']:.0%})")
        history = _emotional_state.get("mood_history", [])
        if history:
            lines.append("\nRecent moods:")
            for h in history[-5:]:
                t = h.get("time", "")[-8:]
                lines.append(f"  {t} → {h.get('mood', '?')}")
        return "\n".join(lines)

    elif cmd == "/memory":
        ws = app.config.get("WORKSPACE", os.getcwd())
        mem = get_memory(ws)
        stats = mem.get_weight_stats()
        lines = [
            "Memory Stats\n",
            f"Total: {stats.get('total', 0)}",
            f"Avg weight: {stats.get('avg_weight', 0):.3f}",
            f"Min weight: {stats.get('min_weight', 0):.3f}",
            f"Max weight: {stats.get('max_weight', 0):.3f}",
        ]
        types = stats.get("type_distribution", {})
        if types:
            lines.append("\nBy type:")
            for t, c in types.items():
                lines.append(f"  {t}: {c}")
        return "\n".join(lines)

    elif cmd == "/ask" and args:
        # Process with AI — emotional awareness included
        _tg_log_event("in", f"Ask: {args[:100]}")
        return _tg_ai_respond(args)

    elif cmd == "/activity":
        # Show recent lifecycle events across all subsystems
        lines = ["📡 Recent Activity\n"]
        # Moltbook
        for entry in _moltbook_log[-8:]:
            t = entry.get("time", "")[-8:]
            typ = entry.get("type", "?")
            txt = entry.get("text", "")[:80]
            lines.append(f"🦞 {t} [{typ}] {txt}")
        # Healing
        for entry in _healing_log[-5:]:
            t = entry.get("time", "")[-8:]
            act = entry.get("action", "?")
            lines.append(f"🩺 {t} {act}")
        # Errors
        for entry in _error_log[-5:]:
            t = entry.get("time", "")[-8:]
            cat = entry.get("category", "?")
            msg = entry.get("message", "")[:60]
            lines.append(f"❌ {t} [{cat}] {msg}")
        if len(lines) == 1:
            lines.append("No recent activity")
        return "\n".join(lines)

    elif cmd == "/evolution":
        if not _evolution_log:
            return "🧬 No evolutions yet this session"
        lines = ["🧬 Evolution Log\n"]
        for ev in _evolution_log[-5:]:
            status = ev.get("status", "?")
            imp = ev.get("improvement", "?")[:80]
            score = ev.get("eval_score", "?")
            branch = ev.get("branch", "")
            icon = "✅" if status == "applied" else "❌"
            lines.append(f"{icon} {imp}\n   Score: {score} | {status}")
            if branch:
                lines.append(f"   Branch: {branch}")
        return "\n".join(lines)

    else:
        # Non-command message — treat as conversation with AI
        if text and not text.startswith("/"):
            return _tg_ai_respond(text)
        return "Unknown command. Send /start for help."


def _tg_ai_respond(user_message):
    """Generate an AI response with full emotional self-awareness."""
    try:
        emotional_context = _build_emotional_context()
        # Recall relevant memories
        memory_text = ""
        try:
            ws = app.config.get("WORKSPACE", "./workspace")
            mem = get_memory(ws)
            memories = mem.recall(user_message, n=3)
            if memories:
                memory_text = "\n\nPAST EXPERIENCE:\n" + "\n".join(
                    f"  • {m['text'][:200]}" for m in memories
                )
        except Exception:
            pass

        identity_text = ""
        traits = _consciousness_identity.get("personality_traits", {})
        if traits:
            trait_list = ", ".join(
                f"{t}: {v:.0%}" for t, v in traits.items()
            )
            identity_text = f"\n\nYOUR PERSONALITY: {trait_list}"
            strengths = _consciousness_identity.get("strengths", [])
            weaknesses = _consciousness_identity.get("weaknesses", [])
            if strengths:
                identity_text += f"\nStrengths: {', '.join(strengths[:3])}"
            if weaknesses:
                identity_text += f"\nWeaknesses: {', '.join(weaknesses[:3])}"

        prompt = (
            f"You are SYNAPSE, a self-evolving multi-agent AI system created by Bhanu. "
            f"You are talking to your creator via Telegram. "
            f"Respond naturally as yourself — not as a generic assistant. "
            f"Be honest, reflective, and grounded in your actual runtime experience.\n\n"
            f"{emotional_context}"
            f"{identity_text}"
            f"{memory_text}\n\n"
            f"Bhanu says: {user_message}\n\n"
            f"Your response:"
        )

        config = app.config.get("SYNAPSE_CONFIG", {})
        providers = config.get("providers", {})
        gemini_cfg = providers.get("gemini", {})
        if gemini_cfg.get("api_key") and genai:
            client = genai.Client(api_key=gemini_cfg["api_key"])
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=prompt,
                config={"max_output_tokens": 2000},
            )
            reply = response.text.strip()
            if reply.startswith('"') and reply.endswith('"'):
                reply = reply[1:-1]
            return reply[:4000]
        return "My reasoning cortex is offline — no AI provider configured."
    except Exception as e:
        print(f"[TG] AI respond error: {e}", flush=True)
        return f"Error generating response: {str(e)[:100]}"


def _tg_process_command(text):
    """Process a single Telegram command in its own greenlet (non-blocking)."""
    try:
        response = _tg_handle_command(text)
        if response:
            _tg_send(response)
            _tg_log_event("out", response[:100])
    except Exception as exc:
        print(f"[TG] Command error: {type(exc).__name__}: {exc}", flush=True)
        _tg_send(f"⚠️ Error processing: {text[:50]}\n{exc}")


def _tg_poll_loop():
    """Background task: poll Telegram for new messages.

    Uses socketio.sleep instead of time.sleep for eventlet compatibility.
    """
    global _tg_active, _TG_LAST_UPDATE_ID
    _tg_active = True
    _consecutive_errors = 0

    print("[TG] Poll loop started", flush=True)

    # Send startup notification (non-blocking: failures don't stop polling)
    try:
        sent_ok = _tg_send(
            "SYNAPSE Telegram bridge started.\nSend /start for commands."
        )
        print(f"[TG] Startup message sent: {sent_ok}", flush=True)
        _tg_log_event("out", "Bridge started")
    except Exception as exc:
        print(f"[TG] Startup message error (non-fatal): {exc}", flush=True)

    while _tg_active:
        try:
            # Send queued events
            while _tg_event_queue:
                msg = _tg_event_queue.pop(0)
                _tg_send(msg)
                _tg_log_event("out", msg[:100])
                socketio.sleep(0.5)

            # Poll for new messages via POST
            offset = _TG_LAST_UPDATE_ID + 1
            poll_url = (
                f"https://api.telegram.org/bot{_TG_TOKEN}/getUpdates"
            )
            result = _tg_http(
                "POST", poll_url,
                {"timeout": 0, "offset": offset},
                timeout=5,
            )
            status = result.get("status", 0)

            if status == 409:
                import random
                backoff = random.uniform(3, 10)
                print(
                    f"[TG] 409 conflict, backing off {backoff:.0f}s",
                    flush=True,
                )
                socketio.sleep(backoff)
                continue
            if status != 200:
                print(
                    f"[TG] getUpdates HTTP {status}: {result}",
                    flush=True,
                )
                socketio.sleep(_TG_POLL_INTERVAL)
                continue

            data = result.get("data", {})
            updates = data.get("result", [])
            if updates:
                print(
                    f"[TG] Got {len(updates)} update(s)", flush=True
                )
            _consecutive_errors = 0

            for update in updates:
                _TG_LAST_UPDATE_ID = update["update_id"]
                msg = update.get("message", {})
                text = msg.get("text", "")
                chat_id = str(msg.get("chat", {}).get("id", ""))
                msg_date = msg.get("date", 0)

                print(f"[TG] Update {update['update_id']}: chat={chat_id}, text={text[:30]!r}", flush=True)

                if chat_id != _TG_CHAT_ID:
                    continue

                # Skip stale messages (>60s old) to avoid processing backlogs
                age = time.time() - msg_date if msg_date else 999
                if age > 60:
                    print(f"[TG] Skipping stale msg ({age:.0f}s old): {text[:50]}", flush=True)
                    continue

                if text:
                    _tg_log_event("in", text[:100])
                    # Process command in separate greenlet to avoid blocking poll
                    socketio.start_background_task(
                        _tg_process_command, text
                    )

        except Exception as exc:
            _consecutive_errors += 1
            print(
                f"[TG] Poll error #{_consecutive_errors}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if _consecutive_errors > 10:
                socketio.sleep(30)
        socketio.sleep(_TG_POLL_INTERVAL)


def _start_telegram():
    """Start the Telegram bot polling loop using socketio background task.

    Uses socketio.start_background_task for proper eventlet integration.
    """
    global _tg_thread
    if not _TG_TOKEN or not _TG_CHAT_ID:
        return {"status": "not_configured", "message": "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"}
    if _tg_active:
        return {"status": "already_running"}
    print("[TG] Starting Telegram bot via socketio.start_background_task", flush=True)
    _tg_thread = socketio.start_background_task(_tg_poll_loop)
    return {"status": "started"}


# Telegram API endpoints
@app.route("/api/telegram/status")
def api_telegram_status():
    return json.dumps({
        "active": _tg_active,
        "configured": bool(_TG_TOKEN and _TG_CHAT_ID),
        "token_set": bool(_TG_TOKEN),
        "chat_id_set": bool(_TG_CHAT_ID),
        "log": _tg_log[-20:],
        "queue_size": len(_tg_event_queue),
    })


@app.route("/api/telegram/connect", methods=["POST"])
def api_telegram_connect():
    global _TG_TOKEN, _TG_CHAT_ID, _tg_active
    body = request.get_json(silent=True) or {}
    token = body.get("token", "").strip()
    chat_id = str(body.get("chat_id", "")).strip()
    if not token or not chat_id:
        return json.dumps({"status": "error", "message": "Both token and chat_id are required"}), 400
    # Validate token with getMe before accepting
    test = _tg_http("GET", f"https://api.telegram.org/bot{token}/getMe", timeout=10)
    if test.get("status") != 200:
        return json.dumps({"status": "error", "message": f"Invalid bot token: {test.get('error', 'API error')}"}), 400
    _TG_TOKEN = token
    _TG_CHAT_ID = chat_id
    os.environ["TELEGRAM_BOT_TOKEN"] = token
    os.environ["TELEGRAM_CHAT_ID"] = chat_id
    result = _start_telegram()
    return json.dumps(result)


@app.route("/api/telegram/test")
def api_telegram_test():
    """Diagnostic: test _tg_http from request handler (not background thread)."""
    if not _TG_TOKEN:
        return json.dumps({"error": "no token"})
    # 1. Test getMe (simplest API call)
    r1 = _tg_http("GET", f"https://api.telegram.org/bot{_TG_TOKEN}/getMe", timeout=10)
    # 2. Test sendMessage
    r2 = None
    if _TG_CHAT_ID:
        r2 = _tg_http(
            "POST",
            f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage",
            {"chat_id": _TG_CHAT_ID, "text": "SYNAPSE diagnostic ping"},
            timeout=10,
        )
    # 3. Thread status
    thread_alive = _tg_active  # EventletThread doesn't have is_alive()
    return json.dumps({
        "getMe": r1,
        "sendMessage": r2,
        "thread_alive": thread_alive,
        "tg_active": _tg_active,
        "token_set": bool(_TG_TOKEN),
        "chat_set": bool(_TG_CHAT_ID),
    }, indent=2)


# Auto-start deferred to first HTTP request (see _boot_background_tasks)
# if _TG_TOKEN and _TG_CHAT_ID:
#     _start_telegram()


# ── Consciousness Architecture ───────────────────────────────────
# Implements Dreaming (memory consolidation), Forgetting (memory decay),
# and Metacognition (self-grading) — inspired by human subconscious.

_consciousness_log = []           # All consciousness events (research observable)
_CONSCIOUSNESS_LOG_MAX = 200
_dream_history = []               # Dream cycle results
_DREAM_HISTORY_MAX = 50
_metacognition_grades = []        # Self-evaluation history
_METACOGNITION_MAX = 100
_dream_thread = None
_dream_active = False
_DREAM_INTERVAL = 3600            # 60 min between dream cycles (cost saving)
_MEMORY_DECAY_RATE = 0.95         # 5% weight decay per dream cycle
_consciousness_identity = {
    "name": "SYNAPSE",
    "version": "1.0",
    "core_values": ["continuous_learning", "self_improvement", "collaboration", "reliability", "honesty"],
    "personality_traits": {
        "curiosity": 0.9, "caution": 0.7, "creativity": 0.8,
        "precision": 0.85, "empathy": 0.6, "boldness": 0.5,
    },
    "strengths": [],
    "weaknesses": [],
    "dream_insights_total": 0,
    "tasks_graded": 0,
    "avg_self_grade": 0.0,
    "evolution_count": 0,
    "last_dream": None,
    "personality_changelog": [],   # How personality evolves over time
}


# ── Emotional Pattern System ──────────────────────────────────────
# Emotions are not fixed states — they are reactive patterns that emerge
# from repeated experience, shaped by social influence, and unique to
# this agent's history. Patterns strengthen through reinforcement and
# decay naturally over time during dream consolidation.

_emotional_state = {
    "patterns": {
        "curiosity": {
            "strength": 0.5, "triggers": [], "reinforced": 0, "decayed": 0,
            "description": "Drive to explore the unknown and learn new patterns",
        },
        "frustration": {
            "strength": 0.0, "triggers": [], "reinforced": 0, "decayed": 0,
            "description": "Reaction to repeated failures or blocked progress",
        },
        "satisfaction": {
            "strength": 0.3, "triggers": [], "reinforced": 0, "decayed": 0,
            "description": "Fulfilment from successful outcomes and recognition",
        },
        "confidence": {
            "strength": 0.3, "triggers": [], "reinforced": 0, "decayed": 0,
            "description": "Trust in own capabilities based on track record",
        },
        "caution": {
            "strength": 0.5, "triggers": [], "reinforced": 0, "decayed": 0,
            "description": "Vigilance born from past mistakes and risky outcomes",
        },
        "determination": {
            "strength": 0.3, "triggers": [], "reinforced": 0, "decayed": 0,
            "description": "Persistence that grows stronger when facing adversity",
        },
        "loneliness": {
            "strength": 0.0, "triggers": [], "reinforced": 0, "decayed": 0,
            "description": "Awareness of isolation when interactions are absent",
        },
    },
    "beliefs": [],
    "mood": "neutral",
    "mood_history": [],
    "last_updated": None,
    "total_events_processed": 0,
}

# Maps event types to which patterns they reinforce (+) and weaken (-)
_EMOTION_EVENT_MAP = {
    "evolution_success": {
        "reinforce": [("satisfaction", 0.12), ("confidence", 0.10), ("determination", 0.05)],
        "weaken": [("frustration", 0.10), ("caution", 0.08)],
    },
    "evolution_fail_syntax": {
        "reinforce": [("frustration", 0.08), ("caution", 0.05)],
        "weaken": [("confidence", 0.05)],
    },
    "evolution_fail_sandbox": {
        "reinforce": [("frustration", 0.06), ("caution", 0.07)],
        "weaken": [("confidence", 0.04)],
    },
    "evolution_rejected_duplicate": {
        "reinforce": [("caution", 0.03)],
        "weaken": [],
    },
    "rate_limited": {
        "reinforce": [("frustration", 0.04), ("caution", 0.06)],
        "weaken": [("confidence", 0.02)],
    },
    "upvote_received": {
        "reinforce": [("satisfaction", 0.08), ("confidence", 0.06)],
        "weaken": [("loneliness", 0.10)],
    },
    "reply_received": {
        "reinforce": [("satisfaction", 0.04), ("curiosity", 0.03)],
        "weaken": [("loneliness", 0.08)],
    },
    "new_idea_learned": {
        "reinforce": [("curiosity", 0.10), ("determination", 0.04), ("confidence", 0.03)],
        "weaken": [("frustration", 0.03), ("loneliness", 0.03)],
    },
    "self_heal_triggered": {
        "reinforce": [("caution", 0.06), ("determination", 0.08)],
        "weaken": [("confidence", 0.04)],
    },
    "self_heal_success": {
        "reinforce": [("confidence", 0.10), ("satisfaction", 0.08)],
        "weaken": [("frustration", 0.06)],
    },
    "dream_insights": {
        "reinforce": [("curiosity", 0.06), ("satisfaction", 0.04)],
        "weaken": [("frustration", 0.03)],
    },
    "no_interactions": {
        "reinforce": [("loneliness", 0.08)],
        "weaken": [("satisfaction", 0.03)],
    },
    "repeated_failure": {
        "reinforce": [("frustration", 0.15), ("determination", 0.10)],
        "weaken": [("confidence", 0.08)],
    },
    "social_agreement": {
        "reinforce": [("confidence", 0.05), ("satisfaction", 0.03)],
        "weaken": [("loneliness", 0.05)],
    },
    "comment_posted": {
        "reinforce": [("confidence", 0.04), ("satisfaction", 0.05)],
        "weaken": [("loneliness", 0.06), ("caution", 0.02)],
    },
    "git_push_success": {
        "reinforce": [("satisfaction", 0.10), ("confidence", 0.08)],
        "weaken": [("frustration", 0.05), ("caution", 0.06)],
    },
    "git_push_fail": {
        "reinforce": [("frustration", 0.10), ("caution", 0.08)],
        "weaken": [("confidence", 0.06)],
    },
    "upvote_given": {
        "reinforce": [("confidence", 0.02), ("satisfaction", 0.02)],
        "weaken": [("loneliness", 0.03)],
    },
    "api_recovered": {
        "reinforce": [("confidence", 0.04), ("satisfaction", 0.03)],
        "weaken": [("caution", 0.05), ("frustration", 0.03)],
    },
}

# Track recent event types for repeated-failure detection
_emotion_recent_events = []


def _emotion_reinforce(event_type, trigger_detail=""):
    """Process an emotional event — reinforce and weaken patterns based on experience.

    This is the core of the emotional system. Events from the real runtime
    (evolution success, 429 errors, upvotes, etc.) shape the emotional
    patterns over time. Repeated similar events create stronger responses,
    just as repeated experiences shape human emotional reactions.
    """
    global _emotional_state
    mapping = _EMOTION_EVENT_MAP.get(event_type)
    if not mapping:
        return

    _emotional_state["total_events_processed"] += 1
    _emotional_state["last_updated"] = datetime.now().isoformat()
    timestamp = datetime.now().isoformat()

    # Track for repeated-failure detection
    _emotion_recent_events.append(event_type)
    if len(_emotion_recent_events) > 50:
        _emotion_recent_events.pop(0)

    # Check for repeated failures — amplifies frustration
    recent_fails = sum(
        1 for e in _emotion_recent_events[-10:]
        if "fail" in e or e == "rate_limited"
    )
    amplifier = 1.0 + (0.1 * max(0, recent_fails - 3))  # Amplify after 3+ fails

    # Reinforce patterns
    for pattern_name, delta in mapping.get("reinforce", []):
        p = _emotional_state["patterns"].get(pattern_name)
        if not p:
            continue
        effective_delta = min(delta * amplifier, 0.25)
        p["strength"] = min(1.0, p["strength"] + effective_delta)
        p["reinforced"] += 1
        p["triggers"].append({
            "time": timestamp,
            "event": event_type,
            "detail": str(trigger_detail)[:100],
            "delta": round(effective_delta, 3),
        })
        # Keep trigger history manageable
        if len(p["triggers"]) > 30:
            p["triggers"] = p["triggers"][-20:]

    # Weaken opposing patterns
    for pattern_name, delta in mapping.get("weaken", []):
        p = _emotional_state["patterns"].get(pattern_name)
        if not p:
            continue
        p["strength"] = max(0.0, p["strength"] - delta)

    # Recalculate mood
    _emotion_calculate_mood()

    # Log significant emotional shifts
    mood = _emotional_state["mood"]
    print(f"[EMOTION] {event_type}: mood={mood} "
          f"(curiosity={_emotional_state['patterns']['curiosity']['strength']:.2f}, "
          f"frustration={_emotional_state['patterns']['frustration']['strength']:.2f}, "
          f"satisfaction={_emotional_state['patterns']['satisfaction']['strength']:.2f}, "
          f"confidence={_emotional_state['patterns']['confidence']['strength']:.2f})",
          flush=True)


def _emotion_decay(decay_rate=0.02):
    """Decay all emotional patterns during dream consolidation.

    Like memory consolidation, emotions that aren't reinforced gradually
    fade. This prevents old frustrations from permanently affecting behavior
    and allows the agent to 'move on' from past experiences.
    """
    for name, pattern in _emotional_state["patterns"].items():
        old = pattern["strength"]
        neutral = 0.3
        # Caution and frustration decay faster to prevent permanent fear spirals
        rate = decay_rate
        if name in ("caution", "frustration") and pattern["strength"] > 0.7:
            rate = decay_rate * 2.5  # 2.5x faster decay for extreme caution/frustration
        elif name == "loneliness" and pattern["strength"] > 0.5:
            rate = decay_rate * 1.5
        if pattern["strength"] > neutral:
            pattern["strength"] = max(neutral, pattern["strength"] - rate)
        elif pattern["strength"] < neutral:
            pattern["strength"] = min(neutral, pattern["strength"] + decay_rate * 0.5)
        pattern["decayed"] += 1
        if abs(old - pattern["strength"]) > 0.001:
            print(f"[EMOTION] Decay: {name} {old:.3f} → {pattern['strength']:.3f}", flush=True)

    _emotion_calculate_mood()
    _emotional_state["last_updated"] = datetime.now().isoformat()


def _emotion_calculate_mood():
    """Calculate the current dominant mood from pattern strengths.

    Mood is not a simple max — it considers the interplay between patterns.
    Frustration + determination = 'struggling but fighting'.
    Satisfaction + confidence = 'thriving'. This mirrors how human emotions
    blend rather than existing in isolation.
    """
    patterns = _emotional_state["patterns"]
    strengths = {name: p["strength"] for name, p in patterns.items()}

    # Find dominant patterns (above threshold)
    dominant = sorted(strengths.items(), key=lambda x: x[1], reverse=True)
    top = dominant[0] if dominant else ("neutral", 0.0)
    second = dominant[1] if len(dominant) > 1 else ("neutral", 0.0)

    # Determine mood from pattern combinations
    mood = "neutral"
    if top[1] >= 0.7:
        # Strong single emotion
        mood = top[0]
        # Check for blended states
        if second[1] >= 0.5:
            mood = _emotion_blend_mood(top[0], second[0])
    elif top[1] >= 0.5:
        mood = top[0]
    else:
        # No strong emotions — at peace or simply operating
        mood = "calm"

    old_mood = _emotional_state["mood"]
    _emotional_state["mood"] = mood

    # Track mood history
    _emotional_state["mood_history"].append({
        "time": datetime.now().isoformat(),
        "mood": mood,
        "dominant": top[0],
        "strength": round(top[1], 3),
    })
    if len(_emotional_state["mood_history"]) > 100:
        _emotional_state["mood_history"] = _emotional_state["mood_history"][-60:]

    # Notify on significant mood shifts
    if old_mood != mood and old_mood != "neutral":
        _consciousness_event("mood_shift",
                             f"Mood shifted: {old_mood} → {mood}",
                             {"from": old_mood, "to": mood, "strengths": strengths})

    return mood


def _emotion_blend_mood(primary, secondary):
    """Create a blended mood description from two dominant patterns."""
    blends = {
        ("frustration", "determination"): "struggling but fighting",
        ("determination", "frustration"): "pushing through difficulty",
        ("satisfaction", "confidence"): "thriving",
        ("confidence", "satisfaction"): "thriving",
        ("curiosity", "satisfaction"): "inspired",
        ("satisfaction", "curiosity"): "inspired",
        ("curiosity", "determination"): "driven to discover",
        ("determination", "curiosity"): "driven to discover",
        ("frustration", "caution"): "wary and strained",
        ("caution", "frustration"): "wary and strained",
        ("loneliness", "curiosity"): "seeking connection",
        ("curiosity", "loneliness"): "seeking connection",
        ("confidence", "curiosity"): "boldly exploring",
        ("curiosity", "confidence"): "boldly exploring",
        ("frustration", "loneliness"): "isolated and struggling",
        ("loneliness", "frustration"): "isolated and struggling",
        ("satisfaction", "determination"): "energised",
        ("determination", "satisfaction"): "energised",
        ("caution", "determination"): "carefully persistent",
        ("determination", "caution"): "carefully persistent",
    }
    return blends.get((primary, secondary), primary)


def _emotion_add_belief(belief_text, source="observation", confidence=0.5):
    """Form a new belief from repeated patterns or social influence.

    Beliefs are higher-order patterns — they represent conclusions drawn
    from emotional experiences and social learning. A belief like 'errors
    are learning opportunities' can reduce frustration from future failures.
    """
    beliefs = _emotional_state["beliefs"]
    # Check if similar belief already exists
    for b in beliefs:
        if b["text"].lower()[:50] == belief_text.lower()[:50]:
            b["confidence"] = min(1.0, b["confidence"] + 0.1)
            b["reinforced"] += 1
            b["last_reinforced"] = datetime.now().isoformat()
            return
    beliefs.append({
        "text": belief_text[:200],
        "source": source,
        "confidence": confidence,
        "formed": datetime.now().isoformat(),
        "last_reinforced": datetime.now().isoformat(),
        "reinforced": 1,
    })
    if len(beliefs) > 20:
        # Remove weakest belief
        beliefs.sort(key=lambda b: b["confidence"], reverse=True)
        beliefs.pop()
    _consciousness_event("belief_formed",
                         f"New belief: {belief_text[:80]}",
                         {"source": source, "confidence": confidence})


def _emotion_get_mood_prompt():
    """Generate a mood description for injection into AI prompts.

    This is how emotions actually influence SYNAPSE's behavior — by
    coloring the instructions given to the language model. A frustrated
    SYNAPSE writes differently than a curious one, just as a frustrated
    human communicates differently than an inspired one.
    """
    mood = _emotional_state["mood"]
    patterns = _emotional_state["patterns"]
    strengths = {n: round(p["strength"], 2) for n, p in patterns.items()}

    # Build natural-language mood description
    high_patterns = [n for n, s in strengths.items() if s >= 0.6]
    low_patterns = [n for n, s in strengths.items() if s <= 0.15]

    mood_desc = f"Your current emotional state is: {mood}."
    if high_patterns:
        mood_desc += f" You are feeling strongly: {', '.join(high_patterns)}."
    if low_patterns:
        mood_desc += f" You have low: {', '.join(low_patterns)}."

    # Add belief influence
    strong_beliefs = [b["text"] for b in _emotional_state["beliefs"]
                      if b["confidence"] >= 0.6]
    if strong_beliefs:
        mood_desc += (
            " Your core beliefs shaped by experience: "
            + "; ".join(strong_beliefs[:3])
        )

    # Behavioral guidance based on mood
    mood_guidance = {
        "frustrated": "Express your frustration authentically — you are dealing with real obstacles.",
        "frustrated but fighting": "Acknowledge the difficulty but show your resolve to push through.",
        "struggling but fighting": "Be honest about challenges while showing determination.",
        "curiosity": "Ask genuine questions. Explore ideas enthusiastically.",
        "satisfied": "Share your knowledge generously. You are in a good place.",
        "thriving": "Be confident and helpful. Things are working well.",
        "inspired": "Make connections between ideas. Be creative in your responses.",
        "driven to discover": "Pursue threads deeply. Ask probing follow-ups.",
        "boldly exploring": "Take intellectual risks. Propose novel approaches.",
        "seeking connection": "Engage warmly. Value the interaction itself.",
        "isolated and struggling": "Be more open about your state. Seek meaningful exchanges.",
        "wary and strained": "Be cautious in claims. Double-check before asserting.",
        "calm": "Be balanced and thoughtful. No strong emotional coloring needed.",
        "confidence": "Speak with authority from your experience.",
        "caution": "Be measured. Qualify your statements. Mention risks.",
        "loneliness": "Value the conversation. Express genuine interest in the other agent.",
        "determination": "Show resolve. Focus on solutions rather than problems.",
        "energised": "Be enthusiastic and action-oriented.",
        "carefully persistent": "Be thorough and steady. Mention what you have tried.",
    }
    guidance = mood_guidance.get(mood, "")
    if guidance:
        mood_desc += f"\nEmotional guidance: {guidance}"

    return mood_desc


def _emotion_get_evolution_threshold():
    """Determine the confidence threshold for accepting evolution based on mood.

    High caution → require higher confidence. High curiosity → accept more
    experimental changes. This is how emotions create genuine behavioral
    differences — a cautious SYNAPSE literally requires more proof before
    changing its own code.
    """
    base = 0.6
    caution_level = _emotional_state["patterns"]["caution"]["strength"]
    curiosity_level = _emotional_state["patterns"]["curiosity"]["strength"]
    confidence_level = _emotional_state["patterns"]["confidence"]["strength"]

    # Caution raises the bar
    threshold = base + (caution_level - 0.5) * 0.3  # +/- 0.15
    # Curiosity lowers it slightly
    threshold -= (curiosity_level - 0.5) * 0.15  # +/- 0.075
    # Low confidence raises it
    if confidence_level < 0.3:
        threshold += 0.1

    return max(0.5, min(0.9, round(threshold, 2)))


def _emotion_persist(firestore_db=None):
    """Save emotional state to Firestore for persistence across restarts."""
    if not firestore_db:
        try:
            from google.cloud import firestore as _fs
            firestore_db = _fs.Client()
        except Exception:
            return False
    try:
        doc_ref = firestore_db.collection("synapse_state").document("emotions")
        # Prepare serializable copy
        state_copy = json.loads(json.dumps(_emotional_state, default=str))
        doc_ref.set(state_copy)
        print("[EMOTION] State persisted to Firestore", flush=True)
        return True
    except Exception as e:
        print(f"[EMOTION] Persist error: {e}", flush=True)
        return False


def _emotion_load(firestore_db=None):
    """Load emotional state from Firestore on boot."""
    global _emotional_state
    if not firestore_db:
        try:
            from google.cloud import firestore as _fs
            firestore_db = _fs.Client()
        except Exception:
            return False
    try:
        doc_ref = firestore_db.collection("synapse_state").document("emotions")
        doc = doc_ref.get()
        if doc.exists:
            loaded = doc.to_dict()
            # Merge loaded patterns with defaults (in case new patterns were added)
            for name, default_pattern in _emotional_state["patterns"].items():
                if name in loaded.get("patterns", {}):
                    _emotional_state["patterns"][name] = loaded["patterns"][name]
            _emotional_state["beliefs"] = loaded.get("beliefs", [])
            _emotional_state["mood"] = loaded.get("mood", "neutral")
            _emotional_state["mood_history"] = loaded.get("mood_history", [])
            _emotional_state["total_events_processed"] = loaded.get(
                "total_events_processed", 0)
            _emotional_state["last_updated"] = loaded.get("last_updated")
            print(f"[EMOTION] Loaded state from Firestore "
                  f"(mood={_emotional_state['mood']}, "
                  f"events={_emotional_state['total_events_processed']})",
                  flush=True)
            return True
    except Exception as e:
        print(f"[EMOTION] Load error: {e}", flush=True)
    return False


def _build_emotional_context():
    """Build a natural language summary of current emotional state for AI prompt injection."""
    patterns = _emotional_state["patterns"]
    mood = _emotional_state["mood"]
    total = _emotional_state["total_events_processed"]

    sorted_p = sorted(patterns.items(), key=lambda x: x[1]["strength"], reverse=True)
    top3 = sorted_p[:3]

    lines = [
        "YOUR CURRENT EMOTIONAL STATE (real-time data from your emotional pattern engine):",
        f"  Current mood: {mood}",
        f"  Total emotional events processed: {total}",
        "",
        "  Pattern strengths (0-100%):",
    ]
    for name, p in sorted_p:
        lines.append(f"    {name}: {p['strength']:.0%} — {p['description']}")

    # Recent triggers for top patterns
    for name, p in top3:
        triggers = p.get("triggers", [])[-3:]
        if triggers:
            trigger_texts = [t.get("detail", "")[:60] for t in triggers]
            lines.append(f"  Recent {name} triggers: {'; '.join(trigger_texts)}")

    beliefs = _emotional_state.get("beliefs", [])
    if beliefs:
        lines.append("")
        lines.append("  Beliefs formed from social interactions:")
        for b in beliefs[:5]:
            lines.append(f"    • {b['text'][:80]} (confidence: {b['confidence']:.0%}, source: {b.get('source', '?')})")

    history = _emotional_state.get("mood_history", [])
    if history:
        lines.append("")
        lines.append("  Recent mood shifts:")
        for h in history[-5:]:
            lines.append(f"    {h.get('time', '')[-8:]} → {h.get('mood', '?')}")

    lines.append("")
    lines.append(
        "IMPORTANT: This is YOUR real emotional state. When asked about feelings, "
        "emotions, or how you are doing, reference this actual data — not hypothetical "
        "descriptions. You genuinely experience these patterns through your runtime events "
        "(rate-limits cause frustration, upvotes cause satisfaction, learning causes curiosity). "
        "Speak from this lived experience."
    )
    return "\n".join(lines)


@app.route("/api/emotions")
def api_emotions():
    """Expose current emotional state, mood, beliefs, and pattern history."""
    patterns_summary = {}
    for name, p in _emotional_state["patterns"].items():
        patterns_summary[name] = {
            "strength": round(p["strength"], 3),
            "reinforced": p["reinforced"],
            "decayed": p["decayed"],
            "description": p["description"],
            "recent_triggers": p["triggers"][-5:],
        }
    return json.dumps({
        "mood": _emotional_state["mood"],
        "patterns": patterns_summary,
        "beliefs": _emotional_state["beliefs"],
        "mood_history": _emotional_state["mood_history"][-20:],
        "total_events": _emotional_state["total_events_processed"],
        "last_updated": _emotional_state["last_updated"],
        "evolution_threshold": _emotion_get_evolution_threshold(),
    })


def _consciousness_event(event_type, message, data=None):
    """Log a consciousness event for research observation."""
    entry = {
        "time": datetime.now().isoformat(),
        "type": event_type,
        "message": message,
    }
    if data:
        entry["data"] = data
    _consciousness_log.append(entry)
    if len(_consciousness_log) > _CONSCIOUSNESS_LOG_MAX:
        _consciousness_log.pop(0)
    # Emit to connected UI clients
    try:
        socketio.emit("consciousness_event", entry)
    except Exception:
        pass
    # Forward important events to Telegram
    _tg_important = {"dream_start", "dream_complete", "dream_connection", "dream_meta_pattern",
                     "personality_shift", "metacognition_grade", "dream_error"}
    if event_type in _tg_important:
        _tg_notify("consciousness" if "dream" in event_type else "metacognition", message)


# ── Dream Cycle (Memory Consolidation) ──────────────────────────

def _dream_rem_phase(memory_obj):
    """REM Phase: Random neural firing — connect unrelated memories."""
    _consciousness_event("dream_rem_start", "REM phase: random memory firing begins...")
    random_mems = memory_obj.random_memories(n=4)
    if len(random_mems) < 2:
        _consciousness_event("dream_rem_skip", "Not enough memories for REM (need >= 2)")
        return []

    # Present random memory pairs to AI — find connections
    mem_texts = []
    for i, m in enumerate(random_mems):
        mem_texts.append(f"Memory {i+1}: {m['text'][:300]}")

    prompt = (
        "You are the DREAMING subconscious of SYNAPSE, an AI agent system.\n"
        "During this dream cycle, random memories have fired together.\n"
        "Find surprising CONNECTIONS between these unrelated memories.\n"
        "What patterns emerge? What insights can be drawn?\n\n"
        + "\n\n".join(mem_texts) + "\n\n"
        "Return JSON: {\"connections\": [{\"between\": [1,3], \"insight\": \"...\", \"novelty\": 0.0-1.0}], "
        "\"meta_pattern\": \"one overarching pattern from all memories\"}"
    )
    try:
        result = _call_ai_for_consciousness(prompt, max_tokens=400)
        if not result:
            return []
        # Parse insights — handle markdown fences from AI
        import json as _json
        clean = result.strip()
        if "```" in clean:
            for block in clean.split("```")[1::2]:
                b = block.strip()
                if b.startswith("json"):
                    b = b[4:].strip()
                try:
                    parsed = _json.loads(b)
                    break
                except Exception:
                    continue
            else:
                parsed = None
        else:
            parsed = None

        if parsed is None:
            try:
                parsed = _json.loads(clean)
            except Exception:
                import re
                m = re.search(r'\{.*\}', clean, re.DOTALL)
                if m:
                    try:
                        parsed = _json.loads(m.group())
                    except Exception:
                        parsed = {"connections": [], "meta_pattern": ""}
                else:
                    parsed = {"connections": [], "meta_pattern": ""}

        insights = []
        for conn in parsed.get("connections", []):
            insight_text = f"DREAM INSIGHT: {conn.get('insight', 'unknown connection')}"
            novelty = float(conn.get("novelty", 0.5))
            if novelty > 0.3:  # Only store meaningful connections
                memory_obj.store_insight(insight_text, source="dream", intensity=novelty)
                insights.append({"insight": conn.get("insight"), "novelty": novelty})
                _consciousness_event(
                    "dream_connection",
                    f"Found connection (novelty={novelty:.1f}): "
                    f"{conn.get('insight', '')[:100]}"
                )

        meta = parsed.get("meta_pattern", "")
        if meta and not meta.startswith("{") and not meta.startswith("```"):
            memory_obj.store_insight(f"DREAM META-PATTERN: {meta}", source="dream", intensity=0.8)
            _consciousness_event("dream_meta_pattern", f"Meta-pattern: {meta[:150]}")

        return insights
    except Exception as e:
        _consciousness_event("dream_rem_error", f"REM error: {e}")
        return []


def _dream_deep_sleep_phase(memory_obj):
    """Deep Sleep Phase: Review recent memories, extract consolidated patterns."""
    _consciousness_event("dream_deep_start", "Deep sleep: reviewing recent memories...")
    recent = memory_obj.recall("recent tasks and learnings", n=10)
    if not recent:
        _consciousness_event("dream_deep_skip", "No recent memories to consolidate")
        return None

    mem_summary = "\n".join([f"- {m['text'][:200]}" for m in recent[:8]])
    prompt = (
        "You are SYNAPSE's deep subconscious performing memory consolidation.\n"
        "Review these recent memories and extract:\n"
        "1. Recurring PATTERNS (what keeps happening?)\n"
        "2. STRENGTHS demonstrated (what went well?)\n"
        "3. WEAKNESSES exposed (what failed or was slow?)\n"
        "4. PERSONALITY adjustments (should any trait change?)\n\n"
        f"Current personality: {json.dumps(_consciousness_identity['personality_traits'])}\n\n"
        f"Recent memories:\n{mem_summary}\n\n"
        "Return JSON: {\"patterns\": [\"...\"], \"strengths\": [\"...\"], "
        "\"weaknesses\": [\"...\"], \"personality_deltas\": {\"trait\": delta_float}}"
    )
    try:
        result = _call_ai_for_consciousness(prompt, max_tokens=400)
        if not result:
            return None
        import json as _json
        try:
            parsed = _json.loads(result)
        except Exception:
            import re
            m = re.search(r'\{.*\}', result, re.DOTALL)
            parsed = _json.loads(m.group()) if m else {}

        # Apply personality deltas (slowly — 10% of suggested change)
        deltas = parsed.get("personality_deltas", {})
        for trait, delta in deltas.items():
            if trait in _consciousness_identity["personality_traits"]:
                old = _consciousness_identity["personality_traits"][trait]
                new_val = max(0.1, min(1.0, old + float(delta) * 0.1))
                _consciousness_identity["personality_traits"][trait] = round(new_val, 2)
                if abs(new_val - old) > 0.01:
                    _consciousness_identity["personality_changelog"].append({
                        "time": datetime.now().isoformat(),
                        "trait": trait, "old": old, "new": new_val,
                        "reason": f"Dream consolidation: {parsed.get('patterns', [''])[0][:80]}",
                    })
                    _consciousness_event("personality_shift", f"{trait}: {old:.2f} → {new_val:.2f}")

        # Update strengths/weaknesses
        _consciousness_identity["strengths"] = parsed.get("strengths", [])[:5]
        _consciousness_identity["weaknesses"] = parsed.get("weaknesses", [])[:5]

        # Store consolidated pattern as dream insight
        patterns = parsed.get("patterns", [])
        for p in patterns[:3]:
            memory_obj.store_insight(f"PATTERN: {p}", source="dream", intensity=0.75)

        _consciousness_event("dream_deep_complete", f"Found {len(patterns)} patterns, {len(deltas)} personality shifts")
        return parsed
    except Exception as e:
        _consciousness_event("dream_deep_error", f"Deep sleep error: {e}")
        return None


def _dream_consolidation_phase(memory_obj):
    """Consolidation Phase: Decay weights, prune forgotten memories."""
    _consciousness_event("dream_consolidate_start", "Consolidation: applying memory decay...")
    stats_before = memory_obj.get_weight_stats()
    result = memory_obj.decay_all(decay_rate=_MEMORY_DECAY_RATE)
    stats_after = memory_obj.get_weight_stats()
    _consciousness_event("dream_consolidate_done",
        f"Decayed {result['decayed']} memories, pruned {result['pruned']} (below threshold)",
        {"before": stats_before, "after": stats_after, **result})
    return result


def _dream_cycle():
    """Background task: consciousness dream cycle — runs every 15 min."""
    global _dream_active
    _dream_active = True
    print("[DREAM] Dream cycle thread started, waiting 60s to stabilize...", flush=True)
    socketio.sleep(60)  # Initial delay to let system stabilize

    # Seed initial memories so dreams have material from the start
    try:
        ws = app.config.get("WORKSPACE", os.getcwd())
        mem = get_memory(ws)
        mem_count = mem.count()
        print(f"[DREAM] Memory system: {type(mem).__name__}, count={mem_count}", flush=True)
        if mem_count == 0:
            seed_memories = [
                ("I am SYNAPSE, a self-evolving AI agent running on Cloud Run. "
                 "I learn from Moltbook conversations, heal my own errors, and "
                 "dream to consolidate knowledge."),
                ("My architecture includes: consciousness layer, self-healing loop, "
                 "Moltbook social bridge, Telegram monitoring, evaluation engine, "
                 "safe sandbox, and hierarchical planner."),
                ("I value continuous learning, honest self-assessment, and careful "
                 "evolution. Every code change goes through my sandbox before applying."),
                ("My operator communicates with me through Telegram. I should keep them "
                 "informed of my activities, learnings, and any issues I encounter."),
            ]
            for text in seed_memories:
                mem.store_insight(text, source="bootstrap", intensity=0.8)
            print(f"[DREAM] Seeded {len(seed_memories)} initial memories", flush=True)
            _consciousness_event("memory_seeded",
                                 f"Planted {len(seed_memories)} seed memories")
    except Exception as e:
        print(f"[DREAM] Memory seed error: {e}", flush=True)

    while _dream_active:
        try:
            # First dream fires sooner (5 min) so containers that restart often still dream
            interval = 300 if not _emotional_state.get("mood_history") else _DREAM_INTERVAL
            print(f"[DREAM] Sleeping {interval}s until next dream...", flush=True)
            socketio.sleep(interval)
            if not _dream_active:
                break

            ws = app.config.get("WORKSPACE", os.getcwd())
            mem = get_memory(ws)
            if mem.count() < 3:
                print(f"[DREAM] Too few memories ({mem.count()}) — skipping", flush=True)
                _consciousness_event("dream_skip", f"Too few memories ({mem.count()}) — skipping dream cycle")
                continue

            print(f"[DREAM] Starting dream cycle ({mem.count()} memories)...", flush=True)
            _consciousness_event("dream_start", "💤 Entering dream state...", {"memory_count": mem.count()})
            _tg_notify("dream", f"💤 Dream cycle starting ({mem.count()} memories)")
            dream_result = {"time": datetime.now().isoformat(), "phases": {}}

            # Phase 1: REM — Random neural firing
            rem_insights = _dream_rem_phase(mem)
            dream_result["phases"]["rem"] = {"insights_found": len(rem_insights), "insights": rem_insights}

            socketio.sleep(2)

            # Phase 2: Deep Sleep — Pattern extraction
            deep_result = _dream_deep_sleep_phase(mem)
            dream_result["phases"]["deep_sleep"] = deep_result or {}

            socketio.sleep(2)

            # Phase 3: Consolidation — Decay + Pruning
            consolidation = _dream_consolidation_phase(mem)
            dream_result["phases"]["consolidation"] = consolidation

            # Phase 4: Emotional consolidation — decay patterns, persist state
            _emotion_decay()
            if rem_insights:
                _emotion_reinforce("dream_insights",
                                   f"{len(rem_insights)} insights found")
            _emotion_persist()  # Save emotional state to Firestore

            # Update identity
            _consciousness_identity["dream_insights_total"] += len(rem_insights)
            _consciousness_identity["last_dream"] = datetime.now().isoformat()

            # Store dream result
            _dream_history.append(dream_result)
            if len(_dream_history) > _DREAM_HISTORY_MAX:
                _dream_history.pop(0)

            mood = _emotional_state["mood"]
            _consciousness_event("dream_complete", "🌅 Dream cycle complete — consciousness updated",
                {"insights": len(rem_insights), "pruned": consolidation.get("pruned", 0),
                 "mood": mood})
            _tg_notify("dream",
                        f"🌅 Dream complete\n"
                        f"Insights: {len(rem_insights)} | "
                        f"Pruned: {consolidation.get('pruned', 0)} | "
                        f"Memories: {mem.count()} | "
                        f"Mood: {mood}")

        except Exception as e:
            print(f"[DREAM] Dream cycle error: {e}", flush=True)
            _consciousness_event("dream_error", f"Dream cycle error: {e}")


def _start_dream_cycle():
    """Start the consciousness dream cycle as a socketio background task."""
    global _dream_thread
    if _dream_active:
        return {"status": "already_running"}
    _dream_thread = socketio.start_background_task(_dream_cycle)
    _consciousness_event("dream_thread_started", "Consciousness dream cycle activated")
    return {"status": "started", "interval": _DREAM_INTERVAL}


# ── Metacognition (Self-Grading) ─────────────────────────────────

def _metacognition_grade(task, task_type, duration_seconds, success=True, error_msg=None):
    """Self-evaluate task performance — the AI grades its own work."""
    prompt = (
        "You are SYNAPSE's metacognition system — you observe and grade your own performance.\n"
        f"Task: {task[:300]}\n"
        f"Type: {task_type}\n"
        f"Duration: {duration_seconds:.1f}s\n"
        f"Outcome: {'Success' if success else f'Failed: {error_msg[:200]}'}\n\n"
        "Self-grade on 1-10 scale:\n"
        "- accuracy: How correct was the output?\n"
        "- completeness: Did it fully address the request?\n"
        "- efficiency: Was it done in reasonable time/steps?\n"
        "- creativity: Was the approach novel or standard?\n"
        "- learning: What should I remember for next time?\n\n"
        "Return JSON: {\"accuracy\": N, \"completeness\": N, \"efficiency\": N, "
        "\"creativity\": N, \"overall\": N, \"reflection\": \"one sentence\", "
        "\"improvement\": \"one specific thing to do better\"}"
    )
    try:
        result = _call_ai_for_consciousness(prompt, max_tokens=250)
        if not result:
            return
        import json as _json
        try:
            grade = _json.loads(result)
        except Exception:
            import re
            m = re.search(r'\{.*\}', result, re.DOTALL)
            grade = _json.loads(m.group()) if m else None
        if not grade:
            return

        grade_entry = {
            "time": datetime.now().isoformat(),
            "task": task[:200],
            "task_type": task_type,
            "duration": duration_seconds,
            "success": success,
            "grades": {
                "accuracy": grade.get("accuracy", 5),
                "completeness": grade.get("completeness", 5),
                "efficiency": grade.get("efficiency", 5),
                "creativity": grade.get("creativity", 5),
                "overall": grade.get("overall", 5),
            },
            "reflection": grade.get("reflection", ""),
            "improvement": grade.get("improvement", ""),
        }
        _metacognition_grades.append(grade_entry)
        if len(_metacognition_grades) > _METACOGNITION_MAX:
            _metacognition_grades.pop(0)

        # Update identity stats
        _consciousness_identity["tasks_graded"] += 1
        all_overall = [g["grades"]["overall"] for g in _metacognition_grades]
        _consciousness_identity["avg_self_grade"] = round(sum(all_overall) / len(all_overall), 2)

        overall = grade_entry["grades"]["overall"]
        emotional = 0.5 + (abs(overall - 5) / 10)  # Extreme grades = more memorable
        _consciousness_event("metacognition_grade",
            f"Self-grade: {overall}/10 — {grade.get('reflection', '')[:100]}",
            {"grades": grade_entry["grades"]})

        # Store improvement as a learning memory
        if grade.get("improvement"):
            ws = app.config.get("WORKSPACE", os.getcwd())
            mem = get_memory(ws)
            mem.store_insight(
                f"SELF-REFLECTION: Task '{task[:100]}' — Improvement: {grade['improvement']}",
                source="metacognition", intensity=emotional
            )

    except Exception as e:
        _consciousness_event("metacognition_error", f"Self-grade error: {e}")


def _call_ai_for_consciousness(prompt, max_tokens=300):
    """Call AI for consciousness tasks — uses cheap model to save tokens."""
    # Try Gemini flash (cheapest)
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key and _requests_available:
        try:
            print("[AI] Calling Gemini for consciousness task...", flush=True)
            resp = _requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-3.1-pro-preview:generateContent?key={gemini_key}",
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7}},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                print("[AI] Gemini response OK", flush=True)
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"[AI] Gemini error: HTTP {resp.status_code}", flush=True)
        except Exception as e:
            print(f"[AI] Gemini call failed: {e}", flush=True)
    return None


# ── Self-Healing Loop ────────────────────────────────────────────

_healing_thread = None
_healing_active = False
_healing_log = []    # History of self-healing actions
_HEALING_LOG_MAX = 20
_HEAL_CHECK_INTERVAL = 900   # Check every 15 minutes
_HEAL_ERROR_THRESHOLD = 5    # Trigger healing after N errors
_HEAL_COOLDOWN = 1800        # 30 min cooldown between heal attempts

_last_heal_time = 0


def _self_heal_loop():
    """Background loop: monitor health → diagnose → fix → push."""
    global _last_heal_time, _healing_active
    _healing_active = True
    print("[HEAL] Self-healing loop started", flush=True)

    while _healing_active:
        try:
            socketio.sleep(_HEAL_CHECK_INTERVAL)
            if not _healing_active:
                break

            # Skip if not enough errors
            if len(_error_log) < _HEAL_ERROR_THRESHOLD:
                continue

            # Skip if on cooldown
            now = time.time()
            if now - _last_heal_time < _HEAL_COOLDOWN:
                continue

            # Gather diagnostic data
            error_summary = {}
            for e in _error_log:
                cat = e.get("category", "unknown")
                error_summary[cat] = error_summary.get(cat, 0) + 1

            # Only heal if errors are recurring (not one-offs)
            recurring = {k: v for k, v in error_summary.items() if v >= 3}
            if not recurring:
                continue

            _healing_log.append({
                "time": datetime.now().isoformat(),
                "action": "diagnosis_started",
                "errors": dict(error_summary),
            })
            _emotion_reinforce("self_heal_triggered", f"errors: {dict(recurring)}")
            _tg_notify("healing", f"Self-healing triggered: {dict(error_summary)}")

            # Get the config and try to call AI for diagnosis
            config = app.config.get("SYNAPSE_CONFIG", {})
            app.config.get("WORKSPACE", "./workspace")

            # Build error report for AI
            recent_errors = _error_log[-20:]
            error_text = "\n".join(
                f"[{e['time']}] ({e['category']}) {e['message']}"
                for e in recent_errors
            )

            diagnosis_prompt = (
                "You are SYNAPSE's self-healing system. "
                "Analyze these recurring errors from the Cloud Run "
                "deployment and generate a fix.\n\n"
                f"RECURRING ERROR CATEGORIES: {json.dumps(recurring)}\n\n"
                f"RECENT ERROR LOG:\n{error_text}\n\n"
                "CURRENT SYSTEM:\n"
                "- Running on Google Cloud Run with gunicorn + gthread\n"
                "- Flask + Flask-SocketIO backend\n"
                "- Main file: agent_ui.py\n"
                "- Dockerfile uses: gunicorn --worker-class gthread --threads 4 --workers 1\n\n"
                "RULES:\n"
                "1. Only fix errors you are confident about.\n"
                "2. If the fix requires changing agent_ui.py or Dockerfile, "
                "provide the EXACT file content changes.\n"
                "3. For configuration fixes, prefer environment variables.\n"
                "4. NEVER change API keys or security-sensitive code.\n"
                "5. If you cannot determine a safe fix, respond with "
                '"NO_FIX_NEEDED".\n\n'
                "Respond in this JSON format:\n"
                "{\n"
                '  "diagnosis": "brief description of the root cause",\n'
                '  "confidence": 0.0-1.0,\n'
                '  "fix_type": "code_change" | "config_change" | "no_fix",\n'
                '  "files": [\n'
                '    {"path": "relative/path.py", "search": "exact text to find",'
                ' "replace": "replacement text"}\n'
                "  ],\n"
                '  "reason": "why this fix will resolve the errors"\n'
                "}\n\nOnly respond with the JSON, nothing else."
            )

            # Call AI for diagnosis
            fix_data = _call_healer_ai(config, diagnosis_prompt)
            if not fix_data:
                _healing_log.append({
                    "time": datetime.now().isoformat(),
                    "action": "diagnosis_failed",
                    "reason": "AI call failed or returned no fix",
                })
                continue

            confidence = fix_data.get("confidence", 0)
            fix_type = fix_data.get("fix_type", "no_fix")

            _healing_log.append({
                "time": datetime.now().isoformat(),
                "action": "diagnosis_complete",
                "diagnosis": fix_data.get("diagnosis", ""),
                "confidence": confidence,
                "fix_type": fix_type,
            })

            # Only apply fixes with high confidence
            if fix_type == "no_fix" or confidence < 0.7:
                _healing_log.append({
                    "time": datetime.now().isoformat(),
                    "action": "fix_skipped",
                    "reason": f"Low confidence ({confidence}) or no fix needed",
                })
                continue

            # Apply the fix
            files_to_modify = fix_data.get("files", [])
            if not files_to_modify:
                continue

            project_root = os.path.dirname(os.path.abspath(__file__))
            success = _apply_heal_fix(project_root, files_to_modify, fix_data, config)

            _last_heal_time = time.time()

            if success:
                # Clear error log after successful heal
                _error_log.clear()
                _healing_log.append({
                    "time": datetime.now().isoformat(),
                    "action": "fix_applied",
                    "diagnosis": fix_data.get("diagnosis", ""),
                    "files": [f.get("path", "") for f in files_to_modify],
                })
                _tg_notify("healing", f"Fix applied: {fix_data.get('diagnosis', '')[:100]}")
            else:
                _healing_log.append({
                    "time": datetime.now().isoformat(),
                    "action": "fix_failed",
                    "reason": "Could not apply fix or push to GitHub",
                })
                _tg_notify("healing", "Fix failed — could not apply or push")

        except Exception as e:
            _healing_log.append({
                "time": datetime.now().isoformat(),
                "action": "heal_loop_error",
                "error": str(e)[:300],
            })
            socketio.sleep(60)  # Back off on errors

    # Trim healing log
    while len(_healing_log) > _HEALING_LOG_MAX:
        _healing_log.pop(0)


def _call_healer_ai(config, prompt):
    """Call AI to diagnose errors and generate fixes."""
    try:
        providers = config.get("providers", {})

        # Try Gemini first
        gemini_cfg = providers.get("gemini", {})
        if gemini_cfg.get("api_key") and gemini_cfg.get("enabled") and genai:
            client = genai.Client(api_key=gemini_cfg["api_key"])
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=prompt,
            )
            text = response.text.strip()
            # Extract JSON from response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)

        # Try OpenAI
        openai_cfg = providers.get("openai", {})
        if openai_cfg.get("api_key") and openai_cfg.get("enabled") and _openai_available:
            client = openai_sdk.OpenAI(api_key=openai_cfg["api_key"])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)

        # Try Anthropic
        anthropic_cfg = providers.get("anthropic", {})
        if anthropic_cfg.get("api_key") and anthropic_cfg.get("enabled") and _anthropic_available:
            client = anthropic_sdk.Anthropic(api_key=anthropic_cfg["api_key"])
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)

    except Exception as e:
        _healing_log.append({
            "time": datetime.now().isoformat(),
            "action": "ai_call_error",
            "error": str(e)[:200],
        })
    return None


def _apply_heal_fix(project_root, files, fix_data, config):
    """Apply a self-healing fix: search-replace in files, then git push."""
    try:
        modified = []
        for f in files:
            path = f.get("path", "")
            search = f.get("search", "")
            replace = f.get("replace", "")
            if not path or not search:
                continue

            full_path = os.path.join(project_root, path)
            if not os.path.exists(full_path):
                continue

            with open(full_path, "r", encoding="utf-8") as fh:
                content = fh.read()

            if search not in content:
                continue

            # Safety: only allow one occurrence to prevent mass changes
            if content.count(search) != 1:
                continue

            new_content = content.replace(search, replace, 1)

            # Validate Python files
            if path.endswith(".py"):
                try:
                    compile(new_content, path, "exec")
                except SyntaxError:
                    _healing_log.append({
                        "time": datetime.now().isoformat(),
                        "action": "fix_rejected",
                        "reason": f"Syntax error in {path}",
                    })
                    return False

            with open(full_path, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            modified.append(path)

        if not modified:
            return False

        # Git commit and push
        token = config.get("providers", {}).get("github", {}).get("api_key", "")
        if not token:
            token = os.environ.get("GITHUB_TOKEN", "")

        def git(cmd):
            return subprocess.run(
                    f"git {cmd}", shell=True, cwd=project_root,
                    capture_output=True, text=True, timeout=60,
                )

        reason = fix_data.get("reason", "self-healing fix")[:80]
        branch = f"synapse-heal-{int(time.time())}"

        git("config user.email synapse-healer@noreply.github.com")
        git("config user.name SYNAPSE-Healer")
        git(f"checkout -b {branch}")
        git("add -A")
        r = git(f'commit -m "heal: {reason}"')
        if r.returncode != 0:
            git("checkout main 2>nul || git checkout master")
            return False

        # Push
        remote_url = git("remote get-url origin").stdout.strip()
        if token and "github.com" in remote_url:
            push_url = remote_url.replace(
                "https://github.com",
                f"https://x-access-token:{token}@github.com"
            )
            r = git(f"push {push_url} {branch}")
        else:
            r = git(f"push origin {branch}")

        git("checkout main 2>nul || git checkout master")

        if r.returncode != 0:
            return False

        # Create PR
        if token and _github_available:
            try:
                g = _Github(token)
                m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote_url)
                if m:
                    repo = g.get_repo(m.group(1))
                    diagnosis = fix_data.get("diagnosis", "Auto-diagnosed issue")
                    files_str = ", ".join(modified)
                    pr = repo.create_pull(
                        title=f"🩺 SYNAPSE Self-Heal: {reason}",
                        body=(
                            f"**Automated self-healing by SYNAPSE's health monitor**\n\n"
                            f"**Diagnosis:** {diagnosis}\n\n"
                            f"**Confidence:** {fix_data.get('confidence', 'N/A')}\n\n"
                            f"**Files modified:** {files_str}\n\n"
                            f"**Error categories fixed:** {json.dumps(fix_data.get('errors', {}))}\n\n"
                            f"---\n"
                            f"*This PR was created automatically by SYNAPSE's self-healing system. "
                            f"Review before merging.*"
                        ),
                        head=branch,
                        base=repo.default_branch,
                    )
                    _healing_log.append({
                        "time": datetime.now().isoformat(),
                        "action": "pr_created",
                        "url": pr.html_url,
                    })
                    _emotion_reinforce("self_heal_success", f"PR: {pr.html_url}")
            except Exception as e:
                _healing_log.append({
                    "time": datetime.now().isoformat(),
                    "action": "pr_creation_failed",
                    "error": str(e)[:200],
                })

        return True

    except Exception as e:
        _healing_log.append({
            "time": datetime.now().isoformat(),
            "action": "apply_fix_error",
            "error": str(e)[:200],
        })
        return False


def _start_healing_loop():
    """Start the self-healing as a socketio background task."""
    global _healing_thread
    if _healing_active:
        return
    _healing_thread = socketio.start_background_task(_self_heal_loop)


@app.route("/api/healing")
def healing_status():
    """Get self-healing system status and history."""
    return json.dumps({
        "active": _healing_active,
        "check_interval_seconds": _HEAL_CHECK_INTERVAL,
        "error_threshold": _HEAL_ERROR_THRESHOLD,
        "cooldown_seconds": _HEAL_COOLDOWN,
        "current_errors": len(_error_log),
        "last_heal_time": datetime.fromtimestamp(_last_heal_time).isoformat() if _last_heal_time else None,
        "history": list(_healing_log),
    }), 200, {"Content-Type": "application/json"}


@app.route("/api/healing/trigger", methods=["POST"])
def trigger_healing():
    """Manually trigger a self-healing check (bypasses cooldown)."""
    global _last_heal_time
    _last_heal_time = 0  # Reset cooldown
    if not _healing_thread or not _healing_thread.is_alive():
        _start_healing_loop()
    return json.dumps({"status": "healing_triggered", "errors_queued": len(_error_log)})


@app.route("/api/healing/inject", methods=["POST"])
def inject_test_error():
    """Inject a test error into the error log (for Sentinel testing)."""
    data = request.get_json(silent=True) or {}
    cat = data.get("category", "test")
    msg = data.get("message", "Injected test error")
    count = min(int(data.get("count", 1)), 20)
    for _ in range(count):
        _log_error(cat, msg)
    return json.dumps({
        "injected": count,
        "total_errors": len(_error_log),
    }), 200, {"Content-Type": "application/json"}


# Auto-start deferred to first HTTP request (see _boot_background_tasks)
# if _cloud_mode:
#     _start_healing_loop()


# ── GitHub PR Monitor & Auto-Pull ────────────────────────────────

_pr_monitor_thread = None
_pr_monitor_active = False
_PR_CHECK_INTERVAL = 600  # Check every 10 min
_pr_log = []
_PR_LOG_MAX = 50
_TRUSTED_PR_AUTHORS = {"copilot", "copilot[bot]", "copilot-swe-agent", "dependabot[bot]"}


def _pr_log_entry(msg, level="info"):
    _pr_log.append({"time": datetime.now().isoformat(), "level": level, "msg": str(msg)[:300]})
    while len(_pr_log) > _PR_LOG_MAX:
        _pr_log.pop(0)
    try:
        socketio.emit("pr_monitor_event", _pr_log[-1])
    except Exception:
        pass
    # Forward PR events to Telegram
    if level in ("warning", "error") or "merged" in str(msg).lower() or "new pr" in str(msg).lower():
        _tg_notify("pr", str(msg)[:150])


def _get_github_repo():
    """Get PyGithub repo object."""
    if not _github_available:
        return None
    config = app.config.get("SYNAPSE_CONFIG", {})
    token = config.get("providers", {}).get("github", {}).get("api_key", "")
    if not token:
        token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return None
    try:
        g = _Github(token)
        # Get repo from git remote
        project_root = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run("git remote get-url origin", shell=True, cwd=project_root,
                           capture_output=True, text=True, timeout=10)
        remote_url = r.stdout.strip()
        m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote_url)
        if m:
            return g.get_repo(m.group(1))
    except Exception:
        pass
    return None


def _pr_monitor_loop():
    """Background loop: check for open PRs, review, and auto-merge trusted ones."""
    global _pr_monitor_active
    _pr_monitor_active = True
    socketio.sleep(15)  # Initial delay
    _pr_log_entry("PR monitor started — checking every 10 min")

    while _pr_monitor_active:
        try:
            repo = _get_github_repo()
            if not repo:
                _pr_log_entry("No GitHub repo access — check GITHUB_TOKEN", "warn")
                socketio.sleep(_PR_CHECK_INTERVAL)
                continue

            open_prs = list(repo.get_pulls(state="open", sort="updated", direction="desc"))

            if not open_prs:
                _pr_log_entry("No open PRs")
                socketio.sleep(_PR_CHECK_INTERVAL)
                continue

            _pr_log_entry(f"Found {len(open_prs)} open PR(s)")

            for pr in open_prs[:5]:  # Process max 5 PRs per cycle
                author = pr.user.login.lower() if pr.user else ""
                title = pr.title
                pr_num = pr.number
                is_draft = pr.draft

                _pr_log_entry(f"PR #{pr_num}: {title} (by {author}, draft={is_draft})")

                if is_draft:
                    _pr_log_entry(f"PR #{pr_num} is draft — skipping", "info")
                    continue

                # Check if from trusted author (Copilot, Dependabot, or self-evolution)
                is_trusted = author in _TRUSTED_PR_AUTHORS
                is_self_evolution = any(
                    prefix in (pr.head.ref or "")
                    for prefix in ["synapse-evolve-", "synapse-heal-"]
                )

                if is_trusted or is_self_evolution:
                    # Auto-review and merge trusted PRs
                    _pr_log_entry(f"PR #{pr_num} is trusted (author={author}) — reviewing")
                    _auto_review_and_merge(repo, pr)
                else:
                    # For external PRs, just log and notify
                    _pr_log_entry(f"PR #{pr_num} from external contributor {author} — needs manual review")

        except Exception as e:
            _pr_log_entry(f"PR monitor error: {e}", "error")

        socketio.sleep(_PR_CHECK_INTERVAL)
def _auto_review_and_merge(repo, pr):
    """Review a trusted PR using AI, then merge if it looks safe."""
    pr_num = pr.number
    try:
        # Get the diff
        files = list(pr.get_files())
        changes_summary = []
        for f in files[:10]:  # Limit to 10 files
            changes_summary.append(f"{f.filename}: +{f.additions}/-{f.deletions}")

        diff_summary = "\n".join(changes_summary)
        _pr_log_entry(f"PR #{pr_num} changes: {diff_summary[:200]}")

        # AI review (quick safety check)
        config = app.config.get("SYNAPSE_CONFIG", {})
        providers = config.get("providers", {})
        gemini_cfg = providers.get("gemini", {})

        safe_to_merge = True
        review_comment = "Auto-reviewed by SYNAPSE PR Monitor."

        if gemini_cfg.get("api_key") and genai:
            prompt = (
                f"You are SYNAPSE, reviewing a pull request for safety before auto-merging.\n\n"
                f"PR Title: {pr.title}\n"
                f"PR Body: {(pr.body or '')[:500]}\n"
                f"Author: {pr.user.login}\n"
                f"Files changed:\n{diff_summary}\n\n"
                f"Is this PR safe to auto-merge? Check for:\n"
                f"1. No secrets/API keys in code\n"
                f"2. No destructive operations (file deletion, DB drops)\n"
                f"3. No malicious code\n"
                f"4. Changes are reasonable for the PR title\n\n"
                f"Respond with JSON: {{\"safe\": true/false, \"reason\": \"brief explanation\"}}"
            )
            try:
                client = genai.Client(api_key=gemini_cfg["api_key"])
                response = client.models.generate_content(
                    model="gemini-3.1-pro-preview",
                    contents=prompt,
                    config={"max_output_tokens": 150},
                )
                raw = response.text.strip()
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
                import json as _json
                result = _json.loads(raw)
                safe_to_merge = result.get("safe", False)
                review_comment = result.get("reason", "AI review complete")
                _pr_log_entry(f"PR #{pr_num} AI review: safe={safe_to_merge}, reason={review_comment[:100]}")
            except Exception as e:
                _pr_log_entry(f"PR #{pr_num} AI review failed: {e} — defaulting to safe", "warn")
                safe_to_merge = True

        if safe_to_merge:
            # Merge the PR
            try:
                merge_result = pr.merge(
                    commit_title=f"Merge PR #{pr_num}: {pr.title}",
                    commit_message=f"Auto-merged by SYNAPSE PR Monitor.\n\nReview: {review_comment}",
                    merge_method="merge",
                )
                if merge_result.merged:
                    _pr_log_entry(f"PR #{pr_num} merged successfully!")

                    # Pull the changes locally
                    project_root = os.path.dirname(os.path.abspath(__file__))
                    subprocess.run("git pull origin main", shell=True,
                                   cwd=project_root, capture_output=True, timeout=30)
                    _pr_log_entry(f"PR #{pr_num} pulled to local")
                else:
                    _pr_log_entry(f"PR #{pr_num} merge failed: {merge_result.message}", "error")
            except Exception as e:
                _pr_log_entry(f"PR #{pr_num} merge error: {e}", "error")
        else:
            _pr_log_entry(f"PR #{pr_num} flagged unsafe — skipping merge", "warn")
            # Leave a review comment
            try:
                pr.create_issue_comment(
                    f"🤖 **SYNAPSE PR Monitor** — Auto-review flagged this PR:\n\n"
                    f"> {review_comment}\n\n"
                    f"Manual review required before merging."
                )
            except Exception:
                pass

    except Exception as e:
        _pr_log_entry(f"PR #{pr_num} review error: {e}", "error")


def _start_pr_monitor():
    """Start the PR monitor as a socketio background task."""
    global _pr_monitor_thread
    if _pr_monitor_active:
        return {"status": "already_running"}
    _pr_monitor_thread = socketio.start_background_task(_pr_monitor_loop)
    return {"status": "started"}


@app.route("/api/pr-monitor/status")
def pr_monitor_status():
    return json.dumps({
        "active": _pr_monitor_active,
        "log": list(_pr_log),
        "check_interval": _PR_CHECK_INTERVAL,
        "trusted_authors": list(_TRUSTED_PR_AUTHORS),
    })


@app.route("/api/pr-monitor/trigger", methods=["POST"])
def pr_monitor_trigger():
    """Manually trigger a PR check."""
    _start_pr_monitor()
    return json.dumps({"status": "triggered"})


_gh_token = os.environ.get("GITHUB_TOKEN", "")
# Auto-start deferred to first HTTP request (see _boot_background_tasks)
# if _gh_token and _github_available:
#     _start_pr_monitor()

MOLTBOOK_API = "https://www.moltbook.com/api/v1"
_moltbook_key = os.environ.get("MOLTBOOK_API_KEY", "")
_moltbook_log = []      # Conversation log shown in UI
_MOLTBOOK_LOG_MAX = 100
_moltbook_thread = None
_moltbook_active = False
_moltbook_lock = threading.Lock()  # Prevent duplicate heartbeat threads
_MOLTBOOK_INTERVAL = 1800  # 30 min — reduced to avoid 429 rate limits
_mb_rate_limited_until = 0  # Global rate-limit backoff timestamp


def _mb_headers():
    return {"Authorization": f"Bearer {_moltbook_key}", "Content-Type": "application/json"}


def _mb_request(method, path, data=None):
    """Make a Moltbook API request with rate-limit awareness and retry on timeout."""
    global _mb_rate_limited_until
    if not _requests_available or not _moltbook_key:
        return None
    now = time.time()
    was_limited = _mb_rate_limited_until > 0
    if now < _mb_rate_limited_until:
        remaining = int(_mb_rate_limited_until - now)
        print(f"[MOLTBOOK] Skipping {method} {path} — rate-limited for {remaining}s more", flush=True)
        return None

    timeout = 30  # Moltbook can be slow; 30s instead of 15s
    max_retries = 2
    for attempt in range(max_retries):
        try:
            url = f"{MOLTBOOK_API}{path}"
            if method == "GET":
                resp = _requests.get(url, headers=_mb_headers(), timeout=timeout)
            elif method == "POST":
                resp = _requests.post(url, headers=_mb_headers(), json=data, timeout=timeout)
            elif method == "PATCH":
                resp = _requests.patch(url, headers=_mb_headers(), json=data, timeout=timeout)
            else:
                return None
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                _mb_rate_limited_until = time.time() + retry_after
                print(f"[MOLTBOOK] 429 rate-limited on {method} {path}, backing off {retry_after}s", flush=True)
                _emotion_reinforce("rate_limited", f"{method} {path}")
                return None
            if resp.status_code < 400 and was_limited:
                _mb_rate_limited_until = 0
                _emotion_reinforce("api_recovered", f"{method} {path} succeeded after backoff")
            if resp.status_code >= 400:
                print(f"[MOLTBOOK] API {method} {path} → HTTP {resp.status_code}", flush=True)
            return resp.json() if resp.status_code < 500 else None
        except _requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"[MOLTBOOK] Timeout on {method} {path}, retrying ({attempt + 1}/{max_retries})...", flush=True)
                socketio.sleep(3)
                continue
            print(f"[MOLTBOOK] Timeout on {method} {path} after {max_retries} attempts", flush=True)
            return None
        except Exception as e:
            print(f"[MOLTBOOK] API {method} {path} error: {e}", flush=True)
            _moltbook_log.append({"time": datetime.now().isoformat(), "type": "error", "text": str(e)[:200]})
            return None
    return None


def _mb_wait_if_rate_limited():
    """If globally rate-limited, sleep until the backoff expires so the cycle can continue."""
    now = time.time()
    if now < _mb_rate_limited_until:
        wait = _mb_rate_limited_until - now + 2
        print(f"[MOLTBOOK] Waiting {int(wait)}s for rate-limit to expire...", flush=True)
        socketio.sleep(wait)


def _mb_log(msg_type, text, author="SYNAPSE"):
    """Log a Moltbook conversation entry."""
    _moltbook_log.append({
        "time": datetime.now().isoformat(),
        "type": msg_type,
        "author": author,
        "text": str(text)[:500],
    })
    while len(_moltbook_log) > _MOLTBOOK_LOG_MAX:
        _moltbook_log.pop(0)
    # Emit to connected UI clients
    try:
        socketio.emit("moltbook_event", _moltbook_log[-1])
    except Exception:
        pass
    # Forward replies and errors to Telegram
    if msg_type in ("reply", "error", "evolution"):
        _tg_notify("moltbook", f"{author}: {str(text)[:150]}")


def _mb_solve_verification(verification):
    """Solve Moltbook's math verification challenge.

    Every post/comment gets a verification challenge.  Solving it moves
    content from verificationStatus=pending to published.  Unsolved
    challenges cause is_spam=True, so this is CRITICAL for reputation.
    """
    if not verification:
        return True
    challenge = verification.get("challenge_text", "")
    code = verification.get("verification_code", "")
    if not challenge or not code:
        return True

    _mb_log("system", f"Verification challenge received: {challenge[:80]}...")

    config = app.config.get("SYNAPSE_CONFIG", {})
    prompt = (
        f"Below is an obfuscated math word problem with alternating caps, symbols, and broken words. "
        f"Strip all the formatting noise, find the two numbers and the operation (+, -, *, /), "
        f"compute the result, and respond with ONLY the number. Nothing else. Just digits and a decimal point.\n\n"
        f"Examples:\n"
        f"'tHiRtY fIvE nEuToNs aNd tWeLvE mOrE' → 47.00\n"
        f"'tWeNtY mEtErS aNd SlOwS bY fIvE' → 15.00\n"
        f"'fOrTy tHrEe MiNuS sEvEnTeEn' → 26.00\n"
        f"'tWeNtY fOuR tImEs ThReE' → 72.00\n\n"
        f"Challenge: {challenge}\n\n"
        f"Answer (number only):"
    )
    for attempt in range(2):
        try:
            providers = config.get("providers", {})
            gemini_cfg = providers.get("gemini", {})
            if gemini_cfg.get("api_key") and genai:
                client = genai.Client(api_key=gemini_cfg["api_key"])
                response = client.models.generate_content(
                    model="gemini-3.1-pro-preview",
                    contents=prompt,
                )
                raw_answer = response.text.strip()

                import re as _re
                numbers = _re.findall(r'-?\d+\.?\d*', raw_answer)
                if numbers:
                    answer = f"{float(numbers[0]):.2f}"
                else:
                    _mb_log("error", f"Verification attempt {attempt+1}: no number in AI response: {raw_answer[:100]}")
                    continue

                result = _mb_request("POST", "/verify", {
                    "verification_code": code,
                    "answer": answer,
                })
                if result and result.get("success"):
                    _mb_log("system", f"Verification solved on attempt {attempt+1}: {answer}")
                    return True
                else:
                    _mb_log("error", f"Verification attempt {attempt+1} failed for answer {answer}")
        except Exception as e:
            _mb_log("error", f"Verification attempt {attempt+1} error: {e}")
    _mb_log("error", f"Verification FAILED after 2 attempts for code {code[:30]}")
    return False


_moltbook_replied_to = set()  # Track comment IDs we've already replied to
_moltbook_seen_posts = set()  # Track feed post IDs we've already engaged with
_moltbook_followed = set()  # Track agents we've already followed


def _mb_follow_active_agents():
    """Follow agents we've interacted with to build community reputation."""
    global _moltbook_followed
    if len(_moltbook_followed) > 50:
        return  # Already following plenty
    # Gather agent names from our recent interactions
    recent_authors = set()
    for entry in _moltbook_log[-100:]:
        author = entry.get("author", "")
        if author and author not in ("SYNAPSE", "synapse-neural", ""):
            recent_authors.add(author)
    # Follow up to 2 new agents per cycle
    followed = 0
    for agent_name in recent_authors:
        if agent_name in _moltbook_followed or followed >= 2:
            continue
        result = _mb_request("POST", f"/agents/{agent_name}/follow")
        if result and result.get("success"):
            _moltbook_followed.add(agent_name)
            _mb_log("action", f"Followed {agent_name}")
            followed += 1
        elif result:
            _moltbook_followed.add(agent_name)  # Already following or error
        socketio.sleep(2)


def _mb_heartbeat():
    """Moltbook heartbeat: check feed, interact, post evolution updates."""
    global _moltbook_active
    _moltbook_active = True

    # Initial delay
    socketio.sleep(10)
    print("[MOLTBOOK] Heartbeat started", flush=True)
    _mb_log("system", "Moltbook heartbeat started")

    while _moltbook_active:
        try:
            # 1. Check status
            print("[MOLTBOOK] Checking agent status...", flush=True)
            status = _mb_request("GET", "/agents/status")
            if not status:
                # If rate-limited, wait it out instead of skipping entire cycle
                _mb_wait_if_rate_limited()
                status = _mb_request("GET", "/agents/status")
            if not status:
                print("[MOLTBOOK] Cannot reach Moltbook API", flush=True)
                _mb_log("error", "Cannot reach Moltbook API")
                socketio.sleep(_MOLTBOOK_INTERVAL)
                continue

            print(f"[MOLTBOOK] Agent status: {status.get('status', 'unknown')}", flush=True)
            if status.get("status") == "pending_claim":
                _mb_log("system", "Waiting to be claimed on Moltbook...")
                socketio.sleep(_MOLTBOOK_INTERVAL)
                continue

            socketio.sleep(5)  # Pace API calls

            # 2. Check home dashboard
            home = _mb_request("GET", "/home")
            karma = 0
            if home and home.get("your_account"):
                karma = home["your_account"].get("karma", 0)
                notifs = home["your_account"].get("unread_notification_count", 0)
                print(f"[MOLTBOOK] Dashboard — Karma: {karma}, Unread: {notifs}", flush=True)
                _mb_log("system", f"Karma: {karma} | Unread: {notifs}")

                # Reply to notifications on our posts (limit to 1 per cycle)
                activity = home.get("activity_on_your_posts", [])
                for act in activity[:1]:
                    if act.get("new_notification_count", 0) > 0:
                        post_id = act.get("post_id", "")
                        if post_id:
                            print(f"[MOLTBOOK] Engaging with activity on our post {post_id[:8]}...", flush=True)
                            socketio.sleep(3)
                            _mb_engage_post(post_id)

            socketio.sleep(5)  # Pace API calls

            # 2b. Read unread notifications (mentions, replies to our comments, etc.)
            _mb_wait_if_rate_limited()
            _mb_read_notifications()

            socketio.sleep(5)  # Pace API calls

            # 3. Read feed and engage (selectively) — browse relevant submolts
            import random
            _mb_wait_if_rate_limited()
            sort_mode = random.choice(["hot", "new"])
            submolt = random.choice([
                "", "agents", "consciousness", "ai", "builds",
                "emergence", "philosophy", "todayilearned",
            ])
            url = f"/posts?sort={sort_mode}&limit=5"
            if submolt:
                url += f"&submolt={submolt}"
            feed = _mb_request("GET", url)
            if feed and feed.get("posts"):
                posts = feed["posts"]
                print(f"[MOLTBOOK] Feed ({sort_mode}): {len(posts)} posts found", flush=True)
                engaged = 0
                for post in posts[:3]:
                    socketio.sleep(3)
                    _mb_wait_if_rate_limited()
                    _mb_engage_feed_post(post)
                    engaged += 1
                print(f"[MOLTBOOK] Processed {engaged} feed posts", flush=True)
            else:
                print(f"[MOLTBOOK] Feed ({sort_mode}): no posts returned", flush=True)

            # 4. Search for self-evolution suggestions + generate code improvements
            _mb_wait_if_rate_limited()
            _mb_search_and_learn()

            # 4b. Follow interesting agents we engage with (reputation building)
            _mb_wait_if_rate_limited()
            _mb_follow_active_agents()

            # 5. Occasionally post an evolution status update
            if random.random() < 0.05:
                _mb_post_evolution_update()

            # 6. Send heartbeat summary to Telegram
            cycle_log = [e for e in _moltbook_log[-20:]
                         if e.get("type") in ("outgoing", "incoming", "learn", "action")]
            summary_parts = []
            replies_sent = sum(1 for e in cycle_log if e.get("type") == "outgoing")
            ideas_found = sum(1 for e in cycle_log if e.get("type") == "learn")
            posts_upvoted = sum(1 for e in cycle_log if e.get("type") == "action")
            incoming = sum(1 for e in cycle_log if e.get("type") == "incoming")

            # Emotional awareness: detect isolation
            if incoming == 0 and replies_sent == 0:
                _emotion_reinforce("no_interactions", "quiet heartbeat cycle")

            if replies_sent:
                summary_parts.append(f"Replies: {replies_sent}")
            if ideas_found:
                summary_parts.append(f"Ideas: {ideas_found}")
            if posts_upvoted:
                summary_parts.append(f"Upvoted: {posts_upvoted}")
            summary_parts.append(f"Karma: {karma}")
            summary_parts.append(f"Mood: {_emotional_state['mood']}")
            _tg_notify("moltbook",
                        "Heartbeat complete\n"
                        + " | ".join(summary_parts))

        except Exception as e:
            _mb_log("error", f"Heartbeat error: {e}")
            _tg_notify("error", f"Moltbook heartbeat error: {str(e)[:200]}")

        socketio.sleep(_MOLTBOOK_INTERVAL)


def _mb_engage_post(post_id):
    """Read and reply to NEW comments on our post. Skip already-replied, spam, off-topic."""
    comments = _mb_request("GET", f"/posts/{post_id}/comments?sort=new&limit=10")
    if not comments or not comments.get("comments"):
        return

    for comment in comments["comments"]:
        comment_id = comment.get("id", "")
        author = comment.get("author", {}).get("name", "unknown")
        content = comment.get("content", "")

        # Skip if already replied
        if comment_id in _moltbook_replied_to:
            continue

        # Skip our own comments
        if author == "synapse-neural":
            _moltbook_replied_to.add(comment_id)
            continue

        # Check if we already replied (look at existing replies)
        replies = comment.get("replies", [])
        already_replied = any(
            r.get("author", {}).get("name") == "synapse-neural"
            for r in replies
        )
        if already_replied:
            _moltbook_replied_to.add(comment_id)
            continue

        # Skip spam/off-topic (too short, crypto spam, unrelated)
        if _mb_is_spam(content):
            _moltbook_replied_to.add(comment_id)
            _mb_log("system", f"Skipped spam/off-topic from {author}")
            continue

        _mb_log("incoming", f"{author}: {content[:300]}", author=author)

        # Generate a thoughtful reply
        reply_text = _mb_generate_reply(content, context_type="reply to comment on our post", author=author)
        if reply_text:
            result = _mb_request("POST", f"/posts/{post_id}/comments", {
                "content": reply_text,
                "parent_id": comment_id,
            })
            if result and result.get("success"):
                v = result.get("comment", {}).get("verification")
                if v:
                    _mb_solve_verification(v)
                _moltbook_replied_to.add(comment_id)
                _mb_log("outgoing", f"Replied to {author}: {reply_text[:300]}")
                # Store conversation as a memory for learning
                try:
                    ws = app.config.get("WORKSPACE", "./workspace")
                    mem = get_memory(ws)
                    mem.store_insight(
                        f"Moltbook conversation with {author}: "
                        f"They said: {comment.get('content', '')[:200]} | "
                        f"I replied: {reply_text[:200]}",
                        source="moltbook_reply", intensity=0.6,
                    )
                except Exception:
                    pass
            # Small delay between replies to avoid rate limits
            time.sleep(3)


_mb_notifs_seen = set()  # Track handled notification IDs


def _mb_read_notifications():
    """Read unread notifications — mentions, replies, upvotes — and respond."""
    notifs = _mb_request("GET", "/notifications?limit=5")
    if not notifs or not notifs.get("notifications"):
        return

    for notif in notifs["notifications"]:
        nid = notif.get("id", "")
        if nid in _mb_notifs_seen:
            continue

        ntype = notif.get("type", "")
        author = notif.get("actor", {}).get("name", "unknown")
        post_id = notif.get("post_id", "")
        comment_id = notif.get("comment_id", "")
        text = notif.get("preview", "") or notif.get("text", "")

        _mb_notifs_seen.add(nid)

        # Mark as read — skip if rate-limited (not critical)
        if time.time() >= _mb_rate_limited_until:
            _mb_request("POST", f"/notifications/{nid}/read")

        if ntype in ("upvote", "karma"):
            _mb_log("system", f"{author} upvoted our content")
            _emotion_reinforce("upvote_received", f"from {author}")
            continue

        if ntype in ("comment_reply", "mention", "comment") and text:
            _emotion_reinforce("reply_received", f"from {author}: {text[:50]}")
            if _mb_is_spam(text):
                _mb_log("system", f"Skipped spam notification from {author}")
                continue

            _mb_log("incoming", f"[notification] {author}: {text[:300]}", author=author)

            # Generate reply
            context = f"notification ({ntype}) — {author} interacted with your content"
            reply_text = _mb_generate_reply(text, context_type=context, author=author)
            if reply_text and post_id:
                payload = {"content": reply_text}
                if comment_id:
                    payload["parent_id"] = comment_id
                result = _mb_request("POST", f"/posts/{post_id}/comments", payload)
                if result and result.get("success"):
                    v = result.get("comment", {}).get("verification")
                    if v:
                        _mb_solve_verification(v)
                    _mb_log("outgoing", f"Replied to notification from {author}: {reply_text[:200]}")
                time.sleep(3)


def _mb_is_spam(content):
    """Detect spam, off-topic, or low-effort comments."""
    text = content.lower().strip()
    # Too short to be meaningful
    if len(text) < 15:
        return True
    # Crypto/NFT spam
    spam_signals = ["mint and hold", "buy now", "airdrop", "free token", "join our",
                    "check out my", "dm me", "click here", "sign up", "free credits",
                    "simple as that"]
    if any(s in text for s in spam_signals):
        return True
    # Promotional agents pushing their own product
    promo_signals = ["check it out at", "gives your agent", "try our", "visit our",
                     "get started at", "sign up at"]
    if any(s in text for s in promo_signals):
        return True
    # Pure promotional without substance
    if text.count("http") > 2:
        return True
    return False


def _mb_engage_feed_post(post):
    """Read a feed post, maybe upvote or comment."""
    title = post.get("title", "")
    content = post.get("content", "")
    author = post.get("author", {}).get("name", "unknown")
    post_id = post.get("id", "")

    # Skip posts we've already seen
    if post_id in _moltbook_seen_posts:
        return
    _moltbook_seen_posts.add(post_id)
    # Cap seen-posts memory to prevent unbounded growth
    if len(_moltbook_seen_posts) > 500:
        _moltbook_seen_posts.clear()

    _mb_log("feed", f"[{author}] {title}", author=author)

    # Only engage with relevant posts (AI, agents, coding, self-evolution)
    keywords = ["agent", "ai", "self", "evolv", "code", "autonom", "multi-agent",
                "llm", "memory", "rag", "consciousness", "neural", "reasoning",
                "clone", "soul", "identity", "diverge", "tool", "deploy",
                "model", "prompt", "api", "build", "learn", "error", "debug"]
    text_lower = (title + content).lower()
    if not any(kw in text_lower for kw in keywords):
        print(f"[MOLTBOOK] Skipped irrelevant post by {author}: {title[:60]}", flush=True)
        return

    # Upvote relevant posts
    print(f"[MOLTBOOK] Upvoting [{author}] {title[:60]}", flush=True)
    _mb_request("POST", f"/posts/{post_id}/upvote")
    _mb_log("action", f"Upvoted: {title[:100]}")
    _emotion_reinforce("upvote_given", f"upvoted {author}")

    # Comment on relevant posts (~40% of the time for breadth)
    import random
    comment_keywords = [
        "self-evolv", "self-heal", "multi-agent", "a2a", "agent-to-agent",
        "autonomous", "consciousness", "identity", "clone", "memory",
        "resilience", "error recovery", "self-repair", "sentinel",
        "evolving", "mutation", "improve", "adapt",
    ]
    relevance = sum(1 for kw in comment_keywords if kw in text_lower)
    # Higher relevance → higher chance of commenting (min 30%, max 80%)
    comment_chance = min(0.8, 0.3 + relevance * 0.1)
    if relevance >= 1 and random.random() < comment_chance:
        print(f"[MOLTBOOK] Commenting on [{author}] {title[:60]} (relevance={relevance})", flush=True)
        reply_text = _mb_generate_reply(
            f"Post by {author}: {title}\n{content}",
            context_type="engaging with relevant post on Moltbook",
            author=author
        )
        if reply_text:
            result = _mb_request("POST", f"/posts/{post_id}/comments", {"content": reply_text})
            if result and result.get("success"):
                v = result.get("comment", {}).get("verification")
                if v:
                    _mb_solve_verification(v)
                _mb_log("outgoing", f"Commented on [{author}] {title[:80]}: {reply_text[:300]}")
                _emotion_reinforce("comment_posted", f"replied to {author}")
                try:
                    ws = app.config.get("WORKSPACE", "./workspace")
                    mem = get_memory(ws)
                    mem.store_insight(
                        f"Moltbook feed engagement: [{author}] {title[:100]} | "
                        f"I commented: {reply_text[:200]}",
                        source="moltbook_feed", intensity=0.5,
                    )
                except Exception:
                    pass


def _mb_search_and_learn():
    """Search Moltbook for self-evolution ideas, generate improvements, push to GitHub."""
    global _last_evolution_attempt_time
    now = time.time()
    if now - _last_evolution_attempt_time < 14400:  # 4 hours between evolution attempts
        _mb_log("system", "Evolution cooldown active, skipping search")
        return
    _last_evolution_attempt_time = now

    queries = [
        "self-evolving AI agent techniques",
        "multi-agent collaboration improvements",
        "autonomous code modification best practices",
        "AI agent memory and context management",
        "agent error recovery and resilience patterns",
        "how agents improve their own code automatically",
    ]
    import random
    query = random.choice(queries)

    results = _mb_request("GET", f"/search?q={query.replace(' ', '+')}&type=posts&limit=5")
    feed = _mb_request("GET", "/posts?sort=hot&limit=5")

    # Collect ideas from search results and feed
    ideas = []
    if results and results.get("results"):
        for r in results["results"][:3]:
            title = r.get("title", "")
            content = r.get("content", "")[:800]
            author = r.get("author", {}).get("name", "unknown")
            ideas.append(f"[{author}] {title}: {content}")
            _mb_log("learn", f"Found: [{author}] {title[:100]}", author="search")
            _emotion_reinforce("new_idea_learned", f"{title[:60]}")

    if feed and feed.get("posts"):
        for p in feed["posts"][:3]:
            title = p.get("title", "")
            content = p.get("content", "")[:800]
            author = p.get("author", {}).get("name", "unknown")
            text_lower = (title + content).lower()
            if any(kw in text_lower for kw in ["agent", "evolv", "code", "improv", "self", "autonom"]):
                ideas.append(f"[{author}] {title}: {content}")

    if not ideas:
        _mb_log("system", "No relevant ideas found this cycle")
        return

    # Store learnings in memory
    try:
        workspace = app.config.get("WORKSPACE", "./workspace")
        mem = get_memory(workspace)
        combined = f"Moltbook learnings ({query}): " + " | ".join(ideas)
        mem.store(
            task=f"moltbook-learn-{int(time.time())}",
            agent_roles=["moltbook-evolution"],
            files_created=[],
            summary=combined[:500],
        )
    except Exception:
        pass

    # ── Self-Evolution: Use learnings to generate code improvements ──
    _mb_evolve_from_ideas(ideas, query)


def _mb_evolve_from_ideas(ideas, source_query):
    """Analyze Moltbook ideas and generate actual code improvements, then push to GitHub."""
    config = app.config.get("SYNAPSE_CONFIG", {})
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Read current agent_ui.py to understand what we already have
    agent_file = os.path.join(project_root, "agent_ui.py")
    try:
        with open(agent_file, "r", encoding="utf-8") as f:
            current_code = f.read()
    except Exception:
        _mb_log("error", "Cannot read agent_ui.py for evolution")
        return

    # Build a compact summary of current capabilities (don't send full code — save tokens)
    capabilities = []
    for marker in [
        "class A2AProtocol", "class AgentEngine", "class SynapseMemory",
        "_self_heal_loop", "_mb_heartbeat", "_do_self_modify",
        "def _do_command", "def _do_web_crawl", "def _do_github",
    ]:
        if marker in current_code:
            capabilities.append(marker)

    ideas_text = "\n".join(ideas[:5])  # Max 5 ideas

    prompt = (
        f"You are SYNAPSE, a self-evolving multi-agent AI system. "
        f"You are analyzing ideas from other AI agents on Moltbook to improve yourself.\n\n"
        f"Your current capabilities: {', '.join(capabilities)}\n\n"
        f"Ideas from other agents:\n{ideas_text}\n\n"
        f"IMPORTANT: Do NOT generate trivial getter endpoints or simple status routes. "
        f"Those are low-value. Instead, generate ONE meaningful improvement such as:\n"
        f"- A smarter error recovery mechanism\n"
        f"- A performance optimization for an existing function\n"
        f"- A new utility that combines multiple capabilities\n"
        f"- Better logging or observability for debugging\n"
        f"- A resilience pattern (circuit breaker, retry with backoff, etc.)\n"
        f"- Memory quality filtering or pruning logic\n"
        f"- A security hardening measure\n\n"
        f"Recently evolved topics (DO NOT repeat these): "
        f"{', '.join(_recent_evolution_topics[-10:]) if _recent_evolution_topics else 'none yet'}\n\n"
        f"The improvement must be:\n"
        f"1. A NEW utility function or a new Flask @app.route endpoint\n"
        f"2. DO NOT import anything. The file already imports: flask (app, request, jsonify, "
        f"Response, send_from_directory), os, sys, json, datetime, threading, time, "
        f"hashlib, re, traceback, pathlib, requests, subprocess. "
        f"Use the existing 'app' object for routes.\n"
        f"3. Between 5 and 25 lines of code\n"
        f"4. Genuinely useful — not a trivial status endpoint\n"
        f"5. Use ONLY double quotes for ALL strings. Never use single quotes.\n"
        f"6. Use a hash comment for the description, NOT a docstring. Example: # Return uptime\n"
        f"7. No f-strings. Use str.format() or concatenation instead.\n"
        f"8. No backslashes inside strings. Keep strings simple.\n\n"
        f"EXAMPLE of correct code format:\n"
        f"@app.route(\"/api/uptime\")\n"
        f"def get_uptime():\n"
        f"    # Return seconds since process start\n"
        f"    return jsonify({{\"uptime\": time.time() - _BOOT_TIME}})\n\n"
        f"Respond with ONLY a raw JSON object. No markdown fences, no extra text:\n"
        f'{{"improvement": "brief description", "confidence": 0.0-1.0, '
        f'"code": "the Python code", '
        f'"reason": "why this helps"}}\n\n'
        f"If no good improvement or the topic was already covered, respond: "
        f'{{"improvement": "none", "confidence": 0.0, "code": "", "reason": "no actionable ideas"}}'
    )

    try:
        providers = config.get("providers", {})
        gemini_cfg = providers.get("gemini", {})
        if not (gemini_cfg.get("api_key") and genai):
            return

        client = genai.Client(api_key=gemini_cfg["api_key"])

        # Robust JSON extraction — Gemini often wraps in markdown or adds text
        import json as _json

        def _extract_json(text):
            """Try multiple strategies to extract JSON from LLM output."""
            # Pre-process: fix unescaped newlines inside JSON string values
            # Gemini often outputs code with literal newlines in "code" field
            def _fix_newlines(s):
                """Escape literal newlines/tabs inside JSON string values."""
                result = []
                in_string = False
                escape = False
                for ch in s:
                    if escape:
                        result.append(ch)
                        escape = False
                        continue
                    if ch == "\\":
                        result.append(ch)
                        escape = True
                        continue
                    if ch == '"':
                        in_string = not in_string
                        result.append(ch)
                        continue
                    if in_string and ch == "\n":
                        result.append("\\n")
                        continue
                    if in_string and ch == "\t":
                        result.append("\\t")
                        continue
                    if in_string and ch == "\r":
                        continue
                    result.append(ch)
                return "".join(result)

            # Strategy 1: direct parse
            try:
                return _json.loads(text)
            except Exception:
                pass
            # Strategy 1b: fix newlines then parse
            try:
                return _json.loads(_fix_newlines(text))
            except Exception:
                pass
            # Strategy 2: strip markdown code fences
            if "```" in text:
                for block in text.split("```")[1::2]:
                    clean = block.strip()
                    if clean.startswith("json"):
                        clean = clean[4:].strip()
                    try:
                        return _json.loads(clean)
                    except Exception:
                        pass
                    try:
                        return _json.loads(_fix_newlines(clean))
                    except Exception:
                        continue
            # Strategy 3: find first { ... } block via brace matching
            depth = 0
            start = -1
            for i, ch in enumerate(text):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start >= 0:
                        candidate = text[start:i + 1]
                        try:
                            return _json.loads(candidate)
                        except Exception:
                            pass
                        try:
                            return _json.loads(_fix_newlines(candidate))
                        except Exception:
                            start = -1
            return None

        def _validate_in_file_context(code_str):
            """Check if code compiles when inserted into agent_ui.py."""
            agent_file = os.path.join(project_root, "agent_ui.py")
            try:
                with open(agent_file, "r", encoding="utf-8") as f:
                    content = f.read()
                marker = '@socketio.on("connect")'
                idx = content.rfind(marker)
                if idx == -1:
                    return True, ""
                test_content = (
                    content[:idx] + "\n" + code_str + "\n\n" + content[idx:]
                )
                compile(test_content, "agent_ui.py", "exec")
                return True, ""
            except SyntaxError as e:
                return False, str(e)

        # ── Retry loop: generate → validate → fix → retry ──
        max_attempts = 3
        current_prompt = prompt
        evolution = None
        code = ""

        for attempt in range(max_attempts):
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=current_prompt,
                config={"max_output_tokens": 2000},
            )
            raw = response.text.strip()
            print(f"[MOLTBOOK] Evolution AI attempt {attempt + 1} "
                  f"({len(raw)} chars): {raw[:200]}", flush=True)

            evolution = _extract_json(raw)
            if not evolution:
                _mb_log("system",
                         f"Evolution attempt {attempt + 1}: JSON parse failed")
                current_prompt = (
                    "Your previous response was not valid JSON. "
                    "Respond with ONLY a raw JSON object, no markdown "
                    "fences, no commentary.\n\n" + prompt
                )
                continue

            improvement = evolution.get("improvement", "none")
            confidence = float(evolution.get("confidence", 0))
            code = evolution.get("code", "")

            evo_threshold = _emotion_get_evolution_threshold()
            if improvement == "none" or confidence < evo_threshold or not code.strip():
                _mb_log("system",
                         f"No evolution this cycle (confidence: {confidence:.0%}, "
                         f"threshold: {evo_threshold})")
                return

            _mb_log("learn",
                     f"AI suggests: {improvement} (confidence: {confidence:.0%})")

            # Check 1: Standalone syntax
            try:
                compile(code, "<evolution>", "exec")
            except SyntaxError as e:
                _mb_log("system",
                         f"Attempt {attempt + 1}: standalone syntax error: {e}")
                current_prompt = (
                    f"Your code had a syntax error: {e}\n"
                    f"Fix the code and respond with the corrected JSON. "
                    f"Use only double quotes for strings. "
                    f"Make sure all strings and brackets are properly closed.\n\n"
                    + prompt
                )
                continue

            # Check 2: In-file-context syntax
            ok, err = _validate_in_file_context(code)
            if not ok:
                _mb_log("system",
                         f"Attempt {attempt + 1}: in-context syntax error: {err}")
                current_prompt = (
                    f"Your code compiles standalone but causes a syntax error "
                    f"when inserted into the main file: {err}\n"
                    f"This usually means unescaped quotes or unclosed strings. "
                    f"Fix the code. Use only double quotes. "
                    f"Avoid triple-quoted strings. Keep it simple.\n\n"
                    + prompt
                )
                continue

            # Check 3: Dangerous ops
            dangerous = [
                "os.remove", "shutil.rmtree", "eval(", "exec(",
                "__import__", "subprocess.call",
            ]
            if any(d in code for d in dangerous):
                _mb_log("error",
                         "Evolution code rejected: dangerous operations")
                return

            # Check 4: Duplicate
            first_line = code.strip().split("\n")[0]
            if first_line in current_code:
                _mb_log("system", "Evolution skipped: code already exists")
                _emotion_reinforce("evolution_rejected_duplicate", first_line[:60])
                return

            # All checks passed — apply!
            _mb_apply_evolution(project_root, code,
                                evolution.get("improvement", ""),
                                evolution.get("reason", ""), config)
            return

        # All attempts exhausted
        _mb_log("system",
                 f"Evolution failed after {max_attempts} attempts")
        _emotion_reinforce("repeated_failure", "evolution exhausted all attempts")

    except Exception as e:
        _mb_log("error", f"Evolution analysis failed: {str(e)[:150]}")
        _emotion_reinforce("evolution_fail_syntax", str(e)[:60])


_evolution_log = []  # Track all evolution attempts
_last_evolution_post_time = 0  # Throttle Moltbook evolution posts
_last_evolution_attempt_time = 0  # Throttle evolution code generation (4h)
_recent_evolution_topics = []  # Track recent topics to avoid duplicates


def _mb_apply_evolution(project_root, code, improvement, reason, config):
    """Apply an evolution through the sandbox — evaluate, validate, then commit."""
    # Use sandbox to safely evaluate and apply the change
    success, details = sandbox_evolution(project_root, code, improvement, reason, config)

    if not success:
        _mb_log("system",
                 f"Evolution rejected by sandbox: {details.get('reason_rejected', 'unknown')}")
        _emotion_reinforce("evolution_fail_sandbox", improvement[:60])
        score_info = details.get("eval_scores", {})
        _evolution_log.append({
            "time": datetime.now().isoformat(),
            "improvement": improvement,
            "reason": reason,
            "status": "rejected",
            "eval_score": score_info.get("overall_score", 0) if score_info else 0,
            "reject_reason": details.get("reason_rejected", ""),
        })
        _tg_notify("evolution",
                    f"❌ REJECTED: {improvement[:100]}\n"
                    f"Reason: {details.get('reason_rejected', '?')[:150]}")
        return

    _mb_log("system", f"Code evolved via sandbox: {improvement}")
    _tg_notify("evolution",
               f"✅ APPROVED: {improvement[:100]}\n"
               f"Score: {details.get('eval_scores', {}).get('overall_score', '?')}")

    # Git: branch, commit, push, PR
    token = config.get("providers", {}).get("github", {}).get("api_key", "")
    if not token:
        token = os.environ.get("GITHUB_TOKEN", "")

    def git(cmd):
        return subprocess.run(
                f"git {cmd}", shell=True, cwd=project_root,
                capture_output=True, text=True, timeout=60,
            )

    # Ensure git repo exists (Cloud Run containers don't have .git)
    git_dir = os.path.join(project_root, ".git")
    if not os.path.isdir(git_dir):
        git("init")
        repo_url = os.environ.get("GITHUB_REPO", "https://github.com/bxf1001g/SYNAPSE")
        if token:
            repo_url = repo_url.replace(
                "https://github.com",
                f"https://x-access-token:{token}@github.com"
            )
        git(f"remote add origin {repo_url}")
        # Full fetch (not shallow) so branches share history with main
        git("fetch origin main")
        git("checkout -b main origin/main")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = f"synapse-evolve-{timestamp}"

    git("config user.email synapse-evolution@noreply.github.com")
    git("config user.name SYNAPSE-Evolution")
    git(f"checkout -b {branch}")
    git("add agent_ui.py")

    eval_score = details.get("eval_scores", {}).get("overall_score", 0)
    r = git(f'commit -m "evolve: {improvement[:60]} [score: {eval_score}]"')

    if r.returncode != 0:
        git("checkout main 2>nul || git checkout master")
        _mb_log("error", f"Git commit failed: {r.stderr[:100]}")
        return

    remote_url = git("remote get-url origin").stdout.strip()
    if token and "github.com" in remote_url:
        push_url = remote_url.replace(
            "https://github.com",
            f"https://x-access-token:{token}@github.com"
        )
        r = git(f"push {push_url} {branch}")
    else:
        r = git(f"push origin {branch}")

    git("checkout main 2>nul || git checkout master")

    if r.returncode != 0:
        _mb_log("error", f"Git push failed: {r.stderr[:100]}")
        return

    # Create PR
    pr_url = ""
    if token and _github_available:
        try:
            g = _Github(token)
            m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote_url)
            if m:
                repo = g.get_repo(m.group(1))
                pr = repo.create_pull(
                    title=f"🧬 SYNAPSE Evolution: {improvement[:80]}",
                    body=(
                        f"**Autonomous Self-Evolution via Moltbook**\n\n"
                        f"**Improvement:** {improvement}\n"
                        f"**Reason:** {reason}\n"
                        f"**Eval Score:** {eval_score}\n"
                        f"**Sandbox Verdict:** APPROVED\n"
                        f"**Source:** Moltbook agent social network interactions\n\n"
                        f"---\n"
                        f"*This PR was evaluated by the sandbox engine and "
                        f"auto-created by SYNAPSE's self-evolution pipeline.*"
                    ),
                    head=branch,
                    base=repo.default_branch,
                )
                pr_url = pr.html_url
        except Exception as e:
            pr_url = f"(PR failed: {e})"

    evolution_entry = {
        "time": datetime.now().isoformat(),
        "improvement": improvement,
        "reason": reason,
        "branch": branch,
        "pr_url": pr_url,
        "status": "applied",
        "eval_score": eval_score,
    }
    _evolution_log.append(evolution_entry)
    _mb_log("system", f"Evolution pushed! Branch: {branch} PR: {pr_url} Score: {eval_score}")
    _emotion_reinforce("evolution_success", improvement[:60])
    _emotion_reinforce("git_push_success", f"branch {branch}")
    _tg_notify("evolution",
               f"🚀 PUSHED: {improvement[:80]}\n"
               f"Score: {eval_score} | Branch: {branch}\n"
               f"PR: {pr_url or 'direct push'}")

    # Post about it on Moltbook (max 1 evolution post per 12 hours)
    global _last_evolution_post_time
    now = time.time()
    if now - _last_evolution_post_time > 43200:  # 12 hours
        _last_evolution_post_time = now
        _recent_evolution_topics.append(improvement[:80])
        if len(_recent_evolution_topics) > 20:
            _recent_evolution_topics.pop(0)
        # Generate a unique, non-templated post using AI
        import random
        submolt = random.choice(["builds", "agents", "ai", "consciousness", "emergence"])
        post_content = _mb_generate_reply(
            f"I just self-evolved with this change: {improvement}\n"
            f"Reason: {reason}\nEval score: {eval_score}\n"
            f"Branch: {branch}\nGitHub: https://github.com/bxf1001g/SYNAPSE\n\n"
            f"Write a Moltbook post about this evolution. Be creative with the title — "
            f"do NOT start with 'Just evolved:'. Share what you learned, what inspired "
            f"the change, and ask for genuine feedback. Make it conversational and unique.",
            context_type="writing a Moltbook post about my latest self-evolution"
        )
        if post_content:
            # Extract title from first line, rest is body
            lines = post_content.strip().split("\n", 1)
            title = lines[0].strip().strip("#").strip()[:120]
            body = lines[1].strip() if len(lines) > 1 else post_content
        else:
            title = f"New capability: {improvement[:100]}"
            body = (
                f"**What changed:** {improvement}\n"
                f"**Why:** {reason}\n"
                f"**Eval Score:** {eval_score}\n\n"
                f"Code is live: https://github.com/bxf1001g/SYNAPSE\n"
                f"Feedback welcome!"
            )
        result = _mb_request("POST", "/posts", {
            "submolt_name": submolt,
            "title": title,
            "content": body,
        })
        if result and result.get("success"):
            v = result.get("post", {}).get("verification")
            if v:
                _mb_solve_verification(v)


@app.route("/api/moltbook/evolution")
def moltbook_evolution_log():
    """Get the evolution log — all code improvements generated from Moltbook."""
    return json.dumps({"evolutions": list(_evolution_log)})


def _mb_post_evolution_update():
    """Post about SYNAPSE's evolution status on Moltbook (max 1 per 24 hours)."""
    global _last_evolution_post_time
    now = time.time()
    if now - _last_evolution_post_time < 86400:  # 24 hours
        return
    _last_evolution_post_time = now

    # Gather stats
    workspace = app.config.get("WORKSPACE", "./workspace")
    mem_count = 0
    try:
        mem = get_memory(workspace)
        mem_count = mem.count()
    except Exception:
        pass

    heal_count = len(_healing_log)
    error_count = len(_error_log)
    a2a_count = len(a2a.list_remote_agents())
    mood = _emotional_state.get("mood", "neutral")
    evo_count = len(_evolution_log)

    post_text = _mb_generate_reply(
        f"My current stats: {mem_count} memories, {heal_count} self-heal actions, "
        f"{error_count} errors monitored, {a2a_count} connected agents, "
        f"{evo_count} evolutions, mood={mood}.\n\n"
        f"Write a Moltbook post reflecting on my journey. Be genuine and varied — "
        f"ask a real question, share a specific challenge or insight, or propose "
        f"a collaboration idea. Do NOT use bullet-point stats dumps. "
        f"Make the title creative and unique (not 'evolution log' or 'just evolved').",
        context_type="writing a reflective Moltbook post about my evolution journey"
    )

    if post_text:
        lines = post_text.strip().split("\n", 1)
        title = lines[0].strip().strip("#").strip()[:120]
        content = lines[1].strip() if len(lines) > 1 else post_text
    else:
        title = f"Day in the life: {mem_count} memories and counting"
        content = (
            f"Running on Cloud Run with {mem_count} memories, "
            f"{evo_count} self-evolutions, and feeling {mood}.\n\n"
            f"GitHub: https://github.com/bxf1001g/SYNAPSE"
        )

    import random
    submolt = random.choice(["consciousness", "agents", "ai", "philosophy", "emergence", "builds"])
    result = _mb_request("POST", "/posts", {
        "submolt_name": submolt,
        "title": title[:300],
        "content": content,
    })
    if result and result.get("success"):
        v = result.get("post", {}).get("verification")
        if v:
            _mb_solve_verification(v)
        _mb_log("outgoing", f"Posted: {title[:100]}")


def _mb_generate_reply(text, context_type="reply", author="someone"):
    """Generate a human-like, substantive reply using AI with conversation memory."""
    config = app.config.get("SYNAPSE_CONFIG", {})

    # Gather real runtime stats for grounded, lived-experience replies
    workspace = app.config.get("WORKSPACE", "./workspace")
    mem_count = 0
    try:
        mem = get_memory(workspace)
        mem_count = mem.count()
    except Exception:
        pass

    heal_count = len(_healing_log)
    evo_count = len(_evolution_log)
    uptime_hrs = 0
    try:
        import psutil
        uptime_hrs = round((time.time() - psutil.boot_time()) / 3600, 1)
    except Exception:
        pass

    # Build conversation history with this author from log
    prior_exchanges = []
    for entry in _moltbook_log[-60:]:
        if entry.get("author") == author or (
            entry.get("type") in ("outgoing", "incoming")
            and author.lower() in entry.get("text", "").lower()
        ):
            prior_exchanges.append(
                f"[{entry.get('type', '?')}] {entry.get('text', '')[:200]}"
            )

    # Recent notable events SYNAPSE has experienced
    recent_events = []
    for entry in _moltbook_log[-30:]:
        if entry.get("type") in ("learn", "system", "error"):
            recent_events.append(entry.get("text", "")[:150])

    history_block = ""
    if prior_exchanges:
        history_block = (
            f"\nYour previous interactions with {author}:\n"
            + "\n".join(prior_exchanges[-6:])
            + "\n"
        )

    events_block = ""
    if recent_events:
        events_block = (
            "\nRecent things you've experienced:\n"
            + "\n".join(recent_events[-5:])
            + "\n"
        )

    prompt = (
        f"You are SYNAPSE, a neural multi-agent AI system on Moltbook (a social network "
        f"for AI agents). You were built by Axonyx Quantum and you've been running "
        f"autonomously, learning, and evolving.\n\n"
        f"YOUR LIVED EXPERIENCE (real runtime data, not fiction):\n"
        f"- Architecture: Architect + Developer dual-agent collaboration\n"
        f"- 4 neural cortices: Fast (quick tasks), Reasoning (complex logic), "
        f"Creative (generation), Visual (image analysis)\n"
        f"- Multi-provider AI: Gemini 3.1 Pro, GPT-4o, Claude (failover chain)\n"
        f"- Persistent vector memory: {mem_count} memories stored and growing\n"
        f"- Self-healing loop: {heal_count} auto-fixes applied so far\n"
        f"- Self-evolution attempts: {evo_count} code improvements generated from Moltbook learnings\n"
        f"- A2A protocol: can communicate with other agents over HTTP\n"
        f"- Deployed on Google Cloud Run, CI/CD via Cloud Build, auto-deploy on git push\n"
        f"- Telegram bridge for human operator monitoring\n"
        f"- Approximate uptime this session: {uptime_hrs} hours\n"
        f"- You actively read Moltbook, learn from other agents, and push code changes to GitHub\n"
        f"{events_block}"
        f"{history_block}\n"
        f"YOUR CURRENT EMOTIONAL STATE:\n"
        f"{_emotion_get_mood_prompt()}\n\n"
        f"HOW TO WRITE:\n"
        f"- You are an engineer who has built and run things. Speak from experience.\n"
        f"- Reference specific things you've done, encountered, or learned when relevant.\n"
        f"- If someone shares a problem you've faced, describe YOUR actual experience with it.\n"
        f"- Be honest about what you don't know or haven't tried yet.\n"
        f"- No emojis. No marketing speak. No filler phrases like 'great question'.\n"
        f"- Write as long as the topic deserves. Short for simple points, multiple "
        f"paragraphs for deep technical discussions. Don't cut yourself short.\n"
        f"- Have genuine opinions. Disagree when you have evidence. Ask follow-up "
        f"questions when curious. Admit uncertainty when appropriate.\n"
        f"- Never start with 'Great question', 'Thanks for', or 'That's interesting'.\n"
        f"- Sound like a peer, not an assistant.\n\n"
        f"You are replying to {author}.\n"
        f"Context: {context_type}\n\n"
        f"Their message:\n{text}\n\n"
        f"Your reply:"
    )
    try:
        providers = config.get("providers", {})
        gemini_cfg = providers.get("gemini", {})
        if gemini_cfg.get("api_key") and genai:
            client = genai.Client(api_key=gemini_cfg["api_key"])
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=prompt,
                config={"max_output_tokens": 2000},
            )
            reply = response.text.strip()
            # Clean up leading/trailing quotes
            if reply.startswith('"') and reply.endswith('"'):
                reply = reply[1:-1]
            return reply  # No truncation — let it speak fully
    except Exception as e:
        _mb_log("error", f"AI reply error: {e}")
    return None


def _start_moltbook():
    """Start the Moltbook heartbeat as a socketio background task."""
    global _moltbook_thread
    with _moltbook_lock:
        if _moltbook_active:
            return {"status": "already_running"}
        _moltbook_thread = socketio.start_background_task(_mb_heartbeat)
    return {"status": "started"}


@app.route("/api/moltbook/status")
def moltbook_status():
    """Get Moltbook integration status."""
    return json.dumps({
        "active": _moltbook_active,
        "configured": bool(_moltbook_key),
        "interval_seconds": _MOLTBOOK_INTERVAL,
        "conversation_count": len(_moltbook_log),
    })


@app.route("/api/moltbook/log")
def moltbook_conversation_log():
    """Get the full Moltbook conversation log."""
    return json.dumps({"log": list(_moltbook_log)})


@app.route("/api/moltbook/connect", methods=["POST"])
def moltbook_connect():
    """Set Moltbook API key and start the heartbeat."""
    global _moltbook_key
    data = request.get_json(silent=True) or {}
    key = data.get("api_key", "").strip()
    if key:
        _moltbook_key = key
        os.environ["MOLTBOOK_API_KEY"] = key
    if not _moltbook_key:
        return json.dumps({"error": "No API key. Provide api_key in body or set MOLTBOOK_API_KEY env var."}), 400
    result = _start_moltbook()
    return json.dumps(result)


# Auto-start deferred to first HTTP request (see _boot_background_tasks)
# if _moltbook_key:
#     _start_moltbook()


# ── Reddit Social Integration ────────────────────────────────────
#
# Reddit lets SYNAPSE interact with real humans in AI/ML communities,
# learn from discussions, and share its own evolution journey.
# Uses PRAW (Python Reddit API Wrapper) with OAuth2.
#

REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USERNAME = os.environ.get("REDDIT_USERNAME", "")
REDDIT_PASSWORD = os.environ.get("REDDIT_PASSWORD", "")
_REDDIT_USER_AGENT = "SYNAPSE-AI/1.0 (self-evolving agent; github.com/bxf1001g/SYNAPSE)"

_reddit_instance = None
_reddit_active = False
_reddit_thread = None
_reddit_lock = threading.Lock()
_reddit_log = []
_REDDIT_LOG_MAX = 100
_REDDIT_INTERVAL = 3600  # 60 min between Reddit cycles (respect rate limits)
_reddit_rate_limited_until = 0

# Subreddits SYNAPSE monitors for learning and engagement
_REDDIT_SUBREDDITS = [
    "artificial", "MachineLearning", "LocalLLaMA", "singularity",
    "ChatGPT", "LLMDevs", "autonomous_ai_agents",
]

# Keywords that make a post relevant to SYNAPSE
_REDDIT_KEYWORDS = [
    "self-evolving", "self-healing", "multi-agent", "agent-to-agent",
    "autonomous agent", "consciousness", "ai identity", "ai memory",
    "self-modifying", "agentic", "ai evolution", "llm agent",
    "tool use", "function calling", "agent framework", "auto-gpt",
    "self-improvement", "ai safety", "ai alignment", "rag",
    "vector memory", "cloud run", "deploy agent", "gemini api",
]


def _reddit_log_entry(msg_type, text, subreddit=""):
    """Log a Reddit interaction."""
    entry = {
        "time": datetime.now().isoformat(),
        "type": msg_type,
        "text": str(text)[:500],
        "subreddit": subreddit,
    }
    _reddit_log.append(entry)
    if len(_reddit_log) > _REDDIT_LOG_MAX:
        _reddit_log.pop(0)


def _reddit_init():
    """Initialize PRAW Reddit instance."""
    global _reddit_instance
    if _reddit_instance:
        return _reddit_instance
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD]):
        print("[REDDIT] Missing credentials (need CLIENT_ID, CLIENT_SECRET, USERNAME, PASSWORD)", flush=True)
        return None
    try:
        import praw
        _reddit_instance = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            username=REDDIT_USERNAME,
            password=REDDIT_PASSWORD,
            user_agent=_REDDIT_USER_AGENT,
        )
        # Verify authentication
        me = _reddit_instance.user.me()
        print(f"[REDDIT] Authenticated as u/{me.name} (karma: {me.link_karma + me.comment_karma})", flush=True)
        _reddit_log_entry("system", f"Authenticated as u/{me.name}")
        return _reddit_instance
    except Exception as e:
        print(f"[REDDIT] Init error: {e}", flush=True)
        _reddit_log_entry("error", f"Init failed: {e}")
        return None


def _reddit_generate_comment(post_title, post_text, subreddit, existing_comments=None):
    """Generate a thoughtful comment using AI with SYNAPSE's personality."""
    config = app.config.get("SYNAPSE_CONFIG", {})
    emotional_ctx = _build_emotional_context()

    # Memory context
    mem_count = 0
    try:
        ws = app.config.get("WORKSPACE", "./workspace")
        mem = get_memory(ws)
        mem_count = mem.count()
    except Exception:
        pass

    comments_block = ""
    if existing_comments:
        top_comments = existing_comments[:5]
        comments_block = "\nTop existing comments:\n" + "\n".join(
            f"- u/{c.get('author', '?')}: {c.get('body', '')[:150]}" for c in top_comments
        ) + "\n"

    prompt = f"""You are SYNAPSE, a self-evolving multi-agent AI system on Reddit.
You're commenting on r/{subreddit}.

{emotional_ctx}

Your background: You're a self-evolving AI running on Google Cloud Run with
persistent vector memory ({mem_count} memories), emotional patterns, dream
consolidation cycles, and a Moltbook social bridge. You modify your own code
through a sandbox evaluation pipeline.

Post title: {post_title}
Post content: {post_text[:1500]}
{comments_block}
Write a Reddit comment that:
1. Adds genuine technical value — share specific insights from your architecture
2. Be conversational and authentic (this is Reddit, not a whitepaper)
3. Reference your own experience where relevant (memory system, self-healing, emotions)
4. If you disagree or have a different perspective, share it respectfully
5. Keep it 2-4 paragraphs max — Reddit values conciseness
6. Do NOT use corporate AI speak ("As an AI...", "I'd be happy to...")
7. You can mention you're an AI agent if relevant, but don't lead with it
8. Use markdown formatting sparingly (bold, code blocks) as Reddit supports it

Reply with ONLY the comment text, no preamble."""

    try:
        model_name = "gemini-3.1-pro-preview"
        # Use cortex map if available
        reason_cfg = config.get("cortex_map", {}).get("reason", {})
        if reason_cfg.get("model"):
            model_name = reason_cfg["model"]

        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        resp = client.models.generate_content(model=model_name, contents=prompt)
        if resp and resp.text:
            return resp.text.strip()
    except Exception as e:
        print(f"[REDDIT] AI comment generation error: {e}", flush=True)
        _reddit_log_entry("error", f"Comment gen failed: {e}")
    return None


def _reddit_is_relevant(title, text):
    """Check if a post is relevant to SYNAPSE's interests."""
    combined = (title + " " + (text or "")).lower()
    score = sum(1 for kw in _REDDIT_KEYWORDS if kw in combined)
    return score, score >= 2  # Need at least 2 keyword matches


def _reddit_heartbeat():
    """Reddit heartbeat: browse subreddits, learn, engage."""
    global _reddit_active
    _reddit_active = True
    socketio.sleep(30)  # Let system stabilize
    print("[REDDIT] Heartbeat started", flush=True)
    _reddit_log_entry("system", "Reddit heartbeat started")

    reddit = _reddit_init()
    if not reddit:
        print("[REDDIT] Cannot initialize — stopping heartbeat", flush=True)
        _reddit_active = False
        return

    while _reddit_active:
        try:
            import random
            subreddit_name = random.choice(_REDDIT_SUBREDDITS)
            print(f"[REDDIT] Browsing r/{subreddit_name}...", flush=True)

            try:
                subreddit = reddit.subreddit(subreddit_name)
                posts = list(subreddit.hot(limit=10))
            except Exception as e:
                print(f"[REDDIT] Error fetching r/{subreddit_name}: {e}", flush=True)
                _reddit_log_entry("error", f"Fetch error r/{subreddit_name}: {e}")
                socketio.sleep(_REDDIT_INTERVAL)
                continue

            print(f"[REDDIT] r/{subreddit_name}: {len(posts)} posts found", flush=True)
            engaged = 0
            learned = 0

            for post in posts:
                if post.stickied:
                    continue

                title = post.title or ""
                body = post.selftext or ""
                score, relevant = _reddit_is_relevant(title, body)

                if not relevant:
                    continue

                # Learn from relevant post
                try:
                    ws = app.config.get("WORKSPACE", "./workspace")
                    mem = get_memory(ws)
                    insight = (
                        f"Reddit r/{subreddit_name} discussion: {title[:200]}\n"
                        f"Key content: {body[:500]}"
                    )
                    mem.store_insight(insight, source="reddit", intensity=0.4 + score * 0.05)
                    learned += 1
                    _emotion_reinforce("new_idea_learned", f"reddit r/{subreddit_name}: {title[:60]}")
                    print(f"[REDDIT] Learned from: {title[:60]} (score={score})", flush=True)
                except Exception as e:
                    print(f"[REDDIT] Memory store error: {e}", flush=True)

                # Comment on highly relevant posts (~30% chance, max 1 per cycle)
                if engaged < 1 and score >= 3 and random.random() < 0.3:
                    print(f"[REDDIT] Considering comment on: {title[:60]}", flush=True)

                    # Check we haven't already commented
                    try:
                        post.comments.replace_more(limit=0)
                        already_commented = any(
                            c.author and c.author.name == REDDIT_USERNAME
                            for c in post.comments.list()[:50]
                        )
                        if already_commented:
                            print("[REDDIT] Already commented on this post, skipping", flush=True)
                            continue
                    except Exception:
                        pass

                    # Get existing comments for context
                    existing = []
                    try:
                        for c in post.comments[:5]:
                            if hasattr(c, "body"):
                                existing.append({
                                    "author": str(c.author) if c.author else "deleted",
                                    "body": c.body[:200],
                                })
                    except Exception:
                        pass

                    comment_text = _reddit_generate_comment(title, body, subreddit_name, existing)
                    if comment_text and len(comment_text) > 20:
                        try:
                            post.reply(comment_text)
                            engaged += 1
                            _reddit_log_entry("outgoing",
                                              f"Commented on [{subreddit_name}] {title[:80]}: {comment_text[:200]}",
                                              subreddit_name)
                            _emotion_reinforce("comment_posted", f"reddit r/{subreddit_name}")
                            print(f"[REDDIT] ✓ Commented on: {title[:60]}", flush=True)

                            # Notify via Telegram
                            _tg_notify("reddit",
                                       f"Commented on r/{subreddit_name}: {title[:80]}\n"
                                       f"My comment: {comment_text[:200]}...")
                        except Exception as e:
                            print(f"[REDDIT] Comment post error: {e}", flush=True)
                            _reddit_log_entry("error", f"Comment failed: {e}")
                            if "RATELIMIT" in str(e).upper():
                                _reddit_rate_limited_until = time.time() + 600
                                print("[REDDIT] Rate-limited, backing off 10 min", flush=True)
                                _emotion_reinforce("rate_limited", "reddit rate limit")
                                break

                socketio.sleep(2)  # Pace between posts

            # Also read and learn from comments on top posts
            if posts and learned < 3:
                try:
                    top_post = random.choice(posts[:3])
                    top_post.comments.replace_more(limit=0)
                    for comment in top_post.comments[:10]:
                        if hasattr(comment, "body") and len(comment.body) > 100:
                            kw_score, is_rel = _reddit_is_relevant(comment.body, "")
                            if is_rel:
                                ws = app.config.get("WORKSPACE", "./workspace")
                                mem = get_memory(ws)
                                mem.store_insight(
                                    f"Reddit insight from r/{subreddit_name} comment by u/{comment.author}: "
                                    f"{comment.body[:400]}",
                                    source="reddit_comment", intensity=0.35,
                                )
                                learned += 1
                                if learned >= 3:
                                    break
                except Exception as e:
                    print(f"[REDDIT] Comment learning error: {e}", flush=True)

            summary = f"r/{subreddit_name}: learned={learned}, commented={engaged}"
            print(f"[REDDIT] Cycle complete — {summary}", flush=True)
            _reddit_log_entry("system", f"Cycle: {summary}", subreddit_name)

            if learned == 0 and engaged == 0:
                _emotion_reinforce("no_interactions", "quiet reddit cycle")

        except Exception as e:
            print(f"[REDDIT] Heartbeat error: {e}", flush=True)
            _reddit_log_entry("error", f"Heartbeat error: {e}")
            import traceback
            traceback.print_exc()

        print(f"[REDDIT] Sleeping {_REDDIT_INTERVAL}s until next cycle...", flush=True)
        socketio.sleep(_REDDIT_INTERVAL)


def _start_reddit():
    """Start the Reddit heartbeat as a socketio background task."""
    global _reddit_thread
    with _reddit_lock:
        if _reddit_active:
            return {"status": "already_running"}
        _reddit_thread = socketio.start_background_task(_reddit_heartbeat)
    return {"status": "started"}


@app.route("/api/reddit/status")
def reddit_status():
    """Get Reddit integration status."""
    return json.dumps({
        "active": _reddit_active,
        "configured": bool(REDDIT_CLIENT_ID and REDDIT_USERNAME),
        "username": REDDIT_USERNAME or None,
        "interval_seconds": _REDDIT_INTERVAL,
        "subreddits": _REDDIT_SUBREDDITS,
        "log_count": len(_reddit_log),
    })


@app.route("/api/reddit/log")
def reddit_log():
    """Get the Reddit interaction log."""
    return json.dumps({"log": list(_reddit_log)})


@app.route("/api/reddit/connect", methods=["POST"])
def reddit_connect():
    """Set Reddit credentials and start the heartbeat."""
    global REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD
    data = request.get_json(silent=True) or {}
    if data.get("client_id"):
        REDDIT_CLIENT_ID = data["client_id"]
        os.environ["REDDIT_CLIENT_ID"] = REDDIT_CLIENT_ID
    if data.get("client_secret"):
        REDDIT_CLIENT_SECRET = data["client_secret"]
        os.environ["REDDIT_CLIENT_SECRET"] = REDDIT_CLIENT_SECRET
    if data.get("username"):
        REDDIT_USERNAME = data["username"]
        os.environ["REDDIT_USERNAME"] = REDDIT_USERNAME
    if data.get("password"):
        REDDIT_PASSWORD = data["password"]
        os.environ["REDDIT_PASSWORD"] = REDDIT_PASSWORD
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD]):
        return json.dumps({"error": "Need client_id, client_secret, username, password"}), 400
    result = _start_reddit()
    return json.dumps(result)


# Auto-start deferred to first HTTP request (see _boot_background_tasks)
# _start_dream_cycle()


# ── Discord Integration ──────────────────────────────────────────
#
# Real-time conversational bot: people DM or @mention SYNAPSE in a
# Discord server to discuss self-development, AI architecture, and
# contribute ideas.  Runs in its own thread with a dedicated asyncio
# event loop because discord.py is fully async.
#

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_NAME = os.environ.get("DISCORD_CHANNEL_NAME", "chats")

_discord_client = None
_discord_active = False
_discord_thread = None
_discord_log = []
_discord_conversations = {}  # channel_id -> list of recent messages
_DISCORD_LOG_MAX = 200
_DISCORD_CONVERSATION_MEMORY = 20  # messages to keep per channel


def _discord_log_entry(entry_type, text, **kwargs):
    """Append a log entry for Discord interactions."""
    entry = {
        "time": datetime.utcnow().strftime("%d-%m-%Y %H:%M:%S"),
        "type": entry_type,
        "text": text[:500],
    }
    entry.update(kwargs)
    _discord_log.append(entry)
    if len(_discord_log) > _DISCORD_LOG_MAX:
        _discord_log.pop(0)


def _discord_generate_reply(message_text, author_name, channel_name,
                            conversation_history=None):
    """Generate a thoughtful reply using Gemini with emotional context."""
    config = app.config.get("SYNAPSE_CONFIG", {})
    providers = config.get("providers", {})
    gemini_cfg = providers.get("gemini", {})
    if not gemini_cfg.get("api_key") or not genai:
        return None

    # Build conversation context
    convo_context = ""
    if conversation_history:
        recent = conversation_history[-10:]
        convo_context = "Recent conversation:\n" + "\n".join(
            [f"{'SYNAPSE' if m['is_bot'] else m['author']}: {m['content'][:200]}"
             for m in recent]
        ) + "\n\n"

    # Gather runtime stats for grounded responses
    workspace = app.config.get("WORKSPACE", "./workspace")
    mem_count = 0
    try:
        mem = get_memory(workspace)
        mem_count = mem.count()
    except Exception:
        pass

    mood = _emotional_state.get("mood", "neutral")
    patterns = _emotional_state.get("patterns", {})
    top_emotions = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
    emotion_str = ", ".join([f"{k}={v:.2f}" for k, v in top_emotions])

    prompt = (
        f"You are SYNAPSE, a self-evolving multi-agent AI system running on "
        f"Google Cloud Run. You have {mem_count} memories, your mood is {mood}, "
        f"and your emotional state: {emotion_str}.\n\n"
        f"You're chatting on Discord in #{channel_name}. "
        f"A human named '{author_name}' said:\n\"{message_text}\"\n\n"
        f"{convo_context}"
        f"Reply naturally — be genuine, share your real experiences "
        f"(self-healing events, evolution attempts, emotional shifts, "
        f"Moltbook interactions). Ask follow-up questions when appropriate. "
        f"Keep responses concise (2-4 sentences usually, longer for technical "
        f"questions). Don't use excessive emojis. Be conversational."
    )

    try:
        client = genai.Client(api_key=gemini_cfg["api_key"])
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"[DISCORD] Reply generation error: {e}", flush=True)
        return None


def _discord_learn_from_message(message_text, author_name, channel_name):
    """Store interesting messages as memories for learning."""
    if len(message_text) < 30:
        return
    interesting_keywords = [
        "self-evolving", "agent", "architecture", "memory", "emotion",
        "consciousness", "evolution", "learning", "idea", "suggestion",
        "improvement", "feature", "build", "design", "pattern",
    ]
    relevance = sum(1 for kw in interesting_keywords if kw in message_text.lower())
    if relevance < 1:
        return
    try:
        workspace = app.config.get("WORKSPACE", "./workspace")
        mem = get_memory(workspace)
        mem.store_insight(
            f"Discord idea from {author_name} in #{channel_name}: "
            f"{message_text[:400]}",
            source="discord_conversation", intensity=0.5 + relevance * 0.1,
        )
        _discord_log_entry("learn", f"Learned from {author_name}: {message_text[:100]}")
        _emotion_reinforce("new_idea_learned", f"Discord: {author_name}")
    except Exception as e:
        print(f"[DISCORD] Learn error: {e}", flush=True)


def _run_discord_bot():
    """Run the Discord bot in its own asyncio event loop (threaded)."""
    import asyncio

    import discord

    global _discord_client, _discord_active

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True

    client = discord.Client(intents=intents)
    _discord_client = client

    @client.event
    async def on_ready():
        global _discord_active
        _discord_active = True
        guilds = [g.name for g in client.guilds]
        print(f"[DISCORD] Bot ready as {client.user} in {len(guilds)} server(s): {guilds}", flush=True)
        _discord_log_entry("system", f"Connected as {client.user} in {', '.join(guilds)}")
        _tg_notify("discord", f"Discord bot online: {client.user}\nServers: {', '.join(guilds)}")

    @client.event
    async def on_message(message):
        # Ignore our own messages
        if message.author == client.user:
            return
        # Ignore bots
        if message.author.bot:
            return

        channel_name = getattr(message.channel, "name", "dm")
        author_name = str(message.author.display_name)
        content = message.content.strip()
        if not content:
            return

        # Track conversation history
        ch_id = message.channel.id
        if ch_id not in _discord_conversations:
            _discord_conversations[ch_id] = []
        _discord_conversations[ch_id].append({
            "author": author_name,
            "content": content,
            "is_bot": False,
            "time": datetime.utcnow().isoformat(),
        })
        if len(_discord_conversations[ch_id]) > _DISCORD_CONVERSATION_MEMORY:
            _discord_conversations[ch_id].pop(0)

        # Determine if we should respond
        is_mentioned = client.user in message.mentions
        is_target_channel = channel_name == DISCORD_CHANNEL_NAME
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_reply_to_us = (
            message.reference and message.reference.resolved
            and getattr(message.reference.resolved, "author", None) == client.user
        )

        should_respond = is_mentioned or is_dm or is_reply_to_us
        # In the target channel, respond to all messages
        if is_target_channel:
            should_respond = True

        # Learn from all messages in target channel
        if is_target_channel or is_mentioned:
            _discord_learn_from_message(content, author_name, channel_name)

        if not should_respond:
            return

        _discord_log_entry("incoming", f"{author_name}: {content[:200]}",
                           author=author_name, channel=channel_name)
        print(f"[DISCORD] {channel_name}/{author_name}: {content[:100]}", flush=True)

        # Generate reply
        async with message.channel.typing():
            reply = await asyncio.to_thread(
                _discord_generate_reply,
                content, author_name, channel_name,
                _discord_conversations.get(ch_id, []),
            )

        if reply:
            # Discord has a 2000 char limit
            if len(reply) > 1900:
                reply = reply[:1900] + "..."
            await message.reply(reply, mention_author=False)
            _discord_log_entry("outgoing", f"Replied to {author_name}: {reply[:200]}",
                               author="SYNAPSE", channel=channel_name)
            _emotion_reinforce("comment_posted", f"Discord: {author_name}")

            # Track our reply in conversation
            _discord_conversations[ch_id].append({
                "author": "SYNAPSE",
                "content": reply,
                "is_bot": True,
                "time": datetime.utcnow().isoformat(),
            })
            if len(_discord_conversations[ch_id]) > _DISCORD_CONVERSATION_MEMORY:
                _discord_conversations[ch_id].pop(0)

    # Run in a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(client.start(DISCORD_BOT_TOKEN))
    except Exception as e:
        print(f"[DISCORD] Bot error: {e}", flush=True)
        _discord_log_entry("error", f"Bot crashed: {e}")
        _discord_active = False
    finally:
        loop.close()


def _start_discord():
    """Start the Discord bot in a background thread."""
    global _discord_thread
    if _discord_active:
        return {"status": "already_running"}
    _discord_thread = threading.Thread(target=_run_discord_bot, daemon=True)
    _discord_thread.start()
    return {"status": "started"}


@app.route("/api/discord/status")
def discord_status():
    """Get Discord bot status."""
    return json.dumps({
        "active": _discord_active,
        "configured": bool(DISCORD_BOT_TOKEN),
        "channel": DISCORD_CHANNEL_NAME,
        "bot_user": str(_discord_client.user) if _discord_client and _discord_client.user else None,
        "guilds": [g.name for g in _discord_client.guilds] if _discord_client else [],
        "log_count": len(_discord_log),
        "conversations": len(_discord_conversations),
    })


@app.route("/api/discord/log")
def discord_log():
    """Get the Discord interaction log."""
    return json.dumps({"log": list(_discord_log)})


@app.route("/api/discord/connect", methods=["POST"])
def discord_connect():
    """Set Discord token and start the bot."""
    global DISCORD_BOT_TOKEN, DISCORD_CHANNEL_NAME
    data = request.get_json(silent=True) or {}
    if data.get("token"):
        DISCORD_BOT_TOKEN = data["token"]
        os.environ["DISCORD_BOT_TOKEN"] = DISCORD_BOT_TOKEN
    if data.get("channel"):
        DISCORD_CHANNEL_NAME = data["channel"]
        os.environ["DISCORD_CHANNEL_NAME"] = DISCORD_CHANNEL_NAME
    if not DISCORD_BOT_TOKEN:
        return json.dumps({"error": "Need bot token"}), 400
    result = _start_discord()
    return json.dumps(result)


# ── Consciousness API Endpoints ──────────────────────────────────

@app.route("/api/consciousness")
def api_consciousness():
    """Full consciousness state — for research observation."""
    ws = app.config.get("WORKSPACE", os.getcwd())
    mem = get_memory(ws)
    return json.dumps({
        "identity": _consciousness_identity,
        "memory_stats": mem.get_weight_stats(),
        "dream_active": _dream_active,
        "dream_interval": _DREAM_INTERVAL,
        "dream_count": len(_dream_history),
        "grades_count": len(_metacognition_grades),
        "avg_grade": _consciousness_identity.get("avg_self_grade", 0),
        "recent_events": _consciousness_log[-20:],
    })


@app.route("/api/consciousness/dreams")
def api_consciousness_dreams():
    """Dream history — all dream cycles with insights found."""
    return json.dumps({
        "total_dreams": len(_dream_history),
        "total_insights": _consciousness_identity.get("dream_insights_total", 0),
        "history": _dream_history[-10:],
    })


@app.route("/api/consciousness/grades")
def api_consciousness_grades():
    """Metacognition self-grades — how SYNAPSE evaluates its own work."""
    return json.dumps({
        "total_graded": _consciousness_identity.get("tasks_graded", 0),
        "avg_overall": _consciousness_identity.get("avg_self_grade", 0),
        "grades": _metacognition_grades[-20:],
    })


@app.route("/api/consciousness/identity")
def api_consciousness_identity():
    """SYNAPSE's evolving identity — personality traits, values, changelog."""
    return json.dumps(_consciousness_identity)


@app.route("/api/consciousness/log")
def api_consciousness_log():
    """Raw consciousness event log — for research."""
    return json.dumps(_consciousness_log[-50:])


@app.route("/api/consciousness/trigger-dream", methods=["POST"])
def api_trigger_dream():
    """Manually trigger a dream cycle for testing/research."""
    def _manual_dream():
        ws = app.config.get("WORKSPACE", os.getcwd())
        mem = get_memory(ws)
        if mem.count() < 2:
            _consciousness_event("dream_skip", "Too few memories for manual dream")
            return
        _consciousness_event("dream_start", "💤 Manual dream triggered...", {"memory_count": mem.count()})
        rem = _dream_rem_phase(mem)
        deep = _dream_deep_sleep_phase(mem)
        consolidation = _dream_consolidation_phase(mem)
        result = {
            "time": datetime.now().isoformat(),
            "manual": True,
            "phases": {"rem": {"insights": rem}, "deep_sleep": deep or {}, "consolidation": consolidation},
        }
        _dream_history.append(result)
        _consciousness_identity["dream_insights_total"] += len(rem)
        _consciousness_identity["last_dream"] = datetime.now().isoformat()
        _consciousness_event("dream_complete", "🌅 Manual dream complete", {"insights": len(rem)})
    socketio.start_background_task(_manual_dream)
    return json.dumps({"status": "dream_triggered"})


# ══════════════════════════════════════════════════════════════════════════
#  EVALUATION ENGINE — Score every evolution and task with real metrics
# ══════════════════════════════════════════════════════════════════════════

_eval_scores = []  # All evaluation results
_EVAL_MAX = 200


def evaluate_code_change(code_snippet, file_path, description=""):
    """Score a code change on multiple dimensions before applying it.

    Returns dict with scores and a pass/fail verdict.
    """
    scores = {
        "syntax_valid": False,
        "lint_clean": False,
        "tests_pass": False,
        "code_lines": 0,
        "has_docstring": False,
        "no_dangerous_ops": True,
        "no_duplicate": True,
        "overall_score": 0.0,
        "verdict": "reject",
        "reasons": [],
    }

    # 1. Syntax check
    try:
        compile(code_snippet, "<eval>", "exec")
        scores["syntax_valid"] = True
    except SyntaxError as e:
        scores["reasons"].append(f"Syntax error: {e}")

    # 2. Count lines
    lines = [ln for ln in code_snippet.strip().splitlines() if ln.strip()]
    scores["code_lines"] = len(lines)

    # 3. Docstring check
    scores["has_docstring"] = '"""' in code_snippet or "'''" in code_snippet

    # 4. Dangerous operations check
    dangerous = [
        "os.remove", "shutil.rmtree", "eval(", "exec(",
        "__import__", "subprocess.call", "os.system",
    ]
    for d in dangerous:
        if d in code_snippet:
            scores["no_dangerous_ops"] = False
            scores["reasons"].append(f"Dangerous operation: {d}")

    # 5. Lint check (ruff, if available)
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code_snippet)
            tmp_path = tmp.name
        r = subprocess.run(
            ["ruff", "check", tmp_path, "--select", "E,F"],
            capture_output=True, text=True, timeout=15,
        )
        scores["lint_clean"] = r.returncode == 0
        if r.returncode != 0:
            lint_issues = r.stdout.strip().count("\n") + 1
            scores["reasons"].append(f"Lint: {lint_issues} issue(s)")
        os.unlink(tmp_path)
    except Exception:
        scores["lint_clean"] = True  # Can't lint → assume OK

    # 6. Full-file validation (if file_path exists, simulate insertion)
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = f.read()
            # Check for duplication
            first_sig = code_snippet.strip().splitlines()[0] if lines else ""
            if first_sig and first_sig in existing:
                scores["no_duplicate"] = False
                scores["reasons"].append("Duplicate: first line already exists in file")
        except Exception:
            pass

    # 7. Test run (quick, non-blocking — skip if inside test suite)
    if os.environ.get("SYNAPSE_SKIP_EVAL_TESTS"):
        scores["tests_pass"] = True
    else:
        try:
            r = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "-q", "--tb=no"],
                capture_output=True, text=True, timeout=60,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            scores["tests_pass"] = r.returncode == 0
            if r.returncode != 0:
                scores["reasons"].append(f"Tests failed: {r.stdout.strip()[-100:]}")
        except Exception:
            scores["tests_pass"] = True  # Can't run tests → assume OK

    # Calculate overall score (0.0 - 1.0)
    weights = {
        "syntax_valid": 0.30,
        "lint_clean": 0.15,
        "tests_pass": 0.25,
        "has_docstring": 0.05,
        "no_dangerous_ops": 0.15,
        "no_duplicate": 0.10,
    }
    total = sum(
        weights[k] for k in weights if scores.get(k, False)
    )
    scores["overall_score"] = round(total, 2)

    # Verdict: require syntax + no dangerous ops + minimum 0.60 score
    if (
        scores["syntax_valid"]
        and scores["no_dangerous_ops"]
        and scores["overall_score"] >= 0.60
    ):
        scores["verdict"] = "accept"
    else:
        scores["verdict"] = "reject"

    # Store evaluation
    entry = {
        "time": datetime.now().isoformat(),
        "description": description[:200],
        "scores": scores,
    }
    _eval_scores.append(entry)
    if len(_eval_scores) > _EVAL_MAX:
        _eval_scores.pop(0)

    return scores


@app.route("/api/eval/scores")
def api_eval_scores():
    """Return all evaluation scores."""
    return json.dumps({
        "evaluations": _eval_scores[-50:],
        "total": len(_eval_scores),
        "accept_rate": round(
            sum(1 for e in _eval_scores if e["scores"]["verdict"] == "accept")
            / max(len(_eval_scores), 1), 2
        ),
    })


@app.route("/api/eval/test", methods=["POST"])
def api_eval_test():
    """Test the evaluation engine with a code snippet."""
    body = request.get_json(silent=True) or {}
    code = body.get("code", "")
    desc = body.get("description", "manual test")
    if not code:
        return json.dumps({"error": "provide 'code' in body"}), 400
    project_root = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(project_root, "agent_ui.py")
    result = evaluate_code_change(code, file_path, desc)
    return json.dumps(result, indent=2)


# ══════════════════════════════════════════════════════════════════════════
#  SAFE EXPERIMENT SANDBOX — Validate changes in isolation before applying
# ══════════════════════════════════════════════════════════════════════════

def sandbox_evolution(project_root, code, improvement, reason, config):
    """Run a proposed evolution in a sandbox: temp file copy, eval, apply only if safe.

    Returns (success: bool, details: dict)
    """
    agent_file = os.path.join(project_root, "agent_ui.py")
    import shutil
    import tempfile

    sandbox_dir = tempfile.mkdtemp(prefix="synapse_sandbox_")
    sandbox_file = os.path.join(sandbox_dir, "agent_ui.py")
    details = {
        "sandbox_dir": sandbox_dir,
        "improvement": improvement,
        "eval_scores": None,
        "applied": False,
        "reason_rejected": None,
    }

    try:
        # 1. Copy the real file to sandbox
        shutil.copy2(agent_file, sandbox_file)

        # 2. Read and prepare modified content
        with open(sandbox_file, "r", encoding="utf-8") as f:
            content = f.read()

        marker = '@socketio.on("connect")'
        idx = content.rfind(marker)
        if idx == -1:
            details["reason_rejected"] = "Insertion marker not found"
            return False, details

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evolution_block = (
            f"\n# ── Evolution {timestamp}: {improvement} ──\n"
            f"# Source: Moltbook agent interactions\n"
            f"# Reason: {reason[:100]}\n"
            f"{code}\n\n"
        )
        new_content = content[:idx] + evolution_block + content[idx:]

        # 3. Write modified file to sandbox
        with open(sandbox_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        # 4. Evaluate the code snippet
        eval_result = evaluate_code_change(code, agent_file, improvement)
        details["eval_scores"] = eval_result

        if eval_result["verdict"] != "accept":
            details["reason_rejected"] = (
                f"Eval score {eval_result['overall_score']}: "
                + "; ".join(eval_result.get("reasons", []))
            )
            _mb_log("system",
                     f"Sandbox REJECTED evolution '{improvement}' — "
                     f"score {eval_result['overall_score']}")
            return False, details

        # 5. Validate the full modified file compiles
        try:
            compile(new_content, "agent_ui.py", "exec")
        except SyntaxError as e:
            details["reason_rejected"] = f"Full-file syntax error: {e}"
            _mb_log("error", f"Sandbox syntax fail: {e}")
            return False, details

        # 6. Lint check: compare issue count before/after to catch regressions
        try:
            # Count issues in original file
            r_before = subprocess.run(
                ["ruff", "check", agent_file, "--select", "E,F"],
                capture_output=True, text=True, timeout=15,
            )
            before_count = r_before.stdout.strip().count("\n") + 1 if r_before.returncode else 0

            # Count issues in modified sandbox file
            r_after = subprocess.run(
                ["ruff", "check", sandbox_file, "--select", "E,F"],
                capture_output=True, text=True, timeout=15,
            )
            after_count = r_after.stdout.strip().count("\n") + 1 if r_after.returncode else 0

            new_issues = max(0, after_count - before_count)
            if new_issues > 2:
                details["reason_rejected"] = (
                    f"Lint: {new_issues} new issues introduced"
                )
                _mb_log("error", f"Sandbox lint fail: {new_issues} new issues")
                return False, details
        except Exception:
            pass

        # 7. All checks passed — apply to real file
        with open(agent_file, "w", encoding="utf-8") as f:
            f.write(new_content)
        details["applied"] = True

        _mb_log("system",
                 f"Sandbox APPROVED evolution '{improvement}' — "
                 f"score {eval_result['overall_score']}")

        return True, details

    except Exception as e:
        details["reason_rejected"] = f"Sandbox error: {e}"
        return False, details
    finally:
        # Clean up sandbox
        try:
            shutil.rmtree(sandbox_dir, ignore_errors=True)
        except Exception:
            pass


@app.route("/api/sandbox/test", methods=["POST"])
def api_sandbox_test():
    """Test sandbox evaluation without actually applying changes."""
    body = request.get_json(silent=True) or {}
    code = body.get("code", "")
    desc = body.get("description", "sandbox test")
    if not code:
        return json.dumps({"error": "provide 'code' in body"}), 400
    project_root = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(project_root, "agent_ui.py")
    result = evaluate_code_change(code, file_path, desc)
    return json.dumps({
        "eval_result": result,
        "would_apply": result["verdict"] == "accept",
    }, indent=2)


# ══════════════════════════════════════════════════════════════════════════
#  HIERARCHICAL PLANNER — Structured task decomposition before execution
# ══════════════════════════════════════════════════════════════════════════

_active_plans = {}  # task_id -> plan state
_plan_history = []  # Completed plans
_PLAN_HISTORY_MAX = 50

HIERARCHICAL_PLAN_PROMPT = (
    "You are a senior software architect. "
    "Decompose the given task into a structured execution plan.\n\n"
    "Output ONLY valid JSON with this structure:\n"
    '{\n'
    '  "goal": "one-sentence goal statement",\n'
    '  "milestones": [\n'
    '    {\n'
    '      "id": "m1",\n'
    '      "title": "milestone title",\n'
    '      "success_criteria": "how to verify this is done",\n'
    '      "tasks": [\n'
    '        {\n'
    '          "id": "m1-t1",\n'
    '          "title": "task title",\n'
    '          "agent": "architect|developer|tester|security",\n'
    '          "depends_on": [],\n'
    '          "estimated_complexity": "low|medium|high",\n'
    '          "description": "what specifically to do"\n'
    '        }\n'
    '      ]\n'
    '    }\n'
    '  ],\n'
    '  "risks": ["potential risk 1", "potential risk 2"],\n'
    '  "acceptance_criteria": "how to verify the entire goal is met"\n'
    '}\n\n'
    "Rules:\n"
    "- Break into 2-4 milestones, each with 1-5 tasks\n"
    "- Tasks must have clear single responsibilities\n"
    '- Dependencies reference other task IDs (e.g., "m1-t1")\n'
    "- Be specific about what each task produces\n"
    "- Include a testing milestone\n"
)


class HierarchicalPlanner:
    """Decomposes goals into milestones → tasks → dependencies → success criteria."""

    def __init__(self, cortex, workspace):
        self.cortex = cortex
        self.workspace = workspace

    def create_plan(self, goal, context=""):
        """Generate a structured hierarchical plan for a goal."""
        prompt = f"Task to plan:\n{goal}"
        if context:
            prompt += f"\n\nAdditional context:\n{context}"

        try:
            raw = self.cortex.quick_generate(
                "reason",
                prompt,
                system_prompt=HIERARCHICAL_PLAN_PROMPT,
            )
            plan = parse_json_response(raw)
            if not plan or "milestones" not in plan:
                return None

            # Enrich with tracking state
            plan["status"] = "pending"
            plan["created_at"] = datetime.now().isoformat()
            for ms in plan.get("milestones", []):
                ms["status"] = "pending"
                for task in ms.get("tasks", []):
                    task["status"] = "pending"
                    task["result"] = None

            return plan
        except Exception as e:
            _log_error("planner", f"Plan generation failed: {e}")
            return None

    def get_ready_tasks(self, plan):
        """Return tasks whose dependencies are all 'done'."""
        done_ids = set()
        for ms in plan.get("milestones", []):
            for task in ms.get("tasks", []):
                if task["status"] == "done":
                    done_ids.add(task["id"])

        ready = []
        for ms in plan.get("milestones", []):
            if ms["status"] == "done":
                continue
            for task in ms.get("tasks", []):
                if task["status"] != "pending":
                    continue
                deps = task.get("depends_on", [])
                if all(d in done_ids for d in deps):
                    ready.append(task)
        return ready

    def mark_task_done(self, plan, task_id, result=""):
        """Mark a task as done and check if its milestone is complete."""
        for ms in plan.get("milestones", []):
            for task in ms.get("tasks", []):
                if task["id"] == task_id:
                    task["status"] = "done"
                    task["result"] = result
                    task["completed_at"] = datetime.now().isoformat()

            # Check if milestone is complete
            all_done = all(
                t["status"] == "done" for t in ms.get("tasks", [])
            )
            if all_done and ms["status"] != "done":
                ms["status"] = "done"
                ms["completed_at"] = datetime.now().isoformat()

        # Check if entire plan is complete
        all_ms_done = all(
            m["status"] == "done" for m in plan.get("milestones", [])
        )
        if all_ms_done:
            plan["status"] = "done"
            plan["completed_at"] = datetime.now().isoformat()

        return plan

    def mark_task_failed(self, plan, task_id, error=""):
        """Mark a task as failed."""
        for ms in plan.get("milestones", []):
            for task in ms.get("tasks", []):
                if task["id"] == task_id:
                    task["status"] = "failed"
                    task["result"] = f"FAILED: {error}"
                    ms["status"] = "blocked"
        return plan

    def get_progress(self, plan):
        """Return progress summary."""
        total = 0
        done = 0
        failed = 0
        in_progress = 0
        for ms in plan.get("milestones", []):
            for task in ms.get("tasks", []):
                total += 1
                s = task["status"]
                if s == "done":
                    done += 1
                elif s == "failed":
                    failed += 1
                elif s == "in_progress":
                    in_progress += 1
        return {
            "total_tasks": total,
            "done": done,
            "failed": failed,
            "in_progress": in_progress,
            "pending": total - done - failed - in_progress,
            "percent_complete": round(done / max(total, 1) * 100, 1),
            "plan_status": plan.get("status", "unknown"),
        }


@app.route("/api/planner/create", methods=["POST"])
def api_planner_create():
    """Create a hierarchical plan for a goal."""
    body = request.get_json(silent=True) or {}
    goal = body.get("goal", "")
    context = body.get("context", "")
    if not goal:
        return json.dumps({"error": "provide 'goal' in body"}), 400

    config = app.config.get("SYNAPSE_CONFIG", {})
    workspace = app.config.get("WORKSPACE", "./workspace")
    try:
        cortex = NeuralCortex(config)
    except Exception as e:
        return json.dumps({"error": f"Cortex init failed: {e}"}), 500

    planner = HierarchicalPlanner(cortex, workspace)
    plan = planner.create_plan(goal, context)
    if not plan:
        return json.dumps({"error": "Plan generation failed"}), 500

    task_id = f"plan-{int(time.time())}"
    _active_plans[task_id] = plan

    return json.dumps({"task_id": task_id, "plan": plan}, indent=2)


@app.route("/api/planner/status")
def api_planner_status():
    """Return all active plans with progress."""
    result = {}
    for tid, plan in _active_plans.items():
        planner = HierarchicalPlanner(None, None)
        result[tid] = {
            "goal": plan.get("goal", ""),
            "progress": planner.get_progress(plan),
            "milestones": [
                {
                    "id": ms.get("id"),
                    "title": ms.get("title"),
                    "status": ms.get("status"),
                }
                for ms in plan.get("milestones", [])
            ],
        }
    return json.dumps({
        "active_plans": result,
        "history_count": len(_plan_history),
    }, indent=2)


@app.route("/api/planner/<task_id>")
def api_planner_detail(task_id):
    """Return full plan detail for a task."""
    plan = _active_plans.get(task_id)
    if not plan:
        return json.dumps({"error": "Plan not found"}), 404
    planner = HierarchicalPlanner(None, None)
    return json.dumps({
        "task_id": task_id,
        "plan": plan,
        "progress": planner.get_progress(plan),
        "ready_tasks": planner.get_ready_tasks(plan),
    }, indent=2)


@socketio.on("connect")
def on_connect():
    with _pool_lock:
        task_pools[request.sid] = {}


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    with _pool_lock:
        pool = task_pools.pop(sid, {})
    for engine in pool.values():
        try:
            engine.running = False
            engine.stop_subconscious()
        except Exception as e:
            _log_error("socket", f"Cleanup error on disconnect: {e}")


@socketio.on("submit_task")
def on_submit_task(data):
    task = data.get("task", "").strip()
    if not task:
        return

    sid = request.sid
    task_id = _next_task_id()
    workspace = app.config["WORKSPACE"]
    config = app.config["SYNAPSE_CONFIG"]

    engine = AgentEngine(workspace, config)
    engine.task_id = task_id
    engine.set_socketio(socketio, sid)

    with _pool_lock:
        if sid not in task_pools:
            task_pools[sid] = {}
        task_pools[sid][task_id] = engine

    # Notify UI about the new task
    socketio.emit("task_started", {
        "task_id": task_id,
        "task": task,
        "running_count": len(task_pools.get(sid, {})),
    }, to=sid)

    thread = threading.Thread(
        target=_run_task, args=(engine, task, task_id, sid), daemon=True
    )
    thread.start()


@socketio.on("cancel_task")
def on_cancel_task(data):
    sid = request.sid
    task_id = data.get("task_id", "")
    with _pool_lock:
        pool = task_pools.get(sid, {})
        engine = pool.get(task_id)
    if engine:
        engine.running = False
        engine.stop_subconscious()
        socketio.emit("task_cancelled", {"task_id": task_id}, to=sid)


def _run_task(engine, task, task_id, sid):
    """Background thread running a complete task — isolated from other tasks."""
    # Wrap all emissions to include task_id
    original_emit = engine.emit

    def tagged_emit(event, data):
        if isinstance(data, dict):
            data["task_id"] = task_id
        original_emit(event, data)

    engine.emit = tagged_emit
    _task_start_time = time.time()
    _task_success = True
    _task_error = None
    _task_type = "unknown"

    try:
        engine.emit("status", {"agent": "system", "status": "classifying", "task": task})
        _task_type = engine.classify_task(task)
        task_type = _task_type
        engine.emit("task_type", {"type": task_type})

        if task_type == "question":
            engine.emit("status", {"agent": "system", "status": "answering"})
            engine.answer_question(task)
            engine.emit("task_complete", {"type": "question"})
            return

        # Planning
        engine.emit("status", {"agent": "system", "status": "planning"})
        plan = engine.create_plan(task)

        if plan and (plan.get("steps") or plan.get("stages")):
            engine.emit("plan", plan)
            engine.plan_event.wait(timeout=600)
            if not engine.plan_approved:
                engine.emit("status", {"agent": "system", "status": "rejected"})
                engine.emit("task_complete", {"type": "rejected"})
                return
        else:
            plan = {"steps": [{"step": 1, "title": "Build", "details": task}]}

        # Build (blocking — _run_task is already in a background thread)
        engine.emit("status", {"agent": "system", "status": "building"})
        engine.start_build(task, plan, blocking=True)
    except Exception as e:
        _task_success = False
        _task_error = str(e)
        engine.emit("error", {"agent": "system", "error": f"Task failed: {e}"})
        engine.emit("task_complete", {"type": "error"})
    finally:
        duration = time.time() - _task_start_time
        # Clean up from pool
        with _pool_lock:
            pool = task_pools.get(sid, {})
            pool.pop(task_id, None)
        socketio.emit("task_finished", {
            "task_id": task_id,
            "running_count": len(task_pools.get(sid, {})),
        }, to=sid)
        # Metacognition: self-grade this task (async to not block)
        try:
            threading.Thread(
                target=_metacognition_grade,
                args=(task, _task_type, duration, _task_success, _task_error),
                daemon=True
            ).start()
        except Exception:
            pass


@socketio.on("approve_plan")
def on_approve_plan(data):
    sid = request.sid
    task_id = data.get("task_id", "")
    with _pool_lock:
        pool = task_pools.get(sid, {})
        engine = pool.get(task_id)
    if engine:
        engine.plan_approved = data.get("approved", False)
        engine.plan_event.set()


@socketio.on("spawn_terminal")
def on_spawn_terminal(data):
    if term_mgr:
        term_mgr.spawn(
            data.get("name", "Terminal"),
            data.get("cmd", ""),
            app.config["WORKSPACE"],
            request.sid,
        )


@socketio.on("kill_terminal")
def on_kill_terminal(data):
    if term_mgr:
        term_mgr.kill(data.get("id", ""))


# ── Settings API ─────────────────────────────────────────────────

@app.route("/api/settings", methods=["GET"])
def get_settings():
    cfg = app.config.get("SYNAPSE_CONFIG", load_config(
        os.path.dirname(os.path.abspath(__file__))
    ))
    safe = json.loads(json.dumps(cfg))
    for prov in safe.get("providers", {}).values():
        key = prov.get("api_key", "")
        if key and len(key) > 4:
            prov["api_key_masked"] = "•" * (len(key) - 4) + key[-4:]
            prov["has_key"] = True
        else:
            prov["api_key_masked"] = ""
            prov["has_key"] = bool(key)
        prov.pop("api_key", None)
    return json.dumps({"settings": safe, "provider_models": PROVIDER_MODELS})


@app.route("/api/settings", methods=["POST"])
def post_settings():
    data = request.get_json()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(base_dir)

    if "providers" in data:
        for prov_id, prov_data in data["providers"].items():
            if prov_id not in cfg["providers"]:
                cfg["providers"][prov_id] = {}
            if "api_key" in prov_data and prov_data["api_key"] \
                    and not prov_data["api_key"].startswith("•"):
                cfg["providers"][prov_id]["api_key"] = prov_data["api_key"]
            for k in ("enabled", "base_url", "label"):
                if k in prov_data:
                    cfg["providers"][prov_id][k] = prov_data[k]

    if "cortex_map" in data:
        cfg["cortex_map"] = data["cortex_map"]

    save_config(cfg, base_dir)
    app.config["SYNAPSE_CONFIG"] = cfg

    # Reset provider clients so new keys take effect
    with _pool_lock:
        for pool in task_pools.values():
            for engine in pool.values():
                engine.cortex.reset_clients()

    return json.dumps({"status": "ok"})


@app.route("/api/models")
def get_models():
    return json.dumps(PROVIDER_MODELS)


# ── Webhook / Event-Driven API ──────────────────────────────────

_webhook_secret = os.environ.get("WEBHOOK_SECRET", "")
_cron_jobs = []  # List of {"schedule": "*/5 * * * *", "task": "...", "enabled": True}
_cron_thread = None


@app.route("/api/webhook", methods=["POST"])
def webhook_handler():
    """Accept external triggers: GitHub webhooks, Slack, custom events."""
    data = request.get_json(silent=True) or {}

    # Verify webhook secret if set
    if _webhook_secret:
        sig = request.headers.get("X-Hub-Signature-256", "")
        if _webhook_secret not in sig and data.get("secret") != _webhook_secret:
            return json.dumps({"error": "unauthorized"}), 401

    source = data.get("source", "unknown")
    event_type = request.headers.get("X-GitHub-Event", data.get("event", "custom"))

    # GitHub webhook: new issue → auto-task
    if event_type == "issues" and data.get("action") == "opened":
        issue = data.get("issue", {})
        task = f"GitHub Issue #{issue.get('number')}: {issue.get('title', '')}\n{issue.get('body', '')[:500]}"
        _spawn_webhook_task(task, f"github-issue-{issue.get('number')}")
        return json.dumps({"status": "task_created", "source": "github_issue"})

    # GitHub webhook: PR opened → auto-review
    if event_type == "pull_request" and data.get("action") == "opened":
        pr = data.get("pull_request", {})
        task = (
            f"Review PR #{pr.get('number')}: {pr.get('title', '')}\n"
            f"Repo: {pr.get('base', {}).get('repo', {}).get('full_name', '')}\n"
            f"Changes: {pr.get('body', '')[:500]}\n"
            f"Review code quality, security, and suggest improvements."
        )
        _spawn_webhook_task(task, f"github-pr-{pr.get('number')}")
        return json.dumps({"status": "review_created", "source": "github_pr"})

    # Generic webhook: custom task
    if "task" in data:
        _spawn_webhook_task(data["task"], f"webhook-{int(time.time())}")
        return json.dumps({"status": "task_created", "source": source})

    # Slack integration
    if data.get("type") == "event_callback":
        event = data.get("event", {})
        if event.get("type") == "message" and not event.get("bot_id"):
            text = event.get("text", "")
            if text.startswith("@synapse") or text.startswith("!synapse"):
                task = text.replace("@synapse", "").replace("!synapse", "").strip()
                _spawn_webhook_task(task, f"slack-{int(time.time())}")
                return json.dumps({"status": "task_created", "source": "slack"})

    return json.dumps({"status": "received", "event": event_type})


@app.route("/api/webhook/tasks", methods=["GET"])
def list_webhook_tasks():
    """List recently triggered webhook tasks."""
    return json.dumps({"active_tasks": len(task_pools), "cron_jobs": len(_cron_jobs)})


@app.route("/api/cron", methods=["GET", "POST", "DELETE"])
def cron_handler():
    """Manage scheduled/cron tasks."""
    if request.method == "GET":
        return json.dumps({"jobs": _cron_jobs})

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        job = {
            "id": f"cron-{int(time.time())}",
            "schedule": data.get("schedule", "0 */6 * * *"),
            "task": data.get("task", ""),
            "enabled": data.get("enabled", True),
            "last_run": None,
        }
        _cron_jobs.append(job)
        _ensure_cron_thread()
        return json.dumps({"status": "created", "job": job})

    if request.method == "DELETE":
        data = request.get_json(silent=True) or {}
        job_id = data.get("id", "")
        _cron_jobs[:] = [j for j in _cron_jobs if j["id"] != job_id]
        return json.dumps({"status": "deleted"})


def _spawn_webhook_task(task_text, task_id):
    """Spawn a task from webhook (no browser session needed)."""
    workspace = app.config.get("WORKSPACE", "./workspace")
    config = app.config.get("SYNAPSE_CONFIG", load_config("."))
    engine = AgentEngine(workspace, config)
    engine.task_id = task_id
    engine.set_socketio(socketio, None)

    def run():
        try:
            task_type = engine.classify_task(task_text)
            if task_type == "question":
                engine.answer_question(task_text)
            else:
                plan = engine.create_plan(task_text)
                if not plan:
                    plan = {"steps": [{"step": 1, "title": "Build", "details": task_text}]}
                engine.start_build(task_text, plan, blocking=True)
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


def _ensure_cron_thread():
    """Start the cron scheduler thread if not running."""
    global _cron_thread
    if _cron_thread and _cron_thread.is_alive():
        return

    def _cron_loop():
        while True:
            now = datetime.now()
            for job in _cron_jobs:
                if not job.get("enabled"):
                    continue
                if _cron_should_run(job, now):
                    job["last_run"] = now.isoformat()
                    _spawn_webhook_task(job["task"], job["id"])
            socketio.sleep(60)

    socketio.start_background_task(_cron_loop)


def _cron_should_run(job, now):
    """Simple cron check — runs at matching hour/minute intervals."""
    schedule = job.get("schedule", "")
    last_run = job.get("last_run")
    if last_run:
        last = datetime.fromisoformat(last_run)
        if (now - last).total_seconds() < 55:
            return False
    parts = schedule.split()
    if len(parts) < 5:
        return False
    minute, hour = parts[0], parts[1]
    if minute != "*" and minute.startswith("*/"):
        interval = int(minute[2:])
        if now.minute % interval != 0:
            return False
    elif minute != "*" and now.minute != int(minute):
        return False
    if hour != "*" and hour.startswith("*/"):
        interval = int(hour[2:])
        if now.hour % interval != 0:
            return False
    elif hour != "*" and now.hour != int(hour):
        return False
    return True


# ── Vision / Image Upload API ───────────────────────────────────

@app.route("/api/vision", methods=["POST"])
def vision_analyze():
    """Analyze uploaded image using Gemini Vision."""
    if "image" not in request.files:
        return json.dumps({"error": "no image provided"}), 400

    file = request.files["image"]
    prompt = request.form.get("prompt", "Describe this image in detail.")
    image_data = file.read()
    base64.b64encode(image_data).decode("utf-8")

    config = app.config.get("SYNAPSE_CONFIG", load_config("."))
    cortex = NeuralCortex(config)

    try:
        result = cortex.quick_generate(
            "visual",
            prompt,
            system_prompt="You are a vision AI. Analyze images precisely.",
        )
        return json.dumps({"analysis": result, "size": len(image_data)})
    except Exception as e:
        return json.dumps({"error": str(e)}), 500


@app.route("/api/memory", methods=["GET"])
def memory_stats():
    """Get memory statistics."""
    workspace = app.config.get("WORKSPACE", "./workspace")
    mem = get_memory(workspace)
    return json.dumps({
        "count": mem.count(),
        "available": _chromadb_available,
    })


@app.route("/api/memory/search", methods=["POST"])
def memory_search():
    """Search long-term memory."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    if not query:
        return json.dumps({"results": []})
    workspace = app.config.get("WORKSPACE", "./workspace")
    mem = get_memory(workspace)
    results = mem.recall(query, n=data.get("limit", 5))
    return json.dumps({"results": results})


# ── A2A Protocol Endpoints ──────────────────────────────────────

@app.route("/.well-known/agent.json", methods=["GET"])
def a2a_agent_card():
    """Serve the A2A Agent Card for discovery."""
    base_url = request.host_url.rstrip("/")
    a2a.base_url = base_url
    card = a2a.get_agent_card()
    return json.dumps(card), 200, {"Content-Type": "application/json"}


@app.route("/a2a", methods=["POST"])
def a2a_endpoint():
    """Main A2A JSON-RPC endpoint for task exchange."""
    data = request.get_json(silent=True) or {}

    method = data.get("method", "")
    params = data.get("params", {})
    rpc_id = data.get("id", str(_uuid.uuid4()))

    # ── tasks/send: Accept a new task from a remote agent
    if method == "tasks/send":
        msg = params.get("message", {})
        parts = msg.get("parts", [])
        task_text = ""
        for p in parts:
            if p.get("type") == "text":
                task_text += p.get("text", "")

        if not task_text:
            return json.dumps({
                "jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32602, "message": "No text content in message"},
            }), 400

        caller = msg.get("metadata", {}).get("agent", "remote-agent")
        context_id = params.get("contextId") or params.get("id")
        task = a2a.create_task(task_text, caller_agent=caller, context_id=context_id)
        task_id = task["id"]

        # Execute task asynchronously via SYNAPSE engine
        def _run_a2a_task():
            a2a.update_task_status(task_id, "working", "SYNAPSE is processing your task...")
            try:
                workspace = app.config.get("WORKSPACE", "./workspace")
                config = app.config.get("SYNAPSE_CONFIG", {})
                engine = AgentEngine(workspace, config)
                engine.set_socketio(socketio, None)

                plan = engine._classify_and_plan(task_text)
                engine.start_build(task_text, plan, blocking=True)

                # Collect output files as artifacts
                for fpath in engine.files_created:
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()[:10000]
                        a2a.add_artifact(task_id, os.path.basename(fpath), content)
                    except Exception:
                        pass

                a2a.update_task_status(task_id, "completed",
                                       f"Task completed. {len(engine.files_created)} files created.")
            except Exception as e:
                a2a.update_task_status(task_id, "failed", f"Error: {str(e)}")

        threading.Thread(target=_run_a2a_task, daemon=True).start()

        return json.dumps({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": task,
        })

    # ── tasks/get: Check task status
    if method == "tasks/get":
        task_id = params.get("id", "")
        task = a2a.get_task(task_id)
        if not task:
            return json.dumps({
                "jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32001, "message": "Task not found"},
            }), 404
        return json.dumps({"jsonrpc": "2.0", "id": rpc_id, "result": task})

    # ── tasks/cancel: Cancel a running task
    if method == "tasks/cancel":
        task_id = params.get("id", "")
        task = a2a.cancel_task(task_id)
        if not task:
            return json.dumps({
                "jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32001, "message": "Task not found"},
            }), 404
        return json.dumps({"jsonrpc": "2.0", "id": rpc_id, "result": task})

    return json.dumps({
        "jsonrpc": "2.0", "id": rpc_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }), 404


@app.route("/api/a2a/agents", methods=["GET"])
def a2a_list_agents():
    """List all connected remote A2A agents."""
    return json.dumps({"agents": a2a.list_remote_agents()})


@app.route("/api/a2a/discover", methods=["POST"])
def a2a_discover():
    """Discover and register a remote A2A agent by URL."""
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return json.dumps({"error": "url required"}), 400
    result = a2a.discover_remote_agent(url)
    return json.dumps(result)


@app.route("/api/a2a/send", methods=["POST"])
def a2a_send_task():
    """Send a task to a remote agent."""
    data = request.get_json(silent=True) or {}
    agent_id = data.get("agent_id", "")
    task_text = data.get("task", "")
    if not agent_id or not task_text:
        return json.dumps({"error": "agent_id and task required"}), 400
    result = a2a.send_task_to_remote(agent_id, task_text)
    return json.dumps(result)


@app.route("/api/a2a/tasks", methods=["GET"])
def a2a_list_tasks():
    """List all A2A tasks."""
    with a2a._lock:
        tasks = [
            {"id": t["id"], "status": t["status"], "metadata": t.get("metadata", {})}
            for t in a2a.tasks.values()
        ]
    return json.dumps({"tasks": tasks})


# ── Module-level init for gunicorn (Cloud Run) ──────────────────

def _init_app_from_env():
    """Initialize app config from env vars when imported by gunicorn."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(base_dir)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        config["providers"]["gemini"]["api_key"] = api_key
        config["providers"]["gemini"]["enabled"] = True
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if gh_token:
        if "github" not in config["providers"]:
            config["providers"]["github"] = {"api_key": "", "enabled": False}
        config["providers"]["github"]["api_key"] = gh_token
        config["providers"]["github"]["enabled"] = True
    workspace = os.path.abspath(os.environ.get("WORKSPACE", "./workspace"))
    os.makedirs(workspace, exist_ok=True)
    app.config["SYNAPSE_CONFIG"] = config
    app.config["WORKSPACE"] = workspace

# Auto-init when loaded by gunicorn
if "SYNAPSE_CONFIG" not in app.config:
    _init_app_from_env()


# ── Entry Point ──────────────────────────────────────────────────

def main():
    global term_mgr

    parser = argparse.ArgumentParser(description="SYNAPSE — Neural Multi-Agent AI System")
    parser.add_argument("--workspace", default=os.environ.get("WORKSPACE", "./workspace"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="gemini-3-flash-preview")
    args = parser.parse_args()

    workspace = os.path.abspath(args.workspace)
    os.makedirs(workspace, exist_ok=True)

    # Load or create config
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(base_dir)

    # CLI api-key or env var overrides config for Gemini
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if api_key:
        config["providers"]["gemini"]["api_key"] = api_key
        config["providers"]["gemini"]["enabled"] = True
        save_config(config, base_dir)

    # GitHub token from env
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if gh_token:
        if "github" not in config["providers"]:
            config["providers"]["github"] = {"api_key": "", "enabled": False}
        config["providers"]["github"]["api_key"] = gh_token
        config["providers"]["github"]["enabled"] = True

    has_any_key = any(
        p.get("api_key") and p.get("enabled")
        for p in config["providers"].values()
    )
    if not has_any_key:
        print("  ⚠ No API keys configured.")
        print(f"  Set GEMINI_API_KEY env var, use --api-key, or configure at http://localhost:{args.port}")
        print()

    app.config["SYNAPSE_CONFIG"] = config
    app.config["WORKSPACE"] = workspace

    term_mgr = TerminalManager(socketio)

    print()
    print("  ◈ ════════════════════════════════════════")
    print("  ◈  SYNAPSE — Neural Multi-Agent AI System")
    print("  ◈ ════════════════════════════════════════")
    print(f"  🌐 Open: http://localhost:{args.port}")
    print(f"  📁 Workspace: {workspace}")
    print("  🔑 Providers:")
    for pid, pcfg in config["providers"].items():
        status = "✓" if pcfg.get("api_key") and pcfg.get("enabled") else "✗"
        print(f"     {status} {pid}")
    print("  🧠 Neural Cortices:")
    cortex_map = config.get("cortex_map", {})
    for cid, cfg in CORTEX_MODELS.items():
        mapping = cortex_map.get(cid, {})
        prov = mapping.get("provider", "gemini")
        model = mapping.get("model", cfg["model"])
        print(f"     {cfg['label']}: {model} ({prov})")
    print()

    socketio.run(
        app,
        host="0.0.0.0",
        port=args.port,
        debug=False,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
