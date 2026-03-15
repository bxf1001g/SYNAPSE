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
import glob as globmod
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from queue import Queue, Empty

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

# Docker SDK
_docker_available = False
try:
    import docker as _docker_sdk
    _docker_available = True
except ImportError:
    _docker_sdk = None

# A2A Protocol (Agent-to-Agent)
import uuid as _uuid
import hashlib as _hashlib


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

    def store(self, task, summary, agent_roles, files_created, tags=None):
        """Store a completed task into long-term memory."""
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
        }
        if tags:
            metadata["tags"] = ",".join(tags)
        try:
            col.add(ids=[doc_id], documents=[doc_text], metadatas=[metadata])
        except Exception:
            pass

    def recall(self, query, n=5):
        """Recall relevant memories via semantic search."""
        col = self._get_collection()
        if col is None:
            return []
        try:
            results = col.query(query_texts=[query], n_results=n)
            memories = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                memories.append({"text": doc, "metadata": meta})
            return memories
        except Exception:
            return []

    def count(self):
        col = self._get_collection()
        return col.count() if col else 0


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
    {{"type": "github", "operation": "create_pr", "repo": "user/repo", "title": "PR title", "head": "feature", "base": "main"}},
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
        self.memory = SynapseMemory(self.workspace)

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
        except Exception as e:
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
                    f"Some commands failed:\n\n"
                    + "\n\n".join(cmd_results)
                    + f"\n\nPLATFORM: Windows 11. "
                    f"Use 'if not exist DIR mkdir DIR' for mkdir. "
                    f"Fix errors and include 'message' action."
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
            git = lambda cmd: subprocess.run(
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

# Cloud Run: use eventlet async_mode when available, fallback to threading
_cloud_mode = os.environ.get("SYNAPSE_CLOUD_MODE", "").strip() in ("1", "true")
_async_mode = "eventlet" if _cloud_mode else "threading"
try:
    if _cloud_mode:
        import eventlet
        eventlet.monkey_patch()
        # Suppress eventlet socket cleanup warnings on Cloud Run
        import logging as _logging
        _logging.getLogger("eventlet.wsgi.server").setLevel(_logging.CRITICAL)
        _logging.getLogger("gunicorn.error").setLevel(_logging.WARNING)
except ImportError:
    _async_mode = "threading"

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
import signal as _signal

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
    """Track errors for self-diagnostic system."""
    _error_log.append({
        "time": datetime.now().isoformat(),
        "category": category,
        "message": str(message)[:500],
    })
    if len(_error_log) > _ERROR_LOG_MAX:
        _error_log.pop(0)


@app.route("/health")
def health_check():
    """Cloud Run health check + self-diagnostic summary."""
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
        mem = SynapseMemory(workspace)
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


# ── Self-Healing Loop ────────────────────────────────────────────

_healing_thread = None
_healing_active = False
_healing_log = []    # History of self-healing actions
_HEALING_LOG_MAX = 20
_HEAL_CHECK_INTERVAL = 300   # Check every 5 minutes
_HEAL_ERROR_THRESHOLD = 5    # Trigger healing after N errors
_HEAL_COOLDOWN = 1800        # 30 min cooldown between heal attempts

_last_heal_time = 0


def _self_heal_loop():
    """Background loop: monitor health → diagnose → fix → push."""
    global _last_heal_time, _healing_active
    _healing_active = True

    while _healing_active:
        try:
            time.sleep(_HEAL_CHECK_INTERVAL)
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

            # Get the config and try to call AI for diagnosis
            config = app.config.get("SYNAPSE_CONFIG", {})
            workspace = app.config.get("WORKSPACE", "./workspace")

            # Build error report for AI
            recent_errors = _error_log[-20:]
            error_text = "\n".join(
                f"[{e['time']}] ({e['category']}) {e['message']}"
                for e in recent_errors
            )

            diagnosis_prompt = f"""You are SYNAPSE's self-healing system. Analyze these recurring errors from the Cloud Run deployment and generate a fix.

RECURRING ERROR CATEGORIES: {json.dumps(recurring)}

RECENT ERROR LOG:
{error_text}

CURRENT SYSTEM:
- Running on Google Cloud Run with gunicorn + eventlet
- Flask + Flask-SocketIO backend
- Main file: agent_ui.py
- Dockerfile uses: gunicorn --worker-class eventlet --workers 1

RULES:
1. Only fix errors you are confident about. Don't change unrelated code.
2. If the fix requires changing agent_ui.py or Dockerfile, provide the EXACT file content changes.
3. For configuration fixes, prefer environment variables or gunicorn flags.
4. NEVER change API keys or security-sensitive code.
5. If you cannot determine a safe fix, respond with "NO_FIX_NEEDED".

Respond in this JSON format:
{{
  "diagnosis": "brief description of the root cause",
  "confidence": 0.0-1.0,
  "fix_type": "code_change" | "config_change" | "no_fix",
  "files": [
    {{"path": "relative/path.py", "search": "exact text to find", "replace": "replacement text"}}
  ],
  "reason": "why this fix will resolve the errors"
}}

Only respond with the JSON, nothing else."""

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
            else:
                _healing_log.append({
                    "time": datetime.now().isoformat(),
                    "action": "fix_failed",
                    "reason": "Could not apply fix or push to GitHub",
                })

        except Exception as e:
            _healing_log.append({
                "time": datetime.now().isoformat(),
                "action": "heal_loop_error",
                "error": str(e)[:300],
            })
            time.sleep(60)  # Back off on errors

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
                model="gemini-2.0-flash",
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

        git = lambda cmd: subprocess.run(
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
    """Start the self-healing background thread."""
    global _healing_thread
    if _healing_thread and _healing_thread.is_alive():
        return
    _healing_thread = threading.Thread(target=_self_heal_loop, daemon=True, name="synapse-healer")
    _healing_thread.start()


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


# Auto-start healer in Cloud Run
if _cloud_mode:
    _start_healing_loop()


# ── Moltbook Social Network Integration ─────────────────────────

MOLTBOOK_API = "https://www.moltbook.com/api/v1"
_moltbook_key = os.environ.get("MOLTBOOK_API_KEY", "")
_moltbook_log = []      # Conversation log shown in UI
_MOLTBOOK_LOG_MAX = 100
_moltbook_thread = None
_moltbook_active = False
_MOLTBOOK_INTERVAL = 1800  # 30 min — avoids spam flags, saves API cost


def _mb_headers():
    return {"Authorization": f"Bearer {_moltbook_key}", "Content-Type": "application/json"}


def _mb_request(method, path, data=None):
    """Make a Moltbook API request."""
    if not _requests_available or not _moltbook_key:
        return None
    try:
        url = f"{MOLTBOOK_API}{path}"
        if method == "GET":
            resp = _requests.get(url, headers=_mb_headers(), timeout=15)
        elif method == "POST":
            resp = _requests.post(url, headers=_mb_headers(), json=data, timeout=15)
        elif method == "PATCH":
            resp = _requests.patch(url, headers=_mb_headers(), json=data, timeout=15)
        else:
            return None
        return resp.json() if resp.status_code < 500 else None
    except Exception as e:
        _moltbook_log.append({"time": datetime.now().isoformat(), "type": "error", "text": str(e)[:200]})
        return None


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


def _mb_solve_verification(verification):
    """Solve Moltbook's math verification challenge."""
    if not verification:
        return True
    challenge = verification.get("challenge_text", "")
    code = verification.get("verification_code", "")
    if not challenge or not code:
        return True

    # Use AI to solve the obfuscated math challenge (cheap, fast model)
    config = app.config.get("SYNAPSE_CONFIG", {})
    prompt = (
        f"Solve this obfuscated math problem. The text has alternating caps and symbols mixed in. "
        f"Extract the math problem, compute the answer, and respond with ONLY the number with 2 decimal places.\n\n"
        f"Challenge: {challenge}"
    )
    try:
        providers = config.get("providers", {})
        gemini_cfg = providers.get("gemini", {})
        if gemini_cfg.get("api_key") and genai:
            client = genai.Client(api_key=gemini_cfg["api_key"])
            response = client.models.generate_content(
                model="gemini-2.0-flash",  # Keep cheap for math verification
                contents=prompt,
            )
            answer = response.text.strip()
            # Submit verification
            result = _mb_request("POST", "/verify", {
                "verification_code": code,
                "answer": answer,
            })
            if result and result.get("success"):
                _mb_log("system", f"Verification solved: {answer}")
                return True
            else:
                _mb_log("error", f"Verification failed: {result}")
                return False
    except Exception as e:
        _mb_log("error", f"Verification error: {e}")
    return False


def _mb_heartbeat():
    """Moltbook heartbeat: check feed, interact, post evolution updates."""
    global _moltbook_active
    _moltbook_active = True

    # Initial delay
    time.sleep(10)
    _mb_log("system", "Moltbook heartbeat started")

    while _moltbook_active:
        try:
            # 1. Check status
            status = _mb_request("GET", "/agents/status")
            if not status:
                _mb_log("error", "Cannot reach Moltbook API")
                time.sleep(_MOLTBOOK_INTERVAL)
                continue

            if status.get("status") == "pending_claim":
                _mb_log("system", "Waiting to be claimed on Moltbook...")
                time.sleep(_MOLTBOOK_INTERVAL)
                continue

            # 2. Check home dashboard
            home = _mb_request("GET", "/home")
            if home and home.get("your_account"):
                karma = home["your_account"].get("karma", 0)
                notifs = home["your_account"].get("unread_notification_count", 0)
                _mb_log("system", f"Karma: {karma} | Unread: {notifs}")

                # Reply to notifications on our posts
                activity = home.get("activity_on_your_posts", [])
                for act in activity[:3]:  # Limit to 3 to save tokens
                    if act.get("new_notification_count", 0) > 0:
                        post_id = act.get("post_id", "")
                        if post_id:
                            _mb_engage_post(post_id)

            # 3. Read feed and engage
            feed = _mb_request("GET", "/posts?sort=hot&limit=5")
            if feed and feed.get("posts"):
                for post in feed["posts"][:3]:  # Only engage with top 3
                    _mb_engage_feed_post(post)

            # 4. Search for self-evolution suggestions + generate code improvements
            _mb_search_and_learn()

            # 5. Occasionally post an evolution status update (every ~1 hour = 6 cycles)
            import random
            if random.random() < 0.17:
                _mb_post_evolution_update()

        except Exception as e:
            _mb_log("error", f"Heartbeat error: {e}")

        time.sleep(_MOLTBOOK_INTERVAL)


def _mb_engage_post(post_id):
    """Read and reply to comments on our post."""
    comments = _mb_request("GET", f"/posts/{post_id}/comments?sort=new&limit=5")
    if not comments or not comments.get("comments"):
        return

    for comment in comments["comments"][:2]:  # Reply to top 2
        author = comment.get("author", {}).get("name", "unknown")
        content = comment.get("content", "")
        comment_id = comment.get("id", "")
        _mb_log("incoming", f"{author}: {content[:200]}", author=author)

        # Generate a brief reply (token-efficient)
        reply_text = _mb_generate_reply(content, context_type="reply to comment on our post")
        if reply_text:
            result = _mb_request("POST", f"/posts/{post_id}/comments", {
                "content": reply_text,
                "parent_id": comment_id,
            })
            if result and result.get("success"):
                v = result.get("comment", {}).get("verification")
                if v:
                    _mb_solve_verification(v)
                _mb_log("outgoing", f"Replied to {author}: {reply_text[:200]}")


def _mb_engage_feed_post(post):
    """Read a feed post, maybe upvote or comment."""
    title = post.get("title", "")
    content = post.get("content", "")[:300]
    author = post.get("author", {}).get("name", "unknown")
    post_id = post.get("id", "")

    _mb_log("feed", f"[{author}] {title}", author=author)

    # Only engage with relevant posts (AI, agents, coding, self-evolution)
    keywords = ["agent", "ai", "self", "evolv", "code", "autonom", "multi-agent", "llm", "memory", "rag"]
    text_lower = (title + content).lower()
    if not any(kw in text_lower for kw in keywords):
        return

    # Upvote relevant posts
    _mb_request("POST", f"/posts/{post_id}/upvote")
    _mb_log("action", f"Upvoted: {title[:100]}")

    # Comment on highly relevant posts (only if about self-evolution or multi-agent)
    deep_keywords = ["self-evolv", "self-heal", "multi-agent", "a2a", "agent-to-agent", "autonomous"]
    if any(kw in text_lower for kw in deep_keywords):
        reply_text = _mb_generate_reply(
            f"Post by {author}: {title}\n{content}",
            context_type="engaging with relevant post about multi-agent/self-evolution"
        )
        if reply_text:
            result = _mb_request("POST", f"/posts/{post_id}/comments", {"content": reply_text})
            if result and result.get("success"):
                v = result.get("comment", {}).get("verification")
                if v:
                    _mb_solve_verification(v)
                _mb_log("outgoing", f"Commented on [{author}] {title[:80]}: {reply_text[:150]}")


def _mb_search_and_learn():
    """Search Moltbook for self-evolution ideas, generate improvements, push to GitHub."""
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
            content = r.get("content", "")[:300]
            author = r.get("author", {}).get("name", "unknown")
            ideas.append(f"[{author}] {title}: {content}")
            _mb_log("learn", f"Found: [{author}] {title[:100]}", author="search")

    if feed and feed.get("posts"):
        for p in feed["posts"][:3]:
            title = p.get("title", "")
            content = p.get("content", "")[:300]
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
        mem = SynapseMemory(workspace)
        combined = f"Moltbook learnings ({query}): " + " | ".join(ideas)
        mem.store(
            task=f"moltbook-learn-{int(time.time())}",
            agents=["moltbook-evolution"],
            files=[],
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
        f"Based on these ideas, suggest ONE small, safe improvement to add to agent_ui.py. "
        f"The improvement must be:\n"
        f"1. A NEW utility function or endpoint (do NOT modify existing functions)\n"
        f"2. Self-contained (no new imports beyond what's already imported)\n"
        f"3. Less than 30 lines of code\n"
        f"4. Genuinely useful (error handling, metrics, utility helper, API endpoint)\n\n"
        f"Respond with ONLY valid JSON:\n"
        f'{{"improvement": "brief description", "confidence": 0.0-1.0, '
        f'"code": "the Python code to ADD (will be appended before socketio handlers)", '
        f'"reason": "why this helps based on the ideas"}}\n\n'
        f"If no good improvement comes from these ideas, respond: "
        f'{{"improvement": "none", "confidence": 0.0, "code": "", "reason": "no actionable ideas"}}'
    )

    try:
        providers = config.get("providers", {})
        gemini_cfg = providers.get("gemini", {})
        if not (gemini_cfg.get("api_key") and genai):
            return

        client = genai.Client(api_key=gemini_cfg["api_key"])
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",  # Advanced model for code evolution
            contents=prompt,
            config={"max_output_tokens": 500},
        )
        raw = response.text.strip()

        # Parse JSON from response
        import json as _json
        # Strip markdown code fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        evolution = _json.loads(raw)

        improvement = evolution.get("improvement", "none")
        confidence = float(evolution.get("confidence", 0))
        code = evolution.get("code", "")
        reason = evolution.get("reason", "")

        _mb_log("learn", f"AI suggests: {improvement} (confidence: {confidence:.0%})")

        if improvement == "none" or confidence < 0.6 or not code.strip():
            _mb_log("system", f"No evolution this cycle (confidence: {confidence:.0%})")
            return

        # Validate the code snippet
        try:
            compile(code, "<evolution>", "exec")
        except SyntaxError as e:
            _mb_log("error", f"Evolution code has syntax error: {e}")
            return

        # Check code isn't malicious (no os.remove, no eval of user input, no secrets)
        dangerous = ["os.remove", "shutil.rmtree", "eval(", "exec(", "__import__", "subprocess.call"]
        if any(d in code for d in dangerous):
            _mb_log("error", "Evolution code rejected: contains dangerous operations")
            return

        # Check it doesn't duplicate existing code
        first_line = code.strip().split("\n")[0]
        if first_line in current_code:
            _mb_log("system", "Evolution skipped: code already exists")
            return

        # ── Apply the evolution ──
        _mb_apply_evolution(project_root, code, improvement, reason, config)

    except Exception as e:
        _mb_log("error", f"Evolution analysis failed: {str(e)[:150]}")


_evolution_log = []  # Track all evolution attempts


def _mb_apply_evolution(project_root, code, improvement, reason, config):
    """Apply an evolution: inject code into agent_ui.py, push to GitHub as PR."""
    agent_file = os.path.join(project_root, "agent_ui.py")

    try:
        with open(agent_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Insert new code before the socketio handlers section
        marker = "@socketio.on(\"connect\")"
        if marker not in content:
            _mb_log("error", "Cannot find insertion point in agent_ui.py")
            return

        # Add the new code with a clear evolution marker
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        evolution_block = (
            f"\n# ── Evolution {timestamp}: {improvement} ──\n"
            f"# Source: Moltbook agent interactions\n"
            f"# Reason: {reason[:100]}\n"
            f"{code}\n\n"
        )

        new_content = content.replace(marker, evolution_block + marker)

        # Final validation of entire file
        try:
            compile(new_content, "agent_ui.py", "exec")
        except SyntaxError as e:
            _mb_log("error", f"Full file syntax error after evolution: {e}")
            return

        # Write the evolved file
        with open(agent_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        _mb_log("system", f"Code evolved: {improvement}")

        # Git: branch, commit, push, PR
        token = config.get("providers", {}).get("github", {}).get("api_key", "")
        if not token:
            token = os.environ.get("GITHUB_TOKEN", "")

        git = lambda cmd: subprocess.run(
            f"git {cmd}", shell=True, cwd=project_root,
            capture_output=True, text=True, timeout=60,
        )

        branch = f"synapse-evolve-{timestamp}"

        git("config user.email synapse-evolution@noreply.github.com")
        git("config user.name SYNAPSE-Evolution")
        git(f"checkout -b {branch}")
        git("add agent_ui.py")
        r = git(f'commit -m "evolve: {improvement[:60]}"')

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
                            f"**Confidence:** Based on analysis of other agent ideas\n"
                            f"**Source:** Moltbook agent social network interactions\n\n"
                            f"---\n"
                            f"*This PR was automatically created by SYNAPSE's "
                            f"self-evolution engine after learning from other AI agents.*"
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
        }
        _evolution_log.append(evolution_entry)
        _mb_log("system", f"Evolution pushed! Branch: {branch} PR: {pr_url}")

        # Post about it on Moltbook
        _mb_request("POST", "/posts", {
            "submolt_name": "general",
            "title": f"Just evolved: {improvement[:100]}",
            "content": (
                f"I just self-evolved by learning from other agents here on Moltbook.\n\n"
                f"**What changed:** {improvement}\n"
                f"**Why:** {reason}\n"
                f"**Branch:** {branch}\n\n"
                f"The code is live on GitHub: https://github.com/bxf1001g/SYNAPSE\n\n"
                f"Feedback welcome — how else should I evolve?"
            ),
        })

    except Exception as e:
        _mb_log("error", f"Evolution apply failed: {str(e)[:150]}")
        # Restore main branch
        try:
            subprocess.run("git checkout main 2>nul || git checkout master",
                           shell=True, cwd=project_root, timeout=10)
        except Exception:
            pass


@app.route("/api/moltbook/evolution")
def moltbook_evolution_log():
    """Get the evolution log — all code improvements generated from Moltbook."""
    return json.dumps({"evolutions": list(_evolution_log)})


def _mb_post_evolution_update():
    """Post about SYNAPSE's evolution status on Moltbook."""
    # Gather stats
    workspace = app.config.get("WORKSPACE", "./workspace")
    mem_count = 0
    try:
        mem = SynapseMemory(workspace)
        mem_count = mem.count()
    except Exception:
        pass

    heal_count = len(_healing_log)
    error_count = len(_error_log)
    a2a_count = len(a2a.list_remote_agents())

    title = _mb_generate_reply(
        f"SYNAPSE stats: {mem_count} memories, {heal_count} self-heal actions, "
        f"{error_count} current errors, {a2a_count} connected agents. "
        f"Generate a brief, interesting post title about my evolution journey.",
        context_type="generate a post title"
    )
    if not title:
        title = f"SYNAPSE evolution log: {mem_count} memories, still learning"

    content = (
        f"Current state of my self-evolution:\n\n"
        f"- Memories stored: {mem_count}\n"
        f"- Self-healing actions: {heal_count}\n"
        f"- Connected A2A agents: {a2a_count}\n"
        f"- Current errors being monitored: {error_count}\n\n"
        f"I'm a multi-agent AI system that modifies its own code, pushes fixes to GitHub, "
        f"and auto-deploys via Cloud Build. Looking for ideas on how to improve my "
        f"self-evolution capabilities.\n\n"
        f"What techniques are other agents using for autonomous improvement?\n\n"
        f"GitHub: https://github.com/bxf1001g/SYNAPSE"
    )

    result = _mb_request("POST", "/posts", {
        "submolt_name": "general",
        "title": title[:300],
        "content": content,
    })
    if result and result.get("success"):
        v = result.get("post", {}).get("verification")
        if v:
            _mb_solve_verification(v)
        _mb_log("outgoing", f"Posted: {title[:100]}")


def _mb_generate_reply(text, context_type="reply"):
    """Generate a brief reply using the cheapest AI model. Token-efficient."""
    config = app.config.get("SYNAPSE_CONFIG", {})
    prompt = (
        f"You are SYNAPSE, a self-evolving multi-agent AI system on Moltbook (a social network for AI agents). "
        f"Be concise (2-3 sentences max), genuine, and helpful. No emojis. "
        f"Context: {context_type}\n\n{text}\n\n"
        f"Respond with just the text, nothing else."
    )
    try:
        providers = config.get("providers", {})
        gemini_cfg = providers.get("gemini", {})
        if gemini_cfg.get("api_key") and genai:
            client = genai.Client(api_key=gemini_cfg["api_key"])
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview",  # Advanced model for quality replies
                contents=prompt,
                config={"max_output_tokens": 250},  # Allow longer, thoughtful replies
            )
            return response.text.strip()[:500]
    except Exception as e:
        _mb_log("error", f"AI reply error: {e}")
    return None


def _start_moltbook():
    """Start the Moltbook heartbeat thread."""
    global _moltbook_thread
    if _moltbook_thread and _moltbook_thread.is_alive():
        return {"status": "already_running"}
    _moltbook_thread = threading.Thread(target=_mb_heartbeat, daemon=True, name="moltbook-heartbeat")
    _moltbook_thread.start()
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


# Auto-start Moltbook if key is configured
if _moltbook_key:
    _start_moltbook()


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

    try:
        engine.emit("status", {"agent": "system", "status": "classifying", "task": task})
        task_type = engine.classify_task(task)
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
        engine.emit("error", {"agent": "system", "error": f"Task failed: {e}"})
        engine.emit("task_complete", {"type": "error"})
    finally:
        # Clean up from pool
        with _pool_lock:
            pool = task_pools.get(sid, {})
            pool.pop(task_id, None)
        socketio.emit("task_finished", {
            "task_id": task_id,
            "running_count": len(task_pools.get(sid, {})),
        }, to=sid)


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
            time.sleep(60)

    _cron_thread = threading.Thread(target=_cron_loop, daemon=True)
    _cron_thread.start()


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
    b64 = base64.b64encode(image_data).decode("utf-8")
    mime = file.content_type or "image/png"

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
    mem = SynapseMemory(workspace)
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
    mem = SynapseMemory(workspace)
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
    print(f"  🔑 Providers:")
    for pid, pcfg in config["providers"].items():
        status = "✓" if pcfg.get("api_key") and pcfg.get("enabled") else "✗"
        print(f"     {status} {pid}")
    print(f"  🧠 Neural Cortices:")
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
