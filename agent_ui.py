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


# ── Neural Model Configuration ──────────────────────────────────

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

        self.plan_event = threading.Event()
        self.plan_approved = False

        self.log_path = os.path.join(self.workspace, ".conversation.log")

        self._sio = None
        self._sid = None

        # Neural cortex — multi-model, multi-provider brain
        self.cortex = NeuralCortex(config)
        self.active_cortex = "create"

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

    # ── Question Answering (full scripting power) ────────────

    def answer_question(self, question):
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

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"# Conversation Log — {datetime.now()}\n\n")

        arch_prompt = ARCHITECT_PROMPT.format(workspace=self.workspace)
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

        # Start subconscious monitoring
        self.start_subconscious()

        if blocking:
            # Run build synchronously (when already in a background thread)
            self._run_build(task, plan)
        else:
            thread = threading.Thread(
                target=self._run_build, args=(task, plan), daemon=True
            )
            thread.start()

    def _run_build(self, task, plan):
        plan_text = json.dumps(plan, indent=2)
        first_prompt = (
            f"Approved plan:\n\n{plan_text}\n\n"
            f"Send comprehensive instructions to Developer. "
            f"Combine steps. Include COMPLETE file contents. "
            f"Aim for 2-4 total turns."
        )

        try:
            arch_msg = self._process_turn("architect", self.arch_chat, first_prompt)

            while self.running and self.turn < self.max_turns:
                if arch_msg is None:
                    break

                if arch_msg.get("type") == "done":
                    self._cleanup()
                    self.emit(
                        "done",
                        {
                            "summary": arch_msg.get("body", "Complete"),
                            "files": sorted(set(self.files_created)),
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

                # Developer processes
                dev_prompt = (
                    f"Architect instructs:\n\n{arch_msg['body']}\n\n"
                    f"Implement: create files, run commands, test, report."
                )
                dev_msg = self._process_turn(
                    "developer", self.dev_chat, dev_prompt
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

    def _process_turn(self, agent, chat, input_text):
        """One turn: AI call → execute actions → iterate if needed."""
        self.turn += 1
        current_input = input_text
        peer_message = None
        is_done = False
        done_summary = ""

        # Emit which cortex is driving this agent
        cortex_id = "reason" if agent == "architect" else "create"
        cortex_info = CORTEX_MODELS[cortex_id]
        self.emit("cortex_active", {
            "id": cortex_id,
            "label": cortex_info["label"],
            "desc": f"{agent.title()} thinking...",
            "model": cortex_info["model"],
            "color": cortex_info["color"],
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
        self.emit("command_start", {"agent": agent, "cmd": cmd})
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
        """Handle self-modification: write signal file for synapse.py launcher."""
        # Cloud Run: self-modification disabled (ephemeral containers)
        if os.environ.get("SYNAPSE_CLOUD_MODE", "").strip() in ("1", "true"):
            self.emit("error", {"agent": agent, "error": "Self-modification disabled in cloud mode"})
            return "self_modify: disabled in cloud mode (ephemeral container)"

        files = action.get("files", [])
        reason = action.get("reason", "Agent-requested modification")
        if not files:
            self.emit("error", {"agent": agent, "error": "self_modify: no files specified"})
            return "self_modify: no files (nothing to do)"

        # Resolve file paths relative to the project root (where nexus.py lives)
        project_root = os.path.dirname(os.path.abspath(__file__))
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
except ImportError:
    _async_mode = "threading"

socketio = SocketIO(app, async_mode=_async_mode, cors_allowed_origins="*")

# Per-session task pools: sid → {task_id → AgentEngine}
task_pools = {}
_pool_lock = threading.Lock()
_task_counter = 0
term_mgr = None


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
        engine.running = False
        engine.stop_subconscious()


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
