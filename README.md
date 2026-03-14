<div align="center">

# ◈ SYNAPSE

### Neural Multi-Agent AI System

*Two AI agents. Multiple AI brains. One self-evolving system.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Cloud Run](https://img.shields.io/badge/Cloud%20Run-Ready-4285F4?logo=googlecloud&logoColor=white)](#-cloud-run-deployment)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## What is SYNAPSE?

SYNAPSE is an **autonomous multi-agent AI system** where two specialized AI agents — an **Architect** and a **Developer** — collaborate in real-time to build software, answer questions, generate images, browse the web, interact with GitHub, and even **modify their own source code**.

Unlike single-agent chatbots, SYNAPSE operates as a **team**: the Architect plans and verifies while the Developer implements and tests. They communicate through structured turn-based messages, creating a feedback loop that produces better results than either agent alone.

### 🧠 Neural Architecture

SYNAPSE models its AI processing after the human brain — **four specialized cortices** activate based on task type:

| Cortex | Purpose | Default Model |
|--------|---------|--------------|
| ⚡ **Fast** | Classification, simple queries | gemini-3.1-flash-lite-preview |
| 🧠 **Reasoning** | Architecture, debugging, agent assignment | gemini-3.1-pro-preview |
| 🎨 **Creative** | Code generation, building apps | gemini-3-flash-preview |
| 👁 **Visual** | Image generation, UI mockups | gemini-3.1-flash-image-preview / DALL-E 3 |

**Agent assignment uses the TOP model** (Gemini 3.1 Pro) — the most powerful model classifies and routes tasks at the initial stage for maximum accuracy.

Each cortex can be mapped to **any provider and model** via the UI settings.

---

## ✨ Key Features

### 🔑 Multi-Provider AI
Configure multiple AI providers through the web UI — no code changes needed:
- **Google Gemini** — Gemini 3.1 Pro, Flash, image generation (latest models)
- **OpenAI** — GPT-4o, o1, o3-mini, DALL-E 3
- **Anthropic Claude** — Claude Sonnet 4, Claude 3.5, Claude 3 Opus
- **Any OpenAI-Compatible API** — Ollama (local), Groq, Together, Mistral, LM Studio
- **GitHub** — GitHub API token for repo management, PRs, issues

### 🤝 Dual-Agent Collaboration
Two agents work as a team:
- **Architect** — Plans, reviews, verifies with tests
- **Developer** — Implements, builds, fixes bugs
- Structured turn-based communication with full visibility

### 🌐 Web Crawling
Agents can **browse the web** in real-time:
- Fetch documentation, API specs, latest tech updates
- Read any URL and extract structured content
- Research libraries and best practices before coding
- Agents adapt by crawling for latest technology updates

### 🐙 GitHub Integration
Built-in GitHub API actions:
- **Clone** repositories into workspace
- **Push** code with auto-commit
- **Create repos**, PRs, and manage issues
- Configure GitHub token in Settings or via `GITHUB_TOKEN` env var

### ⚡ Parallel Multi-Tasking
Submit multiple tasks simultaneously — each runs on its own thread:
- Task tabs with independent workspaces
- Cancel individual tasks
- Real-time status per task

### 🧬 Self-Modification
The system can evolve its own code:
- Agents can modify `agent_ui.py` and `templates/index.html`
- Changes go through: **backup → validate → clone-test → swap → restart**
- **Clone testing**: spins up a copy of itself on port+100 to verify changes
- Automatic rollback after 5+ crashes
- `synapse.py` launcher is never modified (immortal supervisor)
- Disabled in Cloud Run mode (ephemeral containers)

### 🎨 Image Generation
Generate images directly from task prompts:
- Gemini Visual Cortex or DALL-E 3
- Images displayed inline in the UI
- Saved to workspace

### 🖥 Full Scripting Power
Agents can:
- Run shell commands, install packages
- Create and execute scripts (Python, Node.js, PowerShell, Batch)
- Spawn additional terminal processes
- Browse the web for live data
- Interact with GitHub repos
- Automate browsers with Playwright/Selenium
- Access any tool available on the host machine

### 🌊 Iridescent Cyber UI
A stunning web interface with:
- Dragonfly-wing iridescent color scheme
- Real-time neural cortex activity display
- Subconscious workspace monitoring pulses
- Three-panel layout (Architect / Communication / Developer)

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/bxf1001g/SYNAPSE.git
cd SYNAPSE
pip install -r requirements.txt
```

### 2. Configure

**Option A: Web UI (recommended)**
Just launch SYNAPSE and click the ⚙ gear icon to configure any provider — Gemini, OpenAI, Anthropic, GitHub, or any OpenAI-compatible API.

**Option B: Environment variables**
```bash
# Windows
set GEMINI_API_KEY=your-gemini-api-key
set GITHUB_TOKEN=your-github-token     # optional

# Linux/Mac
export GEMINI_API_KEY=your-gemini-api-key
export GITHUB_TOKEN=your-github-token   # optional
```

> **Note:** SYNAPSE supports **multiple AI providers simultaneously**. You can configure different models for different cortices — e.g., Claude for reasoning, GPT-4o for code generation, Gemini for fast classification. Add API keys for any combination via the Settings UI.

### 3. Launch

```bash
# With the self-evolving launcher (recommended)
python synapse.py --workspace ./myproject --port 5000

# Or directly (no self-modification support)
python agent_ui.py --workspace ./myproject --port 5000
```

### 4. Open

Navigate to **http://localhost:5000** and start giving tasks!

---

## ☁ Cloud Run Deployment

SYNAPSE is **Cloud Run ready** with WebSocket support.

### Deploy with gcloud

```bash
# Build and deploy
gcloud run deploy synapse \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GEMINI_API_KEY=your-key,GITHUB_TOKEN=your-token" \
  --session-affinity \
  --timeout 300 \
  --concurrency 10 \
  --min-instances 1 \
  --use-http2=false
```

### Deploy with Docker

```bash
# Build locally
docker build -t synapse .
docker run -p 8080:8080 \
  -e GEMINI_API_KEY=your-key \
  -e GITHUB_TOKEN=your-token \
  synapse
```

### Cloud Run Notes
- WebSocket connections supported (up to 60 min timeout)
- Self-modification is **disabled** in cloud mode (ephemeral containers)
- Session affinity enabled for sticky WebSocket connections
- Uses gunicorn + eventlet for production async support
- Set `SYNAPSE_CLOUD_MODE=1` automatically via Dockerfile

---

## 📁 Project Structure

```
SYNAPSE/
├── synapse.py          # Immortal launcher/supervisor
├── agent_ui.py         # Core: Neural cortex, agents, web server
├── templates/
│   └── index.html      # Iridescent cyber UI (single-page app)
├── Dockerfile          # Cloud Run / Docker deployment
├── .dockerignore
├── ai_agent.py         # Original CLI dual-agent version
├── agent.py            # Manual TCP chat agent
├── protocol.py         # TCP framing protocol
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

### Settings UI
Click **⚙** in the header to open settings:
1. Enable providers and paste API keys (Gemini, OpenAI, Anthropic, GitHub)
2. Map each cortex to your preferred provider + model
3. Click Save — takes effect immediately

### Config File
Settings are stored in `.synapse.json` (auto-generated, git-ignored):
```json
{
  "providers": {
    "gemini": { "api_key": "...", "enabled": true },
    "openai": { "api_key": "...", "enabled": true },
    "anthropic": { "api_key": "...", "enabled": false },
    "github": { "api_key": "...", "enabled": true },
    "openai_compatible": { "base_url": "http://localhost:11434/v1", "enabled": false }
  },
  "cortex_map": {
    "fast": { "provider": "gemini", "model": "gemini-3.1-flash-lite-preview" },
    "reason": { "provider": "gemini", "model": "gemini-3.1-pro-preview" },
    "create": { "provider": "gemini", "model": "gemini-3-flash-preview" },
    "visual": { "provider": "gemini", "model": "gemini-3.1-flash-image-preview" }
  }
}
```

### CLI Arguments
```
python synapse.py [options]

--workspace PATH    Project workspace directory (default: ./workspace)
--port PORT         Web UI port (default: 8080)
--api-key KEY       Gemini API key (overrides env var)
--model MODEL       Default model (default: gemini-3-flash-preview)
```

### Environment Variables
```
GEMINI_API_KEY       Google Gemini API key
GITHUB_TOKEN         GitHub personal access token
PORT                 Server port (used by Cloud Run)
WORKSPACE            Workspace directory path
SYNAPSE_CLOUD_MODE   Set to "1" to disable self-modification
```

---

## 🆚 How SYNAPSE Differs from MoltBot & OpenClaw

| Feature | SYNAPSE | MoltBot | OpenClaw |
|---------|---------|---------|----------|
| **Architecture** | Dual-agent (Architect + Developer) | Single agent | Single agent |
| **AI Models** | Gemini 3.1 Pro + OpenAI + Claude + Custom | Model-agnostic | Model-agnostic |
| **Neural Routing** | 4-cortex brain with TOP model assignment | Single model per task | Single model |
| **Self-Modification** | ✅ Clone-test → swap → auto-rollback | ❌ | ❌ |
| **Web Crawling** | ✅ Browse any URL, fetch docs/APIs | ❌ | ❌ |
| **GitHub Integration** | ✅ Clone, push, PRs, issues | ❌ | ❌ |
| **Cloud Deployment** | ✅ Cloud Run ready with Dockerfile | Manual | Manual |
| **Parallel Tasks** | ✅ Thread pool with task tabs | Limited | Limited |
| **Image Generation** | ✅ Gemini 3.1 + DALL-E 3 | ❌ | ❌ |
| **Agent Collaboration** | ✅ Turn-based Architect↔Developer | N/A | N/A |
| **Self-Evolving Launcher** | ✅ Backup→Validate→Clone-test→Swap | ❌ | ❌ |
| **UI** | Iridescent cyber web UI | Web/CLI | Web/CLI |

**SYNAPSE's unique advantages:**
- 🧠 **Multi-model brain** — Different tasks automatically route to the best model (Gemini 3.1 Pro for reasoning)
- 🤝 **Two-agent collaboration** — Plans are verified, code is tested before delivery
- 🧬 **Self-evolution** — The system improves itself through safe clone-tested code modification
- 🌐 **Web-connected** — Agents browse the web for docs, APIs, and latest tech
- 🐙 **GitHub-native** — Clone, push, PRs, issues directly from agent actions
- ☁ **Cloud-ready** — Deploy to Cloud Run with one command

---

## 💡 Example Tasks

```
"Build a Flask REST API with user authentication"
"Create a React todo app with local storage"
"What files are in my workspace?"
"Debug why my Python script crashes on line 42"
"Generate a logo for my project"
"Browse https://docs.python.org and summarize new features"
"Clone https://github.com/user/repo and add tests"
"Build a web scraper for weather data"
"Delete all .tmp files in my workspace"
```

---

## 🛡 Self-Modification Safety

When agents request code changes to themselves:

1. **Backup** — Current code is timestamped and preserved
2. **Validate** — New code is syntax-checked (`py_compile` for Python, size-check for HTML)
3. **Clone-Test** — New version spins up on port+100 for health check
4. **Swap** — Only if healthy, files are atomically swapped
5. **Restart** — System restarts with new code
6. **Rollback** — If 5+ rapid crashes, auto-reverts to last working version

The launcher (`synapse.py`) is **never modified** — it's the immortal anchor.

> **Cloud Run:** Self-modification is automatically disabled in cloud mode since containers are ephemeral.

---

## 📄 License

MIT License — use freely, modify, distribute.

---

<div align="center">

*Built with neural connections between human creativity and AI capability.*

**◈ SYNAPSE**

</div>
