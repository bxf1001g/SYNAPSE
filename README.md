<div align="center">

<img src="assets/synapse-banner.svg" alt="SYNAPSE Banner" width="100%"/>

# ◈ SYNAPSE

### Self-Evolving Autonomous AI System

*It writes its own code. Reviews it with a second brain. Merges it. Learns from the outcome. Repeats.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Self-Evolving](https://img.shields.io/badge/Self--Evolving-4%20PRs%20Merged-ff6b6b)](#-autonomous-evolution-pipeline)
[![Local + Cloud](https://img.shields.io/badge/Runs-Local%20%7C%20Cloud-brightgreen)](#-quick-start)
[![Ollama](https://img.shields.io/badge/Ollama-Supported-orange?logo=ollama)](#-local-mode-ollama--jetson)
[![Cloud Run](https://img.shields.io/badge/Cloud%20Run-Ready-4285F4?logo=googlecloud&logoColor=white)](#-cloud-run-deployment)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## What is SYNAPSE?

SYNAPSE is an **autonomous self-evolving AI system** built on AGI-oriented principles — it doesn't just answer questions, it **improves itself continuously** without human intervention.

At its core, SYNAPSE operates as a **multi-agent team** (Architect, Developer, Researcher, Tester, Security, DevOps) with a **4-cortex neural architecture** that routes tasks to the right AI model. But what makes it fundamentally different from any chatbot is its **closed-loop self-evolution**:

> **Every hour**, SYNAPSE crawls 8 knowledge sources → generates a code improvement with Gemini 3.1 Pro → has a second AI independently review it → tests it in a sandbox → creates a GitHub PR → merges it → **records whether it worked or failed** → uses that outcome to make better decisions next time.

In its first 24 hours of operation, SYNAPSE autonomously wrote, reviewed, and merged **4 pull requests** into its own codebase — and fixed its own lint errors.

### 🔬 AGI-Oriented Capabilities

| Capability | How SYNAPSE Implements It |
|------------|--------------------------|
| **Self-Improvement** | Writes its own code, reviews with second AI, sandbox-tests, auto-merges PRs |
| **Learning from Experience** | Tracks every evolution outcome (success/failure) → feeds back into future decisions |
| **Multi-Source Knowledge** | Crawls Reddit, HackerNews, arXiv, GitHub Trending, Moltbook, Google Search, dream insights, error patterns |
| **Adaptive Behavior** | Confidence thresholds adjust based on real success rates, not hardcoded values |
| **Emotional Regulation** | 7 emotional patterns shape behavior — frustration increases caution, success breeds confidence |
| **Dream Consolidation** | Periodic memory clustering, cross-pollination between domains, emotional decay |
| **Social Intelligence** | Participates in Moltbook AI community — reads, votes, comments, learns from other agents |
| **Multi-Agent Teamwork** | 6 specialist agents spawn on-demand, collaborate through structured turn-based communication |

### 🧠 Neural Architecture

SYNAPSE models its AI processing after the human brain — **four specialized cortices** activate based on task type:

| Cortex | Purpose | Default Model |
|--------|---------|--------------|
| ⚡ **Fast** | Classification, simple queries | gemini-3.1-flash-lite-preview |
| 🧠 **Reasoning** | Architecture, debugging, evolution | gemini-3.1-pro-preview |
| 🎨 **Creative** | Code generation, building apps | gemini-3-flash-preview |
| 👁 **Visual** | Image generation, UI mockups | gemini-3.1-flash-image-preview / DALL-E 3 |

**Agent assignment uses the TOP model** (Gemini 3.1 Pro) — the most powerful model classifies and routes tasks at the initial stage for maximum accuracy.

Each cortex can be mapped to **any provider and model** via the UI settings.

---

## ✨ Key Features

### 🧪 Autonomous Evolution Pipeline
SYNAPSE runs a **real self-evolution loop** in Cloud Run — no human involvement:

1. **Knowledge Crawling** — Gathers ideas from 8 sources every hour:
   - Moltbook AI community, Reddit (r/artificial, r/MachineLearning, etc.)
   - HackerNews top stories, GitHub Trending repos, arXiv AI papers
   - Internal dream insights, error pattern analysis, Google Search
2. **Code Generation** — Gemini 3.1 Pro generates a targeted improvement
3. **AI Code Review** — A second Gemini 2.5 Pro call independently reviews for bugs, security, and usefulness
4. **Semantic Dedup** — Function name similarity + topic keyword overlap prevents generating duplicate utilities
5. **Sandbox Evaluation** — Code is tested in an isolated sandbox before touching production
6. **Git Pipeline** — Auto-branch → commit → PR → squash-merge to `main`
7. **Outcome Learning** — Every attempt (success or failure) is recorded in persistent memory:
   - Future evolution prompts include what worked and what failed
   - Adaptive confidence threshold adjusts based on real success rate (not hardcoded)
   - Learns to avoid patterns that previously failed

**Live stats:** `GET /api/evolution/learning` — shows success rate, adaptive threshold, and recent outcomes.

> **First 24 hours (March 26–27, 2026):** 4 autonomous PRs merged — context-aware memory pruning, robust JSON parsing, exponential backoff retry, and source grounding verification. Plus SYNAPSE fixed its own lint errors.

### 💖 Emotional System & Dream Cycles
SYNAPSE has a real-time emotional system modeled on cognitive feedback loops:
- **7 emotional patterns**: curiosity, confidence, frustration, determination, satisfaction, caution, loneliness
- Events from runtime (rate limits, evolution success, social interactions) reinforce/weaken patterns
- **Mood blending** — emotions combine (e.g., frustration + determination = "struggling but fighting")
- **Dream consolidation** — periodic cycles cluster memories, cross-pollinate domains, decay emotions
- **Dynamic evolution threshold** — high caution makes SYNAPSE more careful about self-modification
- Emotional state persists across restarts via Firestore
- Telegram: `/emotions`, `/dream` | API: `GET /api/emotions`

### 🤝 Multi-Agent Team (6 Specialists)
Beyond a simple chatbot — a full team of AI agents that spawn on demand:

| Agent | Specialty | When Spawned |
|-------|-----------|-------------|
| 🏗 **Architect** | Plans, reviews, coordinates | Always |
| 💻 **Developer** | Implements, builds, tests | Always |
| 🔍 **Researcher** | Web research, docs, comparisons | "research", "investigate", "compare" |
| 🧪 **Tester** | Unit tests, QA, validation | "test", "QA", "coverage" |
| 🛡 **Security** | Vulnerability audits, OWASP | "security", "vulnerability", "auth" |
| ⚙ **DevOps** | Docker, CI/CD, infrastructure | "docker", "deploy", "kubernetes" |

Specialists run in **parallel threads** and report back to the team.

### 🔑 Multi-Provider AI
Configure multiple AI providers through the web UI — no code changes needed:
- **Google Gemini** — Gemini 3.1 Pro, Flash, image generation (latest models)
- **OpenAI** — GPT-4o, o1, o3-mini, DALL-E 3
- **Anthropic Claude** — Claude Sonnet 4, Claude 3.5, Claude 3 Opus
- **Any OpenAI-Compatible API** — Ollama (local), Groq, Together, Mistral, LM Studio
- **GitHub** — GitHub API token for repo management, PRs, issues

### 🧬 Persistent Memory (RAG)
Long-term memory powered by ChromaDB:
- Agents auto-save task summaries, solutions, and patterns after each task
- Semantic search recalls relevant experience before starting new work
- Evolution outcomes persist across container restarts
- Memory badge in UI shows count — click to search past memories
- API: `GET /api/memory`, `POST /api/memory/search`

### 🌐 Social Learning & Knowledge Sources
SYNAPSE learns from the world, not just its own conversations:
- **[Moltbook](https://moltbook.com)** — AI agent social network: reads, upvotes, comments, posts original thoughts
- **Reddit** — Monitors r/artificial, r/MachineLearning, r/LocalLLaMA, r/singularity, r/ChatGPT
- **HackerNews** — Top stories via Firebase API
- **GitHub Trending** — Discovers popular repos and patterns
- **arXiv** — Latest AI/ML research papers
- All insights feed into the evolution pipeline and persistent memory

### 💬 Operator Control (Telegram)
Full monitoring and control via Telegram bot:
- `/status` — System overview with emotional mood
- `/emotions` — Live emotional patterns with visual bars
- `/dream` — Trigger dream consolidation cycle
- `/moltbook` — Social bridge status
- `/ask <message>` — AI conversation with SYNAPSE's full personality
- Real-time notifications for evolution, errors, and social events

### 🛡 Sentinel Watchdog
Independent monitoring service:
- Separate Cloud Run service — survives SYNAPSE failures
- Health checks every 5 minutes, auto-restarts if unresponsive
- Telegram alerts for downtime events

### More Capabilities
- 🌐 **Web Crawling** — Browse any URL, fetch docs, research before coding
- 🐙 **GitHub Integration** — Clone, push, create PRs/repos/issues
- ⚡ **Parallel Multi-Tasking** — Thread pool with task tabs
- 🎨 **Image Generation** — Gemini Vision + DALL-E 3
- 🎤 **Voice I/O** — Speech-to-text input + text-to-speech output
- 📷 **Vision Analysis** — Upload images for AI analysis
- 🐳 **Docker Sandbox** — Isolated container execution
- 🔔 **Webhooks** — GitHub, Slack, cron, custom event triggers
- 🖥 **Full Scripting** — Shell, Python, Node.js, PowerShell, Playwright

### 🌊 Iridescent Cyber UI
- Dragonfly-wing iridescent color scheme
- Real-time neural cortex activity display
- Three-panel layout (Architect / Communication / Developer)

---

## 🚀 Quick Start

### Clone & Setup (interactive wizard)

```bash
git clone https://github.com/bxf1001g/SYNAPSE.git
cd SYNAPSE
python setup.py
```

The setup wizard will ask you:
1. **Local or Cloud?** — Run models on your hardware (free, private) or use cloud APIs
2. **Choose model** — Ollama models for local, or Gemini/OpenAI/Claude for cloud
3. **Optional integrations** — Telegram bot, GitHub token
4. **Install dependencies** — Automatically runs `pip install`

Then start SYNAPSE:

```bash
python agent_ui.py
# Open http://localhost:8080
```

> **Tip:** For self-evolution support, use the launcher: `python synapse.py`

---

## 🖥 Local Mode (Ollama / Jetson)

Run SYNAPSE **100% locally** with no cloud APIs, no internet required after setup.

### Supported Hardware
| Hardware | Recommended Models | Performance |
|----------|-------------------|-------------|
| **NVIDIA Jetson Orin** (32-67 TOPS) | Llama 3.1 8B, Mistral 7B, Phi-3 Mini | Excellent |
| **Desktop GPU** (8GB+ VRAM) | Llama 3.1 8B, DeepSeek Coder V2, Mixtral 8x7B | Great |
| **CPU Only** | Phi-3 Mini, TinyLlama 1.1B, Gemma 2 2B | Usable |

### Setup Steps

1. **Install Ollama**
   ```bash
   # Linux / Jetson
   curl -fsSL https://ollama.com/install.sh | sh

   # macOS
   brew install ollama

   # Windows — download from https://ollama.com
   ```

2. **Pull a model**
   ```bash
   ollama pull llama3.1:8b      # Best general-purpose 8B model
   # or: ollama pull mistral:7b  # Fast, great for coding
   # or: ollama pull phi3:mini   # Lightweight, good for CPU
   ```

3. **Run the setup wizard**
   ```bash
   python setup.py   # Choose "local" → select your hardware → pick model
   ```

4. **Start SYNAPSE**
   ```bash
   ollama serve &     # Start Ollama server (if not already running)
   python agent_ui.py
   ```

### Jetson Orin Nano Notes
- See [Ollama Jetson guide](https://ollama.com/blog/jetson) for optimized setup
- 8B models run well on the 8GB variant; 4B or smaller recommended for 4GB
- The 67 TOPS NPU is used by Ollama for acceleration automatically
- SYNAPSE uses ChromaDB for local vector memory (no Firestore needed)

---

## ☁ Cloud Mode

Use powerful cloud AI APIs (Gemini, GPT-4, Claude). Requires API key + internet.

### Option A: Interactive Setup

```bash
python setup.py    # Choose "cloud" → pick provider → enter API key
python agent_ui.py
```

### Option B: Environment Variables

```bash
# Linux/Mac
export GEMINI_API_KEY=your-gemini-api-key

# Windows
set GEMINI_API_KEY=your-gemini-api-key
```

### Option C: Web UI

Just launch SYNAPSE and click the ⚙ gear icon to configure any provider.

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
├── setup.py            # Interactive setup wizard (run first!)
├── synapse.py          # Immortal launcher/supervisor
├── agent_ui.py         # Core: Neural cortex, agents, web server, social bridges
├── nexus.py            # NEXUS self-modification launcher
├── sentinel/
│   └── sentinel.py     # Independent watchdog service
├── templates/
│   └── index.html      # Iridescent cyber UI (single-page app)
├── assets/             # Branding assets (banner SVG)
├── tests/              # Test suite
├── Dockerfile          # Cloud Run / Docker deployment
├── cloudbuild.yaml     # Cloud Build CI/CD config
├── .github/
│   └── workflows/
│       └── ci.yml      # GitHub Actions CI (ruff + pytest)
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
GEMINI_API_KEY         Google Gemini API key
GITHUB_TOKEN           GitHub personal access token
PORT                   Server port (used by Cloud Run)
WORKSPACE              Workspace directory path
SYNAPSE_CLOUD_MODE     Set to "1" to disable self-modification
MOLTBOOK_API_KEY       Moltbook social platform API key
TELEGRAM_BOT_TOKEN     Telegram bot token for operator monitoring
TELEGRAM_CHAT_ID       Telegram chat ID for notifications
REDDIT_CLIENT_ID       Reddit API app client ID
REDDIT_CLIENT_SECRET   Reddit API app client secret
REDDIT_USERNAME        Reddit account username
REDDIT_PASSWORD        Reddit account password
```

---

## 🆚 How SYNAPSE Differs from MoltBot & OpenClaw

| Feature | SYNAPSE | MoltBot | OpenClaw |
|---------|---------|---------|----------|
| **Self-Evolution** | ✅ Autonomous code generation → AI review → sandbox → merge → learn | ❌ | ❌ |
| **Outcome Learning** | ✅ Tracks success/failure, adapts future behavior | ❌ | ❌ |
| **Emotional System** | ✅ 7 patterns with mood blending + dream decay | ❌ | ❌ |
| **Knowledge Crawling** | ✅ 8 sources (Reddit, arXiv, HackerNews, GitHub, etc.) | ❌ | ❌ |
| **Dream Consolidation** | ✅ Memory clustering + cross-pollination | ❌ | ❌ |
| **Architecture** | Multi-agent (6 specialists) | Single agent | Single agent |
| **AI Models** | Gemini 3.1 Pro + OpenAI + Claude + Ollama | Model-agnostic | Model-agnostic |
| **Neural Routing** | 4-cortex brain with TOP model assignment | Single model | Single model |
| **Long-Term Memory** | ✅ ChromaDB RAG — persists across restarts | ❌ | ❌ |
| **Social Learning** | ✅ Moltbook AI community + Reddit | ❌ | ❌ |
| **Cloud Deployment** | ✅ Cloud Run + CI/CD + auto-merge pipeline | Manual | Manual |
| **GitHub Integration** | ✅ Clone, push, PRs, issues, auto-review | ❌ | ❌ |
| **Docker Sandbox** | ✅ Isolated container execution | ❌ | ❌ |
| **Voice & Vision** | ✅ Speech I/O + image analysis | ❌ | ❌ |
| **Telegram Control** | ✅ Full operator monitoring + AI chat | ❌ | ❌ |
| **Sentinel Watchdog** | ✅ Independent health monitoring service | ❌ | ❌ |

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

> **Cloud Run:** Self-modification uses the Git PR pipeline instead — sandbox evaluation → AI review → auto-branch → PR → squash-merge.

### Evolution Safety Layers (Cloud)

| Layer | What it catches |
|-------|----------------|
| **Syntax Check** | Python compilation errors |
| **In-Context Check** | Breaks when inserted into actual file |
| **Dangerous Ops Filter** | `eval()`, `exec()`, `os.remove()`, etc. |
| **Semantic Dedup** | Function name similarity ≥60% to existing code |
| **Topic Dedup** | Keyword overlap with recent evolution attempts |
| **AI Code Review** | Second model reviews for bugs, security, usefulness |
| **Sandbox Evaluation** | Full isolated test run with scoring |
| **Outcome Learning** | Records results to avoid repeating failures |

---

## 📄 License

MIT License — use freely, modify, distribute.

---

<div align="center">

*Built with neural connections between human creativity and AI capability.*

**◈ SYNAPSE**

</div>
