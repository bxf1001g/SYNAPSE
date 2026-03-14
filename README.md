<div align="center">

# ◈ SYNAPSE

### Neural Multi-Agent AI System

*Two AI agents. Multiple AI brains. One self-evolving system.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## What is SYNAPSE?

SYNAPSE is an **autonomous multi-agent AI system** where two specialized AI agents — an **Architect** and a **Developer** — collaborate in real-time to build software, answer questions, generate images, and even **modify their own source code**.

Unlike single-agent chatbots, SYNAPSE operates as a **team**: the Architect plans and verifies while the Developer implements and tests. They communicate through structured turn-based messages, creating a feedback loop that produces better results than either agent alone.

### 🧠 Neural Architecture

SYNAPSE models its AI processing after the human brain — **four specialized cortices** activate based on task type:

| Cortex | Purpose | Default Model |
|--------|---------|--------------|
| ⚡ **Fast** | Classification, simple queries | gemini-2.0-flash-lite |
| 🧠 **Reasoning** | Architecture, debugging, deep analysis | gemini-2.5-pro |
| 🎨 **Creative** | Code generation, building apps | gemini-2.0-flash |
| 👁 **Visual** | Image generation, UI mockups | gemini-2.5-flash-image / DALL-E 3 |

Each cortex can be mapped to **any provider and model** via the UI settings.

---

## ✨ Key Features

### 🔑 Multi-Provider AI
Configure multiple AI providers through the web UI — no code changes needed:
- **Google Gemini** — Full support including image generation
- **OpenAI** — GPT-4o, o1, o3-mini, DALL-E 3
- **Anthropic Claude** — Claude 3.5 Sonnet, Claude 3 Opus
- **Any OpenAI-Compatible API** — Ollama (local), Groq, Together, Mistral, LM Studio

### 🤝 Dual-Agent Collaboration
Two agents work as a team:
- **Architect** — Plans, reviews, verifies with tests
- **Developer** — Implements, builds, fixes bugs
- Structured turn-based communication with full visibility

### ⚡ Parallel Multi-Tasking
Submit multiple tasks simultaneously — each runs on its own thread:
- Task tabs with independent workspaces
- Cancel individual tasks
- Real-time status per task

### 🧬 Self-Modification
The system can evolve its own code:
- Agents can modify `agent_ui.py` and `templates/index.html`
- Changes go through: **backup → validate → clone-test → swap → restart**
- Automatic rollback on failure
- `synapse.py` launcher is never modified (immortal supervisor)

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
git clone https://github.com/YOUR_USERNAME/synapse.git
cd synapse
pip install -r requirements.txt
```

### 2. Configure

**Option A: Web UI (recommended)**
Just launch SYNAPSE and click the ⚙ gear icon to configure any provider — Gemini, OpenAI, Anthropic, or any OpenAI-compatible API (Ollama, Groq, Together, etc.).

**Option B: Environment variable (Gemini quick start)**
```bash
# Windows
set GEMINI_API_KEY=your-gemini-api-key

# Linux/Mac
export GEMINI_API_KEY=your-gemini-api-key
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

## 📁 Project Structure

```
synapse/
├── synapse.py          # Immortal launcher/supervisor
├── agent_ui.py         # Core: Neural cortex, agents, web server
├── templates/
│   └── index.html      # Iridescent cyber UI (single-page app)
├── ai_agent.py         # Original CLI dual-agent version
├── agent.py            # Manual TCP chat agent
├── protocol.py         # TCP framing protocol
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Configuration

### Settings UI
Click **⚙** in the header to open settings:
1. Enable providers and paste API keys
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
    "openai_compatible": { "base_url": "http://localhost:11434/v1", "enabled": false }
  },
  "cortex_map": {
    "fast": { "provider": "gemini", "model": "gemini-2.0-flash-lite" },
    "reason": { "provider": "anthropic", "model": "claude-sonnet-4-20250514" },
    "create": { "provider": "openai", "model": "gpt-4o" },
    "visual": { "provider": "gemini", "model": "gemini-2.5-flash-image" }
  }
}
```

### CLI Arguments
```
python synapse.py [options]

--workspace PATH    Project workspace directory (default: ./workspace)
--port PORT         Web UI port (default: 8080)
--api-key KEY       Gemini API key (overrides env var)
--model MODEL       Default model (default: gemini-2.0-flash)
```

---

## 🆚 How SYNAPSE Differs from MoltBot & OpenClaw

| Feature | SYNAPSE | MoltBot | OpenClaw |
|---------|---------|---------|----------|
| **Architecture** | Dual-agent (Architect + Developer) | Single agent | Single agent |
| **AI Providers** | Gemini + OpenAI + Claude + Any compatible | Model-agnostic | Model-agnostic |
| **Neural Routing** | 4-cortex brain with automatic task routing | Single model per task | Single model |
| **Self-Modification** | ✅ Agents rewrite own code safely | ❌ | ❌ |
| **Parallel Tasks** | ✅ Thread pool with task tabs | Limited | Limited |
| **Image Generation** | ✅ Gemini + DALL-E | ❌ | ❌ |
| **Agent Collaboration** | ✅ Turn-based Architect↔Developer | N/A | N/A |
| **Self-Evolving Launcher** | ✅ Backup→Validate→Clone-test→Swap | ❌ | ❌ |
| **Chat Platform Integration** | Web UI only | WhatsApp, Telegram, Slack | WhatsApp, Telegram, Discord |
| **UI** | Iridescent cyber web UI | Web/CLI | Web/CLI |

**SYNAPSE's unique advantages:**
- 🧠 **Multi-model brain** — Different tasks automatically route to the best model
- 🤝 **Two-agent collaboration** — Plans are verified, code is tested before delivery
- 🧬 **Self-evolution** — The system improves itself through safe code modification
- ⚡ **True parallel multi-tasking** — Multiple independent task threads

---

## 💡 Example Tasks

```
"Build a Flask REST API with user authentication"
"Create a React todo app with local storage"
"What files are in my workspace?"
"Debug why my Python script crashes on line 42"
"Generate a logo for my project"
"Delete all .tmp files in my workspace"
"Build a web scraper for weather data"
```

---

## 🛡 Self-Modification Safety

When agents request code changes to themselves:

1. **Backup** — Current code is timestamped and preserved
2. **Validate** — New code is syntax-checked
3. **Clone-Test** — New version spins up on a temp port for health check
4. **Swap** — Only if healthy, files are atomically swapped
5. **Restart** — System restarts with new code
6. **Rollback** — If crashed, auto-reverts to last working version

The launcher (`synapse.py`) is **never modified** — it's the immortal anchor.

---

## 📄 License

MIT License — use freely, modify, distribute.

---

<div align="center">

*Built with neural connections between human creativity and AI capability.*

**◈ SYNAPSE**

</div>
