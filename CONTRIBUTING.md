# Contributing to SYNAPSE

First off, thank you for considering contributing to SYNAPSE! 🧠⚡ This is a community-driven project and every contribution matters.

## 🚀 Quick Start for Contributors

1. **Fork** the repo on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SYNAPSE.git
   cd SYNAPSE
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your environment:**
   ```bash
   # Required
   export GEMINI_API_KEY=your_gemini_key
   
   # Optional (for GitHub integration)
   export GITHUB_TOKEN=your_github_token
   ```
5. **Run locally:**
   ```bash
   python agent_ui.py
   ```
6. Open `http://localhost:5000` in your browser

## 🏗 Architecture Overview

| File | Purpose |
|------|---------|
| `agent_ui.py` | Main backend — Flask + SocketIO server, all agent logic |
| `templates/index.html` | Frontend UI — single-page app with real-time comms |
| `protocol.py` | NEXUS communication protocol between agents |
| `synapse.py` | Core neural processing engine |
| `nexus.py` | Self-modification and launcher system |
| `ai_agent.py` | Base AI agent class |
| `agent.py` | Agent coordination layer |
| `cloudbuild.yaml` | CI/CD pipeline for Google Cloud Run |
| `Dockerfile` | Container configuration |

## 📋 How to Contribute

### Reporting Bugs
- Use the [Bug Report](https://github.com/bxf1001g/SYNAPSE/issues/new?template=bug_report.md) issue template
- Include browser console logs and Python terminal output
- Specify which AI provider/model you were using

### Suggesting Features
- Use the [Feature Request](https://github.com/bxf1001g/SYNAPSE/issues/new?template=feature_request.md) issue template
- Describe the use case, not just the solution
- Check existing issues first to avoid duplicates

### Submitting Code

1. Create a **feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes with clear, focused commits
3. **Test locally** — run the app and verify your changes work
4. Push to your fork and open a **Pull Request**

### PR Guidelines
- Keep PRs focused — one feature or fix per PR
- Write a clear description of what changed and why
- Include screenshots for UI changes
- Reference any related issues with `Fixes #123` or `Relates to #123`

## 🎯 Good First Issues

Look for issues labeled [`good first issue`](https://github.com/bxf1001g/SYNAPSE/labels/good%20first%20issue). These are specifically curated for new contributors.

### Areas Where We Need Help

| Area | Skills Needed | Difficulty |
|------|--------------|------------|
| **New AI Providers** | Python, API integration | ⭐⭐ Medium |
| **UI/UX Improvements** | HTML/CSS/JS, Design | ⭐ Easy |
| **Agent Roles** | Python, prompt engineering | ⭐⭐ Medium |
| **Testing** | Python, pytest | ⭐ Easy |
| **Documentation** | Technical writing | ⭐ Easy |
| **MCP Plugins** | Python, protocol design | ⭐⭐⭐ Hard |
| **Performance** | Python, async programming | ⭐⭐⭐ Hard |
| **Security** | Python, Docker, sandboxing | ⭐⭐⭐ Hard |

## 🔌 Adding a New AI Provider

One of the most impactful contributions is adding support for new AI providers. Here's the pattern:

1. Add the API client to `agent_ui.py`
2. Register it in the provider selection logic
3. Add the model names to the cortex mapping
4. Add UI options in `templates/index.html` settings panel
5. Update README with the new provider

## 💡 Design Principles

- **Agents should collaborate, not just execute** — the architect-developer pattern is core
- **Self-evolution over manual updates** — SYNAPSE should improve itself when possible
- **Provider agnostic** — no hard dependency on any single AI provider
- **Safe by default** — sandboxed execution, clone-before-modify, rollback on failure

## 🧪 Testing

```bash
# Run syntax check
python -c "import agent_ui; print('OK')"

# Run the app locally
python agent_ui.py

# Test webhook endpoint
curl -X POST http://localhost:5000/api/webhook \
  -H "Content-Type: application/json" \
  -d '{"task": "test task"}'

# Test memory endpoint
curl http://localhost:5000/api/memory
```

## 📜 Code Style

- Python: Follow PEP 8, use type hints where practical
- JavaScript: Vanilla JS (no frameworks in the frontend)
- Comments: Only where logic isn't self-evident
- Commits: Use [conventional commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, etc.)

## 🤝 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 📬 Questions?

- Open a [Discussion](https://github.com/bxf1001g/SYNAPSE/discussions) on GitHub
- Tag your issue with `question` label

---

Thank you for helping make SYNAPSE better! Every contribution, no matter how small, helps push the boundaries of what multi-agent AI systems can do. 🚀
