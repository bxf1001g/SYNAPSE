#!/usr/bin/env python3
"""SYNAPSE Interactive Setup Wizard.

Run this after cloning the repository:
    python setup.py

It walks you through choosing local vs cloud mode, configuring
an AI provider (Ollama for local, Gemini/OpenAI for cloud), and
generates a .synapse.json config file ready to go.
"""

import json
import os
import shutil
import subprocess
import sys

BANNER = """
  ◈ ═══════════════════════════════════════════
  ◈   SYNAPSE — Interactive Setup Wizard
  ◈ ═══════════════════════════════════════════
"""

CONFIG_FILE = ".synapse.json"

# Popular Ollama models suitable for different hardware tiers
OLLAMA_MODELS = {
    "jetson_orin": {
        "label": "Jetson Orin (32-67 TOPS)",
        "recommended": [
            ("llama3.1:8b", "Meta Llama 3.1 8B — best overall"),
            ("mistral:7b", "Mistral 7B — fast, great for coding"),
            ("phi3:mini", "Phi-3 Mini 3.8B — very fast, lightweight"),
            ("gemma2:9b", "Google Gemma 2 9B — strong reasoning"),
            ("codellama:7b", "Code Llama 7B — optimized for code"),
        ],
    },
    "gpu_desktop": {
        "label": "Desktop GPU (8GB+ VRAM)",
        "recommended": [
            ("llama3.1:8b", "Meta Llama 3.1 8B — best overall"),
            ("deepseek-coder-v2:16b", "DeepSeek Coder V2 16B — best for code"),
            ("mistral:7b", "Mistral 7B — fast, great for coding"),
            ("mixtral:8x7b", "Mixtral 8x7B MoE — powerful, needs 24GB+"),
            ("codellama:34b", "Code Llama 34B — best code, needs 24GB+"),
        ],
    },
    "cpu_only": {
        "label": "CPU Only (no GPU)",
        "recommended": [
            ("phi3:mini", "Phi-3 Mini 3.8B — very fast on CPU"),
            ("tinyllama:1.1b", "TinyLlama 1.1B — ultra lightweight"),
            ("llama3.2:3b", "Llama 3.2 3B — good quality, moderate speed"),
            ("gemma2:2b", "Gemma 2 2B — compact Google model"),
        ],
    },
}

CLOUD_PROVIDERS = {
    "gemini": {
        "label": "Google Gemini (recommended — generous free tier)",
        "env_var": "GEMINI_API_KEY",
        "signup_url": "https://aistudio.google.com/apikey",
        "models": {
            "fast": "gemini-2.0-flash",
            "reason": "gemini-2.5-pro",
            "create": "gemini-2.0-flash",
            "visual": "gemini-2.0-flash",
        },
    },
    "openai": {
        "label": "OpenAI (GPT-4o, o1)",
        "env_var": "OPENAI_API_KEY",
        "signup_url": "https://platform.openai.com/api-keys",
        "models": {
            "fast": "gpt-4o-mini",
            "reason": "gpt-4o",
            "create": "gpt-4o",
            "visual": "gpt-4o",
        },
    },
    "anthropic": {
        "label": "Anthropic Claude (Claude Sonnet 4)",
        "env_var": "ANTHROPIC_API_KEY",
        "signup_url": "https://console.anthropic.com/",
        "models": {
            "fast": "claude-3-5-haiku-20241022",
            "reason": "claude-sonnet-4-20250514",
            "create": "claude-sonnet-4-20250514",
            "visual": "claude-sonnet-4-20250514",
        },
    },
}


def ask(prompt, choices=None, default=None):
    """Ask user a question. Returns their answer."""
    if choices:
        print()
        for i, c in enumerate(choices, 1):
            marker = " (default)" if default and c == default else ""
            print(f"  [{i}] {c}{marker}")
        print()
        while True:
            raw = input(f"  {prompt} [{default or ''}]: ").strip()
            if not raw and default:
                return default
            try:
                idx = int(raw)
                if 1 <= idx <= len(choices):
                    return choices[idx - 1]
            except ValueError:
                if raw in choices:
                    return raw
            print(f"  Please enter a number 1-{len(choices)}")
    else:
        raw = input(f"  {prompt} [{default or ''}]: ").strip()
        return raw or default


def check_ollama():
    """Check if Ollama is installed and running."""
    if shutil.which("ollama"):
        try:
            r = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                models = [
                    line.split()[0]
                    for line in r.stdout.strip().split("\n")[1:]
                    if line.strip()
                ]
                return True, models
        except Exception:
            pass
        return True, []  # installed but not running
    return False, []


def install_ollama_instructions():
    """Print Ollama installation instructions."""
    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │  Ollama is not installed yet.            │")
    print("  │                                          │")
    if sys.platform == "linux":
        print("  │  Install with:                          │")
        print("  │    curl -fsSL https://ollama.com/install.sh | sh │")
        if os.path.exists("/proc/device-tree/model"):
            print("  │                                          │")
            print("  │  For Jetson, also see:                   │")
            print("  │    https://ollama.com/blog/jetson         │")
    elif sys.platform == "darwin":
        print("  │  Install with:                          │")
        print("  │    brew install ollama                   │")
    else:
        print("  │  Download from: https://ollama.com      │")
    print("  │                                          │")
    print("  │  Then run: ollama serve                  │")
    print("  └─────────────────────────────────────────┘")
    print()


def pull_ollama_model(model_name):
    """Pull an Ollama model."""
    print(f"\n  Pulling {model_name}... (this may take a few minutes)")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"  ✓ {model_name} ready!")
        return True
    except subprocess.CalledProcessError:
        print(f"  ✗ Failed to pull {model_name}")
        return False
    except FileNotFoundError:
        print("  ✗ Ollama not found. Install it first.")
        return False


def setup_local():
    """Configure local mode with Ollama."""
    print("\n  ── Local Mode Setup ──")
    print("  SYNAPSE will use Ollama to run AI models on your hardware.")

    installed, existing_models = check_ollama()
    if not installed:
        install_ollama_instructions()
        proceed = ask("Continue setup anyway? (config will be ready when Ollama is installed)",
                       ["yes", "no"], "yes")
        if proceed == "no":
            return None

    if existing_models:
        print(f"\n  ✓ Ollama is running with models: {', '.join(existing_models)}")

    # Hardware tier selection
    hw = ask(
        "What hardware are you running on?",
        [OLLAMA_MODELS[k]["label"] for k in OLLAMA_MODELS],
        OLLAMA_MODELS["jetson_orin"]["label"],
    )
    hw_key = [k for k, v in OLLAMA_MODELS.items() if v["label"] == hw][0]
    models = OLLAMA_MODELS[hw_key]["recommended"]

    print(f"\n  Recommended models for {hw}:")
    for i, (name, desc) in enumerate(models, 1):
        installed_marker = " ✓ (installed)" if name in existing_models else ""
        print(f"  [{i}] {name} — {desc}{installed_marker}")

    chosen_idx = ask("Choose your primary model", [str(i) for i in range(1, len(models) + 1)], "1")
    chosen_model = models[int(chosen_idx) - 1][0]

    # Pull model if not already present
    if installed and chosen_model not in existing_models:
        pull = ask(f"Pull {chosen_model} now?", ["yes", "no"], "yes")
        if pull == "yes":
            pull_ollama_model(chosen_model)

    # Ollama base URL
    ollama_url = ask("Ollama API URL", default="http://localhost:11434/v1")

    config = {
        "mode": "local",
        "providers": {
            "gemini": {"api_key": "", "enabled": False},
            "openai": {"api_key": "", "enabled": False},
            "anthropic": {"api_key": "", "enabled": False},
            "openai_compatible": {
                "api_key": "not-needed",
                "base_url": ollama_url,
                "label": "Ollama (Local)",
                "enabled": True,
            },
        },
        "cortex_map": {
            "fast": {"provider": "openai_compatible", "model": chosen_model},
            "reason": {"provider": "openai_compatible", "model": chosen_model},
            "create": {"provider": "openai_compatible", "model": chosen_model},
            "visual": {"provider": "openai_compatible", "model": chosen_model},
        },
    }
    return config


def setup_cloud():
    """Configure cloud mode with API provider."""
    print("\n  ── Cloud Mode Setup ──")
    print("  SYNAPSE will use cloud AI APIs (requires API key).")

    provider_key = ask(
        "Choose your AI provider",
        [f"{v['label']}" for v in CLOUD_PROVIDERS.values()],
        CLOUD_PROVIDERS["gemini"]["label"],
    )
    prov_id = [k for k, v in CLOUD_PROVIDERS.items() if v["label"] == provider_key][0]
    prov = CLOUD_PROVIDERS[prov_id]

    # Check for existing env var
    existing_key = os.environ.get(prov["env_var"], "")
    if existing_key:
        print(f"\n  ✓ Found {prov['env_var']} in environment")
        api_key = existing_key
    else:
        print(f"\n  Get your API key at: {prov['signup_url']}")
        api_key = ask(f"Enter your {prov_id} API key").strip()
        if not api_key:
            print("  ⚠ No API key provided. You can add it later in the web UI.")

    config = {
        "mode": "cloud",
        "providers": {
            "gemini": {"api_key": "", "enabled": False},
            "openai": {"api_key": "", "enabled": False},
            "anthropic": {"api_key": "", "enabled": False},
            "openai_compatible": {
                "api_key": "", "base_url": "", "label": "Custom", "enabled": False
            },
        },
        "cortex_map": {},
    }

    config["providers"][prov_id]["api_key"] = api_key
    config["providers"][prov_id]["enabled"] = True
    config["cortex_map"] = {
        k: {"provider": prov_id, "model": m}
        for k, m in prov["models"].items()
    }

    return config


def setup_telegram(config):
    """Optional Telegram integration."""
    print("\n  ── Telegram Bot (Optional) ──")
    setup_tg = ask("Set up Telegram bot notifications?", ["yes", "skip"], "skip")
    if setup_tg == "skip":
        return config

    print("  1. Message @BotFather on Telegram → /newbot")
    print("  2. Copy the bot token")
    token = ask("Bot token").strip()
    chat_id = ask("Your Telegram chat ID (message @userinfobot to get it)").strip()
    if token and chat_id:
        config["telegram"] = {"bot_token": token, "chat_id": chat_id}
        print("  ✓ Telegram configured")
    return config


def write_env_file(config):
    """Write a .env file for easy startup."""
    lines = []
    mode = config.get("mode", "local")

    if mode == "cloud":
        lines.append("SYNAPSE_CLOUD_MODE=1")
        for pid, pcfg in config["providers"].items():
            if pcfg.get("api_key") and pcfg.get("enabled"):
                if pid == "gemini":
                    lines.append(f"GEMINI_API_KEY={pcfg['api_key']}")
                elif pid == "openai":
                    lines.append(f"OPENAI_API_KEY={pcfg['api_key']}")
                elif pid == "anthropic":
                    lines.append(f"ANTHROPIC_API_KEY={pcfg['api_key']}")
    else:
        lines.append("SYNAPSE_CLOUD_MODE=0")

    tg = config.get("telegram", {})
    if tg.get("bot_token"):
        lines.append(f"TELEGRAM_BOT_TOKEN={tg['bot_token']}")
        lines.append(f"TELEGRAM_CHAT_ID={tg['chat_id']}")

    if lines:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        with open(env_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  ✓ Wrote {env_path}")


def install_dependencies():
    """Install Python dependencies."""
    print("\n  Installing dependencies...")
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file, "-q"],
            check=True,
        )
        print("  ✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("  ✗ Failed to install dependencies. Run manually:")
        print(f"    pip install -r {req_file}")
        return False


def main():
    print(BANNER)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Check for existing config
    config_path = os.path.join(base_dir, CONFIG_FILE)
    if os.path.exists(config_path):
        overwrite = ask(
            f"Found existing {CONFIG_FILE}. Overwrite?",
            ["yes", "no"],
            "no",
        )
        if overwrite == "no":
            print("  Setup cancelled. Edit .synapse.json manually or run the web UI.")
            return

    # Mode selection
    print("  How do you want to run SYNAPSE?\n")
    print("  🖥  LOCAL  — Run AI models on your own hardware (Jetson, GPU, CPU)")
    print("               Free, private, no internet needed after setup")
    print("               Uses Ollama + open-source models\n")
    print("  ☁  CLOUD  — Use cloud AI APIs (Gemini, GPT-4, Claude)")
    print("               More powerful models, requires API key + internet")
    print("               Free tiers available (Gemini recommended)\n")

    mode = ask("Choose mode", ["local", "cloud"], "local")

    if mode == "local":
        config = setup_local()
    else:
        config = setup_cloud()

    if config is None:
        print("\n  Setup cancelled.")
        return

    # Optional Telegram
    config = setup_telegram(config)

    # Save config
    # Remove 'mode' and 'telegram' from config before saving (not part of .synapse.json schema)
    tg_config = config.pop("telegram", {})
    saved_mode = config.pop("mode", "local")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  ✓ Config saved to {CONFIG_FILE}")

    # Write .env
    config["mode"] = saved_mode
    config["telegram"] = tg_config
    write_env_file(config)

    # Install deps
    install = ask("Install Python dependencies now?", ["yes", "no"], "yes")
    if install == "yes":
        install_dependencies()

    # Summary
    print("\n  ◈ ═══════════════════════════════════════")
    print("  ◈  Setup Complete!")
    print("  ◈ ═══════════════════════════════════════")
    print()
    if saved_mode == "local":
        ollama_url = config.get("providers", {}).get("openai_compatible", {}).get("base_url", "")
        cortex = config.get("cortex_map", {})
        model = list(cortex.values())[0].get("model", "unknown") if cortex else "unknown"
        print("  Mode:     LOCAL (Ollama)")
        print(f"  Model:    {model}")
        print(f"  Ollama:   {ollama_url}")
    else:
        active = [k for k, v in config.get("providers", {}).items() if v.get("enabled")]
        print(f"  Mode:     CLOUD ({', '.join(active)})")

    print(f"  Config:   {CONFIG_FILE}")
    print()
    print("  To start SYNAPSE:")
    print()
    if saved_mode == "local":
        print("    1. Make sure Ollama is running:  ollama serve")
        print("    2. Start SYNAPSE:                python agent_ui.py")
    else:
        print("    python agent_ui.py")
    print()
    print("    Then open http://localhost:8080 in your browser")
    print()


if __name__ == "__main__":
    main()
