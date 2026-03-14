"""
nexus.py — NEXUS Launcher / Supervisor

The immortal anchor of the self-modifying AI system.
This file is NEVER modified by agents. It manages the lifecycle of agent_ui.py:
  - Starts the agent process
  - Handles self-modification requests (backup → validate → clone-test → swap → restart)
  - Auto-rollback on crash
  - Health monitoring

Usage:
    set GEMINI_API_KEY=your-key
    python nexus.py [--workspace ./myproject] [--port 8080]

This replaces running agent_ui.py directly.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────

AGENT_SCRIPT = "agent_ui.py"
STAGING_SUFFIX = ".staged"
BACKUP_SUFFIX = ".bak"
SIGNAL_FILE = ".nexus_restart"
HEALTH_TIMEOUT = 15       # seconds to wait for new process to be healthy
HEALTH_CHECK_INTERVAL = 2  # seconds between health checks
CLONE_TEST_PORT_OFFSET = 100
CRASH_THRESHOLD = 10       # if process dies within N seconds, consider it a crash

# Terminal colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_RED = "\033[91m"
C_YELLOW = "\033[33m"
C_PURPLE = "\033[35m"
C_BLUE = "\033[94m"

BANNER = f"""
{C_PURPLE}{C_BOLD}  ╔══════════════════════════════════════════════╗
  ║  ◈  NEXUS — Self-Evolving AI System          ║
  ║     Launcher / Supervisor v1.0                ║
  ╚══════════════════════════════════════════════╝{C_RESET}
"""


# ── Utilities ────────────────────────────────────────────────────

def log(msg, color=C_DIM):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {C_DIM}[{ts}]{C_RESET} {color}{msg}{C_RESET}")


def log_ok(msg):
    log(f"✓ {msg}", C_GREEN)


def log_warn(msg):
    log(f"⚠ {msg}", C_YELLOW)


def log_err(msg):
    log(f"✖ {msg}", C_RED)


def log_info(msg):
    log(f"◈ {msg}", C_CYAN)


def validate_python_file(filepath):
    """Syntax-check a Python file. Returns (ok, error_msg)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             f"import py_compile; py_compile.compile(r'{filepath}', doraise=True)"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)


def validate_html_file(filepath):
    """Basic validation: file exists and is non-empty."""
    try:
        size = os.path.getsize(filepath)
        return size > 100, f"File too small ({size} bytes)"
    except Exception as e:
        return False, str(e)


def health_check(port, timeout=HEALTH_TIMEOUT):
    """Check if the agent is responding on the given port."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            url = f"http://127.0.0.1:{port}/"
            req = urllib.request.urlopen(url, timeout=3)
            if req.status == 200:
                return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)
    return False


# ── Self-Modification Engine ─────────────────────────────────────

class SelfModEngine:
    """Handles the backup → validate → clone-test → swap → restart cycle."""

    def __init__(self, base_dir, port):
        self.base_dir = Path(base_dir)
        self.port = port
        self.backup_dir = self.base_dir / ".nexus_backups"
        self.backup_dir.mkdir(exist_ok=True)

    def process_request(self, request_data):
        """
        Process a self-modification request.
        request_data = {
            "files": [
                {"path": "agent_ui.py", "content": "..."},
                {"path": "templates/index.html", "content": "..."}
            ],
            "reason": "Improved UI styling"
        }
        Returns (success, message)
        """
        files = request_data.get("files", [])
        reason = request_data.get("reason", "Unknown")

        if not files:
            return False, "No files to modify"

        log_info(f"Self-modification request: {reason}")
        log_info(f"Files to modify: {', '.join(f['path'] for f in files)}")

        # Step 1: Validate all staged files
        log_info("Step 1/4: Validating new code...")
        for f in files:
            filepath = f["path"]
            content = f["content"]

            # Write staged version
            staged_path = self.base_dir / (filepath + STAGING_SUFFIX)
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_text(content, encoding="utf-8")

            # Validate based on file type
            if filepath.endswith(".py"):
                ok, err = validate_python_file(str(staged_path))
                if not ok:
                    staged_path.unlink(missing_ok=True)
                    log_err(f"Validation failed for {filepath}: {err}")
                    return False, f"Syntax error in {filepath}: {err}"
            elif filepath.endswith(".html"):
                ok, err = validate_html_file(str(staged_path))
                if not ok:
                    staged_path.unlink(missing_ok=True)
                    log_err(f"Validation failed for {filepath}: {err}")
                    return False, f"Invalid HTML in {filepath}: {err}"

        log_ok("All files validated")

        # Step 2: Create backups
        log_info("Step 2/4: Creating backups...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / timestamp
        backup_subdir.mkdir(exist_ok=True)

        for f in files:
            original = self.base_dir / f["path"]
            if original.exists():
                backup_path = backup_subdir / f["path"]
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(original), str(backup_path))
                log(f"  Backed up: {f['path']}")

        # Save metadata
        meta = {
            "timestamp": timestamp,
            "reason": reason,
            "files": [f["path"] for f in files],
        }
        (backup_subdir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        log_ok(f"Backups saved to .nexus_backups/{timestamp}/")

        # Step 3: Clone-test (spin up new code on temp port)
        log_info("Step 3/4: Clone-testing new code...")
        clone_ok = self._clone_test(files)
        if not clone_ok:
            # Clean up staged files
            for f in files:
                staged = self.base_dir / (f["path"] + STAGING_SUFFIX)
                staged.unlink(missing_ok=True)
            log_err("Clone test failed — rolling back")
            return False, "Clone test failed: new code didn't start properly"

        log_ok("Clone test passed")

        # Step 4: Swap staged files to live
        log_info("Step 4/4: Swapping to new code...")
        for f in files:
            staged = self.base_dir / (f["path"] + STAGING_SUFFIX)
            target = self.base_dir / f["path"]
            if staged.exists():
                shutil.move(str(staged), str(target))
                log(f"  Swapped: {f['path']}")

        log_ok(f"Self-modification complete: {reason}")
        return True, "Code updated successfully"

    def _clone_test(self, files):
        """Start a test instance with the staged code, verify it's healthy."""
        test_port = self.port + CLONE_TEST_PORT_OFFSET

        # Temporarily swap in staged files
        originals = {}
        for f in files:
            original = self.base_dir / f["path"]
            staged = self.base_dir / (f["path"] + STAGING_SUFFIX)
            tmp_backup = self.base_dir / (f["path"] + ".tmp_clone_backup")
            if original.exists():
                shutil.copy2(str(original), str(tmp_backup))
                originals[f["path"]] = str(tmp_backup)
            if staged.exists():
                shutil.copy2(str(staged), str(original))

        # Start test process
        test_proc = None
        try:
            env = os.environ.copy()
            test_proc = subprocess.Popen(
                [
                    sys.executable, str(self.base_dir / AGENT_SCRIPT),
                    "--port", str(test_port),
                    "--workspace", str(self.base_dir / "workspace"),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self.base_dir),
            )

            # Wait for it to start and health-check
            healthy = health_check(test_port, timeout=HEALTH_TIMEOUT)
            return healthy

        except Exception as e:
            log_err(f"Clone test error: {e}")
            return False

        finally:
            # Kill test process
            if test_proc and test_proc.poll() is None:
                test_proc.terminate()
                try:
                    test_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    test_proc.kill()

            # Restore originals
            for filepath, tmp_path in originals.items():
                original = self.base_dir / filepath
                shutil.move(tmp_path, str(original))

            # Clean up any remaining tmp backups
            for f in files:
                tmp = self.base_dir / (f["path"] + ".tmp_clone_backup")
                if tmp.exists():
                    tmp.unlink()

    def rollback(self, timestamp=None):
        """Rollback to a previous backup."""
        if timestamp is None:
            # Find most recent backup
            backups = sorted(self.backup_dir.iterdir())
            if not backups:
                log_err("No backups available for rollback")
                return False
            backup_subdir = backups[-1]
        else:
            backup_subdir = self.backup_dir / timestamp

        if not backup_subdir.exists():
            log_err(f"Backup not found: {backup_subdir}")
            return False

        meta_file = backup_subdir / "meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            files = meta.get("files", [])
        else:
            files = [
                str(p.relative_to(backup_subdir))
                for p in backup_subdir.rglob("*")
                if p.is_file() and p.name != "meta.json"
            ]

        for filepath in files:
            backup_file = backup_subdir / filepath
            target = self.base_dir / filepath
            if backup_file.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(backup_file), str(target))
                log(f"  Restored: {filepath}")

        log_ok(f"Rolled back to {backup_subdir.name}")
        return True


# ── Main Supervisor Loop ─────────────────────────────────────────

class NexusSupervisor:
    """The immortal supervisor that manages agent_ui.py."""

    def __init__(self, args):
        self.base_dir = Path(os.path.abspath("."))
        self.port = args.port
        self.workspace = os.path.abspath(args.workspace)
        self.api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        self.model = args.model
        self.agent_proc = None
        self.mod_engine = SelfModEngine(self.base_dir, self.port)
        self.running = True
        self.restart_count = 0
        self.max_rapid_restarts = 5

        os.makedirs(self.workspace, exist_ok=True)

        # Signal file for agent to request restart
        self.signal_file = self.base_dir / SIGNAL_FILE

    def start(self):
        """Main supervisor loop."""
        print(BANNER)
        log_info(f"Port: {self.port}")
        log_info(f"Workspace: {self.workspace}")
        log_info(f"Model: {self.model}")
        log_info(f"Agent script: {AGENT_SCRIPT}")
        print()

        if not self.api_key:
            log_err("No API key. Set GEMINI_API_KEY or use --api-key")
            sys.exit(1)

        # Clean up any leftover signal files
        self.signal_file.unlink(missing_ok=True)

        # Start the agent
        self._start_agent()

        try:
            while self.running:
                # Check if agent process is still alive
                if self.agent_proc and self.agent_proc.poll() is not None:
                    rc = self.agent_proc.returncode
                    log_warn(f"Agent process exited (code {rc})")

                    # Check for rapid crash loop
                    self.restart_count += 1
                    if self.restart_count > self.max_rapid_restarts:
                        log_err("Too many rapid restarts — attempting rollback")
                        if self.mod_engine.rollback():
                            self.restart_count = 0
                            self._start_agent()
                        else:
                            log_err("Rollback failed. Manual intervention needed.")
                            break
                    else:
                        log_info(f"Restarting agent (attempt {self.restart_count})...")
                        time.sleep(2)
                        self._start_agent()

                # Check for self-modification signal
                if self.signal_file.exists():
                    self._handle_self_modify()

                time.sleep(1)

        except KeyboardInterrupt:
            log_info("Shutdown requested")
        finally:
            self._stop_agent()
            log_info("NEXUS supervisor stopped")

    def _start_agent(self):
        """Start the agent_ui.py process."""
        self._stop_agent()

        cmd = [
            sys.executable, str(self.base_dir / AGENT_SCRIPT),
            "--port", str(self.port),
            "--workspace", self.workspace,
            "--model", self.model,
        ]
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])

        env = os.environ.copy()
        if self.api_key:
            env["GEMINI_API_KEY"] = self.api_key

        log_info(f"Starting agent on port {self.port}...")
        self.agent_proc = subprocess.Popen(
            cmd, env=env, cwd=str(self.base_dir),
        )

        # Wait for health
        if health_check(self.port, timeout=HEALTH_TIMEOUT):
            log_ok(f"Agent is live at http://localhost:{self.port}")
            self.restart_count = 0  # Reset crash counter on successful start
        else:
            log_warn("Agent started but health check didn't pass (may still be loading)")

    def _stop_agent(self):
        """Stop the running agent process."""
        if self.agent_proc and self.agent_proc.poll() is None:
            log_info("Stopping agent...")
            self.agent_proc.terminate()
            try:
                self.agent_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.agent_proc.kill()
                self.agent_proc.wait(timeout=5)
            log_ok("Agent stopped")

    def _handle_self_modify(self):
        """Process a self-modification request from the agent."""
        log_info("Self-modification signal detected!")

        try:
            data = json.loads(self.signal_file.read_text(encoding="utf-8"))
        except Exception as e:
            log_err(f"Failed to read signal file: {e}")
            self.signal_file.unlink(missing_ok=True)
            return

        # Remove signal file
        self.signal_file.unlink(missing_ok=True)

        # Stop the current agent
        self._stop_agent()

        # Process the modification
        success, message = self.mod_engine.process_request(data)

        if success:
            log_ok(message)
        else:
            log_err(message)
            log_info("Starting agent with original code...")

        # Restart the agent (with new or original code)
        self._start_agent()


# ── Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NEXUS — Self-Evolving AI System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
This is the immortal supervisor. It manages agent_ui.py and handles
self-modification safely (backup → validate → clone-test → swap → restart).

Examples:
    python nexus.py --workspace ./myproject
    python nexus.py --port 9000 --model gemini-2.0-flash
""",
    )
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="gemini-2.0-flash")
    args = parser.parse_args()

    supervisor = NexusSupervisor(args)
    supervisor.start()


if __name__ == "__main__":
    main()
