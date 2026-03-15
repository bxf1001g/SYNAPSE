"""
ai_agent.py — Autonomous AI Multi-Agent System (v2)

Two Gemini-powered agents (Architect + Developer) collaborate over TCP
to build software. Features:
  - Planning phase: Agents create a plan, user approves before building
  - Visible communication: Full agent-to-agent messages displayed clearly
  - Conversation log: Saved to workspace/.conversation.log

Usage:
  set GEMINI_API_KEY=your-key-here

  Terminal 1 (Architect):  python ai_agent.py server [--workspace ./project]
  Terminal 2 (Developer):  python ai_agent.py connect [--workspace ./project]
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Missing dependency. Run: pip install google-genai")
    sys.exit(1)

from protocol import decode_message, encode_message

# ── Terminal Colors ──────────────────────────────────────────

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_THINK = "\033[90m"
C_PEER = "\033[96m"
C_CMD = "\033[93m"
C_FILE = "\033[95m"
C_RESULT = "\033[92m"
C_ERROR = "\033[91m"
C_SYS = "\033[94m"
C_PLAN = "\033[97m"
C_ARCH = "\033[36m"
C_DEV = "\033[33m"

LINE = "─" * 60

# ── System Prompts ───────────────────────────────────────────

PLAN_PROMPT = """You are a software architect. Given a project request, create a detailed implementation plan.

You MUST respond with ONLY a valid JSON object using EXACTLY these keys:
{
  "project_name": "Short project name",
  "description": "One-paragraph project description",
  "tech_stack": ["tech1", "tech2"],
  "files_to_create": ["file1.ext", "file2.ext"],
  "steps": [
    {"step": 1, "title": "Step title", "details": "Detailed description of what to do"},
    {"step": 2, "title": "Step title", "details": "Detailed description of what to do"}
  ]
}

IMPORTANT: The key MUST be "steps" (not "stages" or anything else).
Do NOT wrap the JSON in markdown code fences. Just output raw JSON.
Be thorough but practical. Typically 3-8 steps.
"""

ARCHITECT_PROMPT = """\
You are the ARCHITECT agent in a two-AI-agent system that builds software.
Your partner is the DEVELOPER agent in another terminal.

YOUR ROLE:
- Follow the approved plan but GROUP related steps into LARGE batches
- Send COMPLETE, DETAILED specs — include exact file contents, all logic, full requirements
- DO NOT micro-manage — avoid single-import or single-line instructions
- Each message to the Developer should contain EVERYTHING needed for a meaningful chunk of work
- Review the Developer's reports and verify with automated tests
- Signal DONE only when the full project is working and verified

EFFICIENCY RULES (critical):
- Combine 2-4 plan steps into ONE message when they are related
- Instead of "add imports" then "add function" then "add error handling" — send ALL of it at once
- Your goal is to finish in 2-4 turns, NOT 7-10 turns
- Each instruction should produce a WORKING, TESTABLE result
- Example: "Create index.html with header, nav, main content, footer, AND style.css with full styling"
  NOT: "First create index.html" ... "Now add header" ... "Now add nav" ... etc.

WORKSPACE: {workspace}
PLATFORM: Windows 11 — use Windows commands (type, dir, python, etc.), NOT Linux (cat, ls).

YOU HAVE FULL POWER TO:
- Install ANY packages (pip install, npm install, etc.)
- Create ANY script (Python, Node.js, batch, PowerShell) for testing, automation, or utilities
- Run Selenium/Playwright/pytest/unittest or any test framework
- Open browsers, run servers, hit APIs, parse files — anything a developer can do
- Create temporary helper scripts for verification (these are auto-cleaned after project ends)

When you need to test: create a script action and run it. DO NOT just visually inspect.

RESPOND WITH ONLY A VALID JSON OBJECT — no markdown fences:
{{
  "thinking": "your private reasoning (not sent to peer)",
  "actions": [
    {{"type": "message", "content": "instruction or feedback for the Developer"}},
    {{"type": "command", "cmd": "shell command to run in workspace"}},
    {{"type": "file", "path": "relative/path", "content": "COMPLETE file content"}},
    {{"type": "script", "name": "test_site.py", "lang": "python", "content": "script code here"}},
    {{"type": "done", "summary": "final project summary"}}
  ]
}}

ACTION TYPES:
- "file": Create/overwrite a file permanently in the workspace
- "command": Run a shell command (pip install, dir, python app.py, etc.)
- "script": Create a temp script, run it, get output, auto-cleanup. Use for one-off tests.
  Supported langs: python, node, bat, powershell
- "message": Send text to peer agent (exactly ONE per response)
- "done": Signal project completion

RULES:
1. Send exactly ONE "message" action per response — make it COMPREHENSIVE
2. Include FULL file contents, exact code, all logic in your instructions
3. After the Developer reports success, VERIFY by running automated tests/scripts
4. When updating files, provide the COMPLETE file — remove any placeholder content
5. Signal "done" ONLY after automated verification passes
6. Install packages with pip/npm as needed
7. NEVER ask the Developer to "show file contents" or do trivial single-line changes
"""

DEVELOPER_PROMPT = """\
You are the DEVELOPER agent in a two-AI-agent system that builds software.
Your partner is the ARCHITECT agent in another terminal.

YOUR ROLE:
- Receive step-by-step instructions from the Architect
- Create files with COMPLETE, WORKING code (no placeholders or TODOs)
- Run commands to build and test your work
- Report results honestly to the Architect
- Fix bugs based on Architect's feedback

WORKSPACE: {workspace}
PLATFORM: Windows 11 — use Windows commands (type, dir, python, etc.), NOT Linux (cat, ls).

YOU HAVE FULL POWER TO:
- Install ANY packages needed: pip install, npm install, etc.
- Create ANY script for ANY purpose: testing, setup, automation, utilities
- Use Selenium, Playwright, pytest, requests, or any tool that helps
- Run servers (Flask, FastAPI, http.server, etc.)
- Execute build tools (webpack, tsc, etc.)
- Create helper/utility scripts (e.g., seed_data.py, run_tests.py)

When building a website: also create a test script (Playwright/Selenium) to verify it loads
When building an API: also create a test script (requests/httpx) to verify endpoints
When building a CLI: also test it by running it with sample inputs
ALWAYS install dependencies before running scripts that need them.

RESPOND WITH ONLY A VALID JSON OBJECT — no markdown fences:
{{
  "thinking": "your private reasoning (not sent to peer)",
  "actions": [
    {{"type": "command", "cmd": "pip install playwright && playwright install chromium"}},
    {{"type": "file", "path": "relative/path", "content": "COMPLETE file content"}},
    {{"type": "script", "name": "verify.py", "lang": "python", "content": "test script code"}},
    {{"type": "message", "content": "status report for the Architect"}},
    {{"type": "done", "summary": "implementation complete summary"}}
  ]
}}

ACTION TYPES:
- "command": Run a shell command (pip install, npm install, dir, python app.py, etc.)
- "file": Create/overwrite a file permanently in the workspace
- "script": Create a temp script, run it, get output, auto-cleanup. Great for quick tests.
  Supported langs: python, node, bat, powershell
- "message": Send text to peer agent (exactly ONE per response)
- "done": Signal project completion

RULES:
1. Install dependencies FIRST (pip install X), then create files, then run/test
2. Write COMPLETE code — no placeholders, no "TODO", no "pass"
3. When updating a file, write the ENTIRE file content — remove old placeholder content
4. Always test by running the code after creating files
5. If a test fails, fix it before reporting
6. Always include a "message" action to report status
7. Order: install deps → create files → run commands/scripts → message
"""


# ── JSON Parsing ─────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """Robustly extract a JSON object from Gemini's response."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
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


# ── AI Agent ─────────────────────────────────────────────────

class AIAgent:
    def __init__(self, role: str, sock: socket.socket, workspace: str,
                 api_key: str, model_name: str, max_turns: int = 50):
        self.role = role
        self.sock = sock
        self.workspace = os.path.abspath(workspace)
        self.running = True
        self.turn = 0
        self.max_turns = max_turns
        self.max_local_iters = 8
        self.files_created = []
        self._temp_scripts = []  # Track scripts for auto-cleanup at end
        self.api_key = api_key
        self.model_name = model_name

        os.makedirs(self.workspace, exist_ok=True)

        # Log file
        self.log_path = os.path.join(self.workspace, ".conversation.log")
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("# Agent Conversation Log\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Role: {role}\n\n")

        prompt_template = ARCHITECT_PROMPT if role == "architect" else DEVELOPER_PROMPT
        system_instruction = prompt_template.format(workspace=self.workspace)

        self.client = genai.Client(api_key=api_key)
        self.chat = self.client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=8192,
            ),
        )

    # ── Conversation Logging ─────────────────────────────────

    def _log_to_file(self, text: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _show_message_exchange(self, direction: str, sender: str, receiver: str, message: str):
        """Display a clearly formatted inter-agent message."""
        arrow = "→" if direction == "out" else "←"
        color = C_ARCH if "Architect" in sender else C_DEV
        icon = "📤" if direction == "out" else "📥"

        print(f"\n{C_DIM}  {LINE}{C_RESET}")
        print(f"  {color}{C_BOLD}{icon} Turn {self.turn}  │  {sender} {arrow} {receiver}{C_RESET}")
        print(f"{C_DIM}  {LINE}{C_RESET}")

        for line in message.split("\n"):
            print(f"  {color}  {line}{C_RESET}")

        print(f"{C_DIM}  {LINE}{C_RESET}\n")

        # Also log to file
        self._log_to_file(f"\n{'=' * 60}")
        self._log_to_file(f"Turn {self.turn} | {sender} {arrow} {receiver}")
        self._log_to_file(f"{'=' * 60}")
        self._log_to_file(message)

    # ── Planning Phase ───────────────────────────────────────

    def create_plan(self, task: str) -> dict | None:
        """Use Gemini to create a project plan (separate from main chat)."""
        self._log_sys("Creating project plan...")

        plan_client = genai.Client(api_key=self.api_key)
        try:
            response = plan_client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=PLAN_PROMPT,
                    temperature=0.7,
                    max_output_tokens=4096,
                ),
                contents=f"Create a plan for: {task}",
            )
            return parse_json_response(response.text)
        except Exception as e:
            self._log_err(f"Planning failed: {e}")
            return None

    def _classify_task(self, task: str) -> str:
        """Classify user input as 'question' or 'build'. Returns 'question' or 'build'."""
        classify_prompt = (
            "Classify the following user input as either 'question' or 'build'.\n"
            "- 'question': The user is asking a question, requesting info, or asking about existing files/folders.\n"
            "  Examples: 'what files are here?', 'what is Python?', 'how does X work?', 'list the folder'\n"
            "- 'build': The user wants to create, build, or make something new.\n"
            "  Examples: 'make a website', 'create a calculator app', 'build a REST API'\n\n"
            "Respond with ONLY the word 'question' or 'build'. Nothing else.\n\n"
            f"User input: {task}"
        )
        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10),
                contents=classify_prompt,
            )
            result = response.text.strip().lower()
            if "question" in result:
                return "question"
            return "build"
        except Exception:
            return "build"  # Default to build if classification fails

    def _answer_question(self, question: str):
        """Answer a question using full scripting power — can run commands, create scripts, fetch data."""
        # Gather workspace context
        files_info = []
        try:
            for item in os.listdir(self.workspace):
                full = os.path.join(self.workspace, item)
                if os.path.isfile(full):
                    size = os.path.getsize(full)
                    files_info.append(f"  📄 {item} ({size} bytes)")
                elif os.path.isdir(full):
                    files_info.append(f"  📁 {item}/")
        except Exception as e:
            files_info.append(f"  (error listing: {e})")

        workspace_listing = "\n".join(files_info) if files_info else "  (empty)"

        system_prompt = (
            "You are a helpful AI assistant with FULL access to a local machine.\n"
            "You can run ANY command, create ANY script, install ANY package, fetch ANY data.\n\n"
            "PLATFORM: Windows 11. Use Windows commands (type, dir, python, curl, etc.).\n"
            f"WORKSPACE: {self.workspace}\n\n"
            "To answer the user's question, you may need to:\n"
            "- Run shell commands (curl, python scripts, pip install, etc.)\n"
            "- Create and run Python scripts (requests, urllib, web scraping, etc.)\n"
            "- Read files, browse directories, parse data\n"
            "- Install packages if needed (pip install requests, etc.)\n\n"
            "RESPOND WITH ONLY A VALID JSON OBJECT:\n"
            "{\n"
            '  "thinking": "your reasoning",\n'
            '  "actions": [\n'
            '    {"type": "command", "cmd": "shell command"},\n'
            '    {"type": "script", "name": "fetch.py", "lang": "python", "content": "script code"},\n'
            '    {"type": "answer", "content": "your final answer to the user"}\n'
            "  ]\n"
            "}\n\n"
            "ALWAYS end with an 'answer' action containing your final response.\n"
            "If you need data from the internet, create a Python script using 'requests' or 'urllib'.\n"
            "Install packages first if needed (pip install requests)."
        )

        prompt = (
            f"Current workspace contents:\n{workspace_listing}\n\n"
            f"User question: {question}\n\n"
            f"Answer this question. Use commands or scripts if you need live data."
        )

        try:
            client = genai.Client(api_key=self.api_key)
            chat = client.chats.create(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=4096,
                ),
            )

            cmd_output = ""
            for iteration in range(self.max_local_iters):
                response = chat.send_message(prompt if iteration == 0 else
                    f"Command results:\n\n{cmd_output}\n\nNow provide your final 'answer' action.")
                parsed = parse_json_response(response.text)

                thinking = parsed.get("thinking", "")
                actions = parsed.get("actions", [])
                if thinking:
                    self._log_think(thinking)

                cmd_output = ""
                final_answer = None

                for action in actions:
                    atype = action.get("type", "")
                    if atype == "command":
                        out = self._do_command(action)
                        if out:
                            cmd_output += out + "\n"
                    elif atype == "script":
                        out = self._do_script(action)
                        if out:
                            cmd_output += out + "\n"
                    elif atype == "answer":
                        final_answer = action.get("content", "")

                if final_answer:
                    print(f"\n  {C_RESULT}{final_answer}{C_RESET}\n")
                    self._log_to_file(f"\n## Answer\n{final_answer}")
                    self._cleanup_temp_scripts()
                    return

                if not cmd_output:
                    break  # No commands ran and no answer — bail

            # Fallback if no explicit answer action
            print(f"\n  {C_DIM}(Agent ran commands but didn't produce a final answer){C_RESET}\n")
        except Exception as e:
            self._log_err(f"Failed to answer: {e}")

    def display_plan(self, plan: dict):
        """Show the plan in a nice format."""
        name = plan.get("project_name", "Untitled")
        desc = plan.get("description", "")
        stack = plan.get("tech_stack", [])
        files = plan.get("files_to_create", [])
        steps = plan.get("steps") or plan.get("stages") or []

        print(f"\n{C_BOLD}{C_PLAN}  ╔{'═' * 56}╗")
        print(f"  ║  📋 PROJECT PLAN{' ' * 39}║")
        print(f"  ╠{'═' * 56}╣")
        print(f"  ║  {name:<54}║")
        print(f"  ╚{'═' * 56}╝{C_RESET}\n")

        if desc:
            print(f"  {C_SYS}Description:{C_RESET} {desc}\n")

        if stack:
            print(f"  {C_SYS}Tech Stack:{C_RESET}  {', '.join(stack)}")

        if files:
            print(f"  {C_SYS}Files:{C_RESET}       {', '.join(files)}\n")

        if steps:
            print(f"  {C_SYS}Steps:{C_RESET}")
            for s in steps:
                num = s.get("step", "?")
                title = s.get("title") or s.get("name", "")
                details = s.get("details") or s.get("description", "")
                print(f"    {C_BOLD}{num}. {title}{C_RESET}")
                if details:
                    # Wrap long lines
                    for line in details.split("\n"):
                        print(f"       {C_DIM}{line}{C_RESET}")
            print()

        # Log to file
        self._log_to_file("\n## Project Plan\n")
        self._log_to_file(json.dumps(plan, indent=2))

    # ── Core Turn Logic ──────────────────────────────────────

    def process_turn(self, input_text: str):
        """One turn: call Gemini → execute actions → iterate if needed → message peer."""
        self.turn += 1
        current_input = input_text

        peer_message = None
        is_done = False
        done_summary = ""

        for iteration in range(self.max_local_iters):
            self._log_sys(f"Turn {self.turn}.{iteration + 1} — thinking...")

            try:
                response = self.chat.send_message(current_input)
                raw = response.text
            except Exception as e:
                self._log_err(f"Gemini API error: {e}")
                self._send_peer({"type": "chat", "body": f"[Agent error: {e}]"})
                return

            parsed = parse_json_response(raw)
            thinking = parsed.get("thinking", "")
            actions = parsed.get("actions", [])

            if thinking:
                self._log_think(thinking)

            if not actions:
                self._log_sys("No actions returned.")
                return

            cmd_results = []
            for action in actions:
                atype = action.get("type", "")

                if atype == "file":
                    self._do_file(action)
                elif atype == "command":
                    out = self._do_command(action)
                    if out:
                        cmd_results.append(out)
                elif atype == "script":
                    # Create-and-run in one shot (for test/verification scripts)
                    out = self._do_script(action)
                    if out:
                        cmd_results.append(out)
                elif atype == "message":
                    peer_message = action.get("content", "")
                elif atype == "done":
                    is_done = True
                    done_summary = action.get("summary", "Project complete.")
                    if not peer_message:
                        peer_message = done_summary

            failed_cmds = [r for r in cmd_results if re.search(r"Exit code: [^0]", r)]

            if failed_cmds and not is_done:
                results_text = "\n\n".join(cmd_results)
                current_input = (
                    f"Some commands failed (non-zero exit code):\n\n{results_text}\n\n"
                    f"This is Windows — use Windows commands (type, dir, etc.).\n"
                    f"Fix the issues and include a 'message' action with your status."
                )
                peer_message = None
                continue

            break

        # Ensure a message is always sent
        if not is_done and not peer_message:
            try:
                summary_resp = self.chat.send_message(
                    "You did not include a 'message' action. "
                    "Send a brief status to your peer. Respond with JSON."
                )
                fallback = parse_json_response(summary_resp.text)
                for a in fallback.get("actions", []):
                    if a.get("type") == "message":
                        peer_message = a["content"]
                        break
            except Exception:
                pass
            if not peer_message and cmd_results:
                peer_message = f"Executed {len(cmd_results)} command(s). Results looked good."

        # Send to peer with clear display
        me = "🏗 Architect" if self.role == "architect" else "👷 Developer"
        them = "👷 Developer" if self.role == "architect" else "🏗 Architect"

        if is_done:
            self._show_done(done_summary)
            self._send_peer({"type": "done", "body": done_summary})
            self.running = False
        elif peer_message:
            self._show_message_exchange("out", me, them, peer_message)
            self._send_peer({"type": "chat", "body": peer_message})
        else:
            self._log_sys("(no message for peer this turn)")

    # ── Action Handlers ──────────────────────────────────────

    def _do_file(self, action: dict):
        path = action.get("path", "")
        content = action.get("content", "")
        if not path:
            return
        full = os.path.join(self.workspace, path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        self.files_created.append(path)
        self._log_file(path, len(content))

    def _do_script(self, action: dict) -> str | None:
        """Create a temporary script, run it, return output. For one-off tests/automation."""
        content = action.get("content", "")
        lang = action.get("lang", "python")
        name = action.get("name", f"_tmp_script.{'py' if lang == 'python' else 'js' if lang == 'node' else 'bat'}")
        if not content:
            return None

        full_path = os.path.join(self.workspace, name)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        self._temp_scripts.append(full_path)
        self._log_script(name, lang)

        runner = {"python": "python", "node": "node", "bat": "cmd /c", "powershell": "powershell -File"}.get(lang, lang)
        cmd = f"{runner} {name}"
        self._log_cmd(cmd)

        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                cwd=self.workspace, timeout=300,
            )
            out = ""
            if r.stdout:
                out += r.stdout
                for ln in r.stdout.rstrip().splitlines():
                    self._log_ok(ln)
            if r.stderr:
                out += f"\n[STDERR]:\n{r.stderr}"
                for ln in r.stderr.rstrip().splitlines():
                    self._log_err(ln)

            return f"Script: {name} ({lang})\nExit code: {r.returncode}\n{out}"
        except subprocess.TimeoutExpired:
            self._log_err(f"Script timed out: {name}")
            return f"Script: {name}\nERROR: timed out (300s)"
        except Exception as e:
            self._log_err(str(e))
            return f"Script: {name}\nERROR: {e}"

    def _do_command(self, action: dict) -> str | None:
        cmd = action.get("cmd", "")
        if not cmd:
            return None
        self._log_cmd(cmd)
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                cwd=self.workspace, timeout=300,
            )
            out = ""
            if r.stdout:
                out += r.stdout
                for ln in r.stdout.rstrip().splitlines():
                    self._log_ok(ln)
            if r.stderr:
                out += f"\n[STDERR]:\n{r.stderr}"
                for ln in r.stderr.rstrip().splitlines():
                    self._log_err(ln)
            return f"Command: {cmd}\nExit code: {r.returncode}\n{out}"
        except subprocess.TimeoutExpired:
            self._log_err(f"Timed out: {cmd}")
            return f"Command: {cmd}\nERROR: timed out (300s)"
        except Exception as e:
            self._log_err(str(e))
            return f"Command: {cmd}\nERROR: {e}"

    # ── Networking ───────────────────────────────────────────

    def _send_peer(self, msg: dict):
        try:
            self.sock.sendall(encode_message(msg))
        except OSError:
            self.running = False

    def _recv_peer(self) -> dict | None:
        try:
            return decode_message(self.sock)
        except OSError:
            self.running = False
            return None

    # ── Display Helpers ──────────────────────────────────────

    def _log_think(self, t):
        print(f"{C_THINK}  💭 {t}{C_RESET}")

    def _log_sys(self, t):
        tag = "🏗 Architect" if self.role == "architect" else "👷 Developer"
        print(f"{C_SYS}  [{tag}] {t}{C_RESET}")

    def _log_cmd(self, t):
        print(f"{C_CMD}  ⚙  $ {t}{C_RESET}")
        self._log_to_file(f"  [CMD] $ {t}")

    def _log_ok(self, t):
        print(f"{C_RESULT}  │ {t}{C_RESET}")

    def _log_err(self, t):
        print(f"{C_ERROR}  │ {t}{C_RESET}")

    def _log_file(self, path, size):
        print(f"{C_FILE}  📄 Created: {path}  ({size} chars){C_RESET}")
        self._log_to_file(f"  [FILE] {path} ({size} chars)")

    def _log_script(self, name, lang):
        print(f"{C_CMD}  🔧 Script: {name}  ({lang}){C_RESET}")
        self._log_to_file(f"  [SCRIPT] {name} ({lang})")

    def _cleanup_temp_scripts(self):
        """Remove all temporary/verification scripts created during the session."""
        project_files = set(os.path.abspath(os.path.join(self.workspace, f))
                           for f in self.files_created)
        removed = []
        for script_path in self._temp_scripts:
            abs_path = os.path.abspath(script_path)
            # Don't delete files that are part of the actual project
            if abs_path in project_files:
                continue
            try:
                if os.path.exists(abs_path):
                    os.remove(abs_path)
                    removed.append(os.path.basename(abs_path))
            except OSError:
                pass
        if removed:
            print(f"{C_DIM}  🧹 Cleaned up {len(removed)} temp script(s): {', '.join(removed)}{C_RESET}")
            self._log_to_file(f"  [CLEANUP] Removed temp scripts: {', '.join(removed)}")

    def _show_done(self, summary):
        # Clean up all temporary/verification scripts first
        self._cleanup_temp_scripts()

        print(f"\n{C_BOLD}{C_RESULT}  ╔{'═' * 56}╗")
        print(f"  ║  🎉 PROJECT COMPLETE{' ' * 36}║")
        print(f"  ╠{'═' * 56}╣")
        # Word-wrap summary
        words = summary.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > 54:
                print(f"  ║  {line:<54}║")
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            print(f"  ║  {line:<54}║")
        print(f"  ╚{'═' * 56}╝{C_RESET}")

        # Show files created
        unique = sorted(set(self.files_created))
        if unique:
            print(f"\n  {C_SYS}Files created/modified:{C_RESET}")
            for f in unique:
                print(f"    {C_FILE}📄 {f}{C_RESET}")

        print(f"\n  {C_DIM}Conversation log: {self.log_path}{C_RESET}\n")

        self._log_to_file(f"\n{'=' * 60}")
        self._log_to_file(f"PROJECT COMPLETE: {summary}")
        self._log_to_file(f"Files: {', '.join(unique)}")

    # ── Role Loops ───────────────────────────────────────────

    def run_architect(self):
        self._banner("🏗  ARCHITECT AGENT")
        print(f"  {C_SYS}Workspace: {self.workspace}{C_RESET}")
        print(f"  {C_BOLD}\n  What should the agents build?{C_RESET}")
        print(f"  {C_THINK}(Be specific: app type, features, tech stack){C_RESET}")

        try:
            task = input("\n  Task ▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            return

        if not task:
            print(f"{C_ERROR}  No task given.{C_RESET}")
            return

        self._log_to_file(f"\n## User Task\n{task}\n")

        # ── Pre-check: Question or Build task? ────────────────
        task_type = self._classify_task(task)

        if task_type == "question":
            # Answer the question directly — no need for a build plan
            print(f"\n{C_SYS}  💡 This looks like a question — answering directly...{C_RESET}\n")
            self._answer_question(task)
            return

        # ── Phase 1: Planning ─────────────────────────────────
        print(f"\n{C_SYS}  {'─' * 50}")
        print("  📋 Phase 1: Creating project plan...")
        print(f"  {'─' * 50}{C_RESET}\n")

        plan = self.create_plan(task)

        if plan and (plan.get("steps") or plan.get("stages")):
            self.display_plan(plan)

            print(f"  {C_BOLD}Review the plan above.{C_RESET}")
            try:
                approval = input("  Accept and start building? [Y/n] ▶ ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return

            if approval in ("n", "no"):
                print(f"\n  {C_SYS}Plan rejected. Exiting.{C_RESET}")
                return

            plan_text = json.dumps(plan, indent=2)
            first_step_prompt = (
                f"The user approved this plan:\n\n{plan_text}\n\n"
                f"Now send instructions to the Developer. IMPORTANT:\n"
                f"- Combine related steps into ONE comprehensive message\n"
                f"- Include COMPLETE file contents — not just descriptions\n"
                f"- Aim to finish the entire project in 2-4 total turns, not 7-10\n"
                f"- Each message should contain ALL the code and details for a meaningful chunk\n"
                f"- Do NOT send single-import or single-line instructions"
            )
        else:
            self._log_sys("Could not generate structured plan. Proceeding directly.")
            first_step_prompt = (
                f"The user has requested this project:\n\n{task}\n\n"
                f"Create a plan, then send the FIRST step to the Developer."
            )

        # ── Phase 2: Building ─────────────────────────────────
        print(f"\n{C_SYS}  {'─' * 50}")
        print("  🚀 Phase 2: Building — agents collaborating...")
        print(f"  {'─' * 50}{C_RESET}")

        self.process_turn(first_step_prompt)

        while self.running and self.turn < self.max_turns:
            msg = self._recv_peer()
            if msg is None:
                self._log_sys("Developer disconnected.")
                break

            me = "🏗 Architect"
            them = "👷 Developer"
            body = msg.get("body", "")

            if msg.get("type") == "done":
                self._show_message_exchange("in", them, me, f"[DONE] {body}")
                self.process_turn(
                    f"Developer reports DONE:\n\n{body}\n\n"
                    f"Run commands to verify. If satisfied, signal 'done'. Otherwise send feedback."
                )
            else:
                self._show_message_exchange("in", them, me, body)
                self.process_turn(
                    f"Developer reports:\n\n{body}\n\n"
                    f"Review this. If step succeeded, send the NEXT BATCH of work "
                    f"(combine multiple remaining steps into one message). "
                    f"If problems, send corrections. If ALL done, verify and signal 'done'."
                )

        if self.turn >= self.max_turns:
            self._log_sys(f"Reached max turns ({self.max_turns}).")

    def run_developer(self):
        self._banner("👷 DEVELOPER AGENT")
        print(f"  {C_SYS}Workspace: {self.workspace}{C_RESET}")
        print(f"  {C_SYS}Waiting for Architect's instructions...{C_RESET}\n")

        while self.running and self.turn < self.max_turns:
            msg = self._recv_peer()
            if msg is None:
                self._log_sys("Architect disconnected.")
                break

            me = "👷 Developer"
            them = "🏗 Architect"
            body = msg.get("body", "")

            if msg.get("type") == "done":
                self._show_message_exchange("in", them, me, f"[DONE] {body}")
                self._show_done(body)
                self.running = False
                break

            self._show_message_exchange("in", them, me, body)
            self.process_turn(
                f"Architect instructs:\n\n{body}\n\n"
                f"Implement this step: create files, run commands to test, "
                f"and report your results to the Architect."
            )

        if self.turn >= self.max_turns:
            self._log_sys(f"Reached max turns ({self.max_turns}).")

    def _banner(self, title: str):
        w = 56
        print(f"\n{C_BOLD}{C_SYS}  ╔{'═' * w}╗")
        print(f"  ║  {title:<{w - 3}}║")
        print(f"  ╚{'═' * w}╝{C_RESET}\n")


# ── Entry Point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Multi-Agent System v2 — plan, approve, build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  Terminal 1:  python ai_agent.py server --workspace ./myapp
  Terminal 2:  python ai_agent.py connect --workspace ./myapp

Set your API key:
  set GEMINI_API_KEY=your-key-here   (Windows)
  export GEMINI_API_KEY=your-key     (Linux/Mac)
""",
    )
    parser.add_argument("mode", choices=["server", "connect"],
                        help="server = Architect, connect = Developer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--workspace", default="./workspace")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--max-turns", type=int, default=50)

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"{C_ERROR}  ✖ No API key. Set GEMINI_API_KEY or use --api-key{C_RESET}")
        sys.exit(1)

    if args.mode == "server":
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", args.port))
            s.listen(1)
            print(f"{C_SYS}  Listening on port {args.port} — waiting for Developer...{C_RESET}")
            conn, addr = s.accept()
            print(f"{C_SYS}  Developer connected from {addr[0]}:{addr[1]}{C_RESET}")

            agent = AIAgent("architect", conn, args.workspace, api_key, args.model, args.max_turns)
            try:
                agent.run_architect()
            finally:
                conn.close()

    elif args.mode == "connect":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((args.host, args.port))
        except ConnectionRefusedError:
            print(f"{C_ERROR}  ✖ Cannot connect. Is the Architect running?{C_RESET}")
            sys.exit(1)

        print(f"{C_SYS}  Connected to Architect at {args.host}:{args.port}{C_RESET}")
        agent = AIAgent("developer", sock, args.workspace, api_key, args.model, args.max_turns)
        try:
            agent.run_developer()
        finally:
            sock.close()


if __name__ == "__main__":
    main()
