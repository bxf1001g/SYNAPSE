"""
agent.py — Bidirectional multi-agent terminal communication system.

Both peers can chat and execute commands on each other's machine.

Usage:
  Terminal 1 (listen):   python agent.py server [port]
  Terminal 2 (connect):  python agent.py connect [host] [port]

Commands (type in the prompt):
  /exec <command>    — Execute a command on the REMOTE terminal
  /autoaccept on|off — Toggle auto-accepting incoming commands (default: off)
  /quit              — Disconnect and exit
  Anything else      — Sends as a chat message
"""

import socket
import subprocess
import sys
import threading
import uuid
import os

from protocol import encode_message, decode_message, make_chat, make_cmd, make_cmd_result, make_system

DEFAULT_PORT = 5000
PROMPT = "  You > "

# ANSI colors for terminal output
C_RESET = "\033[0m"
C_PEER = "\033[96m"     # cyan — peer chat
C_CMD = "\033[93m"      # yellow — incoming command requests
C_RESULT = "\033[92m"   # green — command results
C_SYSTEM = "\033[90m"   # gray — system messages
C_ERROR = "\033[91m"    # red — errors


class Agent:
    def __init__(self, sock: socket.socket, name: str):
        self.sock = sock
        self.name = name
        self.auto_accept = False
        self.running = True
        self._lock = threading.Lock()

    # ── Sending ──────────────────────────────────────────────

    def send(self, msg: dict):
        try:
            self.sock.sendall(encode_message(msg))
        except OSError:
            self.running = False

    # ── Receiving ────────────────────────────────────────────

    def receive_loop(self):
        """Continuously read messages from peer and handle them."""
        try:
            while self.running:
                msg = decode_message(self.sock)
                if msg is None:
                    self._print_system("Peer disconnected.")
                    self.running = False
                    break
                self._handle_message(msg)
        except OSError:
            self._print_system("Connection lost.")
            self.running = False

    def _handle_message(self, msg: dict):
        msg_type = msg.get("type")

        if msg_type == "chat":
            self._print_line(f"{C_PEER}  Peer > {msg['body']}{C_RESET}")

        elif msg_type == "cmd":
            self._handle_incoming_cmd(msg)

        elif msg_type == "cmd_result":
            self._print_cmd_result(msg)

        elif msg_type == "system":
            self._print_system(msg["body"])

    # ── Incoming command execution ───────────────────────────

    def _handle_incoming_cmd(self, msg: dict):
        cmd_id = msg["id"]
        cmd_body = msg["body"]

        self._print_line(f"\n{C_CMD}  ⚡ Peer wants to execute: {cmd_body}{C_RESET}")

        if self.auto_accept:
            self._print_line(f"{C_SYSTEM}  [auto-accepted]{C_RESET}")
            accepted = True
        else:
            self._print_line(f"{C_CMD}  Allow? [y/N]: {C_RESET}", end="")
            try:
                answer = input().strip().lower()
                accepted = answer in ("y", "yes")
            except EOFError:
                accepted = False

        if not accepted:
            self.send(make_cmd_result(cmd_id, -1, "", "Command rejected by peer."))
            self._print_system("Command rejected.")
            return

        self._print_system(f"Executing: {cmd_body}")
        threading.Thread(target=self._execute_cmd, args=(cmd_id, cmd_body), daemon=True).start()

    def _execute_cmd(self, cmd_id: str, cmd_body: str):
        """Run a command locally and send the result back to peer."""
        try:
            result = subprocess.run(
                cmd_body,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.send(make_cmd_result(cmd_id, result.returncode, result.stdout, result.stderr))
        except subprocess.TimeoutExpired:
            self.send(make_cmd_result(cmd_id, -1, "", "Command timed out (60s limit)."))
        except Exception as e:
            self.send(make_cmd_result(cmd_id, -1, "", f"Execution error: {e}"))

    # ── Display helpers ──────────────────────────────────────

    def _print_cmd_result(self, msg: dict):
        code = msg["exit_code"]
        stdout = msg.get("stdout", "")
        stderr = msg.get("stderr", "")
        color = C_RESULT if code == 0 else C_ERROR

        self._print_line(f"\n{color}  ┌─ Command Result (exit code: {code}) ─┐{C_RESET}")
        if stdout:
            for line in stdout.rstrip().split("\n"):
                self._print_line(f"{color}  │ {line}{C_RESET}")
        if stderr:
            for line in stderr.rstrip().split("\n"):
                self._print_line(f"{C_ERROR}  │ [err] {line}{C_RESET}")
        self._print_line(f"{color}  └───────────────────────────────────┘{C_RESET}")
        print(PROMPT, end="", flush=True)

    def _print_system(self, text: str):
        self._print_line(f"{C_SYSTEM}  [{text}]{C_RESET}")

    def _print_line(self, text: str, end="\n"):
        with self._lock:
            print(f"\r{text}", end=end, flush=True)
            if end == "\n":
                print(PROMPT, end="", flush=True)

    # ── Input loop ───────────────────────────────────────────

    def input_loop(self):
        """Read user input and dispatch chat messages or commands."""
        print(f"{C_SYSTEM}  [Connected! Commands: /exec <cmd> | /autoaccept on|off | /quit]{C_RESET}")
        print(f"{C_SYSTEM}  [Type a message to chat, or /exec <cmd> to run on the remote peer]{C_RESET}\n")

        try:
            while self.running:
                try:
                    user_input = input(PROMPT).strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                if user_input.lower() == "/quit":
                    self.send(make_system("Peer disconnected."))
                    self.running = False
                    break

                elif user_input.lower().startswith("/exec "):
                    cmd_body = user_input[6:].strip()
                    if not cmd_body:
                        self._print_system("Usage: /exec <command>")
                        continue
                    cmd_id = str(uuid.uuid4())[:8]
                    self._print_system(f"Sending command to peer: {cmd_body}")
                    self.send(make_cmd(cmd_id, cmd_body))

                elif user_input.lower().startswith("/autoaccept"):
                    arg = user_input[11:].strip().lower()
                    if arg == "on":
                        self.auto_accept = True
                        self._print_system("Auto-accept enabled — incoming commands will run without prompt.")
                    elif arg == "off":
                        self.auto_accept = False
                        self._print_system("Auto-accept disabled — you'll be asked before running commands.")
                    else:
                        self._print_system("Usage: /autoaccept on|off")

                else:
                    self.send(make_chat(user_input))

        except KeyboardInterrupt:
            pass

        self.running = False
        try:
            print(f"\n{C_SYSTEM}  [Session ended]{C_RESET}")
        except OSError:
            pass


# ── Entry point ──────────────────────────────────────────────

def start_server(port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        s.listen(1)
        print(f"{C_SYSTEM}  [Agent listening on port {port} — waiting for peer...]{C_RESET}")

        conn, addr = s.accept()
        print(f"{C_SYSTEM}  [Peer connected from {addr[0]}:{addr[1]}]{C_RESET}")
        run_agent(conn, f"server:{port}")


def start_client(host: str, port: int):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"{C_SYSTEM}  [Connecting to {host}:{port}...]{C_RESET}")
    try:
        s.connect((host, port))
    except ConnectionRefusedError:
        print(f"{C_ERROR}  [Error: Connection refused. Is the server agent running?]{C_RESET}")
        return
    print(f"{C_SYSTEM}  [Connected to {host}:{port}]{C_RESET}")
    run_agent(s, f"client:{host}:{port}")


def run_agent(sock: socket.socket, name: str):
    agent = Agent(sock, name)
    receiver = threading.Thread(target=agent.receive_loop, daemon=True)
    receiver.start()
    agent.input_loop()
    sock.close()


def print_usage():
    print(f"""
{C_SYSTEM}╔══════════════════════════════════════════════════╗
║        Multi-Agent Terminal System                ║
╠══════════════════════════════════════════════════╣
║  Server:  python agent.py server [port]          ║
║  Client:  python agent.py connect [host] [port]  ║
║                                                  ║
║  Default host: 127.0.0.1  |  Default port: 5000  ║
╚══════════════════════════════════════════════════╝{C_RESET}
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT
        start_server(port)

    elif mode == "connect":
        host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_PORT
        start_client(host, port)

    else:
        print_usage()
        sys.exit(1)
