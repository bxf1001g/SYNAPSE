"""
protocol.py — Message framing and encoding for the multi-agent system.

Messages are JSON objects sent over TCP with a 4-byte big-endian length prefix.

Message types:
  - chat:       {"type": "chat", "body": "hello"}
  - cmd:        {"type": "cmd", "id": "uuid", "body": "dir /b"}
  - cmd_result: {"type": "cmd_result", "id": "uuid", "exit_code": 0, "stdout": "...", "stderr": "..."}
  - system:     {"type": "system", "body": "peer disconnected"}
"""

import json
import socket
import struct

HEADER_SIZE = 4  # 4-byte big-endian length prefix


def encode_message(msg: dict) -> bytes:
    """Encode a message dict to length-prefixed JSON bytes."""
    payload = json.dumps(msg).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Receive exactly n bytes from a socket. Returns None on disconnect."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def decode_message(sock: socket.socket) -> dict | None:
    """Read one length-prefixed JSON message from a socket. Returns None on disconnect."""
    header = recv_exact(sock, HEADER_SIZE)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    payload = recv_exact(sock, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def make_chat(body: str) -> dict:
    return {"type": "chat", "body": body}


def make_cmd(cmd_id: str, body: str) -> dict:
    return {"type": "cmd", "id": cmd_id, "body": body}


def make_cmd_result(cmd_id: str, exit_code: int, stdout: str, stderr: str) -> dict:
    return {"type": "cmd_result", "id": cmd_id, "exit_code": exit_code, "stdout": stdout, "stderr": stderr}


def make_system(body: str) -> dict:
    return {"type": "system", "body": body}
