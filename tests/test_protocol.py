"""Tests for protocol.py — message framing and encoding for the multi-agent system."""

import json
import struct

from protocol import (
    HEADER_SIZE,
    decode_message,
    encode_message,
    make_chat,
    make_cmd,
    make_cmd_result,
    make_system,
    recv_exact,
)

# ── encode_message tests ────────────────────────────────────────


class TestEncodeMessage:
    def test_basic_chat_message(self):
        msg = {"type": "chat", "body": "hello"}
        result = encode_message(msg)
        payload = json.dumps(msg).encode("utf-8")
        expected = struct.pack(">I", len(payload)) + payload
        assert result == expected

    def test_header_is_4_bytes(self):
        msg = {"type": "chat", "body": "test"}
        result = encode_message(msg)
        assert len(result[:HEADER_SIZE]) == 4

    def test_length_prefix_matches_payload(self):
        msg = {"type": "cmd", "id": "abc", "body": "ls -la"}
        result = encode_message(msg)
        length = struct.unpack(">I", result[:4])[0]
        payload = result[4:]
        assert length == len(payload)

    def test_payload_is_valid_json(self):
        msg = {"type": "system", "body": "peer disconnected"}
        result = encode_message(msg)
        payload = result[4:]
        decoded = json.loads(payload.decode("utf-8"))
        assert decoded == msg

    def test_unicode_content(self):
        msg = {"type": "chat", "body": "héllo wörld 🌍"}
        result = encode_message(msg)
        length = struct.unpack(">I", result[:4])[0]
        payload = result[4:]
        assert len(payload) == length
        assert json.loads(payload.decode("utf-8")) == msg

    def test_empty_body(self):
        msg = {"type": "chat", "body": ""}
        result = encode_message(msg)
        payload = result[4:]
        assert json.loads(payload.decode("utf-8"))["body"] == ""

    def test_large_message(self):
        msg = {"type": "chat", "body": "x" * 100_000}
        result = encode_message(msg)
        length = struct.unpack(">I", result[:4])[0]
        assert length == len(result[4:])


# ── recv_exact tests ────────────────────────────────────────────


class _FakeSocket:
    """Mimics socket.recv() for testing."""

    def __init__(self, data: bytes, chunk_size: int = 0):
        self._data = data
        self._pos = 0
        self._chunk_size = chunk_size

    def recv(self, n: int) -> bytes:
        if self._pos >= len(self._data):
            return b""
        if self._chunk_size > 0:
            n = min(n, self._chunk_size)
        end = min(self._pos + n, len(self._data))
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk


class TestRecvExact:
    def test_receives_exact_bytes(self):
        sock = _FakeSocket(b"hello world")
        result = recv_exact(sock, 5)
        assert result == b"hello"

    def test_returns_none_on_empty_socket(self):
        sock = _FakeSocket(b"")
        result = recv_exact(sock, 4)
        assert result is None

    def test_handles_chunked_delivery(self):
        sock = _FakeSocket(b"abcdefgh", chunk_size=2)
        result = recv_exact(sock, 8)
        assert result == b"abcdefgh"

    def test_returns_none_on_partial_disconnect(self):
        sock = _FakeSocket(b"ab")
        result = recv_exact(sock, 5)
        assert result is None


# ── decode_message tests ────────────────────────────────────────


class TestDecodeMessage:
    def _make_socket(self, msg: dict, chunk_size: int = 0) -> _FakeSocket:
        data = encode_message(msg)
        return _FakeSocket(data, chunk_size=chunk_size)

    def test_roundtrip_chat(self):
        msg = make_chat("hello")
        sock = self._make_socket(msg)
        assert decode_message(sock) == msg

    def test_roundtrip_cmd(self):
        msg = make_cmd("uuid-1", "ls -la")
        sock = self._make_socket(msg)
        assert decode_message(sock) == msg

    def test_roundtrip_cmd_result(self):
        msg = make_cmd_result("uuid-1", 0, "output", "")
        sock = self._make_socket(msg)
        assert decode_message(sock) == msg

    def test_roundtrip_system(self):
        msg = make_system("peer disconnected")
        sock = self._make_socket(msg)
        assert decode_message(sock) == msg

    def test_returns_none_on_empty(self):
        sock = _FakeSocket(b"")
        assert decode_message(sock) is None

    def test_returns_none_on_truncated_header(self):
        sock = _FakeSocket(b"\x00\x00")
        assert decode_message(sock) is None

    def test_returns_none_on_truncated_payload(self):
        header = struct.pack(">I", 100)
        sock = _FakeSocket(header + b"short")
        assert decode_message(sock) is None

    def test_chunked_delivery(self):
        msg = make_chat("chunked test")
        sock = self._make_socket(msg, chunk_size=3)
        assert decode_message(sock) == msg


# ── Message factory tests ───────────────────────────────────────


class TestMessageFactories:
    def test_make_chat(self):
        msg = make_chat("hi there")
        assert msg == {"type": "chat", "body": "hi there"}

    def test_make_cmd(self):
        msg = make_cmd("id-42", "echo hello")
        assert msg == {"type": "cmd", "id": "id-42", "body": "echo hello"}

    def test_make_cmd_result(self):
        msg = make_cmd_result("id-42", 0, "hello\n", "")
        assert msg == {
            "type": "cmd_result",
            "id": "id-42",
            "exit_code": 0,
            "stdout": "hello\n",
            "stderr": "",
        }

    def test_make_cmd_result_with_error(self):
        msg = make_cmd_result("id-99", 1, "", "not found")
        assert msg["exit_code"] == 1
        assert msg["stderr"] == "not found"

    def test_make_system(self):
        msg = make_system("agent joined")
        assert msg == {"type": "system", "body": "agent joined"}
