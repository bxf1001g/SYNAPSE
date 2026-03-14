"""Tests for JSON parsing and configuration utilities from agent_ui.py."""

import json
import os
import tempfile

import pytest

# Import the specific functions we want to test.
# parse_json_response and config functions are defined in agent_ui.py,
# but agent_ui.py has heavy imports (flask, genai, etc.).  We extract and
# test the pure-logic helpers that don't need the Flask app running.

# We import from agent_ui but only the standalone utility functions.
# If import fails due to missing optional deps, we skip gracefully.
_import_error = None
try:
    from agent_ui import (
        DEFAULT_CONFIG,
        load_config,
        parse_json_response,
        save_config,
    )
except Exception as exc:
    _import_error = str(exc)

pytestmark = pytest.mark.skipif(
    _import_error is not None,
    reason=f"Cannot import agent_ui utilities: {_import_error}",
)


# ── parse_json_response tests ──────────────────────────────────


class TestParseJsonResponse:
    def test_plain_json(self):
        text = '{"thinking": "ok", "actions": []}'
        result = parse_json_response(text)
        assert result == {"thinking": "ok", "actions": []}

    def test_json_in_code_block(self):
        text = '```json\n{"thinking": "plan", "actions": [{"type": "done"}]}\n```'
        result = parse_json_response(text)
        assert result["thinking"] == "plan"
        assert result["actions"] == [{"type": "done"}]

    def test_json_in_plain_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"thinking": "hi", "actions": []} and more text.'
        result = parse_json_response(text)
        assert result["thinking"] == "hi"

    def test_fallback_on_invalid_json(self):
        text = "This is just plain text with no JSON."
        result = parse_json_response(text)
        assert "actions" in result
        assert result["actions"][0]["type"] == "message"
        assert "plain text" in result["actions"][0]["content"]

    def test_whitespace_around_json(self):
        text = '   \n  {"thinking": "yes", "actions": []}  \n  '
        result = parse_json_response(text)
        assert result == {"thinking": "yes", "actions": []}

    def test_nested_json_objects(self):
        text = '{"thinking": "deep", "actions": [{"type": "file", "path": "x.py", "content": "print(1)"}]}'
        result = parse_json_response(text)
        assert len(result["actions"]) == 1
        assert result["actions"][0]["type"] == "file"


# ── load_config / save_config tests ────────────────────────────


class TestConfig:
    def test_load_config_returns_defaults_when_no_file(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = load_config(base_dir=td)
            assert cfg["providers"]["gemini"]["enabled"] is True
            assert cfg["providers"]["openai"]["enabled"] is False

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = load_config(base_dir=td)
            cfg["providers"]["openai"]["api_key"] = "sk-test-key"
            cfg["providers"]["openai"]["enabled"] = True
            save_config(cfg, base_dir=td)

            loaded = load_config(base_dir=td)
            assert loaded["providers"]["openai"]["api_key"] == "sk-test-key"
            assert loaded["providers"]["openai"]["enabled"] is True

    def test_load_config_merges_partial(self):
        with tempfile.TemporaryDirectory() as td:
            partial = {"providers": {"gemini": {"api_key": "my-key"}}}
            path = os.path.join(td, ".synapse.json")
            with open(path, "w") as f:
                json.dump(partial, f)

            cfg = load_config(base_dir=td)
            # Partial key merged
            assert cfg["providers"]["gemini"]["api_key"] == "my-key"
            # Default fields preserved
            assert cfg["providers"]["gemini"]["enabled"] is True
            assert "openai" in cfg["providers"]

    def test_load_config_handles_corrupt_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, ".synapse.json")
            with open(path, "w") as f:
                f.write("not valid json {{{")

            cfg = load_config(base_dir=td)
            # Falls back to defaults
            assert cfg == DEFAULT_CONFIG

    def test_cortex_map_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = load_config(base_dir=td)
            assert "fast" in cfg["cortex_map"]
            assert "reason" in cfg["cortex_map"]
            assert "create" in cfg["cortex_map"]
            assert "visual" in cfg["cortex_map"]

    def test_cortex_map_override(self):
        with tempfile.TemporaryDirectory() as td:
            custom = {
                "cortex_map": {
                    "fast": {"provider": "openai", "model": "gpt-4o-mini"},
                }
            }
            path = os.path.join(td, ".synapse.json")
            with open(path, "w") as f:
                json.dump(custom, f)

            cfg = load_config(base_dir=td)
            assert cfg["cortex_map"]["fast"]["provider"] == "openai"
            assert cfg["cortex_map"]["fast"]["model"] == "gpt-4o-mini"
            # Other cortex entries preserved
            assert cfg["cortex_map"]["reason"]["provider"] == "gemini"
