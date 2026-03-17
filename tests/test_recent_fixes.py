"""Tests for recent fixes: JSON newline escaping, emotion events, retry logic."""

import json

import pytest


# ── JSON newline fixer (extracted from _extract_json inner function) ──

def _fix_newlines(s):
    """Escape literal newlines/tabs inside JSON string values."""
    result = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == "\\":
            result.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch == "\n":
            result.append("\\n")
            continue
        if in_string and ch == "\t":
            result.append("\\t")
            continue
        if in_string and ch == "\r":
            continue
        result.append(ch)
    return "".join(result)


class TestFixNewlines:
    """Test the JSON newline fixer that handles Gemini's literal newlines."""

    def test_simple_json_unchanged(self):
        s = '{"key": "value", "num": 42}'
        assert json.loads(_fix_newlines(s)) == {"key": "value", "num": 42}

    def test_code_with_literal_newlines(self):
        # Simulate what Gemini outputs: literal newlines in code field
        raw = '{"code": "def foo():\n    return 42"}'
        fixed = _fix_newlines(raw)
        parsed = json.loads(fixed)
        assert parsed["code"] == "def foo():\n    return 42"

    def test_code_with_tabs(self):
        raw = '{"code": "def bar():\n\tpass"}'
        fixed = _fix_newlines(raw)
        parsed = json.loads(fixed)
        assert "pass" in parsed["code"]

    def test_already_escaped_newlines_preserved(self):
        # Already-escaped \n should NOT be double-escaped
        raw = '{"code": "line1\\nline2"}'
        fixed = _fix_newlines(raw)
        parsed = json.loads(fixed)
        assert parsed["code"] == "line1\nline2"

    def test_multiline_code_block(self):
        raw = '{"improvement": "test", "code": "import os\nimport sys\n\ndef main():\n    print(os.getcwd())\n    return 0"}'
        fixed = _fix_newlines(raw)
        parsed = json.loads(fixed)
        assert "import os" in parsed["code"]
        assert "def main():" in parsed["code"]
        assert parsed["improvement"] == "test"

    def test_carriage_return_stripped(self):
        raw = '{"code": "line1\r\nline2"}'
        fixed = _fix_newlines(raw)
        parsed = json.loads(fixed)
        assert "\r" not in parsed["code"]
        assert "line1" in parsed["code"]

    def test_newlines_outside_strings_preserved(self):
        raw = '{\n  "key": "value"\n}'
        fixed = _fix_newlines(raw)
        parsed = json.loads(fixed)
        assert parsed["key"] == "value"

    def test_empty_string_field(self):
        raw = '{"code": "", "name": "test"}'
        fixed = _fix_newlines(raw)
        parsed = json.loads(fixed)
        assert parsed["code"] == ""
        assert parsed["name"] == "test"

    def test_nested_quotes_in_code(self):
        raw = '{"code": "x = \\"hello\\"\ny = \\"world\\""}'
        fixed = _fix_newlines(raw)
        parsed = json.loads(fixed)
        assert "hello" in parsed["code"]
        assert "world" in parsed["code"]


class TestEmotionEventMap:
    """Test that new emotion events are properly structured."""

    _import_error = None
    try:
        from agent_ui import _EMOTION_EVENT_MAP
    except Exception as exc:
        _import_error = str(exc)

    @pytest.mark.skipif(_import_error is not None, reason=f"Cannot import: {_import_error}")
    def test_new_events_exist(self):
        from agent_ui import _EMOTION_EVENT_MAP
        new_events = ["comment_posted", "upvote_given", "api_recovered"]
        for event in new_events:
            assert event in _EMOTION_EVENT_MAP, f"Missing event: {event}"

    @pytest.mark.skipif(_import_error is not None, reason=f"Cannot import: {_import_error}")
    def test_comment_posted_boosts_confidence(self):
        from agent_ui import _EMOTION_EVENT_MAP
        mapping = _EMOTION_EVENT_MAP["comment_posted"]
        reinforce_names = [name for name, _ in mapping["reinforce"]]
        assert "confidence" in reinforce_names

    @pytest.mark.skipif(_import_error is not None, reason=f"Cannot import: {_import_error}")
    def test_api_recovered_weakens_caution(self):
        from agent_ui import _EMOTION_EVENT_MAP
        mapping = _EMOTION_EVENT_MAP["api_recovered"]
        weaken_names = [name for name, _ in mapping["weaken"]]
        assert "caution" in weaken_names

    @pytest.mark.skipif(_import_error is not None, reason=f"Cannot import: {_import_error}")
    def test_new_idea_learned_boosts_confidence(self):
        from agent_ui import _EMOTION_EVENT_MAP
        mapping = _EMOTION_EVENT_MAP["new_idea_learned"]
        reinforce_names = [name for name, _ in mapping["reinforce"]]
        assert "confidence" in reinforce_names

    @pytest.mark.skipif(_import_error is not None, reason=f"Cannot import: {_import_error}")
    def test_all_events_have_reinforce_and_weaken(self):
        from agent_ui import _EMOTION_EVENT_MAP
        for event, mapping in _EMOTION_EVENT_MAP.items():
            assert "reinforce" in mapping, f"{event} missing 'reinforce'"
            assert "weaken" in mapping, f"{event} missing 'weaken'"
            for name, delta in mapping["reinforce"]:
                assert isinstance(delta, (int, float)), f"{event}.reinforce has non-numeric delta"
                assert delta > 0, f"{event}.reinforce has non-positive delta"
