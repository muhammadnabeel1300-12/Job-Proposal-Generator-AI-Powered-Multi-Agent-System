"""Tests 10-11 plus error handling for the Claude client (no real API calls)."""

from types import SimpleNamespace

import anthropic
import pytest
from pydantic import BaseModel

from utils.claude_client import (ClaudeAuthError, ClaudeClient, ClaudeConnectionError,
                                 ClaudeRateLimitError, ClaudeResponseError,
                                 MissingAPIKeyError, parse_json_response)


class Tiny(BaseModel):
    name: str
    count: int


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Response:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason


class FakeSDK:
    """Mimics anthropic.Anthropic().messages.create with scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Response(reply)


def make_client(replies):
    return ClaudeClient(api_key="test-key", model="test-model", sdk_client=FakeSDK(replies))


# 10. Missing API key
def test_missing_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setattr("utils.claude_client.load_dotenv", lambda *a, **k: None)
    with pytest.raises(MissingAPIKeyError):
        ClaudeClient(model="x")


def test_placeholder_api_key_is_rejected():
    with pytest.raises(MissingAPIKeyError):
        ClaudeClient(api_key="your_api_key_here", model="x")


# 11. Invalid Claude response
def test_invalid_json_twice_raises():
    client = make_client(["I cannot do that", "still not json"])
    with pytest.raises(ClaudeResponseError):
        client.generate_structured("sys", "user", Tiny)
    assert client._sdk.calls == 2


def test_wrong_schema_then_valid_retry():
    client = make_client(['{"name": "a"}', '{"name": "a", "count": 3}'])
    result = client.generate_structured("sys", "user", Tiny)
    assert result.count == 3 and client._sdk.calls == 2


def test_fenced_json_is_accepted():
    client = make_client(['```json\n{"name": "a", "count": 1}\n```'])
    assert client.generate_structured("s", "u", Tiny).name == "a"


def test_json_with_extra_words():
    assert parse_json_response('Here you go: {"a": 1} Hope it helps')["a"] == 1


def test_empty_response():
    with pytest.raises(ClaudeResponseError):
        parse_json_response("")


# API error handling
def _fake_request():
    return SimpleNamespace(method="POST", url="https://api.anthropic.com/v1/messages")


def _status_error(cls, code):
    # A tiny stand-in for the HTTP response object (avoids depending on httpx).
    response = SimpleNamespace(status_code=code, headers={}, request=_fake_request())
    return cls("boom", response=response, body=None)


@pytest.mark.parametrize("error,expected", [
    (_status_error(anthropic.AuthenticationError, 401), ClaudeAuthError),
    (_status_error(anthropic.RateLimitError, 429), ClaudeRateLimitError),
    (anthropic.APIConnectionError(request=_fake_request()),
     ClaudeConnectionError),
])
def test_api_errors_are_translated(error, expected):
    client = make_client([error])
    with pytest.raises(expected):
        client.generate_text("sys", "user")
