"""
claude_client.py - One reusable Claude client used by ALL four agents.

It handles: loading the API key from .env, calling the API, asking for JSON,
validating the JSON with Pydantic, retrying once if Claude's JSON is invalid,
and turning technical API errors into clear messages.
"""

import json
import os
import re

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

DEFAULT_MAX_TOKENS = 4096
MAX_TOKENS_CAP = 16000

# ------------------------------------------------------------------ errors
class ClaudeError(Exception):
    """Base class. The message is always safe and friendly to show to the user."""


class MissingAPIKeyError(ClaudeError):
    pass


class ClaudeAuthError(ClaudeError):
    pass


class ClaudeRateLimitError(ClaudeError):
    pass


class ClaudeConnectionError(ClaudeError):
    pass


class ClaudeAPIError(ClaudeError):
    pass


class ClaudeResponseError(ClaudeError):
    """Claude answered, but the answer was not valid JSON / did not match the schema."""


# ------------------------------------------------------------------ helpers
def parse_json_response(raw: str) -> dict:
    """Turn Claude's reply into a dict. Tolerates ```json fences and extra words."""
    text = (raw or "").strip()
    if not text:
        raise ClaudeResponseError("Claude returned an empty response.")

    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ClaudeResponseError("Claude's response did not contain JSON.")
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as error:
            raise ClaudeResponseError(f"Claude returned malformed JSON: {error}") from error

    if not isinstance(data, dict):
        raise ClaudeResponseError("Claude returned JSON, but not a JSON object.")
    return data


def _format_validation_error(error: ValidationError) -> str:
    parts = []
    for item in error.errors()[:8]:
        location = ".".join(str(p) for p in item["loc"])
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)


# ------------------------------------------------------------------ client
class ClaudeClient:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 sdk_client=None, max_retries: int = 3, timeout: float = 180.0):
        load_dotenv()
        self.api_key = (api_key or os.getenv("ANTHROPIC_API_KEY", "")).strip()
        self.model = (model or os.getenv("CLAUDE_MODEL", "")).strip()

        if not self.api_key or self.api_key == "your_api_key_here":
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY is missing. Add your key to the .env file "
                "and restart the app.")
        if not self.model:
            raise ClaudeError(
                "CLAUDE_MODEL is missing. Add a model ID to the .env file.")

        # sdk_client lets the tests inject a fake SDK (no real API calls).
        self._sdk = sdk_client or anthropic.Anthropic(
            api_key=self.api_key, max_retries=max_retries, timeout=timeout)

    # ---- low level call -------------------------------------------------
    def _complete(self, system: str, messages: list[dict], max_tokens: int
                  ) -> tuple[str, bool]:
        """Call the API. Returns (text, was_truncated)."""
        try:
            response = self._sdk.messages.create(
                model=self.model, max_tokens=max_tokens,
                system=system, messages=messages)
        except anthropic.AuthenticationError as error:
            raise ClaudeAuthError(
                "The Claude API key was rejected. Check ANTHROPIC_API_KEY in .env."
            ) from error
        except anthropic.PermissionDeniedError as error:
            raise ClaudeAuthError(
                "Your API key does not have permission for this request.") from error
        except anthropic.RateLimitError as error:
            raise ClaudeRateLimitError(
                "Rate limit reached. Please wait a minute and try again.") from error
        except anthropic.APITimeoutError as error:
            raise ClaudeConnectionError(
                "The request to Claude timed out. Please try again.") from error
        except anthropic.APIConnectionError as error:
            raise ClaudeConnectionError(
                "Could not connect to the Claude API. Check your internet connection."
            ) from error
        except anthropic.NotFoundError as error:
            raise ClaudeAPIError(
                f"Model '{self.model}' was not found. Check CLAUDE_MODEL in .env."
            ) from error
        except anthropic.APIStatusError as error:
            raise ClaudeAPIError(
                f"Claude API error (status {error.status_code}). Please try again."
            ) from error

        text = "".join(block.text for block in response.content
                       if getattr(block, "type", "") == "text")
        return text, getattr(response, "stop_reason", "") == "max_tokens"

    # ---- public: plain text --------------------------------------------
    def generate_text(self, system: str, user: str,
                      max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        text, truncated = self._complete(
            system, [{"role": "user", "content": user}], max_tokens)
        if not text.strip():
            raise ClaudeResponseError("Claude returned an empty response.")
        if truncated:
            text += "\n\n[Note: the response was cut off because it reached the length limit.]"
        return text.strip()

    # ---- public: validated JSON ----------------------------------------
    def generate_structured(self, system: str, user: str, schema: type[BaseModel],
                            max_tokens: int = DEFAULT_MAX_TOKENS,
                            max_attempts: int = 2) -> BaseModel:
        """
        Ask Claude for JSON and validate it with a Pydantic model.
        On bad JSON we retry once, telling Claude what was wrong.
        """
        full_system = (
            f"{system}\n\nOUTPUT FORMAT\nReturn ONLY one valid JSON object that "
            "matches the JSON Schema below. No explanations, no Markdown, no code "
            f"fences.\nJSON Schema:\n{json.dumps(schema.model_json_schema())}"
        )
        messages = [{"role": "user", "content": user}]
        last_problem = "unknown problem"

        for _ in range(max_attempts):
            raw, truncated = self._complete(full_system, messages, max_tokens)
            if truncated:
                # Output was cut off: retry with a bigger limit.
                max_tokens = min(max_tokens * 2, MAX_TOKENS_CAP)
                last_problem = "the response was cut off (too long)"
                continue
            try:
                return schema.model_validate(parse_json_response(raw))
            except ClaudeResponseError as error:
                last_problem = str(error)
            except ValidationError as error:
                last_problem = _format_validation_error(error)

            messages = messages + [
                {"role": "assistant", "content": raw.strip() or "(empty)"},
                {"role": "user", "content": (
                    f"Your previous reply was invalid: {last_problem}. "
                    "Return ONLY the corrected JSON object.")},
            ]

        raise ClaudeResponseError(
            f"Claude's answer could not be used ({last_problem}). Please try again.")
