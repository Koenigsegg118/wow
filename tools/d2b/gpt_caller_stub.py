"""GPT caller interface and implementations for D2b hard-case labeling.

Provides:
- GPTCallerInterface: Protocol defining the calling contract
- GPTCallerStub: Returns null labels for end-to-end pipeline testing
- GPTCallerOpenAI: OpenAI-compatible API caller (fill in api_key to use)

Usage:
    # For testing (no API needed):
    caller = GPTCallerStub()

    # For production (requires api_key):
    caller = GPTCallerOpenAI(
        api_key=os.environ["D2B_GPT_API_KEY"],
        base_url="https://api.openai.com/v1",
    )

    response = caller.call_single(request)
    # response is format_C: {"label": {...}, "quality_hint": {...}}
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from typing import Protocol, runtime_checkable

from tools.d2b.d2b_config import (
    ALLOWED_PROFILE_HINTS,
    ALLOWED_ROLES,
    ALLOWED_SEMANTIC_STATES,
    ALLOWED_TARGET_REFS,
    GPT_MAX_TOKENS,
    GPT_MODEL_NAME,
    GPT_TEMPERATURE,
)

# ── Structured output JSON Schema for format_C ───────────────

FORMAT_C_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "label": {
            "type": "object",
            "properties": {
                "semantic_state": {
                    "type": ["string", "null"],
                    "enum": sorted(ALLOWED_SEMANTIC_STATES) + [None],
                },
                "target_ref": {
                    "type": ["string", "null"],
                    "enum": sorted(ALLOWED_TARGET_REFS) + [None],
                },
                "role": {
                    "type": ["string", "null"],
                    "enum": sorted(ALLOWED_ROLES) + [None],
                },
                "constraints": {
                    "type": ["object", "null"],
                    "properties": {
                        "prefer_energy_preserve": {"type": ["boolean", "null"]},
                        "allow_aggressive_turn": {"type": ["boolean", "null"]},
                        "must_support_teammate": {"type": ["boolean", "null"]},
                        "should_abort_if_threatened": {"type": ["boolean", "null"]},
                    },
                    "required": [
                        "prefer_energy_preserve",
                        "allow_aggressive_turn",
                        "must_support_teammate",
                        "should_abort_if_threatened",
                    ],
                    "additionalProperties": False,
                },
                "profile_hint": {
                    "type": ["string", "null"],
                    "enum": sorted(ALLOWED_PROFILE_HINTS) + [None],
                },
                "event_flags": {
                    "type": ["object", "null"],
                    "properties": {
                        "target_switch": {"type": ["boolean", "null"]},
                    },
                    "required": ["target_switch"],
                    "additionalProperties": False,
                },
                "rationale_short": {"type": ["string", "null"]},
            },
            "required": [
                "semantic_state", "target_ref", "role", "constraints",
                "profile_hint", "event_flags", "rationale_short",
            ],
            "additionalProperties": False,
        },
        "quality_hint": {
            "type": "object",
            "properties": {
                "confidence": {"type": ["number", "null"]},
                "conflict_flag": {"type": ["boolean", "null"]},
                "needs_review": {"type": ["boolean", "null"]},
            },
            "required": ["confidence", "conflict_flag", "needs_review"],
            "additionalProperties": False,
        },
    },
    "required": ["label", "quality_hint"],
    "additionalProperties": False,
}

logger = logging.getLogger(__name__)

# ── prompt loading ────────────────────────────────────────────

_PROMPT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPT_DIR / "d2b_gpt_system_prompt.txt"
_USER_TEMPLATE_PATH = _PROMPT_DIR / "d2b_gpt_user_template.txt"


def _load_prompt(path: pathlib.Path) -> str:
    """Load prompt text from file."""
    return path.read_text(encoding="utf-8").strip()


def build_user_prompt(obs: dict, evidence_hints: list[str] | None = None) -> str:
    """Build user prompt by filling obs JSON into template.

    Args:
        obs: The obs-sidecar dict (format_A input).
        evidence_hints: Optional list of hint strings for hard-case context.

    Returns:
        Formatted user prompt string.
    """
    template = _load_prompt(_USER_TEMPLATE_PATH)
    obs_json = json.dumps(obs, indent=2, ensure_ascii=False)

    # Build evidence block: structured JSON if hints exist, otherwise "null"
    if evidence_hints:
        evidence_obj = {"hints": evidence_hints}
        evidence_json = json.dumps(evidence_obj, indent=2, ensure_ascii=False)
    else:
        evidence_json = "null"

    # Replace template placeholders (match user template format)
    prompt = template.replace("{{OBS_JSON}}", obs_json)
    prompt = prompt.replace("{{EVIDENCE_JSON_OR_NULL}}", evidence_json)

    return prompt


# ── interface ─────────────────────────────────────────────────


@runtime_checkable
class GPTCallerInterface(Protocol):
    """Protocol for GPT calling implementations."""

    def call_single(self, request: dict) -> dict:
        """Send a single request, return format_C response.

        Args:
            request: Dict with at least 'obs' key (format_A).
                     May also contain 'evidence_hints' and 'meta'.

        Returns:
            Dict with 'label' and 'quality_hint' keys (format_C).
        """
        ...

    def call_batch(
        self,
        requests_path: pathlib.Path,
        output_path: pathlib.Path,
    ) -> pathlib.Path:
        """Process a batch of requests from JSONL file.

        Args:
            requests_path: Path to input JSONL (one request per line).
            output_path: Path to write output JSONL (one response per line).

        Returns:
            Path to the output file.
        """
        ...


# ── stub implementation ───────────────────────────────────────


class GPTCallerStub:
    """Stub implementation for end-to-end pipeline testing.

    Returns null labels with needs_review=True for every request.
    No API call is made.
    """

    def call_single(self, request: dict) -> dict:
        """Return a null-label response."""
        return {
            "label": {
                "semantic_state": None,
                "target_ref": None,
                "role": None,
                "constraints": {
                    "prefer_energy_preserve": None,
                    "allow_aggressive_turn": None,
                    "must_support_teammate": None,
                    "should_abort_if_threatened": None,
                },
                "profile_hint": None,
                "event_flags": {"target_switch": None},
                "rationale_short": None,
            },
            "quality_hint": {
                "confidence": 0.0,
                "conflict_flag": False,
                "needs_review": True,
            },
        }

    def call_batch(
        self,
        requests_path: pathlib.Path,
        output_path: pathlib.Path,
    ) -> pathlib.Path:
        """Process batch: read requests, return null labels."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(requests_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                request = json.loads(line)
                response = self.call_single(request)
                # Attach request_id for traceability
                response["request_id"] = request.get("request_id", f"stub_{count}")
                fout.write(json.dumps(response, ensure_ascii=False) + "\n")
                count += 1
        logger.info("GPTCallerStub: processed %d requests → %s", count, output_path)
        return output_path


# ── OpenAI-compatible implementation ──────────────────────────


class GPTCallerOpenAI:
    """OpenAI SDK compatible GPT caller.

    Requires:
        pip install openai

    API key is read from:
        1. Constructor argument ``api_key``
        2. Environment variable ``D2B_GPT_API_KEY``

    Usage:
        caller = GPTCallerOpenAI(
            api_key="sk-...",           # or set D2B_GPT_API_KEY env var
            base_url="https://api.openai.com/v1",
            model_name="gpt-5.4",
        )
        resp = caller.call_single({"obs": {...}})
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = GPT_MODEL_NAME,
        max_tokens: int = GPT_MAX_TOKENS,
        temperature: float = GPT_TEMPERATURE,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec

        resolved_key = api_key or os.environ.get("D2B_GPT_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key required. Pass api_key= or set D2B_GPT_API_KEY env var."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required: pip install openai"
            )

        self._client = OpenAI(api_key=resolved_key, base_url=base_url)
        self._system_prompt = _load_prompt(_SYSTEM_PROMPT_PATH)

    def call_single(self, request: dict) -> dict:
        """Call GPT API for a single request.

        Args:
            request: Dict with 'obs' key and optional 'evidence_hints'.

        Returns:
            Parsed format_C dict: {"label": {...}, "quality_hint": {...}}.

        Raises:
            ValueError: If response cannot be parsed as valid JSON.
        """
        obs = request.get("obs", request)
        evidence_hints = request.get("evidence_hints")
        user_prompt = build_user_prompt(obs, evidence_hints)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "format_c_output",
                            "strict": True,
                            "schema": FORMAT_C_SCHEMA,
                        },
                    },
                )
                raw_text = response.choices[0].message.content.strip()
                parsed = json.loads(raw_text)

                # Ensure top-level keys exist
                if "label" not in parsed:
                    parsed = {"label": parsed, "quality_hint": {
                        "confidence": 0.5,
                        "conflict_flag": False,
                        "needs_review": True,
                    }}
                if "quality_hint" not in parsed:
                    parsed["quality_hint"] = {
                        "confidence": 0.5,
                        "conflict_flag": False,
                        "needs_review": True,
                    }

                return parsed

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    "GPT response not valid JSON (attempt %d/%d): %s",
                    attempt, self.max_retries, str(e)[:200],
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "GPT API error (attempt %d/%d): %s",
                    attempt, self.max_retries, str(e)[:200],
                )

            if attempt < self.max_retries:
                time.sleep(self.retry_delay_sec * attempt)

        # All retries exhausted — return fallback
        logger.error("GPT call failed after %d retries: %s", self.max_retries, last_error)
        return {
            "label": {
                "semantic_state": None,
                "target_ref": None,
                "role": None,
                "constraints": {
                    "prefer_energy_preserve": None,
                    "allow_aggressive_turn": None,
                    "must_support_teammate": None,
                    "should_abort_if_threatened": None,
                },
                "profile_hint": None,
                "event_flags": {"target_switch": None},
                "rationale_short": None,
            },
            "quality_hint": {
                "confidence": 0.0,
                "conflict_flag": False,
                "needs_review": True,
            },
        }

    def call_batch(
        self,
        requests_path: pathlib.Path,
        output_path: pathlib.Path,
    ) -> pathlib.Path:
        """Process batch of requests sequentially.

        Each line in requests_path is a JSON object with at least 'obs'.
        Results are written one-per-line to output_path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        errors = 0

        with open(requests_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                request = json.loads(line)
                request_id = request.get("request_id", f"req_{count}")

                try:
                    response = self.call_single(request)
                except Exception as e:
                    logger.error("Fatal error for %s: %s", request_id, e)
                    response = GPTCallerStub().call_single(request)
                    errors += 1

                response["request_id"] = request_id
                fout.write(json.dumps(response, ensure_ascii=False) + "\n")
                count += 1

                if count % 50 == 0:
                    logger.info("Processed %d requests...", count)

        logger.info(
            "GPTCallerOpenAI: processed %d requests (%d errors) → %s",
            count, errors, output_path,
        )
        return output_path
