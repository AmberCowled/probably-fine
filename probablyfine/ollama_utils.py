"""Shared Ollama response utilities for probablyfine modules."""

from __future__ import annotations

import json
import re

import ollama

from probablyfine.config import get_context_size
from probablyfine.log_utils import get_module_logger

_log = get_module_logger("probablyfine.ollama_utils", "tokens.log")


class HangDetected(Exception):
    """Raised when generation appears hung (no tokens for too long)."""


ZERO_TOKEN_ABORT_S = 15  # Abort streaming if no tokens after this many seconds


def create_client(timeout: float = 30) -> ollama.Client:
    """Create an Ollama client with the given timeout."""
    return ollama.Client(timeout=timeout)


def build_chat_options(
    model: str | None = None,
    num_predict: int = 500,
    temperature: float = 0,
) -> dict:
    """Build an options dict for ollama chat calls.

    If model is provided, includes num_ctx from config.
    """
    opts: dict = {"temperature": temperature, "num_predict": num_predict}
    if model:
        opts["num_ctx"] = get_context_size(model)
    return opts


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> blocks from qwen3 thinking mode output."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def strip_llm_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def parse_llm_json(raw: str) -> dict | list | None:
    """Parse JSON from LLM response: strip fences, try direct, regex fallback.

    Returns the parsed dict/list, or None if parsing fails entirely.
    Callers should handle None and apply their own domain-specific fallback.
    """
    cleaned = strip_llm_fences(raw)

    # 1. Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2. Regex extraction of outermost JSON object or array
    match = re.search(r"[\{\[][\s\S]*[\}\]]", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Model-specific prompt variants
# ---------------------------------------------------------------------------

_MODEL_PROMPT_SUFFIXES: dict[str, dict[str, str]] = {
    "deepseek": {
        "checker": (
            '\n\nCRITICAL: Respond with a JSON object containing exactly these keys: '
            '"verdict" (PASS/FAIL/ESCALATE), "confidence" (0.0-1.0), "issues" (array), "summary" (string). '
            "No other keys or format."
        ),
        "classify": (
            '\n\nCRITICAL: Respond with a JSON object containing exactly these keys: '
            '"intent", "complexity", "clarity", "clarification_questions". '
            "No other keys or format."
        ),
        "decompose": (
            '\n\nCRITICAL: Respond with a JSON object containing exactly these keys: '
            '"reasoning" (string), "steps" (array of objects with id/action/description/files/depends_on). '
            "No other keys or format."
        ),
        "file_selector": (
            "\n\nCRITICAL: Respond with ONLY a JSON array of file path strings. "
            "No other keys or format."
        ),
    },
}


def get_prompt_suffix(model: str, phase: str) -> str:
    """Return model-specific prompt suffix for a phase, or empty string.

    Phases: 'checker', 'classify', 'decompose', 'file_selector'.
    Matches model name substrings (e.g. 'deepseek' matches 'deepseek-coder:6.7b').
    """
    model_lower = model.lower()
    for key, suffixes in _MODEL_PROMPT_SUFFIXES.items():
        if key in model_lower:
            return suffixes.get(phase, "")
    return ""


def extract_content(chunk) -> str:
    """Extract text content from an Ollama response or stream chunk.

    Handles both the object-style API (newer ollama library) and
    dict-style API (older ollama library).
    """
    # Object-style (newer ollama library)
    msg = getattr(chunk, "message", None)
    if msg is not None:
        content = getattr(msg, "content", None)
        if content is not None:
            return content
    # Dict-style (older ollama library)
    if isinstance(chunk, dict):
        return chunk.get("message", {}).get("content", "")
    return ""


def log_token_usage(phase: str, model: str, response) -> None:
    """Extract and log token usage from an Ollama response (non-streaming).

    Logs prompt tokens, completion tokens, and context utilization percentage.
    Silently does nothing if usage fields are missing.
    """
    prompt_tokens = getattr(response, "prompt_eval_count", None)
    completion_tokens = getattr(response, "eval_count", None)
    if prompt_tokens is None and isinstance(response, dict):
        prompt_tokens = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")
    if prompt_tokens is None:
        return
    num_ctx = get_context_size(model)
    pct = int(prompt_tokens / num_ctx * 100) if num_ctx else 0
    _log.info(
        "[tokens] phase=%s model=%s prompt=%d/%d (%d%%) completion=%d",
        phase, model, prompt_tokens, num_ctx, pct, completion_tokens or 0,
    )
