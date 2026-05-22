"""Checker module: reviews diffs via Ollama and returns structured feedback."""

from __future__ import annotations

import sys
import time

from probablyfine.log_utils import get_module_logger
from probablyfine.models import CheckerRequest, CheckerResult, Issue
from probablyfine.ollama_utils import (
    HangDetected as _HangDetected,
    ZERO_TOKEN_ABORT_S as _ZERO_TOKEN_ABORT_S,
    build_chat_options,
    create_client,
    extract_content as _extract_content,
    get_prompt_suffix,
    parse_llm_json,
)

MAX_DIFF_LINES = 5000
MAX_CHECKER_TIME_S = 120          # Wall-clock cap for checker generation
MAX_CHECKER_CONTEXT = 4096        # Cap num_ctx for checker (prevents stalls on 8GB VRAM)
_DEFAULT_CONFIDENCE = 0.5         # Fallback when checker omits confidence
_LOW_CONFIDENCE_THRESHOLD = 0.6   # FAIL below this → treat as PASS
_CHECKER_NUM_PREDICT = 2048       # Max tokens for checker response
_PROGRESS_CLEAR_WIDTH = 60        # Columns to clear on progress line wipe

log = get_module_logger("probablyfine.checker", "checker.log")

CHECKER_SYSTEM_PROMPT = """\
You are a senior code reviewer. You review diffs produced by an AI coding assistant.

Your job is to identify real problems -- not style preferences. Focus on:
- Logical errors and bugs
- Missing edge cases that will cause runtime failures
- Security issues (injection, path traversal, etc.)
- Violations of the original task requirements
- Broken imports or undefined references
- Off-by-one errors, null/None handling gaps

Do NOT flag:
- Style preferences (naming conventions, formatting)
- Minor documentation gaps
- Hypothetical future issues
- Anything that works correctly as-is

Example of something that is NOT an issue (do not flag):
- A function parameter named `x` instead of `descriptive_name` — this is style, not a bug
- A missing docstring on a new helper function — this is documentation, not a defect

Example of something that IS an issue (do flag):
- A function calls `items[index]` without checking if `index < len(items)` — this will crash at runtime

You must respond with ONLY a JSON object in this exact format:
{
  "verdict": "PASS or FAIL or ESCALATE",
  "confidence": 0.0,
  "issues": [
    {
      "severity": "critical or warning",
      "file": "path/to/file.py",
      "line": 42,
      "description": "What is wrong",
      "suggestion": "How to fix it"
    }
  ],
  "summary": "One-sentence overall assessment"
}

Rules:
- PASS: No critical issues. Warnings are acceptable.
- FAIL: At least one critical issue found. Provide actionable fixes.
- ESCALATE: The changes are so complex or risky that a deeper review is needed.
- If you are uncertain, lean toward PASS. Do not over-correct working code.
- Keep suggestions minimal and surgical. Do not rewrite the implementation.
- Respond with the JSON object ONLY. No markdown fences, no extra text."""

CHECKER_USER_TEMPLATE = """\
## Original Task
{task}

## Files in Context
{file_list}

## Code Changes (Diff)
{diff}

## Iteration
This is review iteration {iteration} of maximum {max_iterations}.

Review the diff above against the original task. Respond with JSON only."""


def _truncate_diff(diff: str) -> str:
    """Truncate large diffs to keep checker prompts manageable."""
    lines = diff.splitlines(keepends=True)
    if len(lines) <= MAX_DIFF_LINES:
        return diff
    kept = lines[:MAX_DIFF_LINES]
    omitted = len(lines) - MAX_DIFF_LINES
    kept.append(f"\n... ({omitted} lines omitted) ...\n")
    return "".join(kept)


def _build_user_prompt(
    task: str,
    diff: str,
    files: list[str] | None,
    iteration: int,
    max_iterations: int,
) -> str:
    file_list = "\n".join(files) if files else "(no specific files)"
    return CHECKER_USER_TEMPLATE.format(
        task=task,
        file_list=file_list,
        diff=_truncate_diff(diff),
        iteration=iteration,
        max_iterations=max_iterations,
    )


def _parse_response(raw: str) -> CheckerResult:
    """Parse checker JSON response with fallback extraction."""
    data = parse_llm_json(raw)
    if data is None or not isinstance(data, dict):
        return CheckerResult(
            verdict="PASS",
            confidence=0.0,
            issues=[],
            summary="Checker response was not valid JSON -- accepting changes.",
            raw_response=raw,
        )

    verdict = str(data.get("verdict", "PASS")).upper()
    if verdict not in ("PASS", "FAIL", "ESCALATE"):
        verdict = "PASS"

    confidence = float(data.get("confidence", _DEFAULT_CONFIDENCE))
    confidence = max(0.0, min(1.0, confidence))

    issues = []
    for item in data.get("issues", []):
        issues.append(Issue(
            severity=str(item.get("severity", "warning")).lower(),
            file=str(item.get("file", "")),
            description=str(item.get("description", "")),
            suggestion=str(item.get("suggestion", "")),
            line=item.get("line"),
        ))

    summary = str(data.get("summary", ""))

    # Low-confidence FAIL -> treat as PASS with warnings
    if verdict == "FAIL" and confidence < _LOW_CONFIDENCE_THRESHOLD:
        verdict = "PASS"
        summary = f"(low confidence {confidence:.1f}) {summary}"

    return CheckerResult(
        verdict=verdict,
        confidence=confidence,
        issues=issues,
        summary=summary,
        raw_response=raw,
    )


def _clear_progress():
    sys.stderr.write("\r" + " " * _PROGRESS_CLEAR_WIDTH + "\r")
    sys.stderr.flush()


def _try_checker_fallback(
    fallback: str | None,
    reason: str,
    user_prompt: str,
    timeout: float,
) -> CheckerResult | None:
    """Retry checker with a fallback model. Returns result or None on failure."""
    if not fallback:
        return None
    try:
        log.info("Retrying checker with fallback: %s", fallback)
        sys.stderr.write(f"\r  {reason}, retrying with {fallback}...  ")
        sys.stderr.flush()
        return _run_checker_stream(fallback, user_prompt, timeout, None)
    except Exception as e:
        log.warning("Fallback checker also failed: %s", e)
        return None


def run_checker(req: CheckerRequest) -> CheckerResult:
    """Send diff to checker model via Ollama and return structured result.

    Streams the response so the user sees live progress.
    Includes hang detection (no tokens for 60s) and OOM recovery.
    Returns a PASS result with a warning on any failure (timeout, API error, etc.).
    """
    user_prompt = _build_user_prompt(req.task, req.diff, req.files, req.iteration, req.max_iterations)
    diff_lines = len(req.diff.strip().splitlines())
    log.info("Checker start: model=%s, diff_lines=%d, iteration=%d", req.model, diff_lines, req.iteration)

    # Get DRM manager (imported once for hang/OOM recovery)
    _drm = None
    watchdog = None
    try:
        from probablyfine.drm import get_manager as _get_drm
        _drm = _get_drm()
        if _drm.enabled:
            watchdog = _drm.watchdog
    except Exception:
        pass

    try:
        result = _run_checker_stream(req.model, user_prompt, req.timeout, watchdog)
        # Auto-retry on zero-token stall with reduced context to relieve VRAM pressure
        if result.confidence == 0.0 and "no tokens" in result.summary:
            reduced_ctx = MAX_CHECKER_CONTEXT // 2
            log.info("Zero-token stall, retrying with reduced context (%d)", reduced_ctx)
            result = _run_checker_stream(
                req.model, user_prompt, req.timeout, watchdog,
                num_ctx_cap=reduced_ctx,
            )
        return result
    except _HangDetected:
        _clear_progress()
        log.warning("Checker hung on model %s, attempting recovery", req.model)

        if watchdog and _drm:
            _drm.emergency_unload_all()
            result = _try_checker_fallback(
                _drm.get_fallback_model(req.model), "Checker hung", user_prompt, req.timeout,
            )
            if result:
                return result

        return CheckerResult(
            verdict="PASS",
            confidence=0.0,
            issues=[],
            summary="Checker hung (no tokens for 60s) -- accepting changes unchecked.",
            raw_response="",
        )
    except Exception as e:
        _clear_progress()

        # Check for OOM and attempt recovery
        if watchdog and _drm and watchdog.is_oom_error(e):
            log.warning("OOM during checker, attempting recovery")
            result = _try_checker_fallback(
                _drm.handle_failure(e, req.model), "OOM recovered", user_prompt, req.timeout,
            )
            if result:
                return result

        log.exception("Checker error: %s", e)
        return CheckerResult(
            verdict="PASS",
            confidence=0.0,
            issues=[],
            summary=f"Checker error: {e} -- accepting changes unchecked.",
            raw_response="",
        )


def _run_checker_stream(
    model: str,
    user_prompt: str,
    timeout: float,
    watchdog,
    num_ctx_cap: int | None = None,
) -> CheckerResult:
    """Run the actual streaming checker call. Raises _HangDetected on hang."""
    sys.stderr.write(f"\r  Checker: waiting for first token...  ")
    sys.stderr.flush()

    # Build options then cap num_ctx to prevent stalls on constrained VRAM.
    # Checker doesn't need large context — the diff is already truncated.
    cap = num_ctx_cap or MAX_CHECKER_CONTEXT
    options = build_chat_options(model=model, num_predict=_CHECKER_NUM_PREDICT)
    if options.get("num_ctx", 0) > cap:
        log.debug("Capping checker num_ctx from %d to %d", options["num_ctx"], cap)
        options["num_ctx"] = cap

    system_prompt = CHECKER_SYSTEM_PROMPT + get_prompt_suffix(model, "checker")

    client = create_client(timeout=timeout)
    log.debug("Calling ollama chat(stream=True) for %s", model)
    stream = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=options,
        stream=True,
    )
    log.debug("Stream object received, beginning iteration")

    raw_parts: list[str] = []
    token_count = 0
    start = time.monotonic()
    last_token_time = start

    for chunk in stream:
        now = time.monotonic()

        # Hang detection: check if we've been waiting too long for tokens
        if watchdog and token_count > 0 and watchdog.detect_hang(last_token_time, now):
            raise _HangDetected()

        # Zero-token early abort: if no tokens at all after _ZERO_TOKEN_ABORT_S,
        # the model is likely stalled (KV cache allocation failure, etc.)
        if token_count == 0 and (now - start) > _ZERO_TOKEN_ABORT_S:
            _clear_progress()
            log.warning("Checker zero-token abort after %.0fs", now - start)
            return CheckerResult(
                verdict="PASS",
                confidence=0.0,
                issues=[],
                summary=f"Checker produced no tokens after {now - start:.0f}s -- accepting changes unchecked.",
                raw_response="",
            )

        # Wall-clock timeout: cap total generation time
        elapsed = now - start
        if elapsed > MAX_CHECKER_TIME_S:
            _clear_progress()
            log.warning("Checker wall-clock timeout after %.0fs (%d tokens)", elapsed, token_count)
            raw = "".join(raw_parts)
            # Try to parse what we have so far
            if raw.strip():
                result = _parse_response(raw)
                result.summary = f"(timeout after {elapsed:.0f}s) {result.summary}"
                return result
            return CheckerResult(
                verdict="PASS",
                confidence=0.0,
                issues=[],
                summary=f"Checker timed out after {elapsed:.0f}s -- accepting changes unchecked.",
                raw_response=raw,
            )

        content = _extract_content(chunk)
        if content:
            raw_parts.append(content)
            token_count += 1
            last_token_time = now
            sys.stderr.write(f"\r  Checker: {token_count} tokens ({elapsed:.0f}s)  ")
            sys.stderr.flush()

    _clear_progress()
    elapsed = time.monotonic() - start
    raw = "".join(raw_parts)
    log.info("Checker done: %d tokens in %.1fs", token_count, elapsed)
    log.debug("Raw response (first 500 chars): %s", raw[:500])

    if not raw.strip():
        log.warning("Checker returned empty response")
        return CheckerResult(
            verdict="PASS",
            confidence=0.0,
            issues=[],
            summary="Checker returned empty response -- accepting changes.",
            raw_response=raw,
        )

    result = _parse_response(raw)
    log.info("Parsed verdict=%s confidence=%.2f issues=%d", result.verdict, result.confidence, len(result.issues))
    return result
