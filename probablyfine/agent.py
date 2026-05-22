"""Custom agent: streams Ollama responses, parses SEARCH/REPLACE edits, applies them to disk.

Executes single-step and multi-step tasks. Every token is visible,
DRM monitors throughput, and 3-tier error recovery handles parse/apply failures.

Public API:
    execute_step(step, file_context, ctx: StepContext) -> StepResult
    execute_plan(plan, model, files, config) -> AgentResult
"""

from __future__ import annotations

import ast
import difflib
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable

from probablyfine.console import ACTION_COLORS as _ACTION_COLORS, console
from probablyfine.edit_parser import MAX_EDITS_PER_RESPONSE, apply_edits_atomic, count_edits_per_file, parse_edits, validate_edits
from probablyfine.git_utils import get_head_sha
from probablyfine.log_utils import get_module_logger
from probablyfine.models import AgentConfig, AgentResult, StepContext, StepResult, TaskPlan, TaskStep
from probablyfine.ollama_utils import (
    HangDetected as _HangDetected,
    ZERO_TOKEN_ABORT_S as _ZERO_TOKEN_ABORT_S,
    build_chat_options,
    create_client,
    extract_content as _extract_content,
)

log = get_module_logger("probablyfine.agent", "agent.log")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_AGENT_TIME_S = 300          # 5-min wall-clock cap per step
_AGENT_NUM_PREDICT = 4096       # max tokens for agent response (2x checker)
_AGENT_TIMEOUT = 300            # Ollama client timeout
_RETRY_NUM_PREDICT = 3072       # extra room for reasoning + corrected edits
_WHOLE_FILE_NUM_PREDICT = 8192  # more tokens for whole-file fallback
_VERIFY_TIMEOUT_S = 60          # subprocess timeout for lint/test commands
_CHECKPOINT_TIMEOUT_S = 10      # subprocess timeout for git operations
_MAX_RECOVERY_ATTEMPTS = 1      # retries per failed step at plan level
_MAX_EDITS_PER_FILE = 20        # Escalate to whole-file fallback above this
_MAX_CONTINUATION_ROUNDS = 3    # Max multi-turn continuation rounds for capped edits

# Dynamic token budget per step type — avoids wasting tokens on simple steps
# and prevents truncation on complex ones
_STEP_NUM_PREDICT: dict[str, int] = {
    "edit": 4096,
    "create": 6144,
    "explain": 4096,
    "read": 512,
    "verify": 512,
    "delete": 256,
}


def _get_step_budget(action: str) -> int:
    """Return num_predict token budget for a step action type."""
    return _STEP_NUM_PREDICT.get(action, _AGENT_NUM_PREDICT)

# ---------------------------------------------------------------------------
# Misbehavior detection
# ---------------------------------------------------------------------------

_DRIFT_STOP_WORDS = frozenset({
    "that", "this", "with", "from", "have", "been", "should",
    "could", "would", "into", "make", "using", "files", "file",
})


class _MisbehaviorObserver:
    """Lightweight observer for agent failure patterns."""

    def __init__(self):
        self._action_history: list[str] = []

    def check_reasoning_loop(self, action: str) -> bool:
        """Detect if the same action is failing repeatedly."""
        self._action_history.append(action)
        if len(self._action_history) >= 3:
            last_three = self._action_history[-3:]
            if len(set(last_three)) == 1:
                log.warning("Reasoning loop detected: '%s' attempted 3x consecutively", action)
                return True
        return False

def _check_specification_drift(response: str, task: str) -> None:
    """Log warning if response has very low keyword overlap with task."""
    task_words = set(re.findall(r'[a-z]{4,}', task.lower()))
    task_words -= _DRIFT_STOP_WORDS
    if len(task_words) < 3:
        return
    response_lower = response.lower()
    hits = sum(1 for w in task_words if w in response_lower)
    ratio = hits / len(task_words)
    if ratio < 0.2:
        log.warning("Specification drift: only %.0f%% of task keywords found in response", ratio * 100)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are a precise code editing assistant. You modify files using SEARCH/REPLACE blocks.

## Edit format

For each file change, produce a block like this:

FILE: path/to/file.py
<<<<<<< SEARCH
exact lines to find
=======
replacement lines
>>>>>>> REPLACE

Rules:
- The SEARCH section must match the file content EXACTLY (whitespace, indentation, etc.)
- Include enough context lines in SEARCH to be unique in the file
- Use SEARCH blocks that span 5+ lines of context. Do not make line-by-line edits.
- One SEARCH/REPLACE block per change (multiple blocks for multiple changes)
- For new files, use:

FILE: path/to/file.py (new)
<<<<<<< CONTENT
full file content here
>>>>>>> END

- For whole-file replacement, use:

FILE: path/to/file.py (whole)
<<<<<<< CONTENT
full updated content
>>>>>>> END

- When you know exact line numbers, use line-anchored edits (replaces lines N through M):

FILE: path/to/file.py LINES 42-50
<<<<<<< REPLACE
replacement content
>>>>>>> END

## Guidelines

- Make only the changes needed to complete the task
- Preserve existing code style and conventions
- Do not add unnecessary comments or docstrings
- If you need to explain something, put it BEFORE the edit blocks as plain text"""

_FEW_SHOT_EXAMPLES = """

## Examples

### Example 1: Editing an existing function
FILE: utils/math.py
<<<<<<< SEARCH
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
=======
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total
>>>>>>> REPLACE

### Example 2: Creating a new file
FILE: config/defaults.py (new)
<<<<<<< CONTENT
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
>>>>>>> END

### Example 3: Multiple changes in one file (separate blocks)
FILE: app/server.py
<<<<<<< SEARCH
import os
import sys
=======
import os
import sys
import logging
>>>>>>> REPLACE

FILE: app/server.py
<<<<<<< SEARCH
def start():
    app.run()
=======
def start():
    logging.info("Starting server")
    app.run()
>>>>>>> REPLACE"""

_CONSERVATIVE_ADDITION = """

IMPORTANT: Make minimal, targeted changes only. Do not refactor surrounding code.
Do not add features beyond what was explicitly requested. Change as few lines as possible."""

_INTENT_ADDITIONS: dict[str, str] = {
    "bug_fix": "\n\nFocus on fixing the root cause. Do not refactor unrelated code.",
    "feature": "\n\nImplement the feature cleanly. Follow existing patterns in the codebase.",
    "refactor": "\n\nPreserve all existing behavior. Only restructure, do not change functionality.",
    "question": "\n\nExplain clearly and concisely. Do not modify any files.",
}

AGENT_USER_TEMPLATE = """\
## Task
{task}

## Files
{file_contents}

Complete the task using SEARCH/REPLACE blocks."""

_RETRY_TEMPLATE = """\
Step failed while editing {file}: {error}

Here is the actual file content near the intended edit location:
{nearby_content}

What went wrong? Think through the problem, then provide corrected SEARCH/REPLACE blocks \
using the exact content shown above."""

_CONTINUATION_TEMPLATE = """\
{applied} edits were applied successfully. \
{remaining} edits remain for the task. Continue editing from where you left off. \
Only produce the remaining SEARCH/REPLACE blocks — do not repeat edits already applied."""

_WHOLE_FILE_TEMPLATE = """\
The SEARCH/REPLACE edit for {file} failed after retry. Please provide the complete \
updated file content using the whole-file format:

FILE: {file} (whole)
<<<<<<< CONTENT
full updated content here
>>>>>>> END"""


# ---------------------------------------------------------------------------
# DRM helpers
# ---------------------------------------------------------------------------

def _get_drm():
    """Return (manager, watchdog) or (None, None) — lazy import like checker.py."""
    try:
        from probablyfine.drm import get_manager as _get_drm_mgr
        mgr = _get_drm_mgr()
        if mgr.enabled:
            return mgr, mgr.watchdog
    except Exception:
        pass
    return None, None


def _handle_hang(step: TaskStep, model: str, drm, start: float) -> StepResult:
    """Handle a detected hang: emergency unload, return failed StepResult."""
    elapsed = time.monotonic() - start
    log.warning("Hang detected on step %d (model=%s, %.0fs elapsed)", step.id, model, elapsed)
    if drm:
        try:
            drm.emergency_unload_all()
        except Exception as e:
            log.error("Emergency unload failed: %s", e)
    return StepResult(
        step_id=step.id,
        status="failed",
        error=f"Generation hung after {elapsed:.0f}s — no tokens received",
        duration_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _stream_response(
    model: str,
    messages: list[dict],
    num_predict: int,
    watchdog,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Stream an Ollama chat response, collecting tokens.

    Adapted from checker.py:250-334 with key differences:
    - on_token callback for live display (checker writes to stderr directly)
    - Wall-clock timeout breaks (partial response may contain parseable edits)
    - Raises _HangDetected on hang
    """
    options = build_chat_options(model=model, num_predict=num_predict)
    client = create_client(timeout=_AGENT_TIMEOUT)

    log.debug("Streaming chat: model=%s, num_predict=%d, messages=%d",
              model, num_predict, len(messages))

    stream = client.chat(
        model=model,
        messages=messages,
        options=options,
        stream=True,
    )

    raw_parts: list[str] = []
    token_count = 0
    start = time.monotonic()
    last_token_time = start

    for chunk in stream:
        now = time.monotonic()

        # Hang detection
        if watchdog and token_count > 0 and watchdog.detect_hang(last_token_time, now):
            raise _HangDetected()

        # Zero-token early abort: model likely stalled on VRAM allocation
        if token_count == 0 and (now - start) > _ZERO_TOKEN_ABORT_S:
            log.warning("Agent zero-token abort after %.0fs", now - start)
            raise _HangDetected()

        # Wall-clock timeout — break so partial response can be parsed
        elapsed = now - start
        if elapsed > MAX_AGENT_TIME_S:
            log.warning("Wall-clock timeout after %.0fs (%d tokens)", elapsed, token_count)
            break

        content = _extract_content(chunk)
        if content:
            raw_parts.append(content)
            token_count += 1
            last_token_time = now
            if on_token:
                on_token(content)

    elapsed = time.monotonic() - start
    raw = "".join(raw_parts)
    log.info("Stream done: %d tokens in %.1fs (%d chars)", token_count, elapsed, len(raw))
    return raw


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def _format_file_contents(files: list[str]) -> str:
    """Read files and format as labeled blocks for the prompt."""
    parts: list[str] = []
    for fpath in files:
        p = Path(fpath)
        if not p.exists():
            parts.append(f"### {fpath}\n(file not found)\n")
            continue
        try:
            content = p.read_text(errors="replace")
            parts.append(f"### {fpath}\n```\n{content}\n```\n")
        except OSError as e:
            parts.append(f"### {fpath}\n(read error: {e})\n")
    return "\n".join(parts)


def _build_messages(
    step: TaskStep,
    files: list[str],
    plan: TaskPlan | None,
    config: AgentConfig,
    model: str = "",
) -> list[dict]:
    """Build the messages list for an agent LLM call."""
    # System prompt with intent and conservative additions
    system = AGENT_SYSTEM_PROMPT
    # Few-shot examples help qwen3 but confuse deepseek-coder (it regurgitates them)
    if "deepseek" not in model:
        system += _FEW_SHOT_EXAMPLES
    if config.conservative:
        system += _CONSERVATIVE_ADDITION
    if plan:
        intent_addition = _INTENT_ADDITIONS.get(plan.intent, "")
        if intent_addition:
            system += intent_addition

    # User prompt
    file_contents = _format_file_contents(files)
    task_desc = step.description
    if plan and plan.original_task != step.description:
        task_desc = f"{plan.original_task}\n\nCurrent step: {step.description}"

    user_content = AGENT_USER_TEMPLATE.format(
        task=task_desc,
        file_contents=file_contents,
    )

    # Append /no_think for complexity-1 tasks (structured output)
    if plan and plan.complexity == 1:
        user_content += " /no_think"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Diff capture
# ---------------------------------------------------------------------------

def _capture_diff(files_before: dict[str, str], changed_files: list[str]) -> str:
    """Generate unified diff between before-snapshots and current file contents."""
    diff_parts: list[str] = []
    for fpath in changed_files:
        before = files_before.get(fpath, "")
        after = ""
        p = Path(fpath)
        if p.exists():
            try:
                after = p.read_text(errors="replace")
            except OSError:
                pass

        if before == after:
            continue

        diff_lines = difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{fpath}",
            tofile=f"b/{fpath}",
        )
        diff_parts.append("".join(diff_lines))

    return "\n".join(diff_parts)


def _get_nearby_content(file_path: str, search_text: str, context_lines: int = 5) -> str:
    """Get actual file content near where search_text was expected.

    Used for retry prompts so the model sees what the file really contains.
    """
    p = Path(file_path)
    if not p.exists():
        return "(file not found)"
    try:
        content = p.read_text(errors="replace")
    except OSError:
        return "(file unreadable)"

    lines = content.splitlines()
    if not lines:
        return "(empty file)"

    # Find the best matching region using the first non-empty line of search_text
    anchor = ""
    for line in search_text.splitlines():
        stripped = line.strip()
        if stripped:
            anchor = stripped
            break

    if not anchor:
        # Return the first chunk of the file
        return "\n".join(lines[:context_lines * 2])

    best_idx = 0
    best_score = 0
    for i, line in enumerate(lines):
        score = sum(len(word) for word in anchor.split() if word in line)
        if score > best_score:
            best_score = score
            best_idx = i

    start = max(0, best_idx - context_lines)
    end = min(len(lines), best_idx + context_lines + 1)
    numbered = [f"{start + i + 1:4d} | {line}" for i, line in enumerate(lines[start:end])]
    return "\n".join(numbered)


# ---------------------------------------------------------------------------
# Error recovery: Tier 2 — retry with error context
# ---------------------------------------------------------------------------

def _retry_with_error_context(
    step: TaskStep,
    ctx: StepContext,
    failed_file: str,
    error_msg: str,
    search_text: str,
) -> StepResult | None:
    """Tier 2: Retry a failed edit by sending error context back to the model."""
    log.info("Tier 2 retry for %s: %s", failed_file, error_msg[:100])
    nearby = _get_nearby_content(failed_file, search_text)

    retry_prompt = _RETRY_TEMPLATE.format(
        file=failed_file,
        error=error_msg,
        nearby_content=nearby,
    )

    # Build minimal messages with /no_think for structured output
    system = AGENT_SYSTEM_PROMPT
    if ctx.config.conservative:
        system += _CONSERVATIVE_ADDITION

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": retry_prompt},
    ]

    try:
        raw = _stream_response(ctx.model, messages, _RETRY_NUM_PREDICT, ctx.watchdog, ctx.on_token)
    except _HangDetected:
        log.warning("Tier 2 retry hung")
        return None
    except Exception as e:
        log.warning("Tier 2 retry error: %s", e)
        return None

    if not raw.strip():
        return None

    edits = parse_edits(raw)
    if not edits:
        return None

    errors = validate_edits(edits)
    if errors:
        log.warning("Tier 2 validation failed: %s", errors[0][1])
        return None

    applied, changed = apply_edits_atomic(edits)
    if applied == 0:
        return None

    return StepResult(
        step_id=step.id,
        status="ok",
        edits_applied=applied,
        files_changed=changed,
    )


# ---------------------------------------------------------------------------
# Error recovery: Tier 3 — whole-file fallback
# ---------------------------------------------------------------------------

def _whole_file_fallback(
    step: TaskStep,
    ctx: StepContext,
    failed_file: str,
) -> StepResult | None:
    """Tier 3: Ask the model for the complete updated file content."""
    log.info("Tier 3 whole-file fallback for %s", failed_file)

    prompt = _WHOLE_FILE_TEMPLATE.format(file=failed_file)

    system = AGENT_SYSTEM_PROMPT
    if ctx.config.conservative:
        system += _CONSERVATIVE_ADDITION

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt + " /no_think"},
    ]

    try:
        raw = _stream_response(ctx.model, messages, _WHOLE_FILE_NUM_PREDICT, ctx.watchdog, ctx.on_token)
    except _HangDetected:
        log.warning("Tier 3 whole-file fallback hung")
        return None
    except Exception as e:
        log.warning("Tier 3 whole-file fallback error: %s", e)
        return None

    if not raw.strip():
        return None

    edits = parse_edits(raw)
    if not edits:
        return None

    errors = validate_edits(edits)
    if errors:
        log.warning("Tier 3 validation failed: %s", errors[0][1])
        return None

    applied, changed = apply_edits_atomic(edits)
    if applied == 0:
        return None

    return StepResult(
        step_id=step.id,
        status="ok",
        edits_applied=applied,
        files_changed=changed,
    )


# ---------------------------------------------------------------------------
# Step handlers
# ---------------------------------------------------------------------------

def _execute_edit(
    step: TaskStep,
    files: list[str],
    ctx: StepContext,
) -> StepResult:
    """Execute an edit/create step with 3-tier error recovery.

    Tier 1: Stream → parse_edits → validate → apply_edits_atomic
    Tier 2: Retry with error context + nearby file content
    Tier 3: Whole-file fallback (ask for complete FILE: path (whole) block)
    """
    start = time.monotonic()

    # --- DRM: ensure model is loaded ---
    drm, _ = _get_drm()
    if drm:
        drm.ensure_loaded(ctx.model)

    # --- Tier 1: primary attempt ---
    messages = _build_messages(step, files, ctx.plan, ctx.config, model=ctx.model)

    try:
        raw = _stream_response(ctx.model, messages, _get_step_budget(step.action), ctx.watchdog, ctx.on_token)
    except _HangDetected:
        return _handle_hang(step, model, drm, start)

    if not raw.strip():
        return StepResult(
            step_id=step.id,
            status="failed",
            error="Model returned empty response",
            duration_s=time.monotonic() - start,
        )

    # Specification drift check (warning only, non-blocking)
    task_desc = ctx.plan.original_task if ctx.plan and ctx.plan.original_task else step.description
    _check_specification_drift(raw, task_desc)

    edits = parse_edits(raw)
    if not edits:
        return StepResult(
            step_id=step.id,
            status="failed",
            error="No edit blocks found in model response",
            duration_s=time.monotonic() - start,
        )

    # Cap edits and apply in batches with multi-turn continuation
    total_parsed = len(edits)
    if total_parsed > MAX_EDITS_PER_RESPONSE:
        log.info("Capping %d edits to %d, will use continuation", total_parsed, MAX_EDITS_PER_RESPONSE)
        edits = edits[:MAX_EDITS_PER_RESPONSE]

    # Excessive edits for one file → whole-file fallback is more reliable
    file_counts = count_edits_per_file(edits)
    excessive = [f for f, c in file_counts.items() if c > _MAX_EDITS_PER_FILE]
    if excessive:
        for f in excessive:
            log.warning("Excessive edits for %s (%d blocks > %d cap), escalating to whole-file",
                        f, file_counts[f], _MAX_EDITS_PER_FILE)
        result = _whole_file_fallback(step, ctx, excessive[0])
        if result:
            result.duration_s = time.monotonic() - start
            return result
        # If whole-file also fails, fall through to normal validation path

    errors = validate_edits(edits)
    if not errors:
        applied, changed = apply_edits_atomic(edits)
        if applied > 0:
            # Multi-turn continuation: if we capped edits, re-invoke for remainder
            if total_parsed > MAX_EDITS_PER_RESPONSE:
                total_applied = applied
                all_changed = list(changed)
                remaining = total_parsed - MAX_EDITS_PER_RESPONSE
                for round_num in range(_MAX_CONTINUATION_ROUNDS):
                    if remaining <= 0:
                        break
                    log.info("Continuation round %d: %d edits remain", round_num + 1, remaining)
                    cont_prompt = _CONTINUATION_TEMPLATE.format(
                        applied=total_applied, remaining=remaining,
                    )
                    cont_messages = [
                        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                        {"role": "user", "content": cont_prompt + " /no_think"},
                    ]
                    try:
                        cont_raw = _stream_response(ctx.model, cont_messages, _get_step_budget(step.action), ctx.watchdog, ctx.on_token)
                    except (_HangDetected, Exception) as e:
                        log.warning("Continuation round %d failed: %s", round_num + 1, e)
                        break
                    cont_edits = parse_edits(cont_raw)
                    if not cont_edits:
                        break
                    cont_errors = validate_edits(cont_edits)
                    if cont_errors:
                        break
                    cont_applied, cont_changed = apply_edits_atomic(cont_edits)
                    if cont_applied == 0:
                        break
                    total_applied += cont_applied
                    for f in cont_changed:
                        if f not in all_changed:
                            all_changed.append(f)
                    remaining -= cont_applied
                    log.info("Continuation round %d applied %d edits", round_num + 1, cont_applied)

                return StepResult(
                    step_id=step.id,
                    status="ok",
                    edits_applied=total_applied,
                    files_changed=all_changed,
                    duration_s=time.monotonic() - start,
                )

            return StepResult(
                step_id=step.id,
                status="ok",
                edits_applied=applied,
                files_changed=changed,
                duration_s=time.monotonic() - start,
            )

    # --- Tier 2: retry with error context ---
    if errors:
        log.warning("Edit match rate: %d/%d passed validation",
                    len(edits) - len(errors), len(edits))
        failed_edit, error_msg = errors[0]
        result = _retry_with_error_context(
            step, ctx, failed_edit.file, error_msg, failed_edit.search,
        )
        if result:
            result.duration_s = time.monotonic() - start
            return result

        # --- Tier 3: whole-file fallback ---
        result = _whole_file_fallback(step, ctx, failed_edit.file)
        if result:
            result.duration_s = time.monotonic() - start
            return result

    return StepResult(
        step_id=step.id,
        status="failed",
        error="All 3 edit tiers failed",
        duration_s=time.monotonic() - start,
    )


def _execute_explain(
    step: TaskStep,
    files: list[str],
    ctx: StepContext,
) -> StepResult:
    """Execute an explain step — stream response, no file changes."""
    start = time.monotonic()

    drm, _ = _get_drm()
    if drm:
        drm.ensure_loaded(ctx.model)

    messages = _build_messages(step, files, ctx.plan, ctx.config, model=ctx.model)

    try:
        raw = _stream_response(ctx.model, messages, _get_step_budget(step.action), ctx.watchdog, ctx.on_token)
    except _HangDetected:
        return _handle_hang(step, model, drm, start)

    return StepResult(
        step_id=step.id,
        status="ok",
        explanation=raw.strip(),
        duration_s=time.monotonic() - start,
    )


def _execute_delete(step: TaskStep) -> StepResult:
    """Execute a delete step — unlink files listed in step.files."""
    start = time.monotonic()
    deleted: list[str] = []
    for fpath in step.files:
        p = Path(fpath)
        if p.exists():
            try:
                p.unlink()
                deleted.append(fpath)
                log.info("Deleted file: %s", fpath)
            except OSError as e:
                return StepResult(
                    step_id=step.id,
                    status="failed",
                    files_changed=deleted,
                    error=f"Failed to delete {fpath}: {e}",
                    duration_s=time.monotonic() - start,
                )
        else:
            log.warning("Delete target not found: %s", fpath)

    return StepResult(
        step_id=step.id,
        status="ok",
        files_changed=deleted,
        duration_s=time.monotonic() - start,
    )


# ---------------------------------------------------------------------------
# Plan execution helpers
# ---------------------------------------------------------------------------

def _topological_sort(steps: list[TaskStep]) -> list[TaskStep]:
    """Sort steps in dependency order using Kahn's algorithm (BFS).

    Falls back to original order on cycle detection. References to
    non-existent step IDs are silently ignored.
    """
    step_map = {s.id: s for s in steps}
    valid_ids = set(step_map)

    # Build adjacency and in-degree
    in_degree: dict[int, int] = {s.id: 0 for s in steps}
    dependents: dict[int, list[int]] = {s.id: [] for s in steps}
    for s in steps:
        for dep_id in s.depends_on:
            if dep_id in valid_ids:
                in_degree[s.id] += 1
                dependents[dep_id].append(s.id)

    queue: deque[int] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    ordered: list[TaskStep] = []

    while queue:
        sid = queue.popleft()
        ordered.append(step_map[sid])
        for dep in dependents[sid]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(ordered) < len(steps):
        log.warning("Cycle detected in step dependencies — falling back to original order")
        return list(steps)

    return ordered


def _execute_read(step: TaskStep, context: list[str]) -> StepResult:
    """Add step files to the mutable context list (deduplicating).

    Missing files are logged as warnings, not errors — they may be
    created by later steps.
    """
    for fpath in step.files:
        if fpath not in context:
            context.append(fpath)
        if not Path(fpath).exists():
            log.warning("Read step %d: file not found (may be created later): %s",
                        step.id, fpath)
    return StepResult(step_id=step.id, status="ok")


def _execute_verify(step: TaskStep, config: AgentConfig) -> StepResult:
    """Run lint/test command and return pass/fail result.

    Uses config.lint_command (preferred) or config.test_command as fallback.
    Returns skipped if no command is configured.
    """
    cmd = config.lint_command or config.test_command
    if not cmd:
        return StepResult(step_id=step.id, status="skipped",
                          error="No lint/test command configured")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=_VERIFY_TIMEOUT_S,
        )
        output = (result.stdout + result.stderr).strip()[:500]
        if result.returncode == 0:
            return StepResult(step_id=step.id, status="ok")
        return StepResult(
            step_id=step.id, status="failed",
            error=f"Verify command exited {result.returncode}: {output}",
        )
    except subprocess.TimeoutExpired:
        return StepResult(step_id=step.id, status="failed",
                          error=f"Verify command timed out after {_VERIFY_TIMEOUT_S}s")
    except Exception as e:
        return StepResult(step_id=step.id, status="failed",
                          error=f"Verify command error: {e}")


def _git_checkpoint(step: TaskStep) -> str:
    """Create a git checkpoint after a successful step.

    Returns new HEAD SHA, or current HEAD on failure.
    """
    current = get_head_sha()
    try:
        subprocess.run(
            ["git", "add", "-A"],
            capture_output=True, text=True, timeout=_CHECKPOINT_TIMEOUT_S,
        )
        msg = f"probablyfine: step {step.id} -- {step.description[:60]}"
        result = subprocess.run(
            ["git", "commit", "-m", msg, "--allow-empty"],
            capture_output=True, text=True, timeout=_CHECKPOINT_TIMEOUT_S,
        )
        if result.returncode == 0:
            new_sha = get_head_sha()
            log.info("Checkpoint after step %d: %s", step.id, new_sha[:8])
            return new_sha
    except (subprocess.TimeoutExpired, Exception) as e:
        log.warning("Checkpoint failed for step %d: %s", step.id, e)
    return current


def _rollback_to(sha: str) -> bool:
    """Roll back to a previous git SHA. Returns True on success. Never raises."""
    if not sha:
        return False
    try:
        result = subprocess.run(
            ["git", "reset", "--hard", sha],
            capture_output=True, text=True, timeout=_CHECKPOINT_TIMEOUT_S,
        )
        if result.returncode == 0:
            log.info("Rolled back to %s", sha[:8])
            return True
        log.warning("Rollback failed: %s", result.stderr.strip())
    except Exception as e:
        log.warning("Rollback error: %s", e)
    return False


def _snapshot_files(files: list[str]) -> dict[str, str]:
    """Read all files into {path: content} dict. Missing files recorded as ""."""
    snapshot: dict[str, str] = {}
    for fpath in files:
        p = Path(fpath)
        if p.exists():
            try:
                snapshot[fpath] = p.read_text(errors="replace")
            except OSError:
                snapshot[fpath] = ""
        else:
            snapshot[fpath] = ""
    return snapshot


def _show_step_header(step: TaskStep, total_steps: int) -> None:
    """Display a Rich header for the current step."""
    color = _ACTION_COLORS.get(step.action, "white")
    files_str = ", ".join(step.files) if step.files else ""
    console.print(
        f"\n[agent.step]Step {step.id}/{total_steps}[/]  "
        f"[{color}]{step.action}[/]  {step.description}"
        + (f"  [dim]{files_str}[/]" if files_str else "")
    )


def _attempt_recovery(
    step: TaskStep,
    result: StepResult,
    context: list[str],
    ctx: StepContext,
) -> StepResult | None:
    """Attempt to recover a failed step by re-executing it.

    Only retries edit/create/explain steps — verify/read/delete are not
    LLM-fixable. Calls execute_step which internally runs all 3 tiers.
    Returns the successful StepResult, or None if retry also failed.
    """
    if step.action not in ("edit", "create", "explain"):
        return None

    log.info("Recovery attempt for step %d (%s)", step.id, step.action)
    retry_result = execute_step(step, context, ctx)
    if retry_result.status == "ok":
        return retry_result
    return None


def _should_replan(step_result: StepResult, remaining_steps: list[TaskStep]) -> bool:
    """Check if completed step changed files that remaining steps also target."""
    if step_result.status != "ok" or not step_result.files_changed:
        return False
    changed = set(step_result.files_changed)
    for remaining in remaining_steps:
        if set(remaining.files) & changed:
            return True
    return False


def _refresh_step_files(remaining_steps: list[TaskStep], changed_files: list[str],
                        context: list[str]) -> None:
    """Ensure changed files are in context so subsequent steps see current content."""
    for f in changed_files:
        if f not in context:
            context.append(f)
    log.info("Refreshed context with %d changed files for %d remaining steps",
             len(changed_files), len(remaining_steps))


_MAX_REPLANS = 1  # Cap replanning attempts per execution


def _verify_cross_file_consistency(changed_files: list[str]) -> list[str]:
    """Check cross-file import/reference consistency after edits.

    Uses ast.parse to extract imports and definitions from changed Python files.
    Returns a list of warning strings (empty = all consistent).
    """
    py_files = [f for f in changed_files if f.endswith(".py") and Path(f).exists()]
    if len(py_files) < 2:
        return []

    # Collect definitions and imports from each changed file
    definitions: dict[str, set[str]] = {}  # file -> {name, ...}
    imports: dict[str, set[str]] = {}      # file -> {module_or_name, ...}

    for fpath in py_files:
        try:
            source = Path(fpath).read_text(errors="replace")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue

        defs: set[str] = set()
        imps: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defs.add(node.name)
            elif isinstance(node, ast.ImportFrom) and node.names:
                for alias in node.names:
                    imps.add(alias.name if alias.name != "*" else "")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imps.add(alias.name.split(".")[0])
        definitions[fpath] = defs
        imports[fpath] = imps

    # Check: if file A imports a name that was defined in file B (among changed files),
    # verify the name still exists in B
    warnings: list[str] = []
    all_defs = {}
    for fpath, defs in definitions.items():
        for d in defs:
            all_defs.setdefault(d, []).append(fpath)

    for fpath, imps in imports.items():
        for imp_name in imps:
            if imp_name in all_defs:
                # This name is defined in another changed file — verify it still exists
                for def_file in all_defs[imp_name]:
                    if def_file != fpath and imp_name not in definitions.get(def_file, set()):
                        warnings.append(
                            f"{fpath} imports '{imp_name}' but it was removed from {def_file}"
                        )

    if warnings:
        log.warning("Cross-file consistency issues: %s", warnings)
    return warnings


def _build_replan_prompt(
    plan: TaskPlan,
    step_results: list[StepResult],
) -> str:
    """Build a replan prompt summarizing what succeeded and what failed."""
    succeeded = [r for r in step_results if r.status == "ok"]
    failed = [r for r in step_results if r.status == "failed"]
    skipped = [r for r in step_results if r.status == "skipped"]

    parts = [plan.original_task]
    parts.append("\nContext from previous attempt:")
    if succeeded:
        parts.append(f"  Completed: steps {', '.join(str(r.step_id) for r in succeeded)}")
    if failed:
        parts.append("  Failed:")
        for r in failed:
            parts.append(f"    - Step {r.step_id}: {r.error or 'unknown error'}")
    parts.append(f"  {len(skipped)} step(s) were skipped due to failed dependencies.")
    parts.append("\nReplan the remaining work. Do not repeat steps that already succeeded.")
    return "\n".join(parts)


def _print_token(token: str) -> None:
    """Stream a token to stderr — matches checker.py's streaming pattern."""
    sys.stderr.write(token)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_step(
    step: TaskStep,
    file_context: list[str],
    ctx: StepContext,
) -> StepResult:
    """Execute a single TaskStep. Never raises to caller.

    Dispatches on step.action:
        edit/create → _execute_edit (3-tier recovery)
        explain     → _execute_explain (stream only)
        read        → immediate ok (context management, no LLM)
        delete      → _execute_delete (unlink files)
        verify      → _execute_verify (lint/test command)
    """
    log.info("execute_step: id=%d action=%s desc=%s",
             step.id, step.action, step.description[:80])

    try:
        if step.action in ("edit", "create"):
            return _execute_edit(step, file_context, ctx)

        if step.action == "explain":
            return _execute_explain(step, file_context, ctx)

        if step.action == "read":
            return StepResult(step_id=step.id, status="ok")

        if step.action == "delete":
            return _execute_delete(step)

        if step.action == "verify":
            return _execute_verify(step, ctx.config)

        log.warning("Unknown action: %s", step.action)
        return StepResult(
            step_id=step.id,
            status="skipped",
            error=f"Unknown action: {step.action}",
        )

    except KeyboardInterrupt:
        log.info("Step %d interrupted by user", step.id)
        return StepResult(
            step_id=step.id,
            status="failed",
            error="Interrupted by user",
        )
    except Exception as e:
        log.exception("Step %d unexpected error: %s", step.id, e)
        return StepResult(
            step_id=step.id,
            status="failed",
            error=str(e),
        )


def execute_plan(
    plan: TaskPlan,
    model: str,
    files: list[str] | None,
    config: AgentConfig,
    _replan_depth: int = 0,
) -> AgentResult:
    """Execute a multi-step TaskPlan with checkpointing and recovery.

    Steps are executed in dependency order (topological sort). Each
    edit/create step is checkpointed via git commit. On failure, recovery
    is attempted once; if that also fails, we rollback to the last
    checkpoint and attempt a single replan via the interpreter.
    """
    start = time.monotonic()

    # Empty plan — nothing to do
    if not plan.steps:
        return AgentResult(
            diff="", exit_code=0, head_before=get_head_sha(),
            duration_s=time.monotonic() - start,
        )

    # 1. Save starting state
    head_before = get_head_sha()
    last_checkpoint = head_before

    # 2. Collect all files across steps + explicit file list
    initial_files: list[str] = list(files or [])
    for step in plan.steps:
        for f in step.files:
            if f not in initial_files:
                initial_files.append(f)

    # 3. Snapshot files before any changes
    files_before = _snapshot_files(initial_files)

    # 4. Sort steps in dependency order
    ordered = _topological_sort(plan.steps)
    total = len(ordered)

    # 5. Mutable context — grows as read steps add files
    context: list[str] = list(files or [])
    _, watchdog = _get_drm()
    ctx = StepContext(
        model=model, config=config, watchdog=watchdog,
        on_token=_print_token, plan=plan,
    )
    observer = _MisbehaviorObserver()

    step_results: list[StepResult] = []
    all_changed: list[str] = []
    failed_step_ids: set[int] = set()

    # 6. Execute each step
    for step in ordered:
        # Cascading failure prevention: skip steps whose dependencies failed
        failed_deps = [d for d in step.depends_on if d in failed_step_ids]
        if failed_deps:
            log.warning("Skipping step %d — dependency %s failed", step.id, failed_deps)
            _show_step_header(step, total)
            console.print(f"  [dim]Skipped: dependency step {failed_deps[0]} failed[/]")
            skip_result = StepResult(
                step_id=step.id, status="skipped",
                error=f"Dependency step {failed_deps[0]} failed",
            )
            step_results.append(skip_result)
            failed_step_ids.add(step.id)
            continue

        _show_step_header(step, total)

        try:
            # Dispatch by action type
            if step.action == "read":
                result = _execute_read(step, context)

            elif step.action == "verify":
                result = _execute_verify(step, ctx.config)

            elif step.action in ("edit", "create"):
                result = execute_step(step, context, ctx)

            elif step.action == "delete":
                result = execute_step(step, context, StepContext(
                    model=model, config=config, watchdog=watchdog,
                    on_token=None, plan=plan,
                ))

            elif step.action == "explain":
                result = execute_step(step, context, ctx)

            else:
                result = execute_step(step, context, ctx)

        except KeyboardInterrupt:
            log.info("Plan execution interrupted at step %d", step.id)
            console.print("\n[warn]Interrupted — returning partial results[/]")
            break

        # Display result status
        if result.status == "ok":
            console.print(f"  [agent.success]OK[/]")
        elif result.status == "skipped":
            console.print(f"  [dim]Skipped[/]" +
                          (f": {result.error}" if result.error else ""))
        else:
            console.print(f"  [agent.error]FAILED[/]: {result.error}")

        step_results.append(result)

        # Track changed files
        for f in result.files_changed:
            if f not in all_changed:
                all_changed.append(f)

        # Misbehavior: reasoning loop detection
        action_key = f"{step.action}:{step.id}:{result.status}"
        if observer.check_reasoning_loop(action_key):
            console.print("  [agent.error]Loop detected — stopping plan[/]")
            break

        # Checkpoint after successful edit/create steps
        if result.status == "ok" and step.action in ("edit", "create"):
            if config.auto_checkpoint:
                last_checkpoint = _git_checkpoint(step)
            # Refresh context if this step changed files targeted by later steps
            step_idx = ordered.index(step)
            remaining = ordered[step_idx + 1:]
            if _should_replan(result, remaining):
                _refresh_step_files(remaining, result.files_changed, context)

        # Handle failure with recovery
        if result.status == "failed" and step.action in ("edit", "create", "explain"):
            console.print(f"  [agent.retry]Attempting recovery...[/]")
            recovery = _attempt_recovery(step, result, context, ctx)
            if recovery:
                console.print(f"  [agent.success]Recovery succeeded[/]")
                step_results[-1] = recovery
                # Update changed files from recovery
                for f in recovery.files_changed:
                    if f not in all_changed:
                        all_changed.append(f)
                if config.auto_checkpoint:
                    last_checkpoint = _git_checkpoint(step)
            else:
                console.print(f"  [agent.error]Recovery failed — rolling back[/]")
                _rollback_to(last_checkpoint)
                failed_step_ids.add(step.id)
                # Continue to next step instead of breaking — dependents will be skipped

    # 7. Replan on failure: if steps were skipped due to deps, try once
    skipped_deps = [r for r in step_results if r.status == "skipped"
                    and r.error and "Dependency" in r.error]
    if skipped_deps and _replan_depth < _MAX_REPLANS:
        log.info("Replan triggered: %d steps skipped due to failed deps", len(skipped_deps))
        replan_prompt = _build_replan_prompt(plan, step_results)
        try:
            from probablyfine.interpreter import interpret_task
            new_plan = interpret_task(replan_prompt, model, file_context=context)
            if new_plan.steps:
                log.info("Replan produced %d new steps", len(new_plan.steps))
                console.print(f"\n  [agent.step]Replanning: {len(new_plan.steps)} step(s)[/]")
                replan_result = execute_plan(
                    new_plan, model, context, config,
                    _replan_depth=_replan_depth + 1,
                )
                # Merge replan results
                step_results.extend(replan_result.steps)
                for f in replan_result.files_changed:
                    if f not in all_changed:
                        all_changed.append(f)
                log.info("Replan completed: exit_code=%d, files_changed=%d",
                         replan_result.exit_code, len(replan_result.files_changed))
            else:
                log.info("Replan produced empty plan, skipping")
        except Exception as e:
            log.warning("Replan failed: %s", e)

    # 8. Cross-file consistency check (warnings only, non-blocking)
    if len(all_changed) >= 2:
        consistency_warnings = _verify_cross_file_consistency(all_changed)
        for w in consistency_warnings:
            console.print(f"  [check.warning]Consistency: {w}[/]")

    # 9. Snapshot any newly created files not in original snapshot
    for f in all_changed:
        if f not in files_before:
            files_before[f] = ""

    # 10. Capture combined diff
    diff = _capture_diff(files_before, all_changed)

    # Determine exit code: 0 if all executed steps succeeded or were skipped
    exit_code = 0
    for r in step_results:
        if r.status == "failed":
            exit_code = 1
            break

    return AgentResult(
        diff=diff,
        exit_code=exit_code,
        head_before=head_before,
        steps=step_results,
        files_changed=all_changed,
        duration_s=time.monotonic() - start,
    )


