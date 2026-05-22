"""Interpreter module: classifies user intent, assesses clarity, and decomposes tasks into plans."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from probablyfine.log_utils import get_module_logger
from probablyfine.models import ClarificationQuestion, TaskPlan, TaskStep
from probablyfine.ollama_utils import (
    build_chat_options,
    create_client,
    extract_content as _extract_content,
    get_prompt_suffix,
    log_token_usage,
    parse_llm_json,
    strip_llm_fences,
    strip_think_tags as _strip_think_tags,
)

log = get_module_logger("probablyfine.interpreter", "interpreter.log")

_LOG_FILE = Path.home() / ".probablyfine" / "interpreter.log"


def get_log_path() -> Path:
    """Return the path to the interpreter log file."""
    return _LOG_FILE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLARITY_THRESHOLD = 0.7
CLASSIFY_TIMEOUT = 30       # seconds (extra margin for thinking tokens + structured questions)
CLASSIFY_NUM_PREDICT = 800  # tokens (thinking tokens + structured clarification questions)
DECOMPOSE_TIMEOUT = 60      # seconds (thinking mode needs more time on complex prompts)
DECOMPOSE_NUM_PREDICT = 2000 # tokens (thinking overhead + 6-step JSON plan + safety margin)
MAX_DECOMPOSITION_STEPS = 6  # cap runaway decomposition

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """\
You are a task classifier for a coding assistant. Analyze the user's request and respond with ONLY a JSON object.

{{
  "intent": "bug_fix or feature or refactor or question",
  "complexity": 1 or 2 or 3,
  "clarity": 0.0 to 1.0,
  "clarification_questions": [
    {{
      "question": "What specific aspect needs work?",
      "options": ["Option A", "Option B", "Option C"]
    }}
  ]
}}

Intent definitions:
- bug_fix: fixing errors, crashes, broken behavior, stack traces
- feature: adding new functionality, building something new
- refactor: restructuring, cleaning up, improving existing code without changing behavior
- question: asking how/why something works, requesting explanation (no file edits)

Complexity levels:
- 1: Single file, single change, obvious what to do (rename, fix typo, add import)
- 2: Needs context from multiple files, moderate scope (fix a bug, add a method)
- 3: Spans multiple files/concerns, requires planning (add authentication, refactor module)

Clarity: How clear and actionable the request is (1.0 = perfectly clear, 0.0 = completely vague).
If clarity < 0.7, provide 1-2 clarification questions with 2-4 suggested answer options each.
Otherwise leave clarification_questions as an empty list.

Task: {task}

JSON: /no_think"""

DECOMPOSE_PROMPT = """\
You are a task planner for a coding assistant. Break this task into ordered steps.

Respond with ONLY a JSON object:
{{
  "reasoning": "Brief explanation of approach",
  "steps": [
    {{
      "id": 1,
      "action": "read or edit or create or delete or verify or explain",
      "description": "What to do in this step",
      "files": ["path/to/file.py"],
      "depends_on": []
    }}
  ]
}}

Rules:
- Each step should be a single, focused action
- Use "read" for steps that only inspect code (no changes)
- Use "edit" for modifying existing files
- Use "create" for new files
- Use "verify" for running tests or linting
- Use "explain" for answering questions (no file changes)
- Keep the total number of steps between 2 and 6
- List file dependencies with depends_on (step IDs that must complete first)
- You MUST populate the files array for every edit/create/delete step using paths from the available files list below
- If no files are listed or you need a file not in the list, leave the files array empty for that step

Task: {task}
Intent: {intent}
{file_context}

JSON: /no_think"""

# ---------------------------------------------------------------------------
# Keyword patterns (fast path -- no LLM call needed)
# ---------------------------------------------------------------------------

_QUESTION_PATTERNS = [
    r"\b(how (does|do|can|should|would|is|are|to))\b",
    r"\b(what (does|is|are|happens|would))\b",
    r"\b(why (does|is|did|would|are))\b",
    r"\b(explain|describe|tell me about|walk me through)\b",
    r"\b(can you explain|help me understand)\b",
]

_BUG_FIX_PATTERNS = [
    r"\b(fix|bug|broken|crash|error|exception|traceback|doesn.t work|not working)\b",
    r"\b(stack trace|runtime error|type ?error|name ?error|key ?error|index ?error)\b",
    r"\b(failing|fails|broke|regression)\b",
]

_REFACTOR_PATTERNS = [
    r"\b(refactor|restructure|clean ?up|reorganize|simplify|extract|inline)\b",
    r"\b(rename|move|split|merge|consolidate|deduplicate)\b",
    r"\b(tech ?debt|code smell|DRY|modularize)\b",
]

_FEATURE_PATTERNS = [
    r"\b(add|build|implement|create|make|develop|introduce|scaffold|update)\b",
    r"\b(new (feature|endpoint|page|component|module|function|class))\b",
    r"\b(support for|integrate|hook up|wire up|set up)\b",
]

# Ordered dispatch: question checked first so "how do I add X" → question, not feature
_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("question", _QUESTION_PATTERNS),
    ("bug_fix", _BUG_FIX_PATTERNS),
    ("refactor", _REFACTOR_PATTERNS),
    ("feature", _FEATURE_PATTERNS),
]

# Complexity signals (checked by heuristics before LLM)
_COMPLEXITY_3_PATTERNS = [
    r"\b(authentication|auth system|user management|payment|checkout)\b",
    r"\b(migration|deploy|CI/?CD|infrastructure|full.?stack)\b",
    r"\b(redesign|rewrite|overhaul|entire|whole|all (files|modules|tests))\b",
]

_COMPLEXITY_2_PATTERNS = [
    r"\b(debug|investigate|figure out|track down)\b",
    r"\b(refactor|restructure|across (multiple|several|all))\b",
    r"\b(test (suite|coverage)|integration test)\b",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _close_brackets(text: str) -> str:
    """Close any open brackets/braces in a JSON fragment."""
    opens = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            opens.append(ch)
        elif ch in ('}', ']'):
            if opens:
                opens.pop()
    closers = {'[': ']', '{': '}'}
    return text + ''.join(closers[o] for o in reversed(opens))


def _repair_truncated_json(text: str) -> dict | None:
    """Attempt to repair truncated JSON by progressively trimming and closing.

    Handles the common case where num_predict runs out mid-JSON.
    Tries multiple repair strategies from least to most aggressive.
    Returns parsed dict on success, None on failure.
    """
    # Strategy 1: just close brackets as-is
    attempt = _close_brackets(text.rstrip())
    try:
        return json.loads(attempt)
    except json.JSONDecodeError:
        pass

    # Strategy 2: trim trailing comma/colon then close
    trimmed = re.sub(r'[,:\s]+$', '', text.rstrip())
    attempt = _close_brackets(trimmed)
    try:
        return json.loads(attempt)
    except json.JSONDecodeError:
        pass

    # Strategy 3: progressively remove trailing content and retry
    # Find positions of commas outside strings (potential element boundaries)
    candidate = trimmed
    for _ in range(5):
        # Remove everything after the last comma outside a string
        last_comma = -1
        in_str = False
        esc = False
        for i, ch in enumerate(candidate):
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
            elif ch == ',' and not in_str:
                last_comma = i

        if last_comma <= 0:
            break

        candidate = candidate[:last_comma]
        attempt = _close_brackets(candidate)
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue

    return None


def _parse_json_response(raw: str) -> dict | None:
    """Parse JSON from LLM response with fallback extraction.

    Strategy: strip thinking tags, shared fence-strip + parse, then
    attempt truncated JSON repair as last resort.
    """
    cleaned = _strip_think_tags(raw)

    # 1-2. Shared fence-strip + direct parse + regex fallback
    result = parse_llm_json(cleaned)
    if result is not None:
        return result

    # 3. Repair truncated JSON (interpreter-specific)
    cleaned = strip_llm_fences(cleaned)
    brace_pos = cleaned.find('{')
    if brace_pos >= 0:
        fragment = cleaned[brace_pos:]
        repaired = _repair_truncated_json(fragment)
        if repaired is not None:
            log.info("Repaired truncated JSON (salvaged %d keys)", len(repaired))
            return repaired

    return None


def _build_file_summary(file_context: list[str] | None, max_entries: int = 30) -> str:
    """Build a compact file summary with line counts for the decompose prompt.

    Returns a formatted string like:
        Available files:
          probablyfine/agent.py (850 lines)
          probablyfine/checker.py (353 lines)
    Or "(no files in context)" if no files.
    """
    if not file_context:
        return "Files in context: (no files in context)"
    entries: list[str] = []
    for fpath in file_context[:max_entries]:
        p = Path(fpath)
        if p.exists():
            try:
                line_count = sum(1 for _ in p.open(errors="replace"))
                entries.append(f"  {fpath} ({line_count} lines)")
            except OSError:
                entries.append(f"  {fpath} (unreadable)")
        else:
            entries.append(f"  {fpath} (not yet created)")
    if len(file_context) > max_entries:
        entries.append(f"  ... and {len(file_context) - max_entries} more files")
    return "Available files:\n" + "\n".join(entries)


def _classify_intent_keywords(task: str) -> str | None:
    """Classify intent via keyword patterns. Returns None if no match."""
    lower = task.lower()
    for intent, patterns in _INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, lower):
                return intent
    return None


def _assess_complexity(task: str, intent: str) -> int:
    """Heuristic complexity assessment. Returns 1, 2, or 3."""
    lower = task.lower()

    for pattern in _COMPLEXITY_3_PATTERNS:
        if re.search(pattern, lower):
            return 3
    for pattern in _COMPLEXITY_2_PATTERNS:
        if re.search(pattern, lower):
            return 2

    if intent == "question":
        return 1

    return 2


# ---------------------------------------------------------------------------
# Clarity heuristic (keyword fast-path only)
# ---------------------------------------------------------------------------

# Specificity signals that indicate a clear, actionable task
_SPECIFICITY_PATTERNS = [
    r"[\w/\\]+\.\w{1,5}\b",              # file paths (foo.py, src/bar.js)
    r"\b(line|row|col)\s*\d+",            # line references
    r"\b(function|method|class|variable|def|const|let|var)\s+\w+",  # code identifiers
    r"`[^`]+`",                           # backtick-quoted code/names
    r"```",                               # code fences
    r"(error|exception|traceback):.+",    # error messages with content
    r"\b\d{3,}\b",                        # numeric IDs / status codes
]

_MIN_SPECIFIC_WORDS = 12  # tasks shorter than this with no signals are vague


def _assess_clarity_heuristic(task: str) -> float:
    """Estimate clarity from task text without LLM.

    Returns 1.0 for specific tasks, lower values for vague ones.
    Used only on keyword-matched tasks to decide if LLM classification
    should also run for clarity assessment.
    """
    specificity_hits = sum(
        1 for p in _SPECIFICITY_PATTERNS if re.search(p, task)
    )
    word_count = len(task.split())

    # Specific enough — file paths, code refs, error messages
    if specificity_hits >= 2:
        return 1.0
    if specificity_hits == 1 and word_count >= _MIN_SPECIFIC_WORDS:
        return 1.0

    # Short + no specificity signals = vague
    if word_count < _MIN_SPECIFIC_WORDS and specificity_hits == 0:
        return 0.5

    # Medium length, some detail but no concrete anchors
    return 0.7


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


def _call_llm(
    prompt: str,
    model: str,
    phase: str,
    timeout: int,
    num_predict: int,
    streaming: bool = False,
) -> dict | None:
    """Call Ollama LLM, extract content, parse JSON, and log timing.

    Shared implementation for classify/decompose/validate calls.
    When streaming=True, captures partial tokens so truncated JSON can be
    salvaged on timeout (used by decompose phase).
    Returns parsed dict or None on any failure.
    Re-raises KeyboardInterrupt; all other exceptions return None.
    """
    t0 = time.monotonic()
    try:
        options = build_chat_options(model=model, num_predict=num_predict)
        client = create_client(timeout=timeout)

        if streaming:
            stream = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                stream=True,
            )
            raw_parts: list[str] = []
            timed_out = False
            for chunk in stream:
                if time.monotonic() - t0 > timeout:
                    timed_out = True
                    log.warning("[%s] Streaming timeout after %.1fs, salvaging %d tokens",
                                phase, time.monotonic() - t0, len(raw_parts))
                    break
                content = _extract_content(chunk)
                if content:
                    raw_parts.append(content)
            raw = "".join(raw_parts)
        else:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options=options,
            )
            raw = _extract_content(response)
            log_token_usage(phase, model, response)
            timed_out = False

        elapsed = time.monotonic() - t0

        if not raw.strip():
            log.warning("[%s] Empty response from %s (%.1fs)", phase, model, elapsed)
            return None

        log.debug("[%s] Raw response (%.1fs, %d chars%s): %s",
                  phase, elapsed, len(raw), ", PARTIAL" if timed_out else "", raw[:500])
        result = _parse_json_response(raw)
        if result is None and timed_out:
            log.info("[%s] Attempting truncated JSON repair on partial response", phase)
            cleaned = strip_llm_fences(_strip_think_tags(raw))
            brace_pos = cleaned.find('{')
            if brace_pos >= 0:
                result = _repair_truncated_json(cleaned[brace_pos:])
                if result:
                    log.info("[%s] Salvaged partial response (%d keys)", phase, len(result))
        if result is None:
            log.warning("[%s] JSON parse failed (%.1fs). Raw: %s", phase, elapsed, raw[:300])
        return result

    except KeyboardInterrupt:
        log.info("[%s] Interrupted by user", phase)
        raise
    except Exception as e:
        elapsed = time.monotonic() - t0
        log.warning("[%s] Error after %.1fs: %s", phase, elapsed, e)
        return None


def _classify_intent_model(task: str, model: str) -> dict | None:
    """Classify task intent via LLM (fast mode). Returns parsed dict or None.

    Retries once with /no_think if first attempt returns empty (thinking
    tokens can consume the entire num_predict budget).
    """
    prompt = CLASSIFY_PROMPT.format(task=task) + get_prompt_suffix(model, "classify")
    result = _call_llm(prompt, model, "classify", CLASSIFY_TIMEOUT, CLASSIFY_NUM_PREDICT)
    if result is not None:
        return result

    # Retry: thinking tokens may have consumed the budget — force no_think
    log.info("[classify] Empty response, retrying with explicit /no_think prefix")
    retry_prompt = f"/no_think\n{prompt}"
    return _call_llm(retry_prompt, model, "classify_retry", CLASSIFY_TIMEOUT, CLASSIFY_NUM_PREDICT)


def _decompose_task(
    task: str,
    intent: str,
    model: str,
    file_context: str = "",
) -> dict | None:
    """Decompose task into steps via LLM. Returns parsed dict or None."""
    prompt = DECOMPOSE_PROMPT.format(
        task=task,
        intent=intent,
        file_context=file_context or "(no files specified)",
    ) + get_prompt_suffix(model, "decompose")
    result = _call_llm(prompt, model, "decompose", DECOMPOSE_TIMEOUT, DECOMPOSE_NUM_PREDICT,
                       streaming=True)
    if result is not None:
        steps = result.get("steps", [])
        log.info("[decompose] OK: %d steps parsed", len(steps))
    return result


def _validate_plan(
    task: str,
    steps: list[TaskStep],
    model: str,
) -> list[TaskStep] | None:
    """Validate and reorder plan steps using rule-based topological sort.

    Rules applied:
    - "read" before "edit" for the same file (understand first, then change)
    - "create" before "edit" for the same file (file must exist to edit)
    - "verify" steps always last
    - Existing depends_on edges are preserved
    Breaks cycles by dropping the edge from the higher-ID step.
    """
    if len(steps) < 2:
        return steps

    id_to_step: dict[int, TaskStep] = {s.id: s for s in steps}
    step_ids = [s.id for s in steps]

    # Build adjacency: edges[a] = {b} means a must come before b
    edges: dict[int, set[int]] = {sid: set() for sid in step_ids}

    # Preserve existing depends_on
    for s in steps:
        for dep_id in s.depends_on:
            if dep_id in id_to_step:
                edges[dep_id].add(s.id)

    # Rule 1 & 2: read/create before edit on same file
    file_steps: dict[str, list[TaskStep]] = {}
    for s in steps:
        for f in s.files:
            file_steps.setdefault(f, []).append(s)

    for _file, group in file_steps.items():
        reads = [s for s in group if s.action == "read"]
        creates = [s for s in group if s.action == "create"]
        edits = [s for s in group if s.action == "edit"]
        for r in reads:
            for e in edits:
                if r.id != e.id:
                    edges[r.id].add(e.id)
        for c in creates:
            for e in edits:
                if c.id != e.id:
                    edges[c.id].add(e.id)

    # Rule 3: verify steps come after all non-verify steps
    verify_ids = {s.id for s in steps if s.action == "verify"}
    non_verify_ids = [s.id for s in steps if s.action != "verify"]
    for nv in non_verify_ids:
        for v in verify_ids:
            edges[nv].add(v)

    # Topological sort (Kahn's algorithm) with cycle breaking
    in_degree: dict[int, int] = {sid: 0 for sid in step_ids}
    for src, dsts in edges.items():
        for d in dsts:
            if d in in_degree:
                in_degree[d] += 1

    queue = sorted([sid for sid in step_ids if in_degree[sid] == 0])
    ordered: list[int] = []
    max_iterations = len(step_ids) * 2  # safety cap

    for _ in range(max_iterations):
        if not queue:
            # Cycle detected — break by removing edge from highest-ID remaining node
            remaining = [sid for sid in step_ids if sid not in ordered]
            if not remaining:
                break
            victim = max(remaining)
            log.info("[validate] Breaking cycle: dropping edges into step %d", victim)
            in_degree[victim] = 0
            queue.append(victim)

        sid = queue.pop(0)
        if sid in ordered:
            continue
        ordered.append(sid)

        for d in edges.get(sid, set()):
            if d in in_degree:
                in_degree[d] -= 1
                if in_degree[d] <= 0 and d not in ordered:
                    queue.append(d)
                    queue.sort()

        if len(ordered) == len(step_ids):
            break

    # Check if order actually changed
    if ordered == step_ids:
        log.info("[validate] Plan order OK, no changes needed")
        return steps

    reordered = []
    for i, sid in enumerate(ordered, 1):
        s = id_to_step[sid]
        reordered.append(TaskStep(
            id=i,
            action=s.action,
            description=s.description,
            files=s.files,
            depends_on=s.depends_on,
        ))
    log.info("[validate] Reordered %d steps (rule-based)", len(reordered))
    return reordered


# ---------------------------------------------------------------------------
# Plan builders
# ---------------------------------------------------------------------------


def _build_fallback_plan(task: str, reason: str = "") -> TaskPlan:
    """Build a minimal single-step plan when interpretation fails.

    Graceful degradation: caller always gets a valid TaskPlan.
    """
    log.info("Falling back to single-step plan: %s", reason or "unknown")
    return TaskPlan(
        original_task=task,
        intent="feature",
        complexity=2,
        clarity=1.0,
        steps=[TaskStep(id=1, action="edit", description=task)],
        reasoning=f"Fallback plan ({reason})" if reason else "Fallback plan",
    )


def _build_single_step_plan(
    task: str,
    intent: str,
    clarity: float,
    complexity: int,
    model: str,
    raw_classification: str = "",
) -> TaskPlan:
    """Build a single-step plan for simple (Type 1) tasks."""
    action = "explain" if intent == "question" else "edit"
    return TaskPlan(
        original_task=task,
        intent=intent,
        complexity=complexity,
        clarity=clarity,
        steps=[TaskStep(id=1, action=action, description=task)],
        suggested_model=model,
        reasoning="Simple task -- single step, no decomposition needed.",
        raw_classification=raw_classification,
    )


def _parse_steps(steps_data: list) -> list[TaskStep]:
    """Parse and validate step dicts from LLM output into TaskStep objects."""
    valid_actions = {"read", "edit", "create", "delete", "verify", "explain"}
    steps: list[TaskStep] = []

    for item in steps_data:
        if not isinstance(item, dict):
            continue

        description = str(item.get("description", ""))
        if not description:
            continue

        action = str(item.get("action", "edit")).lower()
        if action not in valid_actions:
            action = "edit"

        files = item.get("files", [])
        if not isinstance(files, list):
            files = []
        files = [str(f) for f in files if f]

        depends_on = item.get("depends_on", [])
        if not isinstance(depends_on, list):
            depends_on = []
        depends_on = [int(d) for d in depends_on if isinstance(d, (int, float))]

        steps.append(TaskStep(
            id=item.get("id", len(steps) + 1),
            action=action,
            description=description,
            files=files,
            depends_on=depends_on,
        ))

    return steps


# ---------------------------------------------------------------------------
# Phase runners (extracted from interpret_task for readability)
# ---------------------------------------------------------------------------


def _classify_phase(
    task: str,
    model: str,
    _status: callable,
) -> tuple[str, int, float, list[str], str] | None:
    """Phase 1: Classify intent, complexity, clarity via keywords then LLM.

    Returns (intent, complexity, clarity, questions, raw_classification),
    or None if LLM classification fails entirely.
    """
    _status("classify", "Matching keywords...")
    intent = _classify_intent_keywords(task)
    complexity = _assess_complexity(task, intent or "feature")

    if intent is not None:
        log.info("Keyword classification: intent=%s complexity=%d", intent, complexity)
        clarity = _assess_clarity_heuristic(task)
        if clarity >= CLARITY_THRESHOLD:
            _status("classify", f"Matched: {intent}")
            return intent, complexity, 1.0, [], ""
        # Vague keyword match — fall through to LLM for clarity assessment
        log.info("Keyword match but low clarity (%.2f), deferring to LLM", clarity)

    # LLM classification
    _status("classify_llm", "Classifying with LLM...")
    result = _classify_intent_model(task, model)
    if result is None:
        return None

    raw_classification = str(result)

    intent = result.get("intent", "feature")
    if intent not in ("bug_fix", "feature", "refactor", "question"):
        intent = "feature"

    llm_complexity = result.get("complexity")
    if isinstance(llm_complexity, int) and llm_complexity in (1, 2, 3):
        complexity = llm_complexity

    clarity = float(result.get("clarity", 1.0))
    clarity = max(0.0, min(1.0, clarity))

    raw_questions = result.get("clarification_questions", [])
    if not isinstance(raw_questions, list):
        raw_questions = []
    questions = []
    for q in raw_questions:
        if isinstance(q, dict):
            text = str(q.get("question", ""))
            opts = q.get("options", [])
            if not isinstance(opts, list):
                opts = []
            opts = [str(o) for o in opts if o]
            if text:
                questions.append(ClarificationQuestion(question=text, options=opts))
        elif isinstance(q, str) and q:
            # Backward compat: plain string question (no options)
            questions.append(ClarificationQuestion(question=str(q)))

    log.info(
        "LLM classification: intent=%s complexity=%d clarity=%.2f",
        intent, complexity, clarity,
    )
    _status("classify_llm", f"Classified: {intent} (complexity {complexity})")

    return intent, complexity, clarity, questions, raw_classification


def _decompose_and_parse(
    task: str,
    intent: str,
    model: str,
    file_context: list[str] | None,
    _status: callable,
) -> tuple[list[TaskStep], str, str] | None:
    """Phase 4: Decompose task into steps via LLM, parse, and cap.

    Returns (steps, reasoning, raw_decomposition) or None on any failure.
    """
    _status("decompose", "Breaking down task into steps...")
    file_summary = _build_file_summary(file_context)
    decomp_result = _decompose_task(task, intent, model, file_context=file_summary)

    if decomp_result is None:
        return None

    raw_decomposition = str(decomp_result)

    steps_data = decomp_result.get("steps", [])
    if not isinstance(steps_data, list) or len(steps_data) == 0:
        return None

    steps = _parse_steps(steps_data)
    if not steps:
        return None

    if len(steps) > MAX_DECOMPOSITION_STEPS:
        log.warning("Decomposition produced %d steps, capping at %d",
                     len(steps), MAX_DECOMPOSITION_STEPS)
        steps = steps[:MAX_DECOMPOSITION_STEPS]

    reasoning = str(decomp_result.get("reasoning", ""))
    return steps, reasoning, raw_decomposition


def _suggest_model(
    complexity: int,
    intent: str,
    model: str,
    model_map: dict[str, str] | None,
) -> str:
    """Suggest the best model based on task complexity and intent."""
    if not model_map:
        return model
    if complexity == 1 or intent == "question":
        return model_map.get("fast", model)
    if complexity == 3:
        return model_map.get("planning", model)
    return model_map.get("daily", model)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def interpret_task(
    task: str,
    model: str,
    file_context: list[str] | None = None,
    model_map: dict[str, str] | None = None,
    on_status: callable | None = None,
    prior_classification: tuple[str, int] | None = None,
) -> TaskPlan:
    """Interpret a user task and produce a structured execution plan.

    Pipeline:
    1. Keyword-based intent classification (fast path, no LLM)
    2. If no keyword match: LLM classification (fast mode)
    3. If clarity < threshold: return plan with clarification questions
    4. If complexity == 1: return single-step plan (no decomposition)
    5. If complexity >= 2: LLM decomposition (thinking mode)
    6. Validate and return structured TaskPlan

    Never raises -- all errors produce fallback plans.

    Args:
        on_status: Optional callback(phase, detail) for live progress updates.
            phase: "classify", "classify_llm", "clarity", "decompose", "done"
            detail: short description string
        prior_classification: Optional (intent, complexity) from a previous
            clarification round. Skips re-classification and goes straight
            to decomposition with clarity=1.0.
    """
    def _status(phase: str, detail: str = "") -> None:
        if on_status is not None:
            on_status(phase, detail)

    def _finish(plan: TaskPlan) -> TaskPlan:
        elapsed = time.monotonic() - session_start
        step_summary = ", ".join(f"{s.action}" for s in plan.steps) if plan.steps else "none"
        log.info("RESULT: intent=%s complexity=%d clarity=%.0f%% steps=%d [%s] (%.1fs total)",
                 plan.intent, plan.complexity, plan.clarity * 100,
                 len(plan.steps), step_summary, elapsed)
        if plan.reasoning and "Fallback" in plan.reasoning:
            log.warning("FALLBACK: %s", plan.reasoning)
        log.info("-" * 72)
        return plan

    session_start = time.monotonic()
    log.info("=" * 72)
    log.info("INTERPRET: %s", task[:200])
    log.info("  model=%s  files=%d", model, len(file_context or []))

    # --- Phase 1: Classification ---
    if prior_classification is not None:
        # Post-clarification: user already answered questions, skip re-classify
        intent, complexity = prior_classification
        clarity = 1.0
        questions = []
        raw_classification = ""
        log.info("Using prior classification: intent=%s complexity=%d (post-clarification)",
                 intent, complexity)
        _status("classify", f"Reusing: {intent} (clarified)")
    else:
        classified = _classify_phase(task, model, _status)
        if classified is None:
            _status("done", "Classification failed, using fallback")
            return _finish(_build_fallback_plan(task, reason="classification LLM failed"))

        intent, complexity, clarity, questions, raw_classification = classified

    # --- Phase 2: Clarity check ---
    _status("clarity", f"Clarity: {clarity:.0%}")
    if clarity < CLARITY_THRESHOLD and questions:
        _status("done", "Needs clarification")
        return _finish(TaskPlan(
            original_task=task, intent=intent, complexity=complexity,
            clarity=clarity, steps=[], clarification_questions=questions,
            suggested_model=model,
            reasoning="Task is ambiguous -- clarification needed before planning.",
            raw_classification=raw_classification,
        ))

    # --- Phase 3: Simple task fast path ---
    if complexity == 1:
        _status("done", "Simple task -- skipping decomposition")
        return _finish(_build_single_step_plan(
            task=task, intent=intent, clarity=clarity,
            complexity=complexity, model=model,
            raw_classification=raw_classification,
        ))

    # --- Phase 4: Decomposition ---
    decomposed = _decompose_and_parse(task, intent, model, file_context, _status)
    if decomposed is None:
        _status("done", "Decomposition failed, using fallback")
        plan = _build_fallback_plan(task, reason="decomposition failed")
        plan.intent = intent
        plan.complexity = complexity
        plan.clarity = clarity
        plan.raw_classification = raw_classification
        return _finish(plan)

    steps, reasoning, raw_decomposition = decomposed

    # --- Phase 5: Validate step ordering ---
    if len(steps) >= 2:
        _status("validate", "Finalizing the plan...")
        validated = _validate_plan(task, steps, model)
        if validated is not None:
            steps = validated

    _status("done", f"Planned {len(steps)} step(s)")

    return _finish(TaskPlan(
        original_task=task, intent=intent, complexity=complexity,
        clarity=clarity, steps=steps,
        suggested_model=_suggest_model(complexity, intent, model, model_map),
        reasoning=reasoning,
        raw_classification=raw_classification,
        raw_decomposition=raw_decomposition,
    ))
