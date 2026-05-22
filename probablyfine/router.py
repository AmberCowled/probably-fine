"""Task classifier for AUTO mode.

Hybrid approach:
1. Keyword fast-path — instant, no model needed
2. Model classification — uses classifier model (qwen3:8b) via Ollama API
3. Fallback — defaults to DAILY if anything fails
"""

import re

from probablyfine.modes import Mode
from probablyfine.ollama_utils import (
    build_chat_options,
    create_client,
    extract_content as _extract_content,
)

CLASSIFICATION_PROMPT = """\
You are a task classifier. Given a coding task, respond with exactly one word.

FAST - quick fixes, typos, renaming, simple snippets, one-line changes, formatting
DAILY - implementation tasks, features, refactoring, bug fixes, writing functions, tests
PLANNING - architecture decisions, design discussions, system design, trade-off analysis, comparing approaches

Task: {task}

Classification:"""

# Keyword patterns for fast-path classification (checked before calling model)
_PLANNING_PATTERNS = [
    r"\b(design|architect|plan|tradeoff|trade-off|compare|evaluate)\b",
    r"\b(should (i|we)|which approach|pros and cons|best (way|practice))\b",
    r"\b(system design|high level|overview|strategy|roadmap)\b",
]

_FAST_PATTERNS = [
    r"\b(fix typo|rename|one.?liner|formatting|lint|simple fix)\b",
    r"\b(change .{1,30} to|swap|replace .{1,20} with)\b",
    r"\b(quick|trivial|small (fix|change|tweak))\b",
]

# Patterns for simple tasks that can be downgraded from DAILY to FAST
_SIMPLE_TASK_PATTERNS = [
    r"\b(update|change|tweak|adjust|fix)\b.{0,30}\b(css|style|color|font|margin|padding|border)\b",
    r"\b(add|insert|include)\b.{0,20}\b(media quer|breakpoint|responsive)\b",
    r"\b(center|align|position|spacing|gap|flex|grid)\b.{0,20}\b(css|html|element|div|section)\b",
    r"\b(html|css|ui)\b.{0,20}\b(typo|comment|class name|id)\b",
]


def _keyword_classify(task: str) -> Mode | None:
    """Try to classify via keyword patterns. Returns None if no match."""
    lower = task.lower()
    for pattern in _PLANNING_PATTERNS:
        if re.search(pattern, lower):
            return Mode.PLANNING
    for pattern in _FAST_PATTERNS:
        if re.search(pattern, lower):
            return Mode.FAST
    return None


def _model_classify(task: str, model: str, timeout: float = 10.0) -> Mode | None:
    """Classify using the Ollama API. Returns None on failure."""
    prompt = CLASSIFICATION_PROMPT.format(task=task)

    try:
        client = create_client(timeout=timeout)
        options = build_chat_options(model=model, num_predict=10)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
        )
        text = _extract_content(response).strip().upper()

        # Extract the first word that matches a valid mode
        for word in text.split():
            cleaned = word.strip(".,!:;\"'")
            if cleaned == "FAST":
                return Mode.FAST
            if cleaned == "DAILY":
                return Mode.DAILY
            if cleaned == "PLANNING":
                return Mode.PLANNING

        return None
    except Exception:
        return None


def is_simple_task(task: str) -> bool:
    """Return True if the task is simple enough to downgrade from DAILY to FAST.

    Used for task-complexity routing: simple UI/CSS tasks run faster on the
    small model without meaningful quality loss.
    """
    lower = task.lower()
    # Must match a simple pattern AND not match a complex/planning pattern
    for pattern in _SIMPLE_TASK_PATTERNS:
        if re.search(pattern, lower):
            # Don't downgrade if it also looks complex
            for complex_p in _PLANNING_PATTERNS:
                if re.search(complex_p, lower):
                    return False
            return True
    return False


def classify_task(task: str, classifier_model: str) -> tuple[Mode, str]:
    """Classify a task into a mode.

    Uses the classifier model (typically qwen3:8b) for better reasoning
    when keyword matching fails.

    Returns (mode, reason) where reason explains the routing decision.
    """
    # 1. Keyword fast-path
    keyword_result = _keyword_classify(task)
    if keyword_result is not None:
        return keyword_result, "keyword match"

    # 2. Model classification
    model_result = _model_classify(task, classifier_model)
    if model_result is not None:
        return model_result, "model classified"

    # 3. Fallback
    return Mode.DAILY, "default fallback"
