"""Data structures and shared constants for probablyfine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Ctrl+C exit code (128 + SIGINT). Used by cli.py and reflection.py
# to distinguish user interrupts from real failures.
EXIT_SIGINT = 130


@dataclass
class Issue:
    severity: str          # "critical" | "warning"
    file: str              # file path
    description: str       # what's wrong
    suggestion: str        # how to fix
    line: int | None = None


@dataclass
class CheckerResult:
    verdict: str           # "PASS" | "FAIL" | "ESCALATE"
    confidence: float      # 0.0-1.0
    issues: list[Issue]
    summary: str
    raw_response: str = ""


@dataclass
class ReflectionState:
    task: str
    maker_model: str
    checker_model: str
    head_before: str
    iteration: int = 0
    max_iterations: int = 2
    history: list[CheckerResult] = field(default_factory=list)
    status: str = "pending"  # pending | making | checking | revising | passed | failed | exhausted


@dataclass
class ReflectionLog:
    """Record of a complete reflection session for debugging/display."""
    task: str
    iterations: list[dict] = field(default_factory=list)
    # Each entry: {maker_model, checker_model, diff_lines, verdict, issues_count, duration_s}
    final_verdict: str = ""
    total_duration_s: float = 0.0


@dataclass
class CheckerRequest:
    """Parameters for a single checker invocation."""
    task: str
    diff: str
    files: list[str] | None
    model: str
    iteration: int = 1
    max_iterations: int = 2
    timeout: float = 180


@dataclass
class FileEdit:
    """A single file edit parsed from LLM SEARCH/REPLACE output."""
    file: str               # relative path
    search: str             # content to find (empty for new/whole files)
    replace: str            # replacement content
    is_new_file: bool = False
    is_delete_file: bool = False
    line_start: int | None = None   # line-anchored edit: start line (1-based)
    line_end: int | None = None     # line-anchored edit: end line (1-based, inclusive)


@dataclass
class ClarificationQuestion:
    """A clarification question with suggested answer options."""
    question: str
    options: list[str] = field(default_factory=list)  # 2-4 suggested answers


@dataclass
class ReflectionContext:
    """Static context for a reflection session.

    Bundles the parameters that remain constant throughout the
    maker-checker-repair loop, replacing 7-param function signatures.
    """
    task: str
    maker_model: str
    checker_model: str
    files: list[str] | None = None
    current_mode: Any = None       # Mode enum — Any to avoid circular import
    reflection_mode: str = "auto"
    agent_config: AgentConfig | None = None
    plan: TaskPlan | None = None               # interpreter's decomposed plan


@dataclass
class SessionState:
    """Mutable session state for the REPL loop.

    Bundles the per-session configuration that _execute_task and command
    handlers need, avoiding long parameter lists.
    """
    current_mode: Any       # Mode enum — Any to avoid circular import
    config: dict
    model_map: dict[str, str]
    ctx: Any                # FileContext — Any to avoid circular import
    drm: Any                # DynamicResourceManager
    reflect_on: bool
    reflection_mode: str
    checker_model: str
    auto_select: bool
    max_file_select: int


@dataclass
class TaskStep:
    """A single step in a decomposed task plan."""
    id: int
    action: str            # "read" | "edit" | "create" | "delete" | "verify" | "explain"
    description: str
    files: list[str] = field(default_factory=list)
    depends_on: list[int] = field(default_factory=list)


@dataclass
class TaskPlan:
    """Structured plan produced by the interpreter for downstream execution."""
    original_task: str
    intent: str              # "bug_fix" | "feature" | "refactor" | "question"
    complexity: int          # 1 (narrow), 2 (moderate), 3 (large)
    clarity: float           # 0.0-1.0
    steps: list[TaskStep]
    clarification_questions: list[ClarificationQuestion] = field(default_factory=list)
    suggested_model: str = ""
    reasoning: str = ""
    raw_classification: str = ""
    raw_decomposition: str = ""


@dataclass
class StepResult:
    """Result of executing a single TaskStep."""
    step_id: int
    status: str             # "ok" | "failed" | "skipped"
    edits_applied: int = 0
    files_changed: list[str] = field(default_factory=list)
    error: str = ""
    explanation: str = ""   # for "explain" steps
    duration_s: float = 0.0


@dataclass
class AgentResult:
    """Result of executing a complete TaskPlan (or single step)."""
    diff: str
    exit_code: int
    head_before: str
    steps: list[StepResult] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class AgentConfig:
    """Configuration for the custom agent."""
    conservative: bool = False
    auto_checkpoint: bool = True
    lint_command: str = ""
    test_command: str = ""
    num_ctx: int | None = None
    dark_mode: bool = True


@dataclass
class StepContext:
    """Bundles the execution context threaded through agent step functions.

    Replaces the repeated (model, config, watchdog, on_token, plan) parameters.
    """
    model: str
    config: AgentConfig
    watchdog: Any = None          # HealthWatchdog — Any to avoid circular import
    on_token: Any = None          # Callable[[str], None] | None
    plan: TaskPlan | None = None
