"""Maker-Checker reflection orchestration with iterative repair.

Includes intelligent triggering (Phase 3): decides per-task whether to
engage the checker based on mode, diff size, and task complexity signals.
"""

from __future__ import annotations

import subprocess
import time

from rich.rule import Rule

from probablyfine.agent import execute_plan
from probablyfine.checker import MAX_DIFF_LINES, quick_sanity_check, run_checker
from probablyfine.console import console
from probablyfine.drm import get_manager as get_drm
from probablyfine.models import (
    CHECKER_HANG,
    CHECKER_OOM,
    EXIT_SIGINT,
    CheckerRequest,
    CheckerResult,
    ReflectionContext,
    ReflectionLog,
    ReflectionState,
    TaskPlan,
    TaskStep,
)
from probablyfine.git_utils import get_head_sha as _get_head_sha
from probablyfine.log_utils import get_module_logger
from probablyfine.modes import Mode

log_reflect = get_module_logger("probablyfine.reflection", "reflection.log")

# Diff line thresholds for auto-reflection triggering
_TRIVIAL_DIFF_LINES = 10       # below this: skip reflection
_SUBSTANTIAL_DIFF_LINES = 30   # above this: always reflect
_DELETION_GUARD_MIN = 20       # minimum deletions to trigger deletion-ratio guard
_DELETION_GUARD_RATIO = 3      # deletions must exceed additions by this factor

# Task keywords that signal complexity — reflection is more valuable here
_COMPLEX_SIGNALS = [
    "refactor", "security", "auth", "authentication", "database", "migration",
    "api", "deploy", "delete", "remove", "permission", "encrypt", "password",
    "injection", "validation", "concurrency", "async", "thread",
    "review", "audit", "inspect", "debt",
]


def _fallback_plan(task: str, files: list[str] | None) -> TaskPlan:
    """Create a single-step edit plan when no interpreter plan is available."""
    return TaskPlan(
        original_task=task, intent="feature", complexity=2, clarity=1.0,
        steps=[TaskStep(id=1, action="edit", description=task, files=list(files or []))],
    )


def should_reflect(
    current_mode: Mode,
    task: str,
    diff: str,
    reflection_mode: str = "auto",
) -> bool:
    """Decide whether to engage the checker for this task.

    Args:
        current_mode: The active PROBABLYFINE mode (FAST/DAILY/PLANNING/AUTO).
        task: The original user task text.
        diff: The diff produced by the maker phase.
        reflection_mode: Config setting — "auto", "always", or "never".

    Returns True if the checker should review this diff.
    """
    # Config overrides
    if reflection_mode == "always":
        return True
    if reflection_mode == "never":
        return False

    # Deletion-ratio guard: large deletions are risky, force reflection
    plus = 0
    minus = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            plus += 1
        elif line.startswith("-") and not line.startswith("---"):
            minus += 1
    if minus > _DELETION_GUARD_MIN and minus > plus * _DELETION_GUARD_RATIO:
        log_reflect.info(
            "Deletion-ratio guard triggered: +%d -%d (ratio %.1f:1)",
            plus, minus, minus / max(plus, 1),
        )
        return True

    # Auto mode heuristics
    # Never reflect in FAST mode — speed is the priority
    if current_mode == Mode.FAST:
        return False

    # Always reflect in PLANNING mode — safety is the priority
    if current_mode == Mode.PLANNING:
        return True

    # For DAILY and AUTO modes: check diff size and task complexity
    diff_lines = len(diff.strip().splitlines())

    # Trivial diffs: skip reflection
    if diff_lines < _TRIVIAL_DIFF_LINES:
        return False

    # Substantial diffs: always reflect
    if diff_lines > _SUBSTANTIAL_DIFF_LINES:
        return True

    # Medium diffs: check for complexity signals in the task
    task_lower = task.lower()
    if any(signal in task_lower for signal in _COMPLEX_SIGNALS):
        return True

    # Medium diff, no complexity signals — skip
    return False


def _phase_rule(label: str, style: str = "check.info") -> None:
    """Print a visual separator for a reflection phase."""
    console.print()
    console.print(Rule(f" {label} ", style=style, align="left"))


def _format_confidence(confidence: float) -> str:
    """Format confidence as a small bar: [|||   ] 60%."""
    filled = round(confidence * 5)
    bar = "|" * filled + " " * (5 - filled)
    return f"[{bar}] {confidence:.0%}"


def display_result(result: CheckerResult, duration_s: float = 0.0) -> None:
    """Print checker result to terminal with structured formatting."""
    # Verdict line with confidence
    conf_str = _format_confidence(result.confidence)
    time_str = f" ({duration_s:.0f}s)" if duration_s > 0 else ""

    if result.verdict == "PASS":
        console.print(f"\n  [check.pass]PASS[/check.pass] {conf_str}{time_str}")
        if result.summary:
            console.print(f"  {result.summary}")
    elif result.verdict == "FAIL":
        console.print(f"\n  [check.fail]FAIL[/check.fail] {conf_str}{time_str}")
        if result.summary:
            console.print(f"  {result.summary}")
    elif result.verdict == "ESCALATE":
        console.print(f"\n  [check.warn]ESCALATE[/check.warn] {conf_str}{time_str}")
        if result.summary:
            console.print(f"  {result.summary}")

    if result.issues:
        critical = [i for i in result.issues if i.severity == "critical"]
        warnings = [i for i in result.issues if i.severity != "critical"]

        if critical:
            console.print(f"\n  [check.critical]Critical ({len(critical)}):[/check.critical]")
            for issue in critical:
                loc = f"{issue.file}:{issue.line}" if issue.line else issue.file or "?"
                console.print(f"    [check.critical]\u2718[/check.critical] {loc}")
                console.print(f"      {issue.description}")
                if issue.suggestion:
                    console.print(f"      [check.info]Fix: {issue.suggestion}[/check.info]")

        if warnings:
            console.print(f"\n  [check.warning]Warnings ({len(warnings)}):[/check.warning]")
            for issue in warnings:
                loc = f"{issue.file}:{issue.line}" if issue.line else issue.file or "?"
                console.print(f"    [check.warning]\u26a0[/check.warning] {loc}")
                console.print(f"      {issue.description}")
                if issue.suggestion:
                    console.print(f"      [check.info]Fix: {issue.suggestion}[/check.info]")

    console.print()


def build_revision_prompt(task: str, result: CheckerResult) -> str:
    """Build a revision prompt from checker feedback.

    Only includes critical issues — warnings are displayed but don't
    trigger repair.
    """
    critical_issues = [i for i in result.issues if i.severity == "critical"]
    if not critical_issues:
        # Fallback: include all issues if no critical ones
        critical_issues = result.issues

    issues_text = "\n".join(
        f"- [{i.severity.upper()}] {i.file}:{i.line or '?'}: "
        f"{i.description}. Fix: {i.suggestion}"
        for i in critical_issues
    )
    return (
        f"A code review found issues with the previous changes. "
        f"Original task: {task}\n\n"
        f"Issues to fix:\n{issues_text}\n\n"
        f"Please fix ONLY these specific issues. "
        f"Do not refactor or change anything else."
    )


def _issues_fingerprint(result: CheckerResult) -> set[str]:
    """Create a fingerprint of issues for stuck-loop detection.

    Returns a set of (file, description) tuples — if two iterations
    produce the same set, the loop is stuck.
    """
    return {
        f"{i.file}:{i.description}"
        for i in result.issues
        if i.severity == "critical"
    }


def _git_reset_soft(sha: str) -> bool:
    """Soft-reset to a SHA. Returns True on success."""
    if not sha:
        return False
    try:
        result = subprocess.run(
            ["git", "reset", "--soft", sha],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_repair(
    ctx: ReflectionContext,
    result: CheckerResult,
    iteration: int,
    max_iterations: int,
) -> tuple[str, str | None]:
    """Run agent repair for critical issues found by the checker.

    Returns (status, new_diff) where:
    - ("ok", diff) — repair succeeded, use diff for next checker iteration
    - ("interrupted", None) — user cancelled
    - ("failed", None) — agent crashed, rollback attempted
    - ("empty", None) — repair produced no changes
    """
    revision_prompt = build_revision_prompt(ctx.task, result)
    pre_repair_sha = _get_head_sha()

    _phase_rule(
        f"REVISE  {ctx.maker_model}  iter {iteration}/{max_iterations}",
        style="bold yellow",
    )

    # DRM: switch back to maker model if needed
    drm = get_drm()
    if ctx.maker_model != ctx.checker_model:
        drm.prepare_for_checker(ctx.maker_model, ctx.checker_model)

    try:
        repair_plan = TaskPlan(
            original_task=ctx.task, intent="bug_fix", complexity=1, clarity=1.0,
            steps=[TaskStep(id=1, action="edit", description=revision_prompt,
                            files=list(ctx.files or []))],
        )
        repair_result = execute_plan(repair_plan, ctx.maker_model, ctx.files, ctx.agent_config)
        repair_exit = repair_result.exit_code
        repair_diff = repair_result.diff
    except KeyboardInterrupt:
        console.print("\n  [check.info]Repair interrupted -- keeping current state.[/check.info]")
        return "interrupted", None

    if repair_exit != 0 and repair_exit != EXIT_SIGINT:
        console.print(f"  [check.fail]Repair failed (code {repair_exit})[/check.fail]")
        if pre_repair_sha:
            console.print("  [check.info]Rolling back to pre-repair state...[/check.info]")
            if _git_reset_soft(pre_repair_sha):
                console.print("  [check.info]Rollback successful.[/check.info]")
            else:
                console.print("  [check.warn]Rollback failed -- manual recovery may be needed.[/check.warn]")
        return "failed", None

    if not repair_diff.strip():
        console.print("  [check.info]No changes from repair -- stopping loop.[/check.info]")
        return "empty", None

    # DRM: switch back to checker model for re-check
    if ctx.maker_model != ctx.checker_model:
        drm.prepare_for_checker(ctx.checker_model, ctx.maker_model)

    return "ok", repair_diff


def _run_checker_loop(
    ctx: ReflectionContext,
    diff: str,
    state: ReflectionState,
    log: ReflectionLog,
) -> None:
    """Checker + repair iteration loop with stuck-loop detection.

    Runs the checker on the diff, and if critical issues are found,
    invokes repair via agent and re-checks. Terminates on PASS,
    ESCALATE, max iterations, stuck detection, or user interrupt.

    Updates state and log in place.
    """
    # DRM: prepare for model transition (no-op if same model)
    checker_plan = get_drm().prepare_for_checker(ctx.checker_model, ctx.maker_model)
    if checker_plan.needs_swap:
        console.print(
            f"  [check.info]Swapping to checker (~{checker_plan.estimated_swap_time_s:.0f}s)[/check.info]"
        )

    # Track SHA before each repair for rollback safety
    pre_repair_sha = _get_head_sha()

    # Pre-checker sanity filter: catch syntax errors before expensive LLM call
    sanity_result = quick_sanity_check(ctx.files)
    if sanity_result is not None:
        _phase_rule("PRE-CHECK  syntax errors detected", style="bold red")
        state.status = "failed"
        state.history.append(sanity_result)
        log.iterations.append({
            "maker_model": ctx.maker_model,
            "checker_model": "pre-checker",
            "diff_lines": len(diff.strip().splitlines()),
            "verdict": sanity_result.verdict,
            "issues_count": len(sanity_result.issues),
            "duration_s": 0.0,
        })
        log.final_verdict = "failed"
        display_result(sanity_result, duration_s=0.0)
        return

    for iteration in range(1, state.max_iterations + 1):
        state.status = "checking"
        state.iteration = iteration
        diff_lines = len(diff.strip().splitlines())
        truncated = diff_lines > MAX_DIFF_LINES
        display_lines = f"{diff_lines} lines, truncated to {MAX_DIFF_LINES}" if truncated else f"{diff_lines} lines"

        if iteration == 1:
            _phase_rule(f"CHECKER  {ctx.checker_model}  ({display_lines})", style="bold magenta")
        else:
            _phase_rule(
                f"RE-CHECK  {ctx.checker_model}  iter {iteration}/{state.max_iterations}",
                style="bold magenta",
            )

        iter_start = time.monotonic()
        result = run_checker(CheckerRequest(
            task=ctx.task,
            diff=diff,
            files=ctx.files,
            model=ctx.checker_model,
            iteration=iteration,
            max_iterations=state.max_iterations,
        ))
        iter_duration = time.monotonic() - iter_start
        state.history.append(result)

        # Log this iteration
        log.iterations.append({
            "maker_model": ctx.maker_model,
            "checker_model": ctx.checker_model,
            "diff_lines": diff_lines,
            "verdict": result.verdict,
            "issues_count": len(result.issues),
            "duration_s": round(iter_duration, 1),
        })

        display_result(result, duration_s=iter_duration)

        # -- Failure mode: skip remaining iterations on hang/OOM --
        if result.failure_mode in (CHECKER_HANG, CHECKER_OOM):
            log_reflect.warning("Checker failure mode: %s — skipping remaining iterations",
                                result.failure_mode)
            console.print(
                f"  [check.warn]Checker {result.failure_mode} — skipping remaining iterations.[/check.warn]"
            )
            state.status = "passed"
            log.final_verdict = f"PASS ({result.failure_mode})"
            break

        # -- PASS: done --
        if result.verdict == "PASS":
            state.status = "passed"
            log.final_verdict = "PASS"
            break

        # -- ESCALATE: accept with warning (Phase 5 will handle properly) --
        if result.verdict == "ESCALATE":
            state.status = "failed"
            log.final_verdict = "ESCALATE"
            console.print(
                "  [check.info]Escalation to planning model will be added in Phase 5. "
                "Accepting current changes.[/check.info]"
            )
            break

        # -- FAIL: attempt repair if iterations remain --
        has_critical = any(i.severity == "critical" for i in result.issues)
        if not has_critical:
            # Only warnings, no critical issues — treat as soft pass
            console.print("  [check.info]No critical issues found -- accepting changes.[/check.info]")
            state.status = "passed"
            log.final_verdict = "PASS (warnings only)"
            break

        # Stuck-loop detection: same critical issues as previous iteration
        if len(state.history) >= 2:
            prev_fp = _issues_fingerprint(state.history[-2])
            curr_fp = _issues_fingerprint(result)
            if prev_fp and curr_fp and prev_fp == curr_fp:
                console.print(
                    "  [check.warn]Stuck loop detected -- same issues as previous iteration. "
                    "Stopping repair.[/check.warn]"
                )
                state.status = "exhausted"
                log.final_verdict = "STUCK"
                break

        # Check if we've exhausted iterations
        if iteration >= state.max_iterations:
            console.print(
                f"  [check.warn]Max iterations ({state.max_iterations}) reached. "
                f"Keeping current changes.[/check.warn]"
            )
            state.status = "exhausted"
            log.final_verdict = "EXHAUSTED"
            break

        # -- REPAIR PHASE --
        state.status = "revising"
        repair_status, repair_diff = _run_repair(
            ctx, result, iteration, state.max_iterations,
        )

        if repair_status == "interrupted":
            state.status = "failed"
            log.final_verdict = "INTERRUPTED"
            break
        elif repair_status == "failed":
            state.status = "failed"
            log.final_verdict = "REPAIR_FAILED"
            break
        elif repair_status == "empty":
            state.status = "exhausted"
            log.final_verdict = "NO_REPAIR_CHANGES"
            break

        # Use the new diff for the next checker iteration
        diff = repair_diff


def _display_reflection_summary(log: ReflectionLog) -> None:
    """Display final reflection verdict and timing."""
    if log.final_verdict in ("PASS", "PASS (warnings only)"):
        verdict_style = "bold green"
    elif log.final_verdict in ("FAIL", "STUCK", "EXHAUSTED", "REPAIR_FAILED"):
        verdict_style = "bold red"
    else:
        verdict_style = "bold yellow"

    if len(log.iterations) > 1:
        console.print(Rule(style="dim"))
        console.print(
            f"  Reflection: [{verdict_style}]{log.final_verdict}[/{verdict_style}]"
            f"  ({len(log.iterations)} iterations, {log.total_duration_s}s)"
        )
        console.print()
    elif len(log.iterations) == 1:
        console.print(Rule(style="dim"))
        console.print(
            f"  Reflection: [{verdict_style}]{log.final_verdict}[/{verdict_style}]"
            f"  ({log.total_duration_s}s)"
        )
        console.print()


def run_with_reflection(ctx: ReflectionContext) -> int:
    """Execute a task with maker-checker reflection and iterative repair.

    Flow:
    1. Maker phase: run agent, capture diff
    2. Intelligent triggering: decide whether to run checker based on mode/diff/task
    3. Checker phase: review diff
    4. If FAIL and iterations remain: build revision prompt, re-invoke agent, re-check
    5. Loop terminates on PASS, max iterations, stuck detection, or user interrupt

    Returns agent exit code.
    """
    start_time = time.monotonic()

    state = ReflectionState(
        task=ctx.task,
        maker_model=ctx.maker_model,
        checker_model=ctx.checker_model,
        head_before="",
    )
    log = ReflectionLog(task=ctx.task)

    # -- MAKER PHASE --
    state.status = "making"
    _phase_rule(f"MAKER  {ctx.maker_model}", style="bold blue")

    task_plan = ctx.plan or _fallback_plan(ctx.task, ctx.files)
    agent_result = execute_plan(task_plan, ctx.maker_model, ctx.files, ctx.agent_config)
    exit_code = agent_result.exit_code
    diff = agent_result.diff
    head_before = agent_result.head_before

    state.head_before = head_before

    if exit_code != 0 and exit_code != EXIT_SIGINT:
        state.status = "failed"
        return exit_code

    if not diff.strip():
        console.print("  [check.info]No changes detected -- skipping checker.[/check.info]")
        state.status = "passed"
        return exit_code

    # -- INTELLIGENT TRIGGERING --
    if not should_reflect(ctx.current_mode, ctx.task, diff, ctx.reflection_mode):
        diff_lines = len(diff.strip().splitlines())
        console.print(
            f"  [check.info]Reflection skipped ({diff_lines} line diff, "
            f"mode={ctx.current_mode.value}, trigger={ctx.reflection_mode})[/check.info]"
        )
        if diff_lines < 10 and ctx.reflection_mode == "auto":
            console.print(
                "  [check.info]Tip: use /reflect always to review all diffs.[/check.info]"
            )
        state.status = "passed"
        return exit_code

    # -- CHECKER PHASE (with repair loop) --
    _run_checker_loop(ctx, diff, state, log)

    if not log.final_verdict:
        log.final_verdict = state.status
    log.total_duration_s = round(time.monotonic() - start_time, 1)

    _display_reflection_summary(log)

    return exit_code
