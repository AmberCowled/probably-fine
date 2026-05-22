import argparse
import random
import subprocess
import sys
import threading
import time

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from probablyfine import __version__
from probablyfine.console import ACTION_COLORS as _ACTION_COLORS, console
from probablyfine.config import (
    get_agent_auto_checkpoint,
    get_agent_conservative,
    get_agent_lint_command,
    get_agent_test_command,
    get_auto_file_select,
    get_checker_model,
    get_context_size,
    get_dark_mode,
    get_default_mode,
    get_drm_enabled,
    get_drm_fast_keep_alive,
    get_drm_large_keep_alive,
    get_interpreter_enabled,
    get_max_file_select,
    get_model_map,
    get_reflection_enabled,
    get_reflection_mode,
    load_config,
    save_config_value,
)
from probablyfine.agent import execute_plan
from probablyfine.drm import get_manager as get_drm
from probablyfine.file_selector import select_files
from probablyfine.interpreter import get_log_path, interpret_task
from probablyfine.reflection import _fallback_plan, run_with_reflection
from probablyfine.context import FileContext
from probablyfine.models import EXIT_SIGINT, AgentConfig, AgentResult, ClarificationQuestion, ReflectionContext, SessionState, TaskPlan
from probablyfine.git_utils import git_branch_status, git_diff_stat, git_status, git_undo_last_commit
from probablyfine.modes import MODE_DESCRIPTIONS, Mode
from probablyfine.router import classify_task, is_simple_task

# Intent display labels
_INTENT_LABELS = {
    "bug_fix": ("Bug Fix", "red"),
    "feature": ("Feature", "green"),
    "refactor": ("Refactor", "yellow"),
    "question": ("Question", "cyan"),
}

_COMPLEXITY_LABELS = {
    1: ("Simple", "green"),
    2: ("Moderate", "yellow"),
    3: ("Complex", "red"),
}

# Rotating verbs shown during LLM thinking phases
_THINKING_VERBS = [
    "Pondering", "Reasoning", "Synthesizing", "Deliberating",
    "Cogitating", "Plotting", "Mindforging", "Brainspinning",
    "Thinkflowing", "Overclocking", "Mulling", "Deciphering",
    "Unraveling", "Connecting dots", "Mapping it out", "Going deeper",
    "Chewing on it", "Noodling", "Percolating", "Crystallizing",
]

_DECOMPOSE_VERBS = [
    "Breaking it down", "Splitting into steps", "Charting a course",
    "Drawing the blueprint", "Planning the approach", "Laying out the pieces",
    "Structuring the work", "Sequencing steps", "Forging a plan",
    "Architecting", "Orchestrating", "Mapping the path", "Strategizing",
    "Assembling the playbook", "Drafting the gameplan",
]

_VALIDATE_VERBS = [
    "Sanity checking", "Double checking", "Reviewing the plan",
    "Polishing", "Tightening up", "Final pass", "Smoothing edges",
    "Quality control", "Cross-referencing", "Proof reading",
]

# -- Named constants (magic-number extraction) --
_SPINNER_INTERVAL_MIN = 2.0       # seconds between verb rotations
_SPINNER_INTERVAL_MAX = 3.5
_CLARITY_GOOD = 0.8               # clarity >= this: green
_CLARITY_FAIR = 0.5               # clarity >= this: yellow; below: red
_VRAM_CRITICAL_MB = 200           # free MB below this: red
_VRAM_WARNING_MB = 500            # free MB below this: yellow
_LOG_TAIL_LINES = 50              # max lines shown for /logs
_MAX_CLARIFICATION_ROUNDS = 2     # max interactive clarification loops


def _run_interpret_with_spinner(task, model, file_context, model_map, prior_classification=None):
    """Run interpret_task with a multi-phase spinner and rotating status verbs."""
    current_phase = {"name": "init", "verbs": _THINKING_VERBS}
    stop_event = threading.Event()

    def _verb_rotator(spinner):
        """Background thread that rotates verbs every 2-3s during LLM calls."""
        while not stop_event.is_set():
            stop_event.wait(random.uniform(_SPINNER_INTERVAL_MIN, _SPINNER_INTERVAL_MAX))
            if stop_event.is_set():
                break
            verb = random.choice(current_phase["verbs"])
            spinner.update(f"  [bold cyan]{verb}...[/bold cyan]")

    with console.status("  [bold cyan]Analyzing user intent...[/bold cyan]", spinner="dots") as spinner:
        rotator = threading.Thread(target=_verb_rotator, args=(spinner,), daemon=True)
        rotator.start()

        def _update(phase, detail):
            if phase == "classify":
                spinner.update("  [bold cyan]Analyzing user intent...[/bold cyan]")
            elif phase == "classify_llm":
                current_phase["verbs"] = _THINKING_VERBS
                spinner.update("  [bold cyan]Reasoning...[/bold cyan]")
            elif phase == "clarity":
                spinner.update(f"  [bold cyan]Assessing clarity...[/bold cyan] [dim]{detail}[/dim]")
            elif phase == "decompose":
                current_phase["verbs"] = _DECOMPOSE_VERBS
                spinner.update("  [bold cyan]Breaking task into steps...[/bold cyan]")
            elif phase == "validate":
                current_phase["verbs"] = _VALIDATE_VERBS
                spinner.update("  [bold cyan]Finalizing the plan...[/bold cyan]")
            elif phase == "done":
                stop_event.set()
                spinner.update(f"  [bold green]Done[/bold green] [dim]{detail}[/dim]")

        try:
            plan = interpret_task(
                task=task,
                model=model,
                file_context=file_context,
                model_map=model_map,
                on_status=_update,
                prior_classification=prior_classification,
            )
        except KeyboardInterrupt:
            stop_event.set()
            rotator.join(timeout=1)
            console.print("\n  [warn]Cancelled.[/warn]")
            return None
        stop_event.set()
        rotator.join(timeout=1)

    return plan


def _prompt_clarification(
    original_task: str,
    questions: list[ClarificationQuestion],
) -> str | None:
    """Prompt the user to answer clarification questions.

    For each question, the user can select a numbered option or type
    a custom answer. Returns enriched task string, or None on cancel.
    """
    console.print("  [dim]Select an option number or type your own answer. "
                  "Press Enter or /skip to go back.[/dim]")

    answers = []
    for cq in questions:
        if len(questions) > 1:
            console.print(f"  [yellow]?[/yellow] {cq.question}")
        try:
            response = console.input("  [bold cyan]>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            return None

        response = response.strip()
        if not response or response.lower() == "/skip":
            return None

        # Check if response is a number selecting an option
        if cq.options and response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(cq.options):
                answers.append((cq.question, cq.options[idx]))
                continue

        # Otherwise treat as free-form answer
        answers.append((cq.question, response))

    # Build enriched task
    qa_block = "\n".join(f"  Q: {q}\n  A: {a}" for q, a in answers)
    return (
        f"{original_task}\n\n"
        f"Additional context:\n{qa_block}"
    )


def _interpret_with_clarification(
    task: str, model: str, file_context: list[str] | None,
    model_map: dict[str, str],
) -> tuple[TaskPlan | None, str]:
    """Run interpretation with interactive clarification loop.

    Returns (plan, final_task). plan is None if user cancelled.
    """
    plan = _run_interpret_with_spinner(
        task, model, file_context=file_context, model_map=model_map,
    )
    if plan is None:
        return None, task
    _display_plan(plan)

    clarification_round = 0
    while (
        not plan.steps
        and plan.clarification_questions
        and clarification_round < _MAX_CLARIFICATION_ROUNDS
    ):
        enriched = _prompt_clarification(task, plan.clarification_questions)
        if enriched is None:
            return None, task

        clarification_round += 1
        console.print()
        # Skip re-classification — reuse prior intent/complexity, go straight
        # to decomposition. Re-classifying the enriched task often returns
        # empty (thinking tokens consume budget on the longer prompt).
        plan = _run_interpret_with_spinner(
            enriched, model, file_context=file_context, model_map=model_map,
            prior_classification=(plan.intent, plan.complexity),
        )
        if plan is None:
            return None, task
        _display_plan(plan)
        task = enriched

    if not plan.steps and plan.clarification_questions:
        console.print(
            "  [warn]Still unclear after clarification. "
            "Please rephrase your task.[/warn]"
        )

    return plan, task


def _display_plan(plan):
    """Display a TaskPlan with a polished Rich panel layout."""
    # Header: intent + complexity + clarity
    intent_label, intent_color = _INTENT_LABELS.get(plan.intent, (plan.intent, "white"))
    comp_label, comp_color = _COMPLEXITY_LABELS.get(plan.complexity, ("?", "white"))

    clarity_pct = f"{plan.clarity:.0%}"
    if plan.clarity >= _CLARITY_GOOD:
        clarity_color = "green"
    elif plan.clarity >= _CLARITY_FAIR:
        clarity_color = "yellow"
    else:
        clarity_color = "red"

    header = Text()
    header.append(f"  {intent_label}", style=f"bold {intent_color}")
    header.append("  |  ", style="dim")
    header.append(comp_label, style=comp_color)
    header.append("  |  ", style="dim")
    header.append(f"Clarity {clarity_pct}", style=clarity_color)
    console.print(header)

    # Reasoning
    if plan.reasoning and "Fallback" not in plan.reasoning:
        console.print(f"  [dim]{plan.reasoning}[/dim]")

    # Clarification questions
    if plan.clarification_questions:
        console.print()
        console.print("  [bold yellow]Clarification needed:[/bold yellow]")
        for cq in plan.clarification_questions:
            console.print(f"    [yellow]?[/yellow] {cq.question}")
            if cq.options:
                for i, opt in enumerate(cq.options, 1):
                    console.print(f"      [cyan]{i}.[/cyan] {opt}")
        console.print()
        return

    # Steps table
    if plan.steps:
        console.print()
        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 1),
            pad_edge=False,
        )
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("Action", width=8)
        table.add_column("Description")
        table.add_column("Files", style="dim")
        table.add_column("Deps", style="dim", width=6)

        for step in plan.steps:
            color = _ACTION_COLORS.get(step.action, "white")
            action_text = Text(step.action, style=color)
            files = ", ".join(step.files) if step.files else ""
            deps = ", ".join(str(d) for d in step.depends_on) if step.depends_on else ""
            table.add_row(str(step.id), action_text, step.description, files, deps)

        console.print(Panel(table, title="[bold]Execution Plan[/bold]", border_style="dim", padding=(0, 1)))
    elif not plan.clarification_questions:
        console.print("  [dim]No steps generated (fallback plan)[/dim]")

    console.print()


def _display_agent_result(result: AgentResult) -> None:
    """Display agent execution results with a Rich panel."""
    ok = sum(1 for s in result.steps if s.status == "ok")
    failed = sum(1 for s in result.steps if s.status == "failed")
    skipped = sum(1 for s in result.steps if s.status == "skipped")

    # Build summary lines
    steps_total = len(result.steps)
    parts = []
    if ok:
        parts.append(f"[agent.success]{ok} ok[/agent.success]")
    if failed:
        parts.append(f"[agent.error]{failed} failed[/agent.error]")
    if skipped:
        parts.append(f"[dim]{skipped} skipped[/dim]")
    steps_summary = f"  Steps: {steps_total} ({', '.join(parts)})" if parts else f"  Steps: {steps_total}"

    console.print()
    console.print(steps_summary)

    if result.files_changed:
        console.print(f"  Files changed: {', '.join(result.files_changed)}")

    diff_lines = len(result.diff.strip().splitlines()) if result.diff.strip() else 0
    console.print(f"  Diff: {diff_lines} lines  |  Duration: {result.duration_s:.1f}s")

    if result.exit_code != 0:
        console.print(f"  [agent.error]Exit code: {result.exit_code}[/agent.error]")
    console.print()


def check_git_repo() -> bool:
    """Check if the current directory is inside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def model_display(mode: Mode, model_map: dict[str, str]) -> str:
    if mode == Mode.AUTO:
        return "auto-selects per task"
    return model_map.get(mode.value, "unknown")


def print_banner(mode: Mode, model_map: dict[str, str],
                 reflect_on: bool = False, reflection_mode: str = "auto"):
    model = model_display(mode, model_map)
    reflect_label = f"[bold green]{reflection_mode}[/bold green]" if reflect_on else "[bold red]off[/bold red]"
    console.print(f"\n  [banner]PROBABLYFINE[/banner] v{__version__}", highlight=False)
    console.print(f"  Mode: [{f'mode.{mode.value}'}]{mode.value.upper()}[/] ({model})    Reflection: {reflect_label}")
    console.print(f"  [info]Type a task, or /help for commands.[/info]\n")


def print_help():
    console.print("\n  [bold]Commands:[/bold]")
    console.print("  /mode              Show current mode")
    console.print("  /mode <name>       Switch mode (fast, daily, planning, auto)")
    console.print("  /add <file>        Add file(s) to context (supports globs)")
    console.print("  /drop <file>       Remove file from context")
    console.print("  /files             List files in context")
    console.print("  /clear             Clear all files from context")
    console.print("  /git               Show git status")
    console.print("  /diff              Show uncommitted changes summary")
    console.print("  /undo              Undo last git commit (soft reset)")
    console.print("  /reflect           Toggle reflection (maker-checker) on/off")
    console.print("  /reflect on|off    Force reflection on or off")
    console.print("  /reflect auto      Smart triggering (skip trivial, check complex)")
    console.print("  /reflect always    Always run checker")
    console.print("  /reflect never     Never run checker")
    console.print("  /reflect status    Show reflection config")
    console.print("  /autofiles         Toggle automatic file selection on/off")
    console.print("  /autofiles on|off  Force auto file selection on or off")
    console.print("  /resources         Show loaded models and DRM status")
    console.print("  /unload <model>    Unload a model from VRAM")
    console.print("  /drm               Toggle resource manager on/off")
    console.print("  /drm on|off        Force resource manager on or off")
    console.print("  /safemode          Toggle safe mode (restrict to fast model)")
    console.print("  /safemode on|off   Force safe mode on or off")
    console.print("  /interpret <task>  Show interpreter plan (no execution)")
    console.print("  /logs              Show recent interpreter log entries")
    console.print("  /logs clear        Clear the interpreter log")
    console.print("  /help              Show this help")
    console.print("  /quit, /exit       Exit probablyfine")
    console.print()
    console.print("  [bold]Hotkeys:[/bold]")
    console.print("  Ctrl+N             Next mode")
    console.print("  Ctrl+P             Previous mode")
    console.print()
    console.print("  [bold]Modes:[/bold]")
    for m in Mode:
        console.print(f"  {m.value:<12} {MODE_DESCRIPTIONS[m]}")
    console.print()


def get_prompt(mode: Mode, file_count: int = 0) -> str:
    files_tag = f" ({file_count} files)" if file_count > 0 else ""
    return f"[{mode.value}]{files_tag} probablyfine> "


def handle_mode_command(args: str, current_mode: Mode, model_map: dict[str, str]) -> Mode:
    """Handle /mode command. Returns the (possibly new) mode."""
    if not args:
        model = model_display(current_mode, model_map)
        console.print(f"  Current mode: [{f'mode.{current_mode.value}'}]{current_mode.value.upper()}[/] ({model})")
        return current_mode

    target = args.strip().lower()
    try:
        new_mode = Mode(target)
    except ValueError:
        valid = ", ".join(m.value for m in Mode)
        console.print(f"  [err]Unknown mode: {target}[/err]. Valid modes: {valid}")
        return current_mode

    model = model_display(new_mode, model_map)
    console.print(f"  Switched to [{f'mode.{new_mode.value}'}]{new_mode.value.upper()}[/] ({model})")
    return new_mode


def resolve_model(mode: Mode, model_map: dict[str, str], task: str = "") -> str:
    """Resolve the Ollama model name for the current mode.

    In AUTO mode, classifies the task and routes to the best model.
    In DAILY mode, downgrades simple UI/CSS tasks to the fast model.
    """
    if mode == Mode.AUTO:
        classifier_model = model_map["daily"]
        classified_mode, reason = classify_task(task, classifier_model)
        model = model_map[classified_mode.value]
        style = f"mode.{classified_mode.value}"
        console.print(
            f"  [mode.auto]AUTO[/mode.auto] → [{style}]{classified_mode.value.upper()}[/{style}]"
            f" ({reason}) → {model}"
        )
        return model
    if mode == Mode.DAILY and task and is_simple_task(task):
        fast_model = model_map["fast"]
        console.print(
            f"  [mode.daily]DAILY[/mode.daily] → [mode.fast]FAST[/mode.fast]"
            f" (simple task) → {fast_model}"
        )
        return fast_model
    return model_map[mode.value]


def handle_add(args: str, ctx: FileContext):
    if not args:
        console.print("  [err]Usage: /add <file or glob>[/err]")
        return
    added = ctx.add(args.strip())
    if added:
        for f in added:
            console.print(f"  + {f}")
        console.print(f"  [info]{len(added)} file(s) added. {ctx.count} total in context.[/info]")
    else:
        console.print(f"  [warn]No files matched: {args.strip()}[/warn]")


def handle_drop(args: str, ctx: FileContext):
    if not args:
        console.print("  [err]Usage: /drop <file>[/err]")
        return
    removed = ctx.drop(args.strip())
    if removed:
        for f in removed:
            console.print(f"  - {f}")
        console.print(f"  [info]{len(removed)} file(s) removed. {ctx.count} total in context.[/info]")
    else:
        console.print(f"  [warn]No matching file in context: {args.strip()}[/warn]")


def handle_files(ctx: FileContext):
    if ctx.count == 0:
        console.print("  [info]No files in context. Use /add <file> to add files.[/info]")
        return
    console.print(f"\n  [bold]Files in context ({ctx.count}):[/bold]")
    for f in ctx.list_files():
        console.print(f"  {f}")
    console.print()


def handle_clear(ctx: FileContext):
    count = ctx.clear()
    if count:
        console.print(f"  [info]Cleared {count} file(s) from context.[/info]")
    else:
        console.print("  [info]Context was already empty.[/info]")


def handle_git():
    code, output = git_status()
    if code != 0:
        console.print(f"  [err]{output}[/err]")
    elif output:
        console.print(f"\n  {output}\n")
    else:
        console.print("  [info]Working tree clean.[/info]")


def handle_diff():
    code, output = git_diff_stat()
    if code != 0:
        console.print(f"  [err]{output}[/err]")
    elif output:
        console.print(f"\n{output}\n")
    else:
        console.print("  [info]No uncommitted changes.[/info]")


def handle_undo():
    code, output = git_undo_last_commit()
    if code != 0:
        console.print(f"  [err]{output}[/err]")
    else:
        console.print(f"  {output}")


def handle_reflect(args: str, reflect_on: bool, reflection_mode: str, checker_model: str) -> tuple[bool, str]:
    """Handle /reflect command. Returns (new_enabled, new_mode)."""
    arg = args.strip().lower()
    if arg == "on":
        save_config_value("reflection", "enabled", True)
        console.print(f"  Reflection [bold green]enabled[/bold green] (checker: {checker_model}) [info](saved)[/info]")
        return True, reflection_mode
    elif arg == "off":
        save_config_value("reflection", "enabled", False)
        console.print("  Reflection [bold red]disabled[/bold red] [info](saved)[/info]")
        return False, reflection_mode
    elif arg in ("auto", "always", "never"):
        save_config_value("reflection", "mode", arg)
        if arg == "never":
            save_config_value("reflection", "enabled", False)
            console.print("  Reflection mode: [bold red]never[/bold red] [info](saved)[/info]")
            return False, arg
        else:
            save_config_value("reflection", "enabled", True)
            console.print(f"  Reflection mode: [bold green]{arg}[/bold green] [info](saved)[/info]")
            return True, arg
    elif arg == "status":
        status = "[bold green]on[/bold green]" if reflect_on else "[bold red]off[/bold red]"
        console.print(f"  Reflection: {status}")
        console.print(f"  Mode: {reflection_mode}")
        console.print(f"  Checker model: {checker_model}")
        if reflection_mode == "auto":
            console.print("  [info]Auto: skips FAST mode, trivial diffs (<10 lines).[/info]")
            console.print("  [info]Auto: always checks PLANNING mode, large diffs (>30 lines).[/info]")
        return reflect_on, reflection_mode
    elif arg == "":
        # Toggle enabled
        new_state = not reflect_on
        save_config_value("reflection", "enabled", new_state)
        label = "[bold green]enabled[/bold green]" if new_state else "[bold red]disabled[/bold red]"
        console.print(f"  Reflection {label} [info](saved)[/info]")
        return new_state, reflection_mode
    else:
        console.print("  [err]Usage: /reflect [on|off|auto|always|never|status][/err]")
        return reflect_on, reflection_mode


def handle_autofiles(args: str, auto_select: bool) -> bool:
    """Handle /autofiles command. Returns the new auto-select state."""
    arg = args.strip().lower()
    if arg == "on":
        save_config_value("file_selection", "auto_select", True)
        console.print("  Auto file selection [bold green]enabled[/bold green] [info](saved)[/info]")
        return True
    elif arg == "off":
        save_config_value("file_selection", "auto_select", False)
        console.print("  Auto file selection [bold red]disabled[/bold red] [info](saved)[/info]")
        return False
    elif arg == "":
        new_state = not auto_select
        save_config_value("file_selection", "auto_select", new_state)
        label = "[bold green]enabled[/bold green]" if new_state else "[bold red]disabled[/bold red]"
        console.print(f"  Auto file selection {label} [info](saved)[/info]")
        return new_state
    else:
        console.print("  [err]Usage: /autofiles [on|off][/err]")
        return auto_select


def handle_resources():
    """Handle /resources command. Shows DRM status, VRAM, and loaded models."""
    drm = get_drm()
    status = drm.get_status()

    if not status.ollama_reachable:
        console.print("  [err]Ollama is not reachable.[/err]")
        return

    if not status.enabled:
        console.print("  [info]DRM is disabled. Enable with: /drm on[/info]")
        return

    # VRAM bar
    if status.vram_available:
        vram = status.vram
        bar = drm.vram.format_bar(width=20)
        if vram.free_mb < _VRAM_CRITICAL_MB:
            console.print(f"\n  [err]VRAM: {bar}[/err]")
        elif vram.free_mb < _VRAM_WARNING_MB:
            console.print(f"\n  [warn]VRAM: {bar}[/warn]")
        else:
            console.print(f"\n  VRAM: {bar}")
    else:
        console.print("\n  VRAM: [info]nvidia-smi not available[/info]")

    # Loaded models
    if status.loaded_models:
        console.print(f"\n  [bold]Loaded models ({len(status.loaded_models)}):[/bold]")
        for m in status.loaded_models:
            age = ""
            if m.last_used:
                elapsed = time.monotonic() - m.last_used
                if elapsed < 60:
                    age = f"  used {elapsed:.0f}s ago"
                else:
                    age = f"  used {elapsed / 60:.0f}m ago"
            console.print(f"    {m.name}  ({m.size_vram_gb:.1f} GB){age}")
    else:
        console.print("\n  [info]No models currently loaded.[/info]")

    console.print(f"\n  Swaps: {status.total_swaps}    Avoided: {status.swaps_avoided}")

    if status.safe_mode:
        console.print("  [warn]SAFE MODE active — restricted to fast model[/warn]")

    if status.oom_count or status.hang_count or status.recovery_count:
        console.print(
            f"  Watchdog: {status.oom_count} OOM, {status.hang_count} hangs, "
            f"{status.recovery_count} recoveries"
        )

    console.print()


def handle_unload(args: str):
    """Handle /unload command. Unloads a model from VRAM."""
    if not args:
        console.print("  [err]Usage: /unload <model>[/err]")
        return
    model = args.strip()
    drm = get_drm()
    drm.sync(force=True)
    if not drm.registry.is_loaded(model):
        console.print(f"  [warn]{model} is not currently loaded.[/warn]")
        return
    if drm.unload(model):
        console.print(f"  [info]Unloaded {model}.[/info]")
    else:
        console.print(f"  [err]Failed to unload {model}.[/err]")


def handle_drm_toggle(args: str):
    """Handle /drm command. Toggle or set DRM enabled state."""
    drm = get_drm()
    arg = args.strip().lower()
    if arg == "on":
        drm.enabled = True
        save_config_value("drm", "enabled", True)
        console.print("  DRM [bold green]enabled[/bold green] [info](saved)[/info]")
    elif arg == "off":
        drm.enabled = False
        save_config_value("drm", "enabled", False)
        console.print("  DRM [bold red]disabled[/bold red] [info](saved)[/info]")
    elif arg == "":
        new_state = not drm.enabled
        drm.enabled = new_state
        save_config_value("drm", "enabled", new_state)
        label = "[bold green]enabled[/bold green]" if new_state else "[bold red]disabled[/bold red]"
        console.print(f"  DRM {label} [info](saved)[/info]")
    else:
        console.print("  [err]Usage: /drm [on|off][/err]")


def handle_safemode(args: str):
    """Handle /safemode command. Toggle or set safe mode."""
    drm = get_drm()
    arg = args.strip().lower()
    if arg == "on":
        drm.safe_mode = True
        console.print("  Safe mode [bold yellow]activated[/bold yellow] — only fast model will be used")
    elif arg == "off":
        drm.safe_mode = False
        console.print("  Safe mode [bold green]deactivated[/bold green] — all models available")
    elif arg == "":
        new_state = not drm.safe_mode
        drm.safe_mode = new_state
        if new_state:
            console.print("  Safe mode [bold yellow]activated[/bold yellow] — only fast model will be used")
        else:
            console.print("  Safe mode [bold green]deactivated[/bold green] — all models available")
    else:
        console.print("  [err]Usage: /safemode [on|off][/err]")


def handle_interpret(args: str, model_map: dict, ctx: FileContext):
    """Handle /interpret command. Shows interpreter plan without execution."""
    if not args:
        console.print("  [err]Usage: /interpret <task>[/err]")
        return
    _interpret_with_clarification(
        args, model_map["daily"],
        file_context=ctx.files if ctx.count > 0 else None,
        model_map=model_map,
    )


def handle_logs(args: str):
    """Handle /logs command. Shows or clears interpreter log."""
    log_path = get_log_path()
    if args.strip().lower() == "clear":
        if log_path.exists():
            log_path.write_text("")
            console.print("  [info]Interpreter log cleared.[/info]")
        else:
            console.print("  [info]No log file found.[/info]")
        return

    if not log_path.exists():
        console.print("  [info]No log file found. Run a task first.[/info]")
        return

    lines = log_path.read_text(encoding="utf-8").splitlines()
    tail = lines[-_LOG_TAIL_LINES:] if len(lines) > _LOG_TAIL_LINES else lines
    console.print(f"\n  [bold]Interpreter Log[/bold] ({log_path})")
    console.print(f"  [dim]Showing last {len(tail)} of {len(lines)} lines[/dim]\n")
    for line in tail:
        if " WARNING " in line:
            console.print(f"  [yellow]{line}[/yellow]")
        elif " ERROR " in line:
            console.print(f"  [red]{line}[/red]")
        elif "INTERPRET:" in line or "RESULT:" in line or "====" in line:
            console.print(f"  [bold]{line}[/bold]")
        elif "FALLBACK:" in line:
            console.print(f"  [bold red]{line}[/bold red]")
        else:
            console.print(f"  [dim]{line}[/dim]")
    console.print()


def _read_input_simple(mode: Mode, file_count: int) -> str:
    """Fallback input using rich console (no hotkeys)."""
    prompt = get_prompt(mode, file_count)
    return console.input(prompt)


def _retry_with_fallback(
    task: str,
    model: str,
    files: list[str] | None,
    ss: SessionState,
    plan: TaskPlan | None = None,
) -> None:
    """Retry a failed task with the DRM fallback model."""
    fallback = ss.drm.get_fallback_model(model)
    if not fallback or fallback == model:
        return
    console.print(f"  [warn]Retrying with fallback model: {fallback}[/warn]")
    ss.drm.emergency_unload_all()
    ss.drm.ensure_loaded(fallback)

    agent_cfg = _build_agent_config(ss)
    agent_cfg.num_ctx = get_context_size(fallback)
    task_plan = plan or _fallback_plan(task, files)
    if ss.reflect_on:
        rctx = ReflectionContext(
            task=task, maker_model=fallback, checker_model=fallback,
            files=files, current_mode=ss.current_mode,
            reflection_mode=ss.reflection_mode, agent_config=agent_cfg,
            plan=task_plan,
        )
        exit_code = run_with_reflection(rctx)
    else:
        result = execute_plan(task_plan, fallback, files, agent_cfg)
        exit_code = result.exit_code

    ss.drm.task_completed(fallback)
    if exit_code != 0 and exit_code != EXIT_SIGINT:
        console.print(f"  [err]Fallback also failed (code {exit_code})[/err]")


def _build_agent_config(ss: SessionState) -> AgentConfig:
    """Build AgentConfig from session state."""
    return AgentConfig(
        conservative=get_agent_conservative(ss.config),
        auto_checkpoint=get_agent_auto_checkpoint(ss.config),
        lint_command=get_agent_lint_command(ss.config),
        test_command=get_agent_test_command(ss.config),
        dark_mode=get_dark_mode(ss.config),
    )


def _select_files_for_task(
    task: str, ss: SessionState, manual_files: list[str] | None,
) -> list[str] | None:
    """Select files for a task, using LLM auto-selection if enabled."""
    if not ss.auto_select:
        return manual_files
    console.print("  [info]Selecting files...[/info]")
    from probablyfine.config import get_max_context_bytes
    files = select_files(
        task, ss.model_map["fast"],
        existing_files=manual_files,
        max_git_files=ss.max_file_select,
        max_context_bytes=get_max_context_bytes(ss.config),
    )
    auto_count = len(files or []) - len(manual_files or [])
    if auto_count > 0:
        console.print(f"  [info]Auto-selected {auto_count} file(s)[/info]")
    return files


def _execute_task(task: str, ss: SessionState) -> None:
    """Interpret task, resolve model, select files, and execute.

    Handles optional reflection, DRM coordination, and fallback retry.
    """
    plan = None  # set by interpreter if enabled

    # File selection FIRST — interpreter needs file context for good plans
    manual_files = ss.ctx.files if ss.ctx.count > 0 else None
    files = _select_files_for_task(task, ss, manual_files)

    # Interpreter phase: classify intent, assess clarity, decompose
    if get_interpreter_enabled(ss.config):
        plan, task = _interpret_with_clarification(
            task, ss.model_map["daily"],
            file_context=files,
            model_map=ss.model_map,
        )
        if plan is None:
            return
        if not plan.steps:
            return

    # Resolve model
    model = resolve_model(ss.current_mode, ss.model_map, task=task)

    # Safe mode override: force fast model
    original_model = model
    model = ss.drm.resolve_model_for_safe_mode(model)
    if model != original_model:
        console.print(f"  [warn]Safe mode: using {model} instead of {original_model}[/warn]")

    file_info = f" with {len(files)} file(s)" if files else ""

    # DRM: prepare models for this task
    # In auto reflection mode, defer checker loading — it may not be needed
    defer_checker = ss.reflect_on and ss.reflection_mode == "auto"
    swap_plan = ss.drm.prepare_for_task(
        maker_model=model,
        checker_model=ss.checker_model if (ss.reflect_on and not defer_checker) else None,
        reflection_on=ss.reflect_on and not defer_checker,
    )
    if swap_plan.same_model_opt:
        console.print("  [info]Same model for maker+checker (no swap needed)[/info]")
    elif swap_plan.needs_swap:
        console.print(
            f"  [info]Swapping models (~{swap_plan.estimated_swap_time_s:.0f}s): "
            f"unload {len(swap_plan.models_to_unload)}, load {len(swap_plan.models_to_load)}[/info]"
        )

    vram_warning = ss.drm.get_vram_warning()
    if vram_warning:
        console.print(f"  [warn]{vram_warning}[/warn]")

    # --- Execution path ---
    agent_cfg = _build_agent_config(ss)
    agent_cfg.num_ctx = get_context_size(model)

    # Use interpreter plan if available, otherwise fallback
    task_plan = plan if (get_interpreter_enabled(ss.config) and plan and plan.steps) else _fallback_plan(task, files)

    if ss.reflect_on:
        mode_label = f"reflection {ss.reflection_mode}" if ss.reflection_mode != "always" else "reflection on"
        console.print(f"  [info]Agent → {model}{file_info} ({mode_label})...[/info]")
        rctx = ReflectionContext(
            task=task, maker_model=model, checker_model=ss.checker_model,
            files=files, current_mode=ss.current_mode,
            reflection_mode=ss.reflection_mode, agent_config=agent_cfg,
            plan=task_plan,
        )
        exit_code = run_with_reflection(rctx)
    else:
        console.print(f"  [info]Agent → {model}{file_info}...[/info]")
        result = execute_plan(task_plan, model, files, agent_cfg)
        exit_code = result.exit_code
        _display_agent_result(result)

    # DRM: post-task bookkeeping
    ss.drm.task_completed(model)

    if exit_code != 0 and exit_code != EXIT_SIGINT:
        console.print(f"  [err]Task failed (code {exit_code})[/err]")
        _retry_with_fallback(task, model, files, ss, plan=task_plan)


def _setup_tui(ss: SessionState) -> tuple[bool, object, object]:
    """Initialize the TUI session. Returns (use_tui, tui_session, state)."""
    try:
        from probablyfine.tui import AppState, create_session

        def _vram_info():
            if not ss.drm.enabled or not ss.drm.vram.available:
                return None
            snap = ss.drm.vram.get_snapshot()
            if snap.total_mb == 0:
                return None
            return (snap.used_mb, snap.total_mb)

        def _reflect_info():
            return (ss.reflect_on, ss.reflection_mode)

        def _safe_mode_info():
            return ss.drm.safe_mode

        state = AppState(
            mode=ss.current_mode,
            model_map=ss.model_map,
            file_count_fn=lambda: ss.ctx.count,
            git_branch_fn=git_branch_status,
            vram_fn=_vram_info,
            reflect_fn=_reflect_info,
            safe_mode_fn=_safe_mode_info,
        )
        tui_session = create_session(state)
        return True, tui_session, state
    except ImportError:
        console.print("  [warn]prompt_toolkit not available, using simple mode.[/warn]")
        return False, None, None
    except Exception:
        console.print("  [warn]TUI unavailable for this terminal, using simple mode.[/warn]")
        return False, None, None


def main():
    parser = argparse.ArgumentParser(description="PROBABLYFINE - AI coding agent")
    parser.add_argument("--simple", action="store_true", help="Disable TUI (no hotkeys, no toolbar)")
    args = parser.parse_args()

    config = load_config()
    model_map = get_model_map(config)
    default = get_default_mode(config)

    try:
        current_mode = Mode(default)
    except ValueError:
        current_mode = Mode.DAILY

    ctx = FileContext()

    # Initialize DRM with keep-alive config
    drm = get_drm(
        fast_keep_alive_s=get_drm_fast_keep_alive(config),
        large_keep_alive_s=get_drm_large_keep_alive(config),
    )
    drm.enabled = get_drm_enabled(config)
    drm.set_model_map(model_map)

    ss = SessionState(
        current_mode=current_mode,
        config=config,
        model_map=model_map,
        ctx=ctx,
        drm=drm,
        reflect_on=get_reflection_enabled(config),
        reflection_mode=get_reflection_mode(config),
        checker_model=get_checker_model(config, model_map),
        auto_select=get_auto_file_select(config),
        max_file_select=get_max_file_select(config),
    )

    # Set up TUI or simple mode
    if not args.simple:
        use_tui, tui_session, state = _setup_tui(ss)
    else:
        use_tui, tui_session, state = False, None, None

    if not check_git_repo():
        console.print("  [warn]Warning: Not inside a git repository. probablyfine works best in a git repo.[/warn]")

    print_banner(ss.current_mode, model_map, reflect_on=ss.reflect_on, reflection_mode=ss.reflection_mode)

    # Stateful command handlers — closures over ss and tui state
    def _cmd_mode(a):
        ss.current_mode = handle_mode_command(a, ss.current_mode, ss.model_map)
        if state is not None:
            state.mode = ss.current_mode

    def _cmd_reflect(a):
        ss.reflect_on, ss.reflection_mode = handle_reflect(
            a, ss.reflect_on, ss.reflection_mode, ss.checker_model,
        )

    def _cmd_autofiles(a):
        ss.auto_select = handle_autofiles(a, ss.auto_select)

    # Command dispatch table — all commands including stateful ones
    commands = {
        "/help": lambda a: print_help(),
        "/add": lambda a: handle_add(a, ss.ctx),
        "/drop": lambda a: handle_drop(a, ss.ctx),
        "/files": lambda a: handle_files(ss.ctx),
        "/clear": lambda a: handle_clear(ss.ctx),
        "/git": lambda a: handle_git(),
        "/diff": lambda a: handle_diff(),
        "/undo": lambda a: handle_undo(),
        "/resources": lambda a: handle_resources(),
        "/unload": lambda a: handle_unload(a),
        "/drm": lambda a: handle_drm_toggle(a),
        "/safemode": lambda a: handle_safemode(a),
        "/interpret": lambda a: handle_interpret(a, ss.model_map, ss.ctx),
        "/logs": lambda a: handle_logs(a),
        "/mode": _cmd_mode,
        "/reflect": _cmd_reflect,
        "/autofiles": _cmd_autofiles,
    }

    while True:
        # Sync mode from TUI state (hotkeys may have changed it)
        if state is not None:
            ss.current_mode = state.mode

        try:
            if use_tui and tui_session is not None:
                task = tui_session.prompt()
            else:
                task = _read_input_simple(ss.current_mode, ss.ctx.count)
        except (EOFError, KeyboardInterrupt):
            console.print("\n  Goodbye.")
            break

        task = task.strip()
        if not task:
            continue

        # Slash commands
        if task.startswith("/"):
            parts = task.split(maxsplit=1)
            cmd = parts[0].lower()
            cmd_args = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                console.print("  Goodbye.")
                break

            if cmd in commands:
                commands[cmd](cmd_args)
            else:
                console.print(f"  [err]Unknown command: {cmd}[/err]. Type /help for commands.")
            continue

        _execute_task(task, ss)


if __name__ == "__main__":
    main()
