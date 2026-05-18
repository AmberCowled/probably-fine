import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.theme import Theme

from probablyfine import __version__
from probablyfine.aider_session import run_aider
from probablyfine.config import get_dark_mode, get_default_mode, get_model_map, load_config
from probablyfine.context import FileContext
from probablyfine.git_utils import git_branch_status, git_diff_stat, git_status, git_undo_last_commit
from probablyfine.modes import MODE_DESCRIPTIONS, Mode
from probablyfine.router import classify_task

theme = Theme({
    "mode.fast": "bold green",
    "mode.daily": "bold blue",
    "mode.planning": "bold magenta",
    "mode.auto": "bold yellow",
    "banner": "bold cyan",
    "info": "dim",
    "warn": "bold yellow",
    "err": "bold red",
})
console = Console(theme=theme)


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


def print_banner(mode: Mode, model_map: dict[str, str]):
    model = model_display(mode, model_map)
    console.print(f"\n  [banner]PROBABLYFINE[/banner] v{__version__}", highlight=False)
    console.print(f"  Mode: [{f'mode.{mode.value}'}]{mode.value.upper()}[/] ({model})")
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
    """
    if mode == Mode.AUTO:
        fast_model = model_map["fast"]
        classified_mode, reason = classify_task(task, fast_model)
        model = model_map[classified_mode.value]
        style = f"mode.{classified_mode.value}"
        console.print(
            f"  [mode.auto]AUTO[/mode.auto] → [{style}]{classified_mode.value.upper()}[/{style}]"
            f" ({reason}) → {model}"
        )
        return model
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


def _read_input_simple(mode: Mode, file_count: int) -> str:
    """Fallback input using rich console (no hotkeys)."""
    prompt = get_prompt(mode, file_count)
    return console.input(prompt)


def _make_tui_session(state):
    """Create prompt_toolkit session. Returns (session, state) or None on import failure."""
    try:
        from probablyfine.tui import AppState, create_session
        return create_session(state)
    except ImportError:
        return None


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

    # Set up TUI or simple mode
    use_tui = not args.simple
    tui_session = None
    state = None

    if use_tui:
        try:
            from probablyfine.tui import AppState, create_session
            state = AppState(
                mode=current_mode,
                model_map=model_map,
                file_count_fn=lambda: ctx.count,
                git_branch_fn=git_branch_status,
            )
            tui_session = create_session(state)
        except ImportError:
            console.print("  [warn]prompt_toolkit not available, using simple mode.[/warn]")
            use_tui = False
        except Exception:
            # prompt_toolkit fails on non-TTY (piped input, some Windows terminals)
            console.print("  [warn]TUI unavailable for this terminal, using simple mode.[/warn]")
            use_tui = False

    if not check_git_repo():
        console.print("  [warn]Warning: Not inside a git repository. Aider works best in a git repo.[/warn]")

    print_banner(current_mode, model_map)

    while True:
        # Sync mode from TUI state (hotkeys may have changed it)
        if state is not None:
            current_mode = state.mode

        try:
            if use_tui and tui_session is not None:
                task = tui_session.prompt()
            else:
                task = _read_input_simple(current_mode, ctx.count)
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
            elif cmd == "/help":
                print_help()
            elif cmd == "/mode":
                current_mode = handle_mode_command(cmd_args, current_mode, model_map)
                if state is not None:
                    state.mode = current_mode
            elif cmd == "/add":
                handle_add(cmd_args, ctx)
            elif cmd == "/drop":
                handle_drop(cmd_args, ctx)
            elif cmd == "/files":
                handle_files(ctx)
            elif cmd == "/clear":
                handle_clear(ctx)
            elif cmd == "/git":
                handle_git()
            elif cmd == "/diff":
                handle_diff()
            elif cmd == "/undo":
                handle_undo()
            else:
                console.print(f"  [err]Unknown command: {cmd}[/err]. Type /help for commands.")
            continue

        # Run task through Aider
        model = resolve_model(current_mode, model_map, task=task)
        files = ctx.files if ctx.count > 0 else None
        file_info = f" with {ctx.count} file(s)" if files else ""
        console.print(f"  [info]Sending to {model}{file_info}...[/info]")
        dark_mode = get_dark_mode(config)
        exit_code = run_aider(model=model, message=task, files=files, dark_mode=dark_mode)

        if exit_code != 0 and exit_code != 130:
            console.print(f"  [err]Aider exited with code {exit_code}[/err]")


if __name__ == "__main__":
    main()
