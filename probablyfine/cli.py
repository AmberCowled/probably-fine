import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.theme import Theme

from probablyfine import __version__
from probablyfine.aider_session import run_aider
from probablyfine.config import get_default_mode, get_model_map, load_config
from probablyfine.context import FileContext
from probablyfine.git_utils import git_diff_stat, git_status, git_undo_last_commit
from probablyfine.modes import MODE_DESCRIPTIONS, Mode

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


def print_banner(mode: Mode, model_map: dict[str, str]):
    model = model_map.get(mode.value, "unknown")
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
        model = model_map.get(current_mode.value, "unknown")
        console.print(f"  Current mode: [{f'mode.{current_mode.value}'}]{current_mode.value.upper()}[/] ({model})")
        return current_mode

    target = args.strip().lower()
    try:
        new_mode = Mode(target)
    except ValueError:
        valid = ", ".join(m.value for m in Mode)
        console.print(f"  [err]Unknown mode: {target}[/err]. Valid modes: {valid}")
        return current_mode

    model = model_map.get(new_mode.value, "n/a (auto-selects)")
    console.print(f"  Switched to [{f'mode.{new_mode.value}'}]{new_mode.value.upper()}[/] ({model})")
    return new_mode


def resolve_model(mode: Mode, model_map: dict[str, str]) -> str:
    """Resolve the Ollama model name for the current mode.

    In AUTO mode, falls back to DAILY for now (Phase 3 adds classifier).
    """
    if mode == Mode.AUTO:
        console.print("  [mode.auto]AUTO → routing to DAILY (classifier not yet built)[/mode.auto]")
        return model_map["daily"]
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


def main():
    config = load_config()
    model_map = get_model_map(config)
    default = get_default_mode(config)

    try:
        current_mode = Mode(default)
    except ValueError:
        current_mode = Mode.DAILY

    ctx = FileContext()

    if not check_git_repo():
        console.print("  [warn]Warning: Not inside a git repository. Aider works best in a git repo.[/warn]")

    print_banner(current_mode, model_map)

    while True:
        try:
            prompt = get_prompt(current_mode, ctx.count)
            task = console.input(prompt)
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
            args = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                console.print("  Goodbye.")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/mode":
                current_mode = handle_mode_command(args, current_mode, model_map)
            elif cmd == "/add":
                handle_add(args, ctx)
            elif cmd == "/drop":
                handle_drop(args, ctx)
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
        model = resolve_model(current_mode, model_map)
        files = ctx.files if ctx.count > 0 else None
        file_info = f" with {ctx.count} file(s)" if files else ""
        console.print(f"  [info]Sending to {model}{file_info}...[/info]")
        exit_code = run_aider(model=model, message=task, files=files)

        if exit_code != 0 and exit_code != 130:
            console.print(f"  [err]Aider exited with code {exit_code}[/err]")


if __name__ == "__main__":
    main()
