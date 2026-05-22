"""Shared Rich console for probablyfine modules."""

from rich.console import Console
from rich.theme import Theme

theme = Theme({
    # Mode styles (cli)
    "mode.fast": "bold green",
    "mode.daily": "bold blue",
    "mode.planning": "bold magenta",
    "mode.auto": "bold yellow",
    # General styles (cli)
    "banner": "bold cyan",
    "info": "dim",
    "warn": "bold yellow",
    "err": "bold red",
    # Checker styles (reflection)
    "check.pass": "bold green",
    "check.fail": "bold red",
    "check.warn": "bold yellow",
    "check.info": "dim",
    "check.critical": "bold red",
    "check.warning": "yellow",
    # Agent styles
    "agent.step": "bold cyan",
    "agent.token": "dim",
    "agent.success": "bold green",
    "agent.retry": "bold yellow",
    "agent.error": "bold red",
})

console = Console(theme=theme)

ACTION_COLORS: dict[str, str] = {
    "read": "cyan",
    "edit": "yellow",
    "create": "green",
    "delete": "red",
    "verify": "magenta",
    "explain": "blue",
}
