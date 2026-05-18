"""prompt_toolkit based TUI for PROBABLYFINE.

Provides:
- Keybindings: Ctrl+N (next mode), Ctrl+P (previous mode)
- Bottom toolbar showing mode, model, and file count
- Slash command completion
- Colored prompt per mode
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from probablyfine import __version__
from probablyfine.modes import Mode

SLASH_COMMANDS = [
    "/mode", "/mode fast", "/mode daily", "/mode planning", "/mode auto",
    "/add", "/drop", "/files", "/clear",
    "/git", "/diff", "/undo",
    "/help", "/quit", "/exit",
]

MODE_COLORS = {
    Mode.FAST: "#00cc00",      # green
    Mode.DAILY: "#4488ff",     # blue
    Mode.PLANNING: "#cc44cc",  # magenta
    Mode.AUTO: "#cccc00",      # yellow
}

MODE_ORDER = [Mode.FAST, Mode.DAILY, Mode.PLANNING, Mode.AUTO]


class AppState:
    """Shared mutable state accessible from keybinding handlers."""

    def __init__(self, mode: Mode, model_map: dict[str, str], file_count_fn, git_branch_fn=None):
        self.mode = mode
        self.model_map = model_map
        self._file_count_fn = file_count_fn
        self._git_branch_fn = git_branch_fn

    @property
    def file_count(self) -> int:
        return self._file_count_fn()

    @property
    def model_name(self) -> str:
        if self.mode == Mode.AUTO:
            return "auto-selects"
        return self.model_map.get(self.mode.value, "?")

    @property
    def git_info(self) -> tuple[str, bool]:
        if self._git_branch_fn is None:
            return ("", False)
        return self._git_branch_fn()

    def cycle_mode(self, direction: int = 1):
        idx = MODE_ORDER.index(self.mode)
        self.mode = MODE_ORDER[(idx + direction) % len(MODE_ORDER)]


def _build_keybindings(state: AppState) -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-n")
    def next_mode(event):
        state.cycle_mode(1)

    @kb.add("c-p")
    def prev_mode(event):
        state.cycle_mode(-1)

    return kb


def _build_prompt(state: AppState):
    """Build a multi-line prompt: status rows + input line at the bottom."""
    def prompt():
        color = MODE_COLORS.get(state.mode, "white")
        mode_text = state.mode.value.upper()

        # Row 1: mode | model | git branch | file count
        sections = [
            f'<b><style fg="{color}"> {mode_text} </style></b>',
            state.model_name,
        ]

        branch, dirty = state.git_info
        if branch:
            dirty_marker = " *" if dirty else ""
            sections.append(f'\u2387 {branch}{dirty_marker}')

        if state.file_count > 0:
            sections.append(f"{state.file_count} file(s)")

        status_row = " " + " | ".join(sections)

        # Row 2: shortcuts + version
        shortcuts_row = (
            f' <style fg="#888888"><i>Ctrl+N/P: mode</i> | <i>/help</i> | <i>/quit</i>'
            f'   probablyfine v{__version__}</style>'
        )

        # Row 3: input prompt (cursor goes here)
        input_row = f' <style fg="{color}">&gt;</style> '

        return HTML(f"{status_row}\n{shortcuts_row}\n{input_row}")
    return prompt


def create_session(state: AppState) -> PromptSession:
    kb = _build_keybindings(state)
    prompt = _build_prompt(state)
    completer = WordCompleter(SLASH_COMMANDS, sentence=True)

    return PromptSession(
        message=prompt,
        key_bindings=kb,
        completer=completer,
        complete_while_typing=False,
    )
