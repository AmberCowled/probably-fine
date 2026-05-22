"""prompt_toolkit based TUI for PROBABLYFINE.

Provides:
- Keybindings: Shift+Tab (cycle mode)
- Bottom toolbar showing mode, model, and file count
- Slash command completion
- Colored prompt per mode
"""

import time

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from probablyfine import __version__
from probablyfine.modes import Mode

# VRAM bar thresholds (percentage)
_VRAM_DANGER_PCT = 95   # Red when usage >= this
_VRAM_WARN_PCT = 85     # Yellow when usage >= this

SLASH_COMMANDS = {
    "/mode": "Show or switch mode",
    "/mode fast": "Switch to fast mode",
    "/mode daily": "Switch to daily mode",
    "/mode planning": "Switch to planning mode",
    "/mode auto": "Switch to auto mode",
    "/add": "Add file(s) to context",
    "/drop": "Remove file from context",
    "/files": "List files in context",
    "/clear": "Clear all files from context",
    "/git": "Show git status",
    "/diff": "Show uncommitted changes",
    "/undo": "Undo last commit (soft reset)",
    "/reflect": "Toggle reflection on/off",
    "/reflect on": "Enable reflection",
    "/reflect off": "Disable reflection",
    "/reflect auto": "Smart triggering (skip trivial diffs)",
    "/reflect always": "Always run checker",
    "/reflect never": "Never run checker",
    "/reflect status": "Show reflection config",
    "/autofiles": "Toggle auto file selection on/off",
    "/autofiles on": "Enable auto file selection",
    "/autofiles off": "Disable auto file selection",
    "/resources": "Show loaded models and DRM status",
    "/unload": "Unload a model from VRAM",
    "/drm": "Toggle DRM on/off",
    "/drm on": "Enable DRM",
    "/drm off": "Disable DRM",
    "/safemode": "Toggle safe mode on/off",
    "/safemode on": "Activate safe mode (fast model only)",
    "/safemode off": "Deactivate safe mode",
    "/help": "Show all commands",
    "/quit": "Exit probablyfine",
    "/exit": "Exit probablyfine",
}


class SlashCompleter(Completer):
    """Dropdown completer that only activates when input starts with '/'."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return
        for cmd, desc in SLASH_COMMANDS.items():
            if cmd.startswith(text) and cmd != text:
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display_meta=desc,
                )


class SlashAutoSuggest(AutoSuggest):
    """Inline ghost-text suggestion for slash commands (accept with right arrow)."""

    def get_suggestion(self, buffer, document):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return None
        for cmd in SLASH_COMMANDS:
            if cmd.startswith(text) and cmd != text:
                return Suggestion(cmd[len(text):])
        return None

MODE_COLORS = {
    Mode.FAST: "#00cc00",      # green
    Mode.DAILY: "#4488ff",     # blue
    Mode.PLANNING: "#cc44cc",  # magenta
    Mode.AUTO: "#cccc00",      # yellow
}

MODE_ORDER = [Mode.FAST, Mode.DAILY, Mode.PLANNING, Mode.AUTO]


class AppState:
    """Shared mutable state accessible from keybinding handlers."""

    def __init__(self, mode: Mode, model_map: dict[str, str], file_count_fn,
                 git_branch_fn=None, vram_fn=None, reflect_fn=None,
                 safe_mode_fn=None):
        self.mode = mode
        self.model_map = model_map
        self._file_count_fn = file_count_fn
        self._git_branch_fn = git_branch_fn
        self._vram_fn = vram_fn  # () -> (used_mb, total_mb) | None
        self._reflect_fn = reflect_fn  # () -> (enabled: bool, mode: str)
        self._safe_mode_fn = safe_mode_fn  # () -> bool
        self._git_cache: tuple[str, bool] = ("", False)
        self._git_cache_time: float = 0.0
        self._vram_cache: tuple[int, int] | None = None
        self._vram_cache_time: float = 0.0

    _GIT_CACHE_TTL = 5.0  # seconds
    _VRAM_CACHE_TTL = 10.0  # seconds — poll less often than git

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
        now = time.monotonic()
        if now - self._git_cache_time > self._GIT_CACHE_TTL:
            self._git_cache = self._git_branch_fn()
            self._git_cache_time = now
        return self._git_cache

    @property
    def vram_info(self) -> tuple[int, int] | None:
        """Return (used_mb, total_mb) or None if unavailable."""
        if self._vram_fn is None:
            return None
        now = time.monotonic()
        if now - self._vram_cache_time > self._VRAM_CACHE_TTL:
            self._vram_cache = self._vram_fn()
            self._vram_cache_time = now
        return self._vram_cache

    @property
    def reflect_info(self) -> tuple[bool, str]:
        """Return (enabled, mode_str) or (False, 'off') if unavailable."""
        if self._reflect_fn is None:
            return (False, "off")
        return self._reflect_fn()

    @property
    def is_safe_mode(self) -> bool:
        if self._safe_mode_fn is None:
            return False
        return self._safe_mode_fn()

    def cycle_mode(self, direction: int = 1):
        idx = MODE_ORDER.index(self.mode)
        self.mode = MODE_ORDER[(idx + direction) % len(MODE_ORDER)]


def _build_keybindings(state: AppState) -> KeyBindings:
    kb = KeyBindings()

    @kb.add("s-tab")
    def next_mode(event):
        state.cycle_mode(1)

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

        # Reflection status
        reflect_enabled, reflect_mode = state.reflect_info
        if reflect_enabled:
            if reflect_mode == "always":
                sections.append(f'<style fg="#00cc00">\u2713 reflect</style>')
            else:
                sections.append(f'<style fg="#00cc00">\u2713 reflect:{reflect_mode}</style>')
        else:
            sections.append(f'<style fg="#666666">\u2717 reflect</style>')

        # Safe mode
        if state.is_safe_mode:
            sections.append(f'<style fg="#ff8800">SAFE</style>')

        vram = state.vram_info
        if vram is not None:
            used_mb, total_mb = vram
            if total_mb > 0:
                pct = int(used_mb / total_mb * 100)
                if pct >= _VRAM_DANGER_PCT:
                    sections.append(f'<style fg="#ff4444">VRAM {pct}%</style>')
                elif pct >= _VRAM_WARN_PCT:
                    sections.append(f'<style fg="#cccc00">VRAM {pct}%</style>')
                else:
                    sections.append(f'VRAM {pct}%')

        status_row = " " + " | ".join(sections)

        # Row 2: shortcuts + version
        shortcuts_row = (
            f' <style fg="#888888"><i>Shift+Tab: mode</i> | <i>/help</i> | <i>/quit</i>'
            f'   probablyfine v{__version__}</style>'
        )

        # Row 3: input prompt (cursor goes here)
        input_row = f' <style fg="{color}">&gt;</style> '

        return HTML(f"{status_row}\n{shortcuts_row}\n{input_row}")
    return prompt


def create_session(state: AppState) -> PromptSession:
    kb = _build_keybindings(state)
    prompt = _build_prompt(state)

    return PromptSession(
        message=prompt,
        key_bindings=kb,
        completer=SlashCompleter(),
        auto_suggest=SlashAutoSuggest(),
        complete_while_typing=True,
    )
