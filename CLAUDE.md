# PROBABLYFINE

Local AI coding agent wrapping Ollama models. Replacing Aider with custom agent logic.

## Architecture

~5,100 lines across 22 Python files. Core flow:

```
cli.py (REPL + commands)
  -> interpreter.py (classify intent, decompose into steps)
  -> aider_session.py (execute via Aider subprocess — being replaced by custom agent.py)
  -> reflection.py (maker-checker review loop)
     -> checker.py (LLM code review with streaming)
```

Supporting modules:
- `config.py` — TOML config from `~/.probablyfine/config.toml`
- `models.py` — All dataclasses (Issue, CheckerResult, TaskPlan, etc.)
- `context.py` — FileContext tracking files in session
- `router.py` — Task routing and mode classification
- `modes.py` — Mode definitions (FAST / DAILY / PLANNING / AUTO)
- `tui.py` — Terminal UI with prompt_toolkit, hotkeys, status bar
- `file_selector.py` — LLM-powered file selection for tasks
- `drm/` — Dynamic Resource Manager (6 files): VRAM monitoring, model lifecycle, swap scheduling, health watchdog

## Models

- `qwen3:8b` — Daily + planning model (hybrid thinking/fast mode via `/no_think`)
- `deepseek-coder:6.7b` — Fast model for simple tasks
- Single 8GB GPU — only one large model loaded at a time (DRM manages swaps)

## Conventions

- Python 3.12, dataclasses for structured data, TOML config
- New data structures go in `models.py`
- Logging to `~/.probablyfine/` (one log per module: `checker.log`, `interpreter.log`, etc.)
- Rich library for terminal UI (Panel, Table, Status spinners, color themes)
- prompt_toolkit for TUI input with keybindings
- Every LLM call has: explicit timeout, num_predict limit, fallback on failure
- Graceful degradation: never crash, never raise to caller — always return a safe default
- Ollama responses: handle both dict and object API formats via `_extract_content()` pattern
- qwen3:8b prompts use `/no_think` suffix when structured output is needed (thinking tokens consume token budget)
- JSON parsing: strip markdown fences -> json.loads -> regex fallback -> truncated JSON repair

## Config Pattern

Add new options to `DEFAULT_CONFIG` string in `config.py`, then add a `get_*()` accessor function.
All getters follow: `config.get("section", {}).get("key", default)`.

## Slash Commands

Defined in two places that must stay in sync:
- `tui.py` — `SLASH_COMMANDS` dict (autocomplete definitions)
- `cli.py` — Command handler chain (if/elif dispatch)

## Known Debt

Run `/refactor` for a full analysis. Key items:
- `cli.py` main() is a 300+ line monolith — extract command handlers to a registry
- Logger factory, `_extract_content()`, and JSON parsing duplicated across 3-4 modules
- Long parameter lists on `run_aider()` / `run_with_reflection()` — should be dataclasses
