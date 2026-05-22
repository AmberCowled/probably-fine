# PROBABLYFINE

A terminal AI coding assistant that runs entirely on local Ollama models. Routes tasks through a custom agent with streaming edits, a maker-checker reflection loop, and dynamic VRAM management — all on a single 8GB GPU.

## Prerequisites

- **Python 3.12+**
- **Git** installed and available on PATH
- **Ollama** installed and running ([ollama.com](https://ollama.com))
- **8GB+ VRAM GPU** recommended (CPU inference works but is slow)

## 1. Install Ollama Models

Pull the models PROBABLYFINE uses:

```bash
# Fast model — quick code edits, small fixes, execution tasks
ollama pull deepseek-coder:6.7b

# Daily + Planning model — coding, reasoning, architecture, task classification
ollama pull qwen3:8b
```

Verify Ollama is running:

```bash
ollama list
```

## 2. Install PROBABLYFINE

### Recommended: pipx (global command, isolated environment)

`pipx` installs PROBABLYFINE in its own virtual environment and makes the `probablyfine` command available globally — no activation needed, works from any directory in any terminal session.

Install pipx if you don't have it:

```bash
pip install pipx
pipx ensurepath
```

Then install PROBABLYFINE:

```bash
git clone https://github.com/AmberCowled/probably-fine.git
cd probably-fine
pipx install -e .
```

That's it. The `probablyfine` command is now available everywhere.

To update after pulling new changes:

```bash
pipx install -e . --force
```

### Alternative: venv (manual activation)

If you prefer a standard virtual environment:

```bash
git clone https://github.com/AmberCowled/probably-fine.git
cd probably-fine
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\activate

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Then install:

```bash
pip install -e .
```

> **Note:** With the venv method, you need to activate the venv each time you open a new terminal. The `probablyfine` command only exists inside the venv.

## 3. Run It

Navigate to any git repository and launch:

```bash
cd your-project
probablyfine
```

Or run as a Python module:

```bash
python -m probablyfine
```

You'll see:

```
  PROBABLYFINE v0.1.0
  Mode: DAILY (qwen3:8b)
  Type a task, or /help for commands.

[daily] probablyfine>
```

## 4. How It Works

Type a coding task and press Enter. PROBABLYFINE interprets your request, selects relevant files, streams edits from the model, and applies them directly to your codebase:

```
[daily] probablyfine> add input validation to the signup form
  Interpreting task...
  Intent: feature | Complexity: 2 | Clarity: 95%
  Selected 3 files automatically
  Step 1/2: [edit] Add validation logic to signup handler
    ✓ Applied 4 edits to src/auth.py
  Step 2/2: [verify] Run linter
    OK
```

All edits are tracked by git with automatic checkpointing. Use `/undo` to roll back.

### The Agent Pipeline

1. **Interpreter** — classifies your task (bug fix, feature, refactor, question), assesses complexity, and decomposes it into ordered steps
2. **File Selector** — uses keyword matching + LLM to pick which files the agent needs
3. **Agent** — streams Ollama responses, parses SEARCH/REPLACE edit blocks, applies them atomically with 3-tier error recovery
4. **Reflection** (optional) — a maker-checker loop where a second model reviews the diff and requests fixes if needed

### 3-Tier Error Recovery

When an edit fails to apply, the agent tries progressively harder recovery:

1. **Tier 1** — exact SEARCH/REPLACE match
2. **Tier 2** — retry with error context (tells the model what went wrong, asks for corrected edit)
3. **Tier 3** — whole-file fallback (sends the entire file, asks for complete replacement)

## 5. Modes

PROBABLYFINE has four modes, each mapped to a different model:

| Mode | Model | Best for |
|------|-------|----------|
| **FAST** | deepseek-coder:6.7b | Quick code edits, small fixes, execution tasks |
| **DAILY** | qwen3:8b | Implementation, features, refactoring, bug fixes |
| **PLANNING** | qwen3:8b | Architecture, design, reasoning, trade-offs |
| **AUTO** | (selects per task) | Classifies your task and picks the best model |

Switch modes with:

```
/mode fast
/mode daily
/mode planning
/mode auto
```

Or use hotkeys (when the TUI toolbar is active):

- **Ctrl+N** — next mode
- **Ctrl+P** — previous mode

## 6. Reflection (Maker-Checker)

When enabled, PROBABLYFINE runs a review loop after the agent makes changes:

1. The **maker** model generates edits
2. The **checker** model reviews the diff for bugs, logic errors, and style issues
3. If the checker finds problems, the maker gets another pass to fix them
4. Repeats up to 2 iterations or until the checker passes

Toggle reflection:

```
/reflect on        Always reflect
/reflect off       Never reflect
/reflect auto      Reflect based on diff size and task complexity (default)
```

Auto mode triggers reflection when:
- The diff is large (>30 lines changed)
- The deletion ratio is suspicious (>3:1 deletions to additions)
- The task involves complex keywords (refactor, security, auth, etc.)

## 7. Commands

### File Context

```
/add src/auth.py          Add a file to context
/add src/**/*.py          Add files by glob pattern
/drop auth                Remove a file (matches by substring)
/files                    List files in context
/clear                    Remove all files from context
```

### Git

```
/git                      Show git status
/diff                     Show uncommitted changes summary
/undo                     Soft-reset the last commit (keeps changes staged)
```

### DRM (Dynamic Resource Manager)

```
/drm                      Toggle DRM on/off
/drm on                   Enable DRM
/drm off                  Disable DRM
/drm status               Show VRAM usage, loaded models, swap stats
```

The DRM monitors GPU VRAM, manages model loading/unloading, detects OOM errors, and automatically falls back to smaller models when needed.

### Other

```
/mode                     Show current mode and model
/reflect                  Toggle reflection mode
/help                     Show all commands and hotkeys
/quit                     Exit (also: /exit, Ctrl+C, Ctrl+D)
```

## 8. Configuration

On first run, a config file is created at `~/.probablyfine/config.toml`:

```toml
[models]
fast = "deepseek-coder:6.7b"
daily = "qwen3:8b"
planning = "qwen3:8b"

[defaults]
mode = "daily"
dark_mode = true

[reflection]
enabled = true
mode = "auto"

[drm]
enabled = true
```

Edit this file to:
- Swap in different Ollama models
- Change the default startup mode
- Configure reflection behavior
- Set DRM preferences

## 9. CLI Flags

```
probablyfine              Launch with TUI (toolbar, hotkeys, tab completion)
probablyfine --simple     Launch without TUI (plain input, works everywhere)
```

Use `--simple` if you have terminal compatibility issues or prefer a minimal interface.

## Architecture

```
cli.py (REPL + commands)
  → interpreter.py (classify intent, decompose into steps)
  → agent.py (stream Ollama, parse SEARCH/REPLACE, apply edits)
  → reflection.py (maker-checker review loop)
     → checker.py (LLM code review with streaming)
```

Supporting modules:
- `config.py` — TOML config from `~/.probablyfine/config.toml`
- `models.py` — 15 dataclasses (Issue, CheckerResult, TaskPlan, StepContext, etc.)
- `context.py` — FileContext tracking files in session
- `router.py` — Task routing and mode classification
- `modes.py` — Mode definitions (FAST / DAILY / PLANNING / AUTO)
- `tui.py` — Terminal UI with prompt_toolkit, hotkeys, status bar
- `file_selector.py` — LLM-powered file selection for tasks
- `edit_parser.py` — SEARCH/REPLACE block parser with atomic apply
- `ollama_utils.py` — Shared Ollama client, streaming, JSON parsing
- `console.py` — Rich theme and console instance
- `log_utils.py` — Logger factory
- `drm/` — Dynamic Resource Manager (VRAM monitoring, model lifecycle, swap scheduling, health watchdog)

## Troubleshooting

**"Warning: Not inside a git repository"**
PROBABLYFINE works best inside a git repo. Run `git init` first if needed.

**Model is slow**
On low VRAM, responses may be slow. Use FAST mode for quick tasks. Check `/drm status` for VRAM pressure.

**TUI not showing / falling back to simple mode**
This happens when stdin is piped or the terminal doesn't support prompt_toolkit. The `--simple` fallback works identically, just without hotkeys and toolbar.

**Model not found**
Make sure you've pulled the model: `ollama pull <model-name>`. Run `ollama list` to see what's available.

**OOM / model stalls**
The DRM detects OOM errors and automatically falls back to the fast model. If issues persist, try `/drm status` to check VRAM, or manually switch to FAST mode.
