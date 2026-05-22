# Getting Started with PROBABLYFINE

PROBABLYFINE is a terminal AI coding assistant that routes tasks to local Ollama models through Aider. It gives you mode-based model selection, keyboard-driven switching, and git-aware code editing — all running locally.

## Prerequisites

- **Python 3.10+**
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
git clone https://github.com/YOUR_USERNAME/probably-fine.git
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

## 4. Modes

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

## 5. Basic Usage

Type a coding task and press Enter. PROBABLYFINE sends it to Aider with the active model, which edits your files directly:

```
[daily] probablyfine> add input validation to the signup form
  Sending to qwen3:8b...
```

Aider applies changes to your repo. All edits are tracked by git.

### File Context

Control which files Aider sees:

```
/add src/auth.py          Add a file to context
/add src/**/*.py          Add files by glob pattern
/drop auth                Remove a file (matches by substring)
/files                    List files in context
/clear                    Remove all files from context
```

The prompt shows your file count:

```
[daily] (3 files) probablyfine>
```

### Git Commands

```
/git                      Show git status
/diff                     Show uncommitted changes summary
/undo                     Soft-reset the last commit (keeps changes staged)
```

### Other Commands

```
/mode                     Show current mode and model
/help                     Show all commands and hotkeys
/quit                     Exit (also: /exit, Ctrl+C, Ctrl+D)
```

## 6. AUTO Mode

When set to AUTO, PROBABLYFINE classifies your task before routing it:

```
[auto] probablyfine> design a caching layer for the API
  AUTO → PLANNING (keyword match) → qwen3:8b
  Sending to qwen3:8b...
```

Classification uses a two-step approach:
1. **Keyword matching** — instant, catches obvious cases ("fix typo" → FAST, "design" → PLANNING)
2. **Model classification** — if no keyword match, asks qwen3:8b to classify (better reasoning than the fast model)
3. **Fallback** — if both fail, defaults to DAILY

You can always override by switching mode manually.

## 7. Configuration

On first run, a config file is created at `~/.probablyfine/config.toml`:

```toml
[models]
fast = "deepseek-coder:6.7b"
daily = "qwen3:8b"
planning = "qwen3:8b"

[defaults]
mode = "daily"
auto_commit = false
```

Edit this file to:
- Swap in different Ollama models
- Change the default startup mode
- Enable auto-commit (Aider commits changes automatically)

## 8. CLI Flags

```
probablyfine              Launch with TUI (toolbar, hotkeys, tab completion)
probablyfine --simple     Launch without TUI (plain input, works everywhere)
```

Use `--simple` if you have terminal compatibility issues or prefer a minimal interface.

## Troubleshooting

**pip install fails with "Could not install packages due to an OSError" (Windows)**
This happens when pip tries to write to a system-wide `Scripts` folder without permission. Use a virtual environment (see step 2 above) to avoid this entirely. If you already installed without a venv, delete the stale `probablyfine.exe` from `C:\Python312\Scripts\` and reinstall with `pip install -e . --user`, or switch to a venv.

**"aider not found"**
Aider should be installed automatically. If not: `pip install aider-chat`

**"Warning: Not inside a git repository"**
PROBABLYFINE works best inside a git repo. Run `git init` first if needed.

**Model is slow**
On low VRAM, responses may be slow. Use FAST mode for quick tasks. Both models fit comfortably in ~6GB VRAM.

**TUI not showing / falling back to simple mode**
This happens when stdin is piped or the terminal doesn't support prompt_toolkit. The `--simple` fallback works identically, just without hotkeys and toolbar.

**Model not found**
Make sure you've pulled the model: `ollama pull <model-name>`. Run `ollama list` to see what's available.
