import os
import shutil
import subprocess
import sys

OLLAMA_API_BASE = "http://localhost:11434"


def find_aider() -> str | None:
    """Return path to aider executable, or None if not found."""
    return shutil.which("aider")


def _build_env() -> dict[str, str]:
    """Build environment with OLLAMA_API_BASE set."""
    env = os.environ.copy()
    env.setdefault("OLLAMA_API_BASE", OLLAMA_API_BASE)
    return env


def run_aider(
    model: str,
    message: str,
    files: list[str] | None = None,
    auto_commit: bool = False,
    dark_mode: bool = True,
) -> int:
    """
    Run an Aider session as a subprocess.

    Returns the process exit code (0 = success).
    """
    aider_path = find_aider()
    if aider_path is None:
        print("Error: aider not found. Install it with: pip install aider-chat")
        return 1

    cmd = [
        aider_path,
        "--model", f"ollama_chat/{model}",
        "--no-show-model-warnings",
        "--message", message,
    ]

    if dark_mode:
        cmd.append("--dark-mode")

    if not auto_commit:
        cmd.append("--no-auto-commits")

    if files:
        for f in files:
            cmd.extend(["--file", f])

    try:
        result = subprocess.run(
            cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr,
            env=_build_env(),
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\nAider session interrupted.")
        return 130
    except FileNotFoundError:
        print("Error: aider executable not found.")
        return 1
    except Exception as e:
        print(f"Error running aider: {e}")
        return 1
