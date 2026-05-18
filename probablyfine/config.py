import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

CONFIG_DIR = Path.home() / ".probablyfine"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULT_CONFIG = """\
[models]
fast = "deepseek-coder:6.7b"
daily = "qwen3-coder:30b"
planning = "qwen3:32b"

[defaults]
mode = "daily"
auto_commit = false
dark_mode = true
"""


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(DEFAULT_CONFIG)

    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def get_model_map(config: dict) -> dict[str, str]:
    """Return mode-name → model-name mapping from config."""
    models = config.get("models", {})
    return {
        "fast": models.get("fast", "deepseek-coder:6.7b"),
        "daily": models.get("daily", "qwen3-coder:30b"),
        "planning": models.get("planning", "qwen3:32b"),
    }


def get_default_mode(config: dict) -> str:
    return config.get("defaults", {}).get("mode", "daily")


def get_dark_mode(config: dict) -> bool:
    return config.get("defaults", {}).get("dark_mode", True)
