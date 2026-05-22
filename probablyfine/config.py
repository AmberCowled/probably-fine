import re
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
daily = "qwen3:8b"
planning = "qwen3:8b"

[defaults]
mode = "daily"
auto_commit = false
dark_mode = true

[reflection]
enabled = true
mode = "auto"
checker_model = "daily"

[file_selection]
auto_select = true
max_files = 500
max_context_bytes = 48000

[drm]
enabled = true
fast_keep_alive = "10m"
large_keep_alive = "5m"

[interpreter]
enabled = true
clarity_threshold = 0.7

[agent]
conservative = false
auto_checkpoint = true
lint_command = ""
test_command = ""
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
        "daily": models.get("daily", "qwen3:8b"),
        "planning": models.get("planning", "qwen3:8b"),
    }


# Context window sizes (num_ctx) per model.
# 16k fits comfortably in 8GB VRAM alongside model weights + system usage.
MODEL_CONTEXT_SIZES: dict[str, int] = {
    "qwen3:8b": 16384,
    "deepseek-coder:6.7b": 16384,
}

_DEFAULT_CONTEXT_SIZE = 8192


def get_context_size(model: str) -> int:
    """Return the num_ctx context window size for a model."""
    return MODEL_CONTEXT_SIZES.get(model, _DEFAULT_CONTEXT_SIZE)


def get_default_mode(config: dict) -> str:
    return config.get("defaults", {}).get("mode", "daily")


def get_dark_mode(config: dict) -> bool:
    return config.get("defaults", {}).get("dark_mode", True)


def get_auto_commit(config: dict) -> bool:
    return config.get("defaults", {}).get("auto_commit", False)


def get_reflection_enabled(config: dict) -> bool:
    return config.get("reflection", {}).get("enabled", True)


def get_reflection_mode(config: dict) -> str:
    """Return reflection trigger mode: 'auto', 'always', or 'never'."""
    mode = config.get("reflection", {}).get("mode", "auto")
    if mode in ("auto", "always", "never"):
        return mode
    return "auto"


def get_checker_model(config: dict, model_map: dict[str, str]) -> str:
    """Return checker model name.

    Supports mode references ('fast', 'daily', 'planning') which resolve
    to the corresponding model from the model map. Defaults to the fast
    model for best performance on constrained hardware.
    """
    value = config.get("reflection", {}).get("checker_model", "daily")
    if not value:
        value = "daily"
    # Resolve mode references to actual model names
    if value in model_map:
        return model_map[value]
    return value


def get_auto_file_select(config: dict) -> bool:
    return config.get("file_selection", {}).get("auto_select", True)


def get_max_file_select(config: dict) -> int:
    return config.get("file_selection", {}).get("max_files", 500)


def get_max_context_bytes(config: dict) -> int:
    """Return max total bytes of file content to include in agent context.

    Default 48000 (~12k tokens), leaving ~4k tokens for prompt + reasoning
    within a 16k context window.
    """
    val = config.get("file_selection", {}).get("max_context_bytes", 48000)
    try:
        return max(0, int(val))
    except (ValueError, TypeError):
        return 48000


def get_drm_enabled(config: dict) -> bool:
    return config.get("drm", {}).get("enabled", True)


def _parse_duration(value: str) -> float:
    """Parse a duration string like '10m', '5m', '300s', '1h' into seconds."""
    value = value.strip().lower()
    if value.endswith("h"):
        return float(value[:-1]) * 3600
    if value.endswith("m"):
        return float(value[:-1]) * 60
    if value.endswith("s"):
        return float(value[:-1])
    # Plain number = seconds
    return float(value)


def get_drm_fast_keep_alive(config: dict) -> float:
    """Return fast model keep-alive duration in seconds (default 600s / 10m)."""
    raw = config.get("drm", {}).get("fast_keep_alive", "10m")
    try:
        return _parse_duration(str(raw))
    except (ValueError, TypeError):
        return 600.0


def get_drm_large_keep_alive(config: dict) -> float:
    """Return large model keep-alive duration in seconds (default 300s / 5m)."""
    raw = config.get("drm", {}).get("large_keep_alive", "5m")
    try:
        return _parse_duration(str(raw))
    except (ValueError, TypeError):
        return 300.0


def get_interpreter_enabled(config: dict) -> bool:
    return config.get("interpreter", {}).get("enabled", True)


def get_agent_conservative(config: dict) -> bool:
    return config.get("agent", {}).get("conservative", False)


def get_agent_auto_checkpoint(config: dict) -> bool:
    return config.get("agent", {}).get("auto_checkpoint", True)


def get_agent_lint_command(config: dict) -> str:
    val = config.get("agent", {}).get("lint_command", "")
    return str(val) if val else ""


def get_agent_test_command(config: dict) -> str:
    val = config.get("agent", {}).get("test_command", "")
    return str(val) if val else ""


def save_config_value(section: str, key: str, value: str | bool | int) -> bool:
    """Update a single value in the config file, preserving formatting.

    Returns True on success, False on failure.
    """
    if not CONFIG_FILE.exists():
        return False

    text = CONFIG_FILE.read_text()

    # Format the value for TOML
    if isinstance(value, bool):
        toml_value = "true" if value else "false"
    elif isinstance(value, int):
        toml_value = str(value)
    else:
        toml_value = f'"{value}"'

    # Find the [section] then update key = value within it
    section_pattern = re.compile(
        rf"(\[{re.escape(section)}\].*?(?=\n\[|\Z))",
        re.DOTALL,
    )
    section_match = section_pattern.search(text)
    if not section_match:
        # Section missing — append it
        text = text.rstrip() + f"\n\n[{section}]\n{key} = {toml_value}\n"
    else:
        section_text = section_match.group(1)
        key_pattern = re.compile(
            rf"^({re.escape(key)}\s*=\s*).*$",
            re.MULTILINE,
        )
        key_match = key_pattern.search(section_text)
        if key_match:
            new_section = key_pattern.sub(rf"\g<1>{toml_value}", section_text)
            text = text[:section_match.start()] + new_section + text[section_match.end():]
        else:
            # Key missing in section — append before section ends
            insert_pos = section_match.end()
            text = text[:insert_pos] + f"{key} = {toml_value}\n" + text[insert_pos:]

    try:
        CONFIG_FILE.write_text(text)
        return True
    except OSError:
        return False
