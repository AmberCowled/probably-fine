from enum import Enum


class Mode(Enum):
    FAST = "fast"
    DAILY = "daily"
    PLANNING = "planning"
    AUTO = "auto"


MODEL_MAP: dict[Mode, str] = {
    Mode.FAST: "deepseek-coder:6.7b",
    Mode.DAILY: "qwen3-coder:30b",
    Mode.PLANNING: "qwen3:32b",
}

MODE_DESCRIPTIONS: dict[Mode, str] = {
    Mode.FAST: "Quick fixes, snippets, one-liners",
    Mode.DAILY: "Implementation, features, bug fixes",
    Mode.PLANNING: "Architecture, design, reasoning",
    Mode.AUTO: "Auto-classifies task and routes to best model",
}
