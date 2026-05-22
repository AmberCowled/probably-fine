from enum import Enum


class Mode(Enum):
    FAST = "fast"
    DAILY = "daily"
    PLANNING = "planning"
    AUTO = "auto"


MODE_DESCRIPTIONS: dict[Mode, str] = {
    Mode.FAST: "Quick fixes, snippets, one-liners",
    Mode.DAILY: "Implementation, features, bug fixes",
    Mode.PLANNING: "Architecture, design, reasoning",
    Mode.AUTO: "Auto-classifies task and routes to best model",
}
