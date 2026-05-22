"""Dynamic Resource Manager — model lifecycle orchestration for Ollama.

Usage:
    from probablyfine.drm import get_manager
    drm = get_manager()
    drm.ensure_loaded("qwen3:32b")
"""

from __future__ import annotations

from probablyfine.drm.manager import ResourceManager

_instance: ResourceManager | None = None


def get_manager(
    fast_keep_alive_s: float = 600.0,
    large_keep_alive_s: float = 300.0,
) -> ResourceManager:
    """Return the singleton ResourceManager instance.

    Keep-alive parameters are only used on first call (instance creation).
    """
    global _instance
    if _instance is None:
        _instance = ResourceManager(
            fast_keep_alive_s=fast_keep_alive_s,
            large_keep_alive_s=large_keep_alive_s,
        )
    return _instance
