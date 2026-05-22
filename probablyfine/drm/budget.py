"""Context budgeting: determine safe num_ctx per model based on available VRAM.

Prevents large context windows from causing VRAM overflow, partial offloading
to system RAM, and catastrophic generation slowdowns.
"""

from __future__ import annotations

from probablyfine.log_utils import get_module_logger

log = get_module_logger("probablyfine.drm.budget", "drm.log")
from probablyfine.drm.vram import VRAMMonitor

# Base context limits per known model.
# 16k fits comfortably in 8GB VRAM alongside model weights + system usage.
_BASE_LIMITS: dict[str, int] = {
    "deepseek-coder:6.7b": 16384,
    "qwen3:8b": 16384,
}

# Default for unknown models: moderate context
_DEFAULT_BASE_LIMIT = 8192

# Absolute minimum — below this, the model can't do useful work
MIN_CONTEXT = 1024


class ContextBudget:
    """Determines safe context sizes per model based on available VRAM."""

    def __init__(self, vram: VRAMMonitor):
        self._vram = vram

    def get_budget(self, model: str) -> int:
        """Return recommended num_ctx for this model.

        Uses dynamic adjustment if VRAM data is available,
        otherwise returns the static base limit.
        """
        base = _BASE_LIMITS.get(model, _DEFAULT_BASE_LIMIT)

        if not self._vram.available:
            log.debug("No VRAM data — using base limit %d for %s", base, model)
            return base

        snap = self._vram.poll()

        # Dynamic adjustment based on free VRAM after model is loaded.
        # When a model is loaded, most VRAM is consumed by weights.
        # The remaining free VRAM constrains the KV cache (context).
        if snap.free_mb > 1500:
            # Comfortable headroom — full context
            budget = base
        elif snap.free_mb > 800:
            # Moderate pressure — trim 25%
            budget = int(base * 0.75)
        elif snap.free_mb > 400:
            # High pressure — trim 50%
            budget = int(base * 0.50)
        else:
            # Critical — minimum viable context
            budget = MIN_CONTEXT

        budget = max(budget, MIN_CONTEXT)

        if budget < base:
            log.info(
                "Context budget for %s: %d (reduced from %d, free VRAM: %d MB)",
                model, budget, base, snap.free_mb,
            )
        else:
            log.debug("Context budget for %s: %d (full, free VRAM: %d MB)",
                       model, budget, snap.free_mb)

        return budget
