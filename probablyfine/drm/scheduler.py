"""Swap Scheduler: intelligent model swap planning and warm pool management.

Decides which models to unload and in what order, based on priority scoring.
Models within their keep-alive window are protected from eviction unless
VRAM pressure forces it.
"""

from __future__ import annotations

import time

from probablyfine.drm.models import ModelState, SwapPlan
from probablyfine.drm.registry import ModelRegistry
from probablyfine.log_utils import get_module_logger

log = get_module_logger("probablyfine.drm.scheduler", "drm.log")
from probablyfine.drm.vram import VRAMMonitor


# Default keep-alive durations (seconds)
_DEFAULT_FAST_KEEP_ALIVE = 600    # 10 minutes
_DEFAULT_LARGE_KEEP_ALIVE = 300   # 5 minutes

# Estimated load times for priority scoring (seconds)
_ESTIMATED_LOAD_TIME: dict[str, float] = {
    "deepseek-coder:6.7b": 3.0,
    "qwen3:8b": 5.0,
}
_DEFAULT_LOAD_TIME = 5.0

# Unload priority scoring coefficients (higher score = unload first)
_PRIORITY_BASE = 50.0             # Baseline score for every loaded model
_STALENESS_PER_SEC = 0.1          # Bonus per second since last use
_STALENESS_MAX = 30.0             # Cap on staleness bonus
_LARGE_MODEL_BONUS = 10.0         # Bonus for large models (frees more VRAM)
_KEEP_ALIVE_PENALTY = -40.0       # Penalty for models inside keep-alive window
_PROTECTED_SCORE = -100.0         # Sentinel score for protected models
_UNLOAD_TIME_ESTIMATE = 1.0       # Estimated seconds per model unload


class SwapScheduler:
    """Plans model transitions with priority-based unloading.

    Priority scoring determines which models to unload first:
    - Higher score = unload first
    - Models in their keep-alive window get a penalty (lower score)
    - Larger models get a bonus (freeing more VRAM is more valuable)
    - Stale models (not used recently) get a bonus
    """

    def __init__(
        self,
        registry: ModelRegistry,
        vram: VRAMMonitor,
        fast_keep_alive_s: float = _DEFAULT_FAST_KEEP_ALIVE,
        large_keep_alive_s: float = _DEFAULT_LARGE_KEEP_ALIVE,
    ):
        self._registry = registry
        self._vram = vram
        self._fast_keep_alive = fast_keep_alive_s
        self._large_keep_alive = large_keep_alive_s

    def unload_priority(self, model: ModelState, protected: set[str] | None = None) -> float:
        """Score a loaded model for unload priority. Higher = unload first.

        Scoring factors:
        - Base: 50.0
        - Staleness bonus: +0.1 per second since last use (max +30)
        - Size bonus: +10 if large model (frees more VRAM)
        - Keep-alive penalty: -40 if within keep-alive window
        - Protected penalty: -100 if model is in the protected set
        """
        if protected and model.name in protected:
            return _PROTECTED_SCORE

        now = time.monotonic()
        score = _PRIORITY_BASE

        # Staleness: older = higher priority to unload
        if model.last_used > 0:
            age_s = now - model.last_used
            score += min(age_s * _STALENESS_PER_SEC, _STALENESS_MAX)
        else:
            score += _STALENESS_MAX  # Never used this session — very stale

        # Size: large models free more VRAM, slight bonus to unload them
        if model.is_large:
            score += _LARGE_MODEL_BONUS

        # Keep-alive: if within window, strong penalty
        if model.last_used > 0:
            keep_alive = self._large_keep_alive if model.is_large else self._fast_keep_alive
            if (now - model.last_used) < keep_alive:
                score += _KEEP_ALIVE_PENALTY
                log.debug(
                    "Model %s within keep-alive window (%.0fs remaining)",
                    model.name, keep_alive - (now - model.last_used),
                )

        return score

    def plan_task(
        self,
        maker_model: str,
        checker_model: str | None = None,
        reflection_on: bool = False,
    ) -> SwapPlan:
        """Plan model transitions for a task.

        Returns a SwapPlan describing what to unload and load.
        """
        self._registry.sync()

        same_model = reflection_on and checker_model and maker_model == checker_model

        # Models needed for this task
        needed: list[str] = [maker_model]
        if reflection_on and checker_model and checker_model != maker_model:
            needed.append(checker_model)

        # Protected set: models we need, don't unload them
        protected = set(needed)

        # Figure out what's already loaded
        loaded = self._registry.get_loaded()
        loaded_names = {m.name for m in loaded}

        # Models that need to be loaded
        to_load = [m for m in needed if m not in loaded_names]

        # Determine if we need to free VRAM
        to_unload: list[str] = []
        keep_warm: list[str] = []

        if to_load:
            # Check all loaded models that aren't needed for this task
            evictable = [m for m in loaded if m.name not in protected]

            if evictable:
                # Score and sort by unload priority (highest first)
                scored = [(self.unload_priority(m, protected), m) for m in evictable]
                scored.sort(key=lambda x: x[0], reverse=True)

                for score, model in scored:
                    if score < 0:
                        # Protected or strongly within keep-alive — keep warm
                        keep_warm.append(model.name)
                        continue

                    # Free VRAM for the incoming model + its KV cache
                    to_unload.append(model.name)
                    log.info(
                        "Scheduler: unload %s (priority %.1f) for %s",
                        model.name, score, to_load[0],
                    )

        # Estimate swap time
        estimated_time = 0.0
        for name in to_unload:
            estimated_time += _UNLOAD_TIME_ESTIMATE
        for name in to_load:
            estimated_time += _ESTIMATED_LOAD_TIME.get(name, _DEFAULT_LOAD_TIME)

        plan = SwapPlan(
            models_to_unload=to_unload,
            models_to_load=to_load,
            keep_warm=keep_warm,
            same_model_opt=bool(same_model),
            estimated_swap_time_s=estimated_time,
        )

        log.info(
            "SwapPlan: unload=%s load=%s warm=%s same_model=%s est=%.1fs",
            to_unload, to_load, keep_warm, same_model, estimated_time,
        )

        return plan

    def plan_checker_transition(
        self,
        checker_model: str,
        maker_model: str,
    ) -> SwapPlan:
        """Plan transition from maker phase to checker phase.

        If same model, returns a no-op plan. Otherwise plans the swap.
        """
        if checker_model == maker_model:
            log.debug("Scheduler: checker == maker, no swap needed")
            return SwapPlan(same_model_opt=True)

        self._registry.sync()

        # If checker is already loaded, no swap needed
        if self._registry.is_loaded(checker_model):
            self._registry.record_usage(checker_model)
            return SwapPlan()

        # Need to load checker, may need to unload maker
        to_unload: list[str] = []
        loaded_large = self._registry.get_loaded_large()

        for m in loaded_large:
            if m.name == checker_model:
                continue  # Don't unload what we're about to use
            # During reflection, the maker model is expendable
            to_unload.append(m.name)

        estimated_time = len(to_unload) * _UNLOAD_TIME_ESTIMATE
        estimated_time += _ESTIMATED_LOAD_TIME.get(checker_model, _DEFAULT_LOAD_TIME)

        plan = SwapPlan(
            models_to_unload=to_unload,
            models_to_load=[checker_model],
            estimated_swap_time_s=estimated_time,
        )

        log.info(
            "Checker transition plan: unload=%s load=%s est=%.1fs",
            to_unload, [checker_model], estimated_time,
        )

        return plan
