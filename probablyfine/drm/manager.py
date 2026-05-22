"""Resource Manager: coordinates model lifecycle with Ollama."""

from __future__ import annotations

import time

from probablyfine.drm.budget import ContextBudget
from probablyfine.drm.models import DRMStatus, SwapPlan
from probablyfine.drm.registry import ModelRegistry
from probablyfine.log_utils import get_module_logger

log = get_module_logger("probablyfine.drm.manager", "drm.log")
from probablyfine.drm.scheduler import SwapScheduler
from probablyfine.drm.vram import VRAMMonitor
from probablyfine.drm.watchdog import HealthWatchdog


class ResourceManager:
    """Orchestrates model loading, unloading, and swap optimization.

    Sits above Ollama and controls it via the ollama Python library.
    Does not replace Ollama — coordinates it.
    """

    def __init__(
        self,
        fast_keep_alive_s: float = 600.0,
        large_keep_alive_s: float = 300.0,
    ):
        self.registry = ModelRegistry()
        self.vram = VRAMMonitor()
        self.budget = ContextBudget(self.vram)
        self.scheduler = SwapScheduler(
            self.registry, self.vram,
            fast_keep_alive_s=fast_keep_alive_s,
            large_keep_alive_s=large_keep_alive_s,
        )
        self.watchdog = HealthWatchdog()
        self._total_swaps: int = 0
        self._swaps_avoided: int = 0
        self._enabled: bool = True
        self._last_plan: SwapPlan | None = None
        self._model_map: dict[str, str] = {}

    def set_model_map(self, model_map: dict[str, str]) -> None:
        """Store model map for fallback routing."""
        self._model_map = model_map

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def safe_mode(self) -> bool:
        return self.watchdog.safe_mode

    @safe_mode.setter
    def safe_mode(self, value: bool) -> None:
        self.watchdog.safe_mode = value

    def sync(self, force: bool = False) -> None:
        """Sync registry with Ollama. Call periodically or before decisions."""
        self.registry.sync(force=force)

    def resolve_model_for_safe_mode(self, model: str) -> str:
        """If safe mode is active, downgrade to fast model. Otherwise return as-is."""
        if not self.watchdog.safe_mode:
            return model
        fast = self._model_map.get("fast", "")
        if fast and model != fast:
            log.info("Safe mode: downgrading %s to %s", model, fast)
            return fast
        return model

    def ensure_loaded(self, model: str, unload_conflicting: bool = True) -> bool:
        """Ensure a model is ready in Ollama.

        If the model is already loaded, this is a no-op (~0ms).
        If a conflicting large model is loaded and unload_conflicting is True,
        unloads it first to free VRAM.

        Returns True if the model is loaded or can be loaded, False on failure.
        """
        if not self._enabled:
            return True

        self.registry.sync()

        # Already loaded — nothing to do
        if self.registry.is_loaded(model):
            self.registry.record_usage(model)
            log.debug("ensure_loaded(%s): already loaded", model)
            return True

        # Check if we need to unload a conflicting large model
        if unload_conflicting:
            loaded_large = self.registry.get_loaded_large()
            if loaded_large:
                # On 8GB VRAM, only one large model fits.
                # Unload the least-recently-used large model.
                loaded_large.sort(key=lambda m: m.last_used)
                for m in loaded_large:
                    log.info("Unloading %s to make room for %s", m.name, model)
                    self._unload(m.name)
                    self._total_swaps += 1

        # Model will auto-load when Ollama receives the next request.
        # We just track that we expect it.
        self.registry.record_usage(model)
        self.registry.mark_status(model, "loading")
        log.info("ensure_loaded(%s): prepared (will load on next request)", model)
        return True

    def prepare_for_task(
        self,
        maker_model: str,
        checker_model: str | None = None,
        reflection_on: bool = False,
    ) -> SwapPlan:
        """Plan model lifecycle for an entire task using the scheduler.

        Returns a SwapPlan describing what transitions are needed.
        Executes the unload portion of the plan immediately.
        """
        # Safe mode override
        maker_model = self.resolve_model_for_safe_mode(maker_model)
        if checker_model:
            checker_model = self.resolve_model_for_safe_mode(checker_model)

        if not self._enabled:
            same = reflection_on and maker_model == checker_model
            plan = SwapPlan(same_model_opt=same)
            self._last_plan = plan
            return plan

        # Use scheduler to build an intelligent swap plan
        plan = self.scheduler.plan_task(
            maker_model=maker_model,
            checker_model=checker_model,
            reflection_on=reflection_on,
        )
        self._last_plan = plan

        if plan.same_model_opt:
            self._swaps_avoided += 1

        # Execute unloads from the plan
        for model_name in plan.models_to_unload:
            log.info("Executing planned unload: %s", model_name)
            self._unload(model_name)
            self._total_swaps += 1

        # Ensure maker model is ready
        self.registry.record_usage(maker_model)
        if not self.registry.is_loaded(maker_model):
            self.registry.mark_status(maker_model, "loading")
            log.info("Maker model %s will load on next request", maker_model)

        return plan

    def prepare_for_checker(self, checker_model: str, maker_model: str) -> SwapPlan:
        """Transition from maker phase to checker phase.

        Uses the scheduler to plan the transition intelligently.
        Returns a SwapPlan describing what happened.
        """
        # Safe mode override
        checker_model = self.resolve_model_for_safe_mode(checker_model)

        if not self._enabled:
            return SwapPlan(same_model_opt=(checker_model == maker_model))

        plan = self.scheduler.plan_checker_transition(checker_model, maker_model)

        if plan.same_model_opt:
            self.registry.record_usage(checker_model)
            return plan

        # Execute the transition
        for model_name in plan.models_to_unload:
            log.info("Checker transition: unloading %s", model_name)
            self._unload(model_name)
            self._total_swaps += 1

        self.registry.record_usage(checker_model)
        if not self.registry.is_loaded(checker_model):
            self.registry.mark_status(checker_model, "loading")

        return plan

    def handle_failure(self, error: Exception, model: str) -> str | None:
        """Handle a model failure. Returns fallback model name, or None.

        Detects OOM vs transient errors, performs recovery as needed.
        """
        if not self._enabled:
            return None

        if self.watchdog.is_oom_error(error):
            return self.watchdog.handle_oom(model, self._model_map)

        return None

    def get_fallback_model(self, failed_model: str) -> str | None:
        """Get a fallback model for when failed_model can't be used."""
        return self.watchdog.get_fallback_model(failed_model, self._model_map)

    def emergency_unload_all(self) -> int:
        """Emergency: unload all models from VRAM."""
        count = self.watchdog.emergency_unload_all()
        # Sync registry to reflect the unloads
        self.registry._last_sync = 0.0
        self.registry.sync(force=True)
        return count

    def unload(self, model: str) -> bool:
        """Manually unload a model. Returns True on success."""
        if not self._enabled:
            return False
        return self._unload(model)

    def _unload(self, model: str) -> bool:
        """Send unload request to Ollama via keep_alive=0."""
        try:
            from probablyfine.ollama_utils import create_client
            client = create_client(timeout=10)
            client.generate(model=model, prompt="", keep_alive=0)
            self.registry.mark_status(model, "cold")
            log.info("Unloaded model %s", model)
            return True
        except Exception as e:
            log.warning("Failed to unload %s: %s", model, e)
            return False

    def task_completed(self, model: str) -> None:
        """Post-task bookkeeping. Record usage, let sync update state later."""
        if not self._enabled:
            return
        self.registry.record_usage(model)
        # Force a sync next time to see actual loaded state
        self.registry._last_sync = 0.0

    def get_vram_warning(self) -> str | None:
        """Return a warning string if VRAM pressure is dangerous, else None."""
        if not self._enabled or not self.vram.available:
            return None
        if self.vram.is_pressure_critical():
            snap = self.vram.get_snapshot()
            return f"VRAM critically low ({snap.free_mb} MB free)"
        if self.vram.is_pressure_high():
            snap = self.vram.get_snapshot()
            return f"VRAM pressure high ({snap.free_mb} MB free)"
        return None

    def get_status(self) -> DRMStatus:
        """Return current DRM state for display."""
        self.registry.sync()
        snap = self.vram.poll()
        wd_stats = self.watchdog.get_stats()
        return DRMStatus(
            enabled=self._enabled,
            loaded_models=self.registry.get_loaded(),
            vram=snap,
            total_swaps=self._total_swaps,
            swaps_avoided=self._swaps_avoided,
            ollama_reachable=self.registry.ollama_reachable,
            vram_available=self.vram.available,
            safe_mode=self.watchdog.safe_mode,
            oom_count=wd_stats["oom_count"],
            hang_count=wd_stats["hang_count"],
            recovery_count=wd_stats["recovery_count"],
        )
