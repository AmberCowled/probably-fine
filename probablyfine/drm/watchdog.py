"""Health Watchdog: failure detection, recovery, and safe mode.

Detects OOM errors, generation hangs, and Ollama unavailability.
Provides emergency recovery (unload all, fallback to small model)
and safe mode (restrict to fast model only).
"""

from __future__ import annotations

import re
import time

from probablyfine.drm.registry import parse_ps_models
from probablyfine.log_utils import get_module_logger

log = get_module_logger("probablyfine.drm.watchdog", "drm.log")

# OOM error patterns from Ollama / CUDA
_OOM_PATTERNS: list[re.Pattern] = [
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"not enough memory", re.IGNORECASE),
    re.compile(r"allocation failed", re.IGNORECASE),
    re.compile(r"OOM", re.IGNORECASE),
    re.compile(r"memory allocation", re.IGNORECASE),
    re.compile(r"cudaMalloc failed", re.IGNORECASE),
    re.compile(r"CUBLAS_STATUS_ALLOC_FAILED", re.IGNORECASE),
]

# Default hang timeout: no new tokens for this many seconds
HANG_TIMEOUT_S = 60.0


class HealthWatchdog:
    """Monitors Ollama health, detects failures, and orchestrates recovery."""

    def __init__(self):
        self._safe_mode: bool = False
        self._oom_count: int = 0
        self._hang_count: int = 0
        self._recovery_count: int = 0
        self._last_oom_time: float = 0.0

    @property
    def safe_mode(self) -> bool:
        return self._safe_mode

    @safe_mode.setter
    def safe_mode(self, value: bool) -> None:
        if value != self._safe_mode:
            self._safe_mode = value
            if value:
                log.warning("Safe mode ACTIVATED — restricting to fast model only")
            else:
                log.info("Safe mode deactivated")

    def is_oom_error(self, error: str | Exception) -> bool:
        """Check if an error looks like an OOM failure."""
        error_str = str(error)
        for pattern in _OOM_PATTERNS:
            if pattern.search(error_str):
                log.info("OOM error detected: %s", error_str[:200])
                return True
        return False

    def detect_hang(self, last_token_time: float, now: float | None = None) -> bool:
        """Check if generation appears hung (no tokens for HANG_TIMEOUT_S).

        Args:
            last_token_time: monotonic timestamp of the last received token.
            now: current time (defaults to time.monotonic()).

        Returns True if generation appears hung.
        """
        if now is None:
            now = time.monotonic()
        elapsed = now - last_token_time
        if elapsed >= HANG_TIMEOUT_S:
            log.warning("Hang detected: no tokens for %.0fs", elapsed)
            self._hang_count += 1
            return True
        return False

    def emergency_unload_all(self) -> int:
        """Unload ALL models from Ollama to free VRAM. Returns count unloaded."""
        log.warning("Emergency unload: clearing all models from VRAM")
        unloaded = 0
        try:
            import ollama
            response = ollama.ps()

            for item in parse_ps_models(response):
                name = item["name"]
                try:
                    from probablyfine.ollama_utils import create_client
                    client = create_client(timeout=10)
                    client.generate(model=name, prompt="", keep_alive=0)
                    log.info("Emergency unloaded: %s", name)
                    unloaded += 1
                except Exception as e:
                    log.warning("Failed to emergency-unload %s: %s", name, e)

        except Exception as e:
            log.error("Emergency unload failed to list models: %s", e)

        self._recovery_count += 1
        return unloaded

    def get_fallback_model(
        self,
        failed_model: str,
        model_map: dict[str, str],
    ) -> str | None:
        """Return a smaller fallback model when a large model fails.

        Returns None if the failed model IS the fallback (fast model),
        or if no fallback is configured.
        """
        fast_model = model_map.get("fast", "")
        if not fast_model or failed_model == fast_model:
            return None  # Already at smallest model, nowhere to fall back
        log.info("Fallback routing: %s failed, suggesting %s", failed_model, fast_model)
        return fast_model

    def handle_oom(self, failed_model: str, model_map: dict[str, str]) -> str | None:
        """Handle an OOM error: emergency unload, suggest fallback, maybe enter safe mode.

        Returns the fallback model name, or None if no fallback available.
        """
        self._oom_count += 1
        self._last_oom_time = time.monotonic()
        log.warning("OOM #%d for model %s", self._oom_count, failed_model)

        # Emergency: clear all VRAM
        self.emergency_unload_all()

        # If we've had multiple OOMs recently, enter safe mode
        if self._oom_count >= 2:
            self.safe_mode = True

        return self.get_fallback_model(failed_model, model_map)

    def get_stats(self) -> dict[str, int]:
        """Return watchdog statistics for display."""
        return {
            "oom_count": self._oom_count,
            "hang_count": self._hang_count,
            "recovery_count": self._recovery_count,
        }
