"""Model registry: tracks which models are loaded in Ollama."""

from __future__ import annotations

import time

from probablyfine.drm.models import ModelState
from probablyfine.log_utils import get_module_logger

log = get_module_logger("probablyfine.drm", "drm.log")

# How often to re-poll Ollama /api/ps (seconds)
_SYNC_TTL = 5.0


def parse_ps_models(response) -> list[dict[str, object]]:
    """Parse Ollama ps() response into a list of {name, size_vram} dicts.

    Handles both dict and object API formats.
    """
    if isinstance(response, dict):
        running = response.get("models", [])
    else:
        running = getattr(response, "models", []) or []

    results: list[dict[str, object]] = []
    for entry in running:
        if isinstance(entry, dict):
            name = entry.get("name", "") or entry.get("model", "")
            size_vram = entry.get("size_vram", 0) or entry.get("size", 0)
        else:
            name = getattr(entry, "name", "") or getattr(entry, "model", "")
            size_vram = getattr(entry, "size_vram", 0) or getattr(entry, "size", 0)
        if name:
            results.append({"name": name, "size_vram": size_vram})
    return results


class ModelRegistry:
    """Tracks loaded models by polling Ollama's /api/ps endpoint."""

    def __init__(self):
        self._models: dict[str, ModelState] = {}
        self._last_sync: float = 0.0
        self._ollama_reachable: bool = True

    @property
    def ollama_reachable(self) -> bool:
        return self._ollama_reachable

    def sync(self, force: bool = False) -> None:
        """Poll Ollama /api/ps and update loaded model states.

        Respects TTL cache unless force=True.
        """
        now = time.monotonic()
        if not force and (now - self._last_sync) < _SYNC_TTL:
            return

        try:
            import ollama
            response = ollama.ps()
        except Exception as e:
            log.debug("Ollama ps() failed: %s", e)
            self._ollama_reachable = False
            return

        self._ollama_reachable = True
        self._last_sync = now

        parsed = parse_ps_models(response)
        loaded_names: set[str] = set()
        for item in parsed:
            name = item["name"]
            size_vram = item["size_vram"]
            loaded_names.add(name)

            if name in self._models:
                state = self._models[name]
                state.status = "loaded"
                state.size_vram_bytes = size_vram
            else:
                self._models[name] = ModelState(
                    name=name,
                    status="loaded",
                    size_vram_bytes=size_vram,
                    last_used=now,
                )

        # Mark models that disappeared as cold
        for name, state in self._models.items():
            if name not in loaded_names and state.status == "loaded":
                state.status = "cold"
                state.size_vram_bytes = 0
                log.info("Model %s no longer loaded (detected via sync)", name)

    def get_or_create(self, name: str) -> ModelState:
        """Get state for a model, creating a cold entry if new."""
        if name not in self._models:
            self._models[name] = ModelState(name=name)
        return self._models[name]

    def is_loaded(self, name: str) -> bool:
        """Check if model is currently loaded in Ollama."""
        state = self._models.get(name)
        return state is not None and state.status == "loaded"

    def get_loaded(self) -> list[ModelState]:
        """Return all currently loaded models."""
        return [s for s in self._models.values() if s.status == "loaded"]

    def get_loaded_large(self) -> list[ModelState]:
        """Return loaded models that are 'large' (>5 GB VRAM)."""
        return [s for s in self._models.values()
                if s.status == "loaded" and s.is_large]

    def record_usage(self, name: str) -> None:
        """Update last_used and use_count for a model."""
        state = self.get_or_create(name)
        state.last_used = time.monotonic()
        state.use_count += 1

    def mark_status(self, name: str, status: str) -> None:
        """Manually set a model's status."""
        state = self.get_or_create(name)
        state.status = status
        if status == "cold":
            state.size_vram_bytes = 0
