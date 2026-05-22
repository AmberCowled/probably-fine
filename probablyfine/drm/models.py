"""Data structures for the Dynamic Resource Manager."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelState:
    """Tracks the state of a single model known to the DRM."""

    name: str                        # "qwen3:32b"
    status: str = "cold"             # "cold" | "loading" | "loaded" | "unloading"
    size_vram_bytes: int = 0         # VRAM occupied (from Ollama /api/ps)
    last_used: float = 0.0           # time.monotonic() timestamp
    use_count: int = 0               # Times used this session
    load_time_s: float = 0.0         # Last observed load time

    @property
    def size_vram_gb(self) -> float:
        return self.size_vram_bytes / (1024 ** 3) if self.size_vram_bytes else 0.0

    @property
    def is_loaded(self) -> bool:
        return self.status == "loaded"

    @property
    def is_large(self) -> bool:
        """Heuristic: models using > 5 GB VRAM are 'large' (30B+)."""
        return self.size_vram_bytes > 5 * 1024 ** 3


@dataclass
class VRAMSnapshot:
    """Point-in-time GPU memory state."""

    total_mb: int = 0
    used_mb: int = 0
    free_mb: int = 0
    utilization_pct: float = 0.0
    timestamp: float = 0.0

    @property
    def usage_pct(self) -> float:
        """VRAM usage as a percentage (0-100)."""
        if self.total_mb == 0:
            return 0.0
        return (self.used_mb / self.total_mb) * 100

    @property
    def available(self) -> bool:
        """True if we have valid VRAM data."""
        return self.total_mb > 0


@dataclass
class SwapPlan:
    """Describes model transitions needed for a task."""

    models_to_unload: list[str] = field(default_factory=list)
    models_to_load: list[str] = field(default_factory=list)
    keep_warm: list[str] = field(default_factory=list)
    same_model_opt: bool = False
    estimated_swap_time_s: float = 0.0

    @property
    def needs_swap(self) -> bool:
        return bool(self.models_to_unload) or bool(self.models_to_load)


@dataclass
class DRMStatus:
    """Snapshot of DRM state for display."""

    enabled: bool
    loaded_models: list[ModelState] = field(default_factory=list)
    vram: VRAMSnapshot = field(default_factory=VRAMSnapshot)
    total_swaps: int = 0
    swaps_avoided: int = 0
    ollama_reachable: bool = True
    vram_available: bool = False
    safe_mode: bool = False
    oom_count: int = 0
    hang_count: int = 0
    recovery_count: int = 0
