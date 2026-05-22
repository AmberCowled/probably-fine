"""VRAM monitoring via nvidia-smi.

Polls GPU memory usage and provides pressure detection.
Falls back gracefully if nvidia-smi is unavailable.
"""

from __future__ import annotations

import subprocess
import time

from probablyfine.drm.models import VRAMSnapshot
from probablyfine.log_utils import get_module_logger

log = get_module_logger("probablyfine.drm.vram", "drm.log")

# How often to re-poll nvidia-smi (seconds)
_POLL_TTL = 5.0

# Pressure thresholds (MB free)
PRESSURE_HIGH_MB = 500
PRESSURE_CRITICAL_MB = 200


class VRAMMonitor:
    """Polls GPU memory via nvidia-smi and detects memory pressure."""

    def __init__(self):
        self._snapshot: VRAMSnapshot = VRAMSnapshot()
        self._last_poll: float = 0.0
        self._available: bool | None = None  # None = not yet checked

    @property
    def available(self) -> bool:
        """True if nvidia-smi is working and we have VRAM data."""
        if self._available is None:
            self.poll(force=True)
        return self._available is True

    def poll(self, force: bool = False) -> VRAMSnapshot:
        """Return current VRAM state. Uses TTL cache unless force=True."""
        now = time.monotonic()
        if not force and (now - self._last_poll) < _POLL_TTL:
            return self._snapshot

        self._last_poll = now

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                log.debug("nvidia-smi returned code %d", result.returncode)
                self._available = False
                return self._snapshot

            # Parse CSV: "8192, 4200, 3992, 45"
            parts = result.stdout.strip().split(",")
            if len(parts) < 4:
                log.debug("nvidia-smi unexpected output: %s", result.stdout.strip())
                self._available = False
                return self._snapshot

            total = int(parts[0].strip())
            used = int(parts[1].strip())
            free = int(parts[2].strip())
            util = float(parts[3].strip())

            self._snapshot = VRAMSnapshot(
                total_mb=total,
                used_mb=used,
                free_mb=free,
                utilization_pct=util,
                timestamp=now,
            )
            self._available = True

        except FileNotFoundError:
            log.debug("nvidia-smi not found")
            self._available = False
        except subprocess.TimeoutExpired:
            log.debug("nvidia-smi timed out")
            # Don't change _available — might be temporarily slow
        except (ValueError, IndexError) as e:
            log.debug("nvidia-smi parse error: %s", e)
            self._available = False
        except OSError as e:
            log.debug("nvidia-smi OS error: %s", e)
            self._available = False

        return self._snapshot

    def get_snapshot(self) -> VRAMSnapshot:
        """Return the most recent snapshot (does not force a new poll)."""
        return self._snapshot

    def is_pressure_high(self) -> bool:
        """True if free VRAM < 500 MB."""
        if not self.available:
            return False
        snap = self.poll()
        return snap.free_mb < PRESSURE_HIGH_MB

    def is_pressure_critical(self) -> bool:
        """True if free VRAM < 200 MB."""
        if not self.available:
            return False
        snap = self.poll()
        return snap.free_mb < PRESSURE_CRITICAL_MB

    def format_bar(self, width: int = 20) -> str:
        """Return a text bar like '[||||||||...........]  51%'."""
        snap = self.poll()
        if not snap.available:
            return "[no GPU data]"

        pct = snap.usage_pct
        filled = int(width * pct / 100)
        filled = min(filled, width)
        bar = "|" * filled + "." * (width - filled)

        return f"[{bar}]  {snap.used_mb:,} / {snap.total_mb:,} MB ({pct:.0f}%)"
