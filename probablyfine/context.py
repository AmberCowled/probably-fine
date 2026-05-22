import glob
from pathlib import Path


class FileContext:
    """Tracks which files are included in agent context."""

    def __init__(self):
        self._files: list[Path] = []

    @property
    def files(self) -> list[str]:
        """Return file paths as strings for passing to the agent."""
        return [str(f) for f in self._files]

    @property
    def count(self) -> int:
        return len(self._files)

    def add(self, pattern: str) -> list[str]:
        """Add files matching pattern. Returns list of newly added paths."""
        added = []
        matches = glob.glob(pattern, recursive=True)

        if not matches:
            path = Path(pattern)
            if path.is_file():
                matches = [pattern]

        for match in matches:
            p = Path(match).resolve()
            if not p.is_file():
                continue
            if p not in self._files:
                self._files.append(p)
                added.append(str(p))

        return added

    def drop(self, pattern: str) -> list[str]:
        """Remove files matching pattern. Returns list of removed paths."""
        removed = []
        to_remove = []

        for f in self._files:
            if pattern in str(f) or f.name == pattern or str(f) == pattern:
                to_remove.append(f)
                removed.append(str(f))

        for f in to_remove:
            self._files.remove(f)

        return removed

    def clear(self) -> int:
        """Remove all files. Returns count removed."""
        count = len(self._files)
        self._files.clear()
        return count

    def list_files(self) -> list[str]:
        """Return display-friendly list of tracked files."""
        return [str(f) for f in self._files]
