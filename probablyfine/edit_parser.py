"""Parse and apply SEARCH/REPLACE edit blocks from LLM output.

Pure string processing — no LLM dependencies. Handles two block formats:

SEARCH/REPLACE (editing existing files):
    FILE: path/to/file.py
    <<<<<<< SEARCH
    old content
    =======
    new content
    >>>>>>> REPLACE

CONTENT/END (new or whole-file):
    FILE: path/to/file.py (new)
    <<<<<<< CONTENT
    full content
    >>>>>>> END
"""

from __future__ import annotations

import re
from pathlib import Path

from probablyfine.log_utils import get_module_logger
from probablyfine.models import (
    EDIT_ERR_FILE_MISSING,
    EDIT_ERR_GENERIC,
    EDIT_ERR_LINE_RANGE,
    EDIT_ERR_MULTI_MATCH,
    EDIT_ERR_NOT_FOUND,
    FileEdit,
)

log = get_module_logger("probablyfine.edit_parser", "edit_parser.log")

MAX_EDITS_PER_RESPONSE = 10  # Cap to prevent quality degradation on long sequences

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_SEARCH_REPLACE_RE = re.compile(
    r"FILE:\s*(.+?)\s*\n"
    r"<{7}\s*SEARCH\n"
    r"([\s\S]*?)\n"
    r"={7}\n"
    r"([\s\S]*?)\n"
    r">{7}\s*REPLACE",
)

_CONTENT_END_RE = re.compile(
    r"FILE:\s*(.+?)\s*\((new|whole)\)\s*\n"
    r"<{7}\s*CONTENT\n"
    r"([\s\S]*?)\n"
    r">{7}\s*END",
)

_LINE_ANCHORED_RE = re.compile(
    r"FILE:\s*(.+?)\s+LINES?\s+(\d+)\s*[-–]\s*(\d+)\s*\n"
    r"<{7}\s*REPLACE\n"
    r"([\s\S]*?)\n"
    r">{7}\s*END",
)

# Unified diff format: --- a/file\n+++ b/file\n@@ ... @@\n hunk lines
_UNIFIED_DIFF_RE = re.compile(
    r"---\s+a/(.+?)\s*\n"
    r"\+\+\+\s+b/(.+?)\s*\n"
    r"((?:@@\s*-\d+(?:,\d+)?\s*\+\d+(?:,\d+)?\s*@@[^\n]*\n(?:[ +\-].*\n?)*)+"
    r")",
)

# Individual hunk header within a unified diff
_HUNK_HEADER_RE = re.compile(r"@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@")

# Markdown code fences that models sometimes wrap blocks in
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n([\s\S]*?)\n```$", re.MULTILINE)


def _parse_unified_diff(file_path: str, hunk_text: str) -> list[FileEdit]:
    """Convert unified diff hunks into FileEdit objects.

    Each @@ hunk becomes a separate FileEdit with line_start/line_end
    for the removed region and the added lines as replacement.
    """
    edits: list[FileEdit] = []
    # Split into individual hunks
    parts = _HUNK_HEADER_RE.split(hunk_text)
    # parts: [pre, start1, count1, start2, count2, body, start1, ...]
    i = 1
    while i + 4 < len(parts):
        old_start = int(parts[i])
        old_count = int(parts[i + 1]) if parts[i + 1] else 1
        # parts[i+2], parts[i+3] are new start/count — not needed for edit
        body = parts[i + 4]
        i += 5

        search_lines: list[str] = []
        replace_lines: list[str] = []
        for line in body.splitlines():
            if line.startswith("-"):
                search_lines.append(line[1:])
            elif line.startswith("+"):
                replace_lines.append(line[1:])
            elif line.startswith(" "):
                search_lines.append(line[1:])
                replace_lines.append(line[1:])
            # Skip lines without prefix (context/empty)

        if search_lines or replace_lines:
            edits.append(FileEdit(
                file=file_path,
                search="\n".join(search_lines),
                replace="\n".join(replace_lines),
                line_start=old_start,
                line_end=old_start + max(old_count - 1, 0),
            ))

    if edits:
        log.info("Parsed %d hunk(s) from unified diff for %s", len(edits), file_path)
    return edits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_edits(response: str) -> list[FileEdit]:
    """Parse SEARCH/REPLACE and CONTENT/END blocks from LLM output.

    Returns a list of FileEdit objects in the order they appear in the response.
    Empty list if no blocks found.
    """
    # Strip markdown fences that may wrap edit blocks
    cleaned = _strip_outer_fences(response)

    edits: list[tuple[int, FileEdit]] = []

    # SEARCH/REPLACE blocks
    for m in _SEARCH_REPLACE_RE.finditer(cleaned):
        file_path = m.group(1).strip()
        search = m.group(2)
        replace = m.group(3)
        edits.append((m.start(), FileEdit(
            file=file_path,
            search=search,
            replace=replace,
        )))

    # LINE-ANCHORED blocks
    for m in _LINE_ANCHORED_RE.finditer(cleaned):
        file_path = m.group(1).strip()
        line_start = int(m.group(2))
        line_end = int(m.group(3))
        replace = m.group(4)
        edits.append((m.start(), FileEdit(
            file=file_path,
            search="",
            replace=replace,
            line_start=line_start,
            line_end=line_end,
        )))

    # CONTENT/END blocks
    for m in _CONTENT_END_RE.finditer(cleaned):
        file_path = m.group(1).strip()
        kind = m.group(2)  # "new" or "whole"
        content = m.group(3)
        is_new = kind == "new"
        edits.append((m.start(), FileEdit(
            file=file_path,
            search="",
            replace=content,
            is_new_file=is_new,
        )))

    # UNIFIED DIFF blocks (--- a/file ... +++ b/file ... @@ hunks)
    for m in _UNIFIED_DIFF_RE.finditer(cleaned):
        file_a = m.group(1).strip()
        file_b = m.group(2).strip()
        hunk_text = m.group(3)
        # Use the b-side path (destination); fall back to a-side
        file_path = file_b or file_a
        diff_edits = _parse_unified_diff(file_path, hunk_text)
        for de in diff_edits:
            edits.append((m.start(), de))

    # Sort by position in the response (preserve model's intended order)
    edits.sort(key=lambda t: t[0])
    result = [e for _, e in edits]

    if result:
        paths = [e.file for e in result]
        log.info("Parsed %d edit(s) for files: %s", len(result), paths)
    else:
        log.info("No edit blocks found in response (%d chars)", len(response))

    return result


def _calculate_context_hint(content: str, search_text: str, match_count: int) -> str:
    """Calculate how much surrounding context is needed to disambiguate a multi-match SEARCH block.

    Returns a short guidance string like "Include 3+ surrounding lines to make unique."
    """
    lines = content.splitlines(keepends=True)
    search_lines = search_text.splitlines()
    if not search_lines:
        return f"Include more surrounding lines to disambiguate {match_count} matches."

    # Find all match start positions (line-based)
    match_starts: list[int] = []
    search_len = len(search_lines)
    for i in range(len(lines) - search_len + 1):
        window = [l.rstrip("\n\r") for l in lines[i:i + search_len]]
        search_stripped = [l.rstrip("\n\r") for l in search_lines]
        if window == search_stripped:
            match_starts.append(i)

    if len(match_starts) < 2:
        # Couldn't reproduce matches line-by-line; give generic advice
        return f"Include more surrounding lines to disambiguate {match_count} matches."

    # Find minimum extra lines (before or after) needed to distinguish all matches
    for extra in range(1, 10):
        contexts: list[str] = []
        for start in match_starts:
            before_start = max(0, start - extra)
            after_end = min(len(lines), start + search_len + extra)
            ctx = "".join(lines[before_start:after_end])
            contexts.append(ctx)
        if len(set(contexts)) == len(contexts):
            return f"Include {extra}+ surrounding line(s) to make the SEARCH block unique."

    return f"Include 10+ surrounding lines to disambiguate {match_count} matches."


def validate_edits(edits: list[FileEdit]) -> list[tuple[FileEdit, str, str]]:
    """Validate edits against the filesystem.

    Returns a list of (edit, error_message, error_type) for invalid edits.
    error_type is one of the EDIT_ERR_* constants from models.py.
    Empty list means all edits are valid.
    """
    errors: list[tuple[FileEdit, str, str]] = []

    for edit in edits:
        if edit.is_new_file:
            # Check parent dir is writable
            parent = Path(edit.file).parent
            if parent != Path(".") and not parent.exists():
                try:
                    parent.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    errors.append((edit, f"Cannot create parent dir {parent}: {exc}", EDIT_ERR_GENERIC))
                    continue
            # Warn if file already exists (not a hard error)
            if Path(edit.file).exists():
                log.warning("New file target already exists: %s", edit.file)
            continue

        if edit.is_delete_file:
            if not Path(edit.file).exists():
                errors.append((edit, f"Cannot delete — file not found: {edit.file}", EDIT_ERR_FILE_MISSING))
            continue

        # File must exist for edits and whole-file replacements
        path = Path(edit.file)
        if not path.exists():
            errors.append((edit, f"File not found: {edit.file}", EDIT_ERR_FILE_MISSING))
            continue

        try:
            content = path.read_text(errors="replace")
        except OSError as exc:
            errors.append((edit, f"Cannot read {edit.file}: {exc}", EDIT_ERR_GENERIC))
            continue

        # Line-anchored edit: validate line range
        if edit.line_start is not None and edit.line_end is not None:
            total_lines = len(content.splitlines())
            if edit.line_start < 1 or edit.line_end < edit.line_start:
                errors.append((edit, f"Invalid line range {edit.line_start}-{edit.line_end} in {edit.file}", EDIT_ERR_LINE_RANGE))
            elif edit.line_end > total_lines:
                log.warning("Line range %d-%d exceeds file length %d in %s, clamping",
                            edit.line_start, edit.line_end, total_lines, edit.file)
            continue

        # Whole-file replacement: fill in search with full file content
        if not edit.search:
            edit.search = content
            continue

        # SEARCH/REPLACE: verify search text exists in file
        if edit.search in content:
            # Check uniqueness
            count = content.count(edit.search)
            if count > 1:
                hint = _calculate_context_hint(content, edit.search, count)
                errors.append((edit, (
                    f"SEARCH text matches {count} locations in {edit.file} — "
                    f"must be unique. {hint}"
                ), EDIT_ERR_MULTI_MATCH))
            continue

        # Try normalized matching
        norm_content = _normalize_whitespace(content)
        norm_search = _normalize_whitespace(edit.search)
        if norm_search and norm_search in norm_content:
            # Normalized match found — fuzzy replace will handle it
            log.info("Fuzzy whitespace match found for %s", edit.file)
            continue

        # Try indent-normalized matching (handles wrong indentation from 8B models)
        if _indent_fuzzy_replace(content, edit.search, edit.replace) is not None:
            log.info("Indent-fuzzy match found for %s", edit.file)
            continue

        # No match — build an informative error
        snippet = _best_partial_snippet(content, edit.search)
        errors.append((edit, (
            f"SEARCH text not found in {edit.file}. "
            f"Closest content near partial match:\n{snippet}"
        ), EDIT_ERR_NOT_FOUND))

    return errors


def count_edits_per_file(edits: list[FileEdit]) -> dict[str, int]:
    """Count how many edits target each file."""
    counts: dict[str, int] = {}
    for edit in edits:
        counts[edit.file] = counts.get(edit.file, 0) + 1
    return counts


def _sort_edits_bottom_up(edits: list[FileEdit]) -> list[FileEdit]:
    """Sort edits so later positions in each file are applied first.

    Prevents line-number drift when multiple edits target the same file.
    Edits across different files keep their relative order.
    New/delete file edits are placed at the end (order-insensitive).
    """
    def _edit_position(edit: FileEdit) -> int:
        """Estimate the line position of an edit (higher = later in file)."""
        if edit.is_new_file or edit.is_delete_file:
            return -1  # sentinel: sort last
        if edit.line_end is not None:
            return edit.line_end
        # SEARCH/REPLACE: estimate position by finding search text in file
        if edit.search:
            try:
                content = Path(edit.file).read_text(errors="replace")
                pos = content.find(edit.search)
                if pos >= 0:
                    return content[:pos].count("\n") + 1
            except OSError:
                pass
        return 0

    # Group by file, sort each group bottom-up, then flatten
    from collections import OrderedDict
    groups: OrderedDict[str, list[FileEdit]] = OrderedDict()
    for edit in edits:
        groups.setdefault(edit.file, []).append(edit)

    result: list[FileEdit] = []
    for _file, group in groups.items():
        group.sort(key=_edit_position, reverse=True)
        result.extend(group)
    return result


def apply_edits_atomic(edits: list[FileEdit]) -> tuple[int, list[str]]:
    """Apply all edits or none. Rolls back on any failure.

    Returns (count_applied, list_of_changed_file_paths).
    On failure returns (0, []).
    """
    # 0. Sort edits bottom-up to prevent line-number drift
    edits = _sort_edits_bottom_up(edits)

    # 1. Snapshot all target files
    snapshots: dict[str, str] = {}
    for edit in edits:
        if not edit.is_new_file and edit.file not in snapshots:
            path = Path(edit.file)
            if path.exists():
                try:
                    snapshots[edit.file] = path.read_text(errors="replace")
                except OSError:
                    pass

    # 2. Apply all edits
    applied = 0
    changed: list[str] = []
    for edit in edits:
        error = _apply_single_edit(edit)
        if error:
            log.warning("Atomic apply failed at edit %d: %s", applied + 1, error)
            # 3. Rollback: restore all snapshots, delete new files
            for fpath, content in snapshots.items():
                try:
                    Path(fpath).write_text(content)
                except OSError as exc:
                    log.error("Rollback failed for %s: %s", fpath, exc)
            for f in changed:
                if f not in snapshots:  # was a new file
                    p = Path(f)
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError as exc:
                            log.error("Rollback delete failed for %s: %s", f, exc)
            return 0, []
        applied += 1
        if edit.file not in changed:
            changed.append(edit.file)

    return applied, changed


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_outer_fences(text: str) -> str:
    """Remove markdown code fences wrapping edit blocks."""
    # Replace fenced blocks with their contents
    return _FENCE_RE.sub(r"\1", text)


def _normalize_whitespace(text: str) -> str:
    """Minimal normalization for fuzzy matching.

    - \\r\\n -> \\n
    - Strip trailing whitespace from each line
    - Strip trailing newlines

    Does NOT change leading whitespace (indentation).
    """
    text = text.replace("\r\n", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).rstrip("\n")


def _detect_indent(line: str) -> str:
    """Return the leading whitespace of a line."""
    return line[:len(line) - len(line.lstrip())]


def _indent_fuzzy_replace(content: str, search: str, replace: str) -> str | None:
    """Replace using indent-normalized matching.

    Strips leading whitespace for matching, then re-applies the file's
    indentation to the replacement. Handles the common 8B model failure
    where SEARCH text has wrong indentation but correct content.

    Returns the modified content, or None if no match found.
    """
    content_lines = content.replace("\r\n", "\n").split("\n")
    search_lines = search.replace("\r\n", "\n").split("\n")
    replace_lines = replace.replace("\r\n", "\n").split("\n")

    # Strip trailing empty lines from search
    while search_lines and not search_lines[-1].strip():
        search_lines.pop()

    if not search_lines:
        return None

    # Compare with all leading whitespace stripped
    stripped_content = [line.lstrip() for line in content_lines]
    stripped_search = [line.lstrip() for line in search_lines]

    # Find matching range
    match_start = None
    for i in range(len(stripped_content) - len(stripped_search) + 1):
        if stripped_content[i:i + len(stripped_search)] == stripped_search:
            match_start = i
            break

    if match_start is None:
        return None

    # Detect base indent from file's matched region (first non-empty line)
    file_indent = ""
    for line in content_lines[match_start:match_start + len(search_lines)]:
        if line.strip():
            file_indent = _detect_indent(line)
            break

    # Detect base indent from replacement (first non-empty line)
    replace_indent = ""
    for line in replace_lines:
        if line.strip():
            replace_indent = _detect_indent(line)
            break

    # Re-indent replacement: preserve relative indentation, shift to file's base
    reindented: list[str] = []
    for line in replace_lines:
        if not line.strip():
            reindented.append("")
        elif line.startswith(replace_indent):
            relative = line[len(replace_indent):]
            reindented.append(file_indent + relative)
        else:
            reindented.append(file_indent + line.lstrip())

    result_lines = content_lines[:match_start] + reindented + content_lines[match_start + len(search_lines):]
    return "\n".join(result_lines)


def _fuzzy_replace(content: str, search: str, replace: str) -> str | None:
    """Replace using normalized whitespace matching.

    Returns the modified content, or None if no match found.
    """
    norm_content = _normalize_whitespace(content)
    norm_search = _normalize_whitespace(search)

    if not norm_search or norm_search not in norm_content:
        return None

    # Find matching line range in normalized form
    content_lines = content.replace("\r\n", "\n").split("\n")
    search_lines = [line.rstrip() for line in search.replace("\r\n", "\n").split("\n")]

    # Strip trailing empty lines from search
    while search_lines and not search_lines[-1]:
        search_lines.pop()

    norm_content_lines = [line.rstrip() for line in content_lines]

    # Find the start line
    match_start = None
    for i in range(len(norm_content_lines) - len(search_lines) + 1):
        if norm_content_lines[i:i + len(search_lines)] == search_lines:
            match_start = i
            break

    if match_start is None:
        return None

    # Replace original lines with replacement
    replace_lines = replace.replace("\r\n", "\n").split("\n")
    result_lines = content_lines[:match_start] + replace_lines + content_lines[match_start + len(search_lines):]
    return "\n".join(result_lines)


def _fuzzy_anchor_replace(content: str, search: str, replace: str) -> str | None:
    """Replace using anchor-based fuzzy matching.

    Extracts the first and last non-blank, non-comment lines from SEARCH as
    anchors. Finds those anchors in the file content (stripped comparison) and
    replaces the region between them with the replacement text.

    Handles the case where 8B models produce correct core edits but
    surrounding context lines have drifted (added/removed blank lines,
    changed comments, etc.).

    Returns modified content, or None if anchors can't be matched.
    """
    content_lines = content.replace("\r\n", "\n").split("\n")
    search_lines = search.replace("\r\n", "\n").split("\n")
    replace_lines = replace.replace("\r\n", "\n").split("\n")

    # Strip trailing empty lines from search
    while search_lines and not search_lines[-1].strip():
        search_lines.pop()

    if len(search_lines) < 2:
        return None  # need at least 2 lines for meaningful anchors

    # Extract first and last non-blank, non-comment anchors
    _COMMENT_PREFIXES = ("#", "//", "/*", "*", "<!--")

    def _is_anchor(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        return not any(s.startswith(p) for p in _COMMENT_PREFIXES)

    first_anchor = None
    first_anchor_idx = -1
    for i, line in enumerate(search_lines):
        if _is_anchor(line):
            first_anchor = line.strip()
            first_anchor_idx = i
            break

    last_anchor = None
    last_anchor_idx = -1
    for i in range(len(search_lines) - 1, -1, -1):
        if _is_anchor(search_lines[i]):
            last_anchor = search_lines[i].strip()
            last_anchor_idx = i
            break

    if first_anchor is None or last_anchor is None:
        return None
    if first_anchor_idx >= last_anchor_idx:
        return None  # same line — need distinct first/last

    # Find anchors in file content (stripped comparison)
    stripped_content = [line.strip() for line in content_lines]

    first_matches = [i for i, s in enumerate(stripped_content) if s == first_anchor]
    last_matches = [i for i, s in enumerate(stripped_content) if s == last_anchor]

    if not first_matches or not last_matches:
        return None

    # Find the best (first_pos, last_pos) pair where last > first and
    # region size is close to search block size
    expected_span = last_anchor_idx - first_anchor_idx
    best_pair = None
    best_score = float("inf")
    for fi in first_matches:
        for li in last_matches:
            if li <= fi:
                continue
            actual_span = li - fi
            # Reject if region is wildly different from expected (>3x or <0.3x)
            if actual_span > expected_span * 3 or actual_span < max(1, expected_span // 3):
                continue
            score = abs(actual_span - expected_span)
            if score < best_score:
                best_score = score
                best_pair = (fi, li)

    if best_pair is None:
        return None

    file_start, file_end = best_pair

    # Detect file indent from matched region
    file_indent = ""
    for line in content_lines[file_start:file_end + 1]:
        if line.strip():
            file_indent = _detect_indent(line)
            break

    # Detect replace indent
    replace_indent = ""
    for line in replace_lines:
        if line.strip():
            replace_indent = _detect_indent(line)
            break

    # Re-indent replacement
    reindented: list[str] = []
    for line in replace_lines:
        if not line.strip():
            reindented.append("")
        elif line.startswith(replace_indent):
            relative = line[len(replace_indent):]
            reindented.append(file_indent + relative)
        else:
            reindented.append(file_indent + line.lstrip())

    result_lines = content_lines[:file_start] + reindented + content_lines[file_end + 1:]
    log.info("Applied anchor-fuzzy match in file (anchors: %r ... %r)",
             first_anchor[:40], last_anchor[:40])
    return "\n".join(result_lines)


def _apply_single_edit(edit: FileEdit) -> str | None:
    """Apply a single edit to disk.

    Returns error message on failure, None on success.
    """
    path = Path(edit.file)

    # New file
    if edit.is_new_file:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(edit.replace)
            log.info("Created new file: %s", edit.file)
            return None
        except OSError as exc:
            return f"Failed to create {edit.file}: {exc}"

    # Delete file
    if edit.is_delete_file:
        try:
            if path.exists():
                path.unlink()
                log.info("Deleted file: %s", edit.file)
            return None
        except OSError as exc:
            return f"Failed to delete {edit.file}: {exc}"

    # Read current content
    try:
        content = path.read_text(errors="replace")
    except OSError as exc:
        return f"Cannot read {edit.file}: {exc}"

    # Line-anchored replacement: replace specific line range
    if edit.line_start is not None and edit.line_end is not None:
        lines = content.splitlines(keepends=True)
        total = len(lines)
        start = max(0, edit.line_start - 1)  # 1-based to 0-based
        end = min(total, edit.line_end)       # inclusive end
        if start >= total:
            return f"Line {edit.line_start} out of range (file has {total} lines): {edit.file}"
        replace_text = edit.replace
        if lines and lines[-1].endswith("\n") and not replace_text.endswith("\n"):
            replace_text += "\n"
        result_lines = lines[:start] + [replace_text] + lines[end:]
        try:
            path.write_text("".join(result_lines))
            log.info("Applied line-anchored edit (L%d-%d) to %s", edit.line_start, edit.line_end, edit.file)
            return None
        except OSError as exc:
            return f"Cannot write {edit.file}: {exc}"

    # Empty search = whole-file replacement
    if not edit.search:
        try:
            path.write_text(edit.replace)
            log.info("Whole-file replacement: %s", edit.file)
            return None
        except OSError as exc:
            return f"Cannot write {edit.file}: {exc}"

    # SEARCH/REPLACE: try exact match first
    if edit.search in content:
        count = content.count(edit.search)
        if count > 1:
            return (
                f"SEARCH text matches {count} locations in {edit.file} — "
                "must be unique. Add more context lines."
            )
        result = content.replace(edit.search, edit.replace, 1)
        try:
            path.write_text(result)
            log.info("Applied SEARCH/REPLACE to %s", edit.file)
            return None
        except OSError as exc:
            return f"Cannot write {edit.file}: {exc}"

    # Try fuzzy replace (trailing whitespace normalization)
    result = _fuzzy_replace(content, edit.search, edit.replace)
    if result is not None:
        try:
            path.write_text(result)
            log.info("Applied fuzzy SEARCH/REPLACE to %s", edit.file)
            return None
        except OSError as exc:
            return f"Cannot write {edit.file}: {exc}"

    # Try indent-aware fuzzy replace (leading whitespace normalization + re-indent)
    result = _indent_fuzzy_replace(content, edit.search, edit.replace)
    if result is not None:
        try:
            path.write_text(result)
            log.info("Applied indent-fuzzy SEARCH/REPLACE to %s", edit.file)
            return None
        except OSError as exc:
            return f"Cannot write {edit.file}: {exc}"

    # Try anchor-based fuzzy replace (first/last non-comment lines as anchors)
    result = _fuzzy_anchor_replace(content, edit.search, edit.replace)
    if result is not None:
        try:
            path.write_text(result)
            log.info("Applied anchor-fuzzy SEARCH/REPLACE to %s", edit.file)
            return None
        except OSError as exc:
            return f"Cannot write {edit.file}: {exc}"

    return f"SEARCH text not found in {edit.file} (exact, fuzzy, indent-fuzzy, and anchor-fuzzy match failed)"


def _best_partial_snippet(content: str, search: str, context_lines: int = 3) -> str:
    """Find the best partial match and return a snippet of surrounding content.

    Used for error messages when SEARCH text isn't found.
    """
    content_lines_list = content.split("\n")
    search_lines = search.strip().split("\n")

    if not search_lines:
        return "(empty search)"

    # Use the first non-empty line of search as anchor
    anchor = ""
    for line in search_lines:
        stripped = line.strip()
        if stripped:
            anchor = stripped
            break

    if not anchor:
        return "(blank search text)"

    # Find best matching line
    best_idx = 0
    best_score = 0
    for i, line in enumerate(content_lines_list):
        # Simple substring overlap scoring
        score = 0
        for word in anchor.split():
            if word in line:
                score += len(word)
        if score > best_score:
            best_score = score
            best_idx = i

    start = max(0, best_idx - context_lines)
    end = min(len(content_lines_list), best_idx + context_lines + 1)
    snippet_lines = content_lines_list[start:end]

    return "\n".join(f"  {start + i + 1:4d} | {line}" for i, line in enumerate(snippet_lines))
