"""Automatic file selection for agent context.

Before invoking the agent, calls a fast LLM with the task + git ls-files listing
to pick which files are needed. Falls back to keyword extraction if the LLM
is unavailable.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path


from probablyfine.log_utils import get_module_logger
from probablyfine.ollama_utils import (
    build_chat_options,
    create_client,
    extract_content as _extract_content,
    get_prompt_suffix,
    log_token_usage,
    parse_llm_json,
)

log = get_module_logger("probablyfine.file_selector", "file_selector.log")

_SELECTOR_NUM_PREDICT = 500     # Max tokens for file selector response
_BYTES_PER_TOKEN_APPROX = 4000  # Rough bytes-to-tokens divisor for budget logging
_DEFAULT_MAX_GIT_FILES = 500    # Cap on git ls-files results
_DEFAULT_MAX_CONTEXT_BYTES = 48000  # Default context budget for file content

FILE_SELECTOR_PROMPT = """\
You are a file selector for a coding assistant. Given a task and a list of \
files in a repository, select which files the assistant needs.

Rules:
- Select ONLY files from the list below — do not invent or guess paths
- Do NOT include README.md unless the task explicitly mentions documentation
- Include files directly mentioned in the task
- Include related files (e.g., tests for a modified module)
- If unsure, include the file
- Respond with ONLY a JSON array of file paths

Task: {task}

Repository files:
{file_list}

Selected files:"""


def _get_git_files(max_files: int = _DEFAULT_MAX_GIT_FILES) -> list[str]:
    """Return tracked files from git ls-files, capped at max_files.

    Returns [] if not in a git repo or git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        files = [f for f in result.stdout.strip().splitlines() if f]
        return files[:max_files]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def _keyword_file_extract(task: str, git_files: list[str]) -> list[str]:
    """Fast path: extract file-like tokens from the task and match against git files.

    Handles cases like 'update README.md' without an LLM call.
    """
    # Extract tokens that look like filenames (contain a dot with an extension)
    tokens = re.findall(r'[\w./\\-]+\.\w+', task)
    if not tokens:
        return []

    # Build lookup sets for matching
    basename_map: dict[str, list[str]] = {}
    for gf in git_files:
        base = os.path.basename(gf)
        basename_map.setdefault(base, []).append(gf)

    git_set = set(git_files)
    matched: list[str] = []

    for token in tokens:
        # Normalize separators
        normalized = token.replace("\\", "/")

        # Direct full-path match
        if normalized in git_set:
            if normalized not in matched:
                matched.append(normalized)
            continue

        # Basename match
        base = os.path.basename(normalized)
        if base in basename_map:
            for full_path in basename_map[base]:
                if full_path not in matched:
                    matched.append(full_path)

    return matched


def _parse_file_list(raw: str, git_files: list[str], task: str = "") -> list[str] | None:
    """Parse LLM response into a list of file paths, validated against git files.

    Uses fuzzy basename matching to recover near-miss hallucinations before dropping.
    """
    data = parse_llm_json(raw)

    if not isinstance(data, list):
        return None

    # Build basename lookup for fuzzy recovery
    git_set = set(git_files)
    basename_map: dict[str, list[str]] = {}
    for gf in git_files:
        base = os.path.basename(gf)
        basename_map.setdefault(base, []).append(gf)

    task_lower = task.lower()
    valid = []
    for item in data:
        path = str(item).strip().replace("\\", "/")
        if path in git_set:
            valid.append(path)
            continue

        # Skip README.md unless task mentions documentation
        base = os.path.basename(path)
        if base.lower() == "readme.md" and "readme" not in task_lower and "doc" not in task_lower:
            log.info("Dropping hallucinated README.md (task doesn't mention docs)")
            continue

        # Fuzzy recovery: try basename match
        candidates = basename_map.get(base, [])
        if len(candidates) == 1:
            recovered = candidates[0]
            log.info("Recovered hallucinated path: %s -> %s", path, recovered)
            if recovered not in valid:
                valid.append(recovered)
        elif len(candidates) > 1:
            log.warning("Dropping ambiguous hallucinated path: %s (%d matches)", path, len(candidates))
        else:
            log.warning("Dropping hallucinated path: %s", path)

    return valid if valid else None


def _keyword_glob_fallback(task: str, git_files: list[str]) -> list[str]:
    """Match task keywords against git filenames when LLM selection fails."""
    words = set(re.findall(r'[a-z]{3,}', task.lower()))
    words -= {"the", "and", "for", "that", "this", "with", "from", "are", "was",
              "will", "should", "could", "would", "have", "has", "been", "into"}
    matched = []
    for gf in git_files:
        base = os.path.basename(gf).lower()
        stem = os.path.splitext(base)[0]
        if any(w in stem for w in words):
            matched.append(gf)
    return matched[:10]


def _model_select_files(
    task: str,
    git_files: list[str],
    model: str,
    timeout: int = 10,
) -> list[str] | None:
    """Call the fast LLM to select relevant files.

    Returns None on any failure (timeout, parse error, ollama down).
    """
    file_list = "\n".join(git_files)
    prompt = FILE_SELECTOR_PROMPT.format(task=task, file_list=file_list) + get_prompt_suffix(model, "file_selector")

    try:
        options = build_chat_options(model=model, num_predict=_SELECTOR_NUM_PREDICT)

        client = create_client(timeout=timeout)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
        )

        raw = _extract_content(response)
        log_token_usage("file_selector", model, response)
        if not raw.strip():
            log.warning("File selector LLM returned empty response")
            return None

        log.debug("File selector raw response: %s", raw[:500])
        result = _parse_file_list(raw, git_files, task=task)
        if result:
            log.info("LLM selected %d files: %s", len(result), result)
            return result

        # LLM hallucinated all paths — try keyword matching
        log.info("All LLM paths hallucinated, trying keyword fallback")
        fallback = _keyword_glob_fallback(task, git_files)
        if fallback:
            log.info("Keyword fallback selected %d files: %s", len(fallback), fallback)
            return fallback
        return None

    except KeyboardInterrupt:
        log.info("File selection interrupted by user")
        return None
    except Exception as e:
        log.debug("File selector LLM error: %s", e)
        return None


def _filter_by_budget(
    files: list[str],
    max_bytes: int,
    existing_bytes: int = 0,
) -> list[str]:
    """Filter files by cumulative size budget.

    Returns files that fit within max_bytes (accounting for existing context).
    Files are included in order until the budget is exhausted.
    """
    if max_bytes <= 0:
        return files

    remaining = max_bytes - existing_bytes
    kept: list[str] = []
    for f in files:
        try:
            size = Path(f).stat().st_size
        except OSError:
            size = 0
        if size > remaining:
            log.info(
                "Excluding %s (%d bytes, ~%dk tokens) — exceeds remaining budget (%d bytes)",
                f, size, size // _BYTES_PER_TOKEN_APPROX, remaining,
            )
            continue
        remaining -= size
        kept.append(f)
    return kept


_MAX_IMPORT_CHAIN_FILES = 20  # Cap total files from import chain following


def _extract_imports(fpath: str) -> list[str]:
    """Extract imported module names from a Python file via ast."""
    try:
        source = Path(fpath).read_text(errors="replace")
        tree = ast.parse(source, filename=fpath)
    except (SyntaxError, OSError):
        return []

    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


def _follow_import_chain(
    git_files: list[str],
    seed_files: list[str],
    max_depth: int = 1,
) -> list[str]:
    """Follow import chains from seed files to discover additional relevant files.

    Uses ast to extract imports, then maps module paths back to repo files.
    BFS to max_depth, capped at _MAX_IMPORT_CHAIN_FILES total.
    """
    # Build module-to-file mapping from git files
    module_map: dict[str, str] = {}
    for gf in git_files:
        if not gf.endswith(".py"):
            continue
        # Convert path to dotted module: probablyfine/agent.py -> probablyfine.agent
        mod = gf.replace("/", ".").replace("\\", ".")
        if mod.endswith(".py"):
            mod = mod[:-3]
        if mod.endswith(".__init__"):
            mod = mod[:-9]
        module_map[mod] = gf
        # Also map the basename: agent -> probablyfine/agent.py (for relative imports)
        parts = mod.split(".")
        if len(parts) > 1:
            module_map[parts[-1]] = gf

    seed_set = set(seed_files)
    discovered: list[str] = []
    visited: set[str] = set(seed_files)
    queue = list(seed_files)

    for _depth in range(max_depth):
        next_queue: list[str] = []
        for fpath in queue:
            if not fpath.endswith(".py"):
                continue
            imports = _extract_imports(fpath)
            for mod_name in imports:
                # Try full module name, then progressively shorter prefixes
                matched = module_map.get(mod_name)
                if not matched:
                    # Try last component: probablyfine.models -> models
                    short = mod_name.split(".")[-1]
                    matched = module_map.get(short)
                if matched and matched not in visited:
                    visited.add(matched)
                    discovered.append(matched)
                    next_queue.append(matched)
                    if len(discovered) >= _MAX_IMPORT_CHAIN_FILES:
                        log.info("Import chain cap reached (%d files)", len(discovered))
                        return discovered
        queue = next_queue

    if discovered:
        log.info("Import chain discovered %d additional files: %s",
                 len(discovered), discovered[:5])
    return discovered


def select_files(
    task: str,
    model: str,
    existing_files: list[str] | None = None,
    max_git_files: int = _DEFAULT_MAX_GIT_FILES,
    max_context_bytes: int = _DEFAULT_MAX_CONTEXT_BYTES,
) -> list[str] | None:
    """Select files for agent context using keyword extraction + LLM.

    Merges results with any manually /add'd files. Returns absolute paths
    matching FileContext conventions. Returns None if no files to add.
    """
    git_files = _get_git_files(max_files=max_git_files)
    if not git_files:
        return existing_files

    # Keyword extraction (fast, always works)
    keyword_files = _keyword_file_extract(task, git_files)

    # LLM selection (may fail gracefully)
    llm_files = _model_select_files(task, git_files, model)

    # Merge keyword + LLM results
    selected: list[str] = []
    for f in keyword_files:
        if f not in selected:
            selected.append(f)
    if llm_files:
        for f in llm_files:
            if f not in selected:
                selected.append(f)

    if not selected:
        return existing_files

    # Follow import chains from selected files to discover dependencies
    import_files = _follow_import_chain(git_files, selected)
    for f in import_files:
        if f not in selected:
            selected.append(f)

    # Convert to absolute paths (matching FileContext convention)
    cwd = Path.cwd()
    abs_selected = [str((cwd / f).resolve()) for f in selected]

    # Budget-aware filtering: skip files that would blow the context window
    existing_bytes = 0
    if existing_files:
        for ef in existing_files:
            try:
                existing_bytes += Path(ef).stat().st_size
            except OSError:
                pass
    abs_selected = _filter_by_budget(abs_selected, max_context_bytes, existing_bytes)

    if not abs_selected:
        return existing_files

    # Deduplicate with manually added files
    if existing_files:
        existing_set = set(existing_files)
        merged = list(existing_files)
        for f in abs_selected:
            if f not in existing_set:
                merged.append(f)
                existing_set.add(f)
        return merged

    return abs_selected
