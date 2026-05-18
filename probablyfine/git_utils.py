import subprocess


def git_status() -> tuple[int, str]:
    """Run git status and return (exit_code, output)."""
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            capture_output=True, text=True,
        )
        return result.returncode, result.stdout.strip()
    except FileNotFoundError:
        return 1, "git not found"


def git_undo_last_commit() -> tuple[int, str]:
    """Soft-reset the last commit, keeping changes staged.

    Returns (exit_code, message).
    """
    try:
        # Check there's at least one commit
        check = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            return 1, "No commits to undo."

        # Check this isn't the root commit (HEAD~1 won't exist)
        parent_check = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD~1"],
            capture_output=True, text=True,
        )
        if parent_check.returncode != 0:
            return 1, "Cannot undo: this is the first commit in the repository."

        # Get the commit message we're about to undo
        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True, text=True,
        )
        commit_msg = log.stdout.strip()

        result = subprocess.run(
            ["git", "reset", "--soft", "HEAD~1"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return 0, f"Undid commit: {commit_msg}\nChanges are now staged."
        return result.returncode, result.stderr.strip()
    except FileNotFoundError:
        return 1, "git not found"


def git_diff_stat() -> tuple[int, str]:
    """Show a short diff stat of uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True,
        )
        return result.returncode, result.stdout.strip()
    except FileNotFoundError:
        return 1, "git not found"


def git_branch_status() -> tuple[str, bool]:
    """Return (branch_name, is_dirty) cheaply for toolbar display.

    Returns ("", False) if not in a git repo or git is unavailable.
    """
    try:
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if branch_result.returncode != 0:
            return ("", False)
        branch = branch_result.stdout.strip()

        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=2,
        )
        is_dirty = bool(dirty_result.stdout.strip())

        return (branch, is_dirty)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ("", False)
