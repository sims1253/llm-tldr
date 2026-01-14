"""TLDR ignore file handling (.tldrignore + .gitignore).

Provides gitignore-style pattern matching for excluding files from indexing.
Uses pathspec library for gitignore-compatible pattern matching.

Precedence (highest to lowest):
1. .tldrignore patterns (explicit include/exclude)
2. .gitignore patterns (via git check-ignore, if in git repo)
3. Default patterns (if no .tldrignore exists)
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathspec import PathSpec

# Default .tldrignore template
DEFAULT_TEMPLATE = """\
# TLDR ignore patterns (gitignore syntax)
# Auto-generated - review and customize for your project
# Docs: https://git-scm.com/docs/gitignore

# ===================
# Dependencies
# ===================
node_modules/
.venv/
venv/
env/
__pycache__/
.tox/
.nox/
.pytest_cache/
.mypy_cache/
.ruff_cache/
vendor/
Pods/

# ===================
# Build outputs
# ===================
dist/
build/
out/
target/
*.egg-info/
*.whl
*.pyc
*.pyo

# ===================
# Binary/large files
# ===================
*.so
*.dylib
*.dll
*.exe
*.bin
*.o
*.a
*.lib

# ===================
# IDE/editors
# ===================
.idea/
.vscode/
*.swp
*.swo
*~

# ===================
# Security (always exclude)
# ===================
.env
.env.*
*.pem
*.key
*.p12
*.pfx
credentials.*
secrets.*

# ===================
# Version control
# ===================
.git/
.hg/
.svn/

# ===================
# OS files
# ===================
.DS_Store
Thumbs.db

# ===================
# Project-specific
# Add your custom patterns below
# ===================
# large_test_fixtures/
# data/
"""


@lru_cache(maxsize=128)
def is_git_repo(project_dir: str) -> bool:
    """Check if directory is inside a git repository.

    Args:
        project_dir: Directory path to check

    Returns:
        True if inside a git repo, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=project_dir,
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # git not installed, timeout, or other error
        return False


def is_gitignored(file_path: str | Path, project_dir: str | Path) -> bool:
    """Check if a file is ignored by .gitignore using git check-ignore.

    This handles all gitignore complexity including:
    - Nested .gitignore files
    - Pattern precedence
    - Negation patterns (!)
    - Directory-relative patterns

    Args:
        file_path: Path to the file to check
        project_dir: Root directory of the git repo

    Returns:
        True if file is gitignored, False otherwise
    """
    project_path = Path(project_dir)
    file_path = Path(file_path)

    # Make path relative for git check-ignore
    try:
        rel_path = file_path.relative_to(project_path)
    except ValueError:
        rel_path = file_path

    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(rel_path)],
            cwd=str(project_path),
            capture_output=True,
            timeout=5,
        )
        # Return code 0 = ignored, 1 = not ignored, 128 = error
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def load_ignore_patterns(project_dir: str | Path) -> "PathSpec":
    """Load ignore patterns from .tldrignore file.

    Args:
        project_dir: Root directory of the project

    Returns:
        PathSpec matcher for checking if files should be ignored
    """
    import pathspec

    project_path = Path(project_dir)
    tldrignore_path = project_path / ".tldrignore"

    patterns: list[str] = []

    if tldrignore_path.exists():
        content = tldrignore_path.read_text()
        patterns: list[str] = content.splitlines()
    else:
        # Use defaults if no .tldrignore exists
        patterns = list(DEFAULT_TEMPLATE.splitlines())

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def ensure_tldrignore(project_dir: str | Path) -> tuple[bool, str]:
    """Ensure .tldrignore exists, creating with defaults if needed.

    Args:
        project_dir: Root directory of the project

    Returns:
        Tuple of (created: bool, message: str)
    """
    project_path = Path(project_dir)

    if not project_path.exists():
        return False, f"Project directory does not exist: {project_path}"

    tldrignore_path = project_path / ".tldrignore"

    if tldrignore_path.exists():
        return False, f".tldrignore already exists at {tldrignore_path}"

    # Create with default template
    tldrignore_path.write_text(DEFAULT_TEMPLATE)

    return (
        True,
        """Created .tldrignore with sensible defaults:
  - node_modules/, .venv/, __pycache__/
  - dist/, build/, *.egg-info/
  - Binary files (*.so, *.dll, *.whl)
  - Security files (.env, *.pem, *.key)

Review .tldrignore before indexing large codebases.
Edit to exclude vendor code, test fixtures, etc.""",
    )


def should_ignore(
    file_path: str | Path,
    project_dir: str | Path,
    spec: "PathSpec | None" = None,
    use_gitignore: bool = True,
) -> bool:
    """Check if a file should be ignored.

    Precedence:
    1. .gitignore provides baseline (if in git repo)
    2. .tldrignore overrides - can add ignores OR un-ignore via ! patterns

    Args:
        file_path: Path to check (absolute or relative)
        project_dir: Root directory of the project
        spec: Optional pre-loaded PathSpec (for efficiency in loops)
        use_gitignore: Whether to also check .gitignore (default True)

    Returns:
        True if file should be ignored, False otherwise
    """
    if spec is None:
        spec = load_ignore_patterns(project_dir)

    project_path = Path(project_dir)
    file_path = Path(file_path)

    # Make path relative to project for matching
    try:
        rel_path = file_path.relative_to(project_path)
    except ValueError:
        # File is not under project_dir, use as-is
        rel_path = file_path

    rel_path_str = str(rel_path)

    # .tldrignore is the final authority - it can:
    # - Add ignores (positive patterns)
    # - Un-ignore gitignored files (! negation patterns)
    #
    # pathspec.match_file returns True if file matches a positive pattern
    # and wasn't subsequently un-matched by a negation pattern
    tldr_ignored = spec.match_file(rel_path_str)

    # Check if .tldrignore has an explicit opinion via negation
    # by checking if any negation pattern matches this file
    has_negation = _has_negation_for_file(spec, rel_path_str)

    if has_negation:
        # .tldrignore explicitly un-ignores this file - respect that
        return tldr_ignored

    if tldr_ignored:
        # .tldrignore says ignore
        return True

    # .tldrignore has no opinion - check gitignore as fallback
    if use_gitignore and is_git_repo(str(project_path)):
        return is_gitignored(file_path, project_path)

    return False


def _has_negation_for_file(spec: "PathSpec", rel_path: str) -> bool:
    """Check if any negation pattern in the spec would match this file.

    This helps determine if .tldrignore has an explicit opinion about
    including a file (via ! pattern) vs simply not matching it.
    """
    for pattern in spec.patterns:
        # Check if this is a negation pattern
        if hasattr(pattern, 'regex') and pattern.pattern.startswith('!'):
            # This is a negation pattern - check if it matches
            if pattern.match_file(rel_path):
                return True
    return False


def filter_files(
    files: list[Path],
    project_dir: str | Path,
    respect_ignore: bool = True,
    use_gitignore: bool = True,
) -> list[Path]:
    """Filter a list of files, removing those matching ignore patterns.

    Checks both .tldrignore and .gitignore (if in a git repo).
    .tldrignore patterns take precedence over .gitignore.

    Args:
        files: List of file paths to filter
        project_dir: Root directory of the project
        respect_ignore: If False, skip filtering (--no-ignore mode)
        use_gitignore: Whether to also check .gitignore (default True)

    Returns:
        Filtered list of files
    """
    if not respect_ignore:
        return files

    spec = load_ignore_patterns(project_dir)
    return [f for f in files if not should_ignore(f, project_dir, spec, use_gitignore)]
