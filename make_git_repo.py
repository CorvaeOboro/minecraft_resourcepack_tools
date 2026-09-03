"""Initialize this folder as a git repository linked to GitHub.

Run from the repository root:

    python make_git_repo.py

What it does:
1. `git init`
2. Set the default branch to `main`
3. Add the remote `origin` -> https://github.com/CorvaeOboro/minecraft_resourcepack_tools.git
4. Write a minimal `.gitignore` (Python + common local artifacts) if none exists

This script does NOT add, commit, or push any files. It only creates the
`.git` directory and configures the remote so you can stage and commit
selectively afterward.

This script is idempotent: if a remote named `origin` already exists it
updates the URL instead of adding a duplicate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REMOTE_NAME = "origin"
REMOTE_URL = "https://github.com/CorvaeOboro/minecraft_resourcepack_tools.git"
DEFAULT_BRANCH = "main"

GITIGNORE_CONTENT = """\
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.eggs/
build/
dist/
.venv/
venv/
env/

# Pytest / coverage
.pytest_cache/
.coverage
htmlcov/

# IDE / editor
.vscode/
.idea/
*.swp
*.swo
*~

# OS
Thumbs.db
desktop.ini
.DS_Store

# Local artifacts
*.blend1
*.blend2
*_OPT.json
"""


def run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)
    if check and result.returncode != 0:
        raise SystemExit(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    if not (repo_root / ".git").exists():
        run(["git", "init"], cwd=repo_root)
    else:
        print("git repository already exists, skipping init")

    # set default branch
    run(["git", "symbolic-ref", "HEAD", f"refs/heads/{DEFAULT_BRANCH}"], cwd=repo_root)

    # configure remote
    remotes = run(["git", "remote"], cwd=repo_root).stdout.split()
    if REMOTE_NAME in remotes:
        run(["git", "remote", "set-url", REMOTE_NAME, REMOTE_URL], cwd=repo_root)
    else:
        run(["git", "remote", "add", REMOTE_NAME, REMOTE_URL], cwd=repo_root)

    # write .gitignore if missing
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(GITIGNORE_CONTENT, encoding="utf-8")
        print(f"Wrote {gitignore.name}")
    else:
        print(".gitignore already exists, skipping")

    # show final state
    run(["git", "remote", "-v"], cwd=repo_root, check=False)

    print()
    print("Repository initialized (no commits made).")
    print(f"Remote: {REMOTE_URL}")
    print()
    print("To stage and commit files selectively:")
    print("  git add <file>")
    print('  git commit -m "Initial commit"')
    print()
    print("To push to GitHub:")
    print(f'  git push -u {REMOTE_NAME} {DEFAULT_BRANCH}')


if __name__ == "__main__":
    main()
