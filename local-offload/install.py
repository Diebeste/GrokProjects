#!/usr/bin/env python3
"""Copy a local-offload git checkout onto this machine.

Pass a repo URL, or the path of a git checkout the host already has.
The default --repo is https://github.com/Diebeste/GrokProjects, which
keeps this skill in its local-offload subfolder.
The files land in ~/.local-offload. Both skills are copied into the
Cursor and Codex skill directories. This script does not know a store path.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "https://github.com/Diebeste/GrokProjects"
SKILL_SUBDIR = "local-offload"

PAYLOAD = (
    "server.py",
    "config.json",
    "test_server.py",
    "README.md",
    "install.py",
    "SKILL.md",
    "codex/SKILL.md",
)


def default_home() -> Path:
    return Path.home()


def runtime_dir(home: Path) -> Path:
    return home / ".local-offload"


def clone_dir(home: Path) -> Path:
    return home / ".cache" / "local-offload" / "repo"


def cursor_skill(home: Path) -> Path:
    return home / ".cursor" / "skills" / "local-offload" / "SKILL.md"


def codex_skill(home: Path) -> Path:
    return home / ".codex" / "skills" / "local-offload" / "SKILL.md"


def is_git_checkout(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def origin_url(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"no origin remote in {path}")
    return result.stdout.strip()


def clone_or_pull(repo: str, cache: Path) -> Path:
    if cache.exists() and not is_git_checkout(cache):
        raise SystemExit(f"{cache} exists and is not a git checkout")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["git", "clone", "--", repo, str(cache)],
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(f"git clone failed for {repo}")
        return cache
    current = origin_url(cache)
    if current != repo:
        raise SystemExit(f"{cache} already tracks {current}")
    completed = subprocess.run(
        ["git", "-C", str(cache), "pull", "--ff-only"],
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"git pull failed in {cache}")
    return cache


def copy_payload(source: Path, dest: Path) -> None:
    missing = [rel for rel in PAYLOAD if not (source / rel).is_file()]
    if missing:
        raise SystemExit(f"checkout {source} is missing {', '.join(missing)}")
    for rel in PAYLOAD:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, target)


def copy_skills(source: Path, home: Path) -> None:
    pairs = (
        (source / "SKILL.md", cursor_skill(home)),
        (source / "codex" / "SKILL.md", codex_skill(home)),
    )
    for src, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def next_commands(dest: Path, home: Path) -> str:
    return "\n".join(
        [
            f"Installed to {dest}",
            f"Cursor skill: {cursor_skill(home)}",
            f"Codex skill: {codex_skill(home)}",
            "",
            f"cd {dest}",
            "python3 server.py",
            "",
            "curl -sS http://127.0.0.1:8765/tier",
            "curl -sS http://127.0.0.1:8765/job \\",
            "  -H 'Content-Type: application/json' \\",
            '  -d \'{"instruction":"Write a Python function add(a, b) that returns the sum.","test":"add(1, 2) is 3"}\'',
            "",
        ]
    )


def _option_given(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def _same_repo(left: str, right: str) -> bool:
    def norm(url: str) -> str:
        return url.rstrip("/").removesuffix(".git")

    return norm(left) == norm(right)


def skill_source(checkout: Path, repo: str) -> Path:
    if not _same_repo(repo, DEFAULT_REPO):
        return checkout
    nested = checkout / SKILL_SUBDIR
    if not nested.is_dir():
        raise SystemExit(f"{checkout} has no {SKILL_SUBDIR} subfolder")
    return nested


def resolve_source(
    repo: str | None,
    path: str | None,
    home: Path,
    *,
    repo_given: bool,
) -> Path:
    if path and repo_given:
        raise SystemExit("pass either --repo or --path")
    if path:
        checkout = Path(path).expanduser().resolve()
        if not is_git_checkout(checkout):
            raise SystemExit(f"{checkout} is not a git checkout")
        return checkout
    chosen = repo or DEFAULT_REPO
    checkout = clone_or_pull(chosen, clone_dir(home))
    return skill_source(checkout, chosen)


def install(
    repo: str | None,
    path: str | None,
    home: Path,
    *,
    repo_given: bool,
) -> Path:
    source = resolve_source(repo, path, home, repo_given=repo_given)
    dest = runtime_dir(home)
    copy_payload(source, dest)
    copy_skills(source, home)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Install local-offload from a git checkout.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=(
            "git URL to clone, or pull when the cache already exists "
            "(default: %(default)s). That repo keeps the skill in its local-offload subfolder."
        ),
    )
    parser.add_argument("--path", help="path to a git checkout the host already has")
    args = parser.parse_args(argv)
    supplied = list(sys.argv[1:] if argv is None else argv)
    home = default_home()
    dest = install(
        args.repo,
        args.path,
        home,
        repo_given=_option_given(supplied, "--repo"),
    )
    sys.stdout.write(next_commands(dest, home))


if __name__ == "__main__":
    main()
