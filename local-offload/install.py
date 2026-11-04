#!/usr/bin/env python3
"""Copy a local-offload git checkout onto this machine.

Pass a repo URL, or the path of a git checkout the host already has.
The default --repo is https://github.com/Diebeste/GrokProjects, which
keeps this skill in its local-offload subfolder.
The files land in ~/.local-offload. Both skills are copied into the
Cursor and Codex skill directories. This script does not know a store path.

--setup, --pull, and --never run only when you pass them. The server never
calls them. --setup is for after the user says yes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import server

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
            "python3 install.py --setup   # only after the user says yes",
            "python3 install.py --pull",
            "python3 install.py --never",
            "",
        ]
    )


LINUX_INSTALL = [
    "bash",
    "-c",
    "curl -fsSL --retry 3 --max-time 180 https://ollama.com/install.sh | sh",
]
DARWIN_ZIP = (
    "curl -fsSL -o /tmp/Ollama-darwin.zip https://ollama.com/download/Ollama-darwin.zip"
    " && unzip -o /tmp/Ollama-darwin.zip -d /Applications"
)
WINDOWS_INSTALL = (
    "Invoke-WebRequest -Uri https://ollama.com/download/OllamaSetup.exe -OutFile OllamaSetup.exe; "
    "Start-Process .\\OllamaSetup.exe -ArgumentList '/VERYSILENT' -Wait"
)


class LiveSetup:
    def system(self) -> str:
        return platform.system()

    def arch(self) -> str:
        return platform.machine()

    def which(self, name: str):
        return shutil.which(name)

    def is_file(self, path: Path) -> bool:
        return Path(path).is_file()

    def home_dir(self) -> Path:
        return Path.home()

    def localappdata(self) -> str | None:
        value = os.environ.get("LOCALAPPDATA")
        if value:
            return value
        return None

    def run(self, argv: list[str]) -> None:
        sys.stdout.write("setup: " + " ".join(argv) + "\n")
        sys.stdout.flush()
        completed = subprocess.run(argv, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"command failed ({completed.returncode}): {argv[0]}")

    def try_run(self, argv: list[str]) -> int:
        sys.stdout.write("setup: " + " ".join(argv) + "\n")
        sys.stdout.flush()
        completed = subprocess.run(argv, check=False)
        return completed.returncode

    def spawn(self, argv: list[str]) -> None:
        sys.stdout.write("setup: start " + " ".join(argv) + "\n")
        sys.stdout.flush()
        subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None,
            start_new_session=True,
        )

    def memory_gb(self):
        return server.read_memory_gb()

    def installed_models(self) -> list:
        names = []
        for url in (
            "http://127.0.0.1:11434/api/tags",
            "http://127.0.0.1:11434/v1/models",
        ):
            request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=2) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, OSError, json.JSONDecodeError, UnicodeError):
                continue
            if not isinstance(payload, dict):
                continue
            for item in payload.get("models") or []:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    names.append(item["name"])
            for item in payload.get("data") or []:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    names.append(item["id"])
            if names:
                return names
        return names

    def healthy(self, url: str) -> bool:
        request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=1.5) as response:
                return int(getattr(response, "status", 200)) == 200
        except (urllib.error.URLError, OSError):
            return False

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def fetch_tier(self, url: str):
        request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read()
                return int(getattr(response, "status", 200)), raw.decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read().decode("utf-8", "replace")
            finally:
                exc.close()
            return exc.code, raw
        except urllib.error.URLError:
            return 0, ""


OLLAMA_HEALTH_URL = "http://127.0.0.1:11434/api/tags"
OLLAMA_POLL_SECONDS = 30.0
OLLAMA_POLL_INTERVAL = 0.5
OLLAMA_MISSING = (
    "ollama was not found. Checked PATH, "
    "/Applications/Ollama.app/Contents/Resources/ollama, "
    "~/Applications/Ollama.app/Contents/Resources/ollama, "
    "/usr/local/bin/ollama, /opt/homebrew/bin/ollama, "
    "and %LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe."
)


def ollama_search_paths(home: Path, localappdata: str | None) -> list[Path]:
    paths = [
        Path("/Applications/Ollama.app/Contents/Resources/ollama"),
        Path(home) / "Applications" / "Ollama.app" / "Contents" / "Resources" / "ollama",
        Path("/usr/local/bin/ollama"),
        Path("/opt/homebrew/bin/ollama"),
    ]
    if localappdata:
        paths.append(Path(localappdata) / "Programs" / "Ollama" / "ollama.exe")
    return paths


def find_ollama(which, is_file, home: Path, localappdata: str | None = None) -> str:
    found = which("ollama")
    if found:
        return found
    for candidate in ollama_search_paths(home, localappdata):
        if is_file(candidate):
            return str(candidate)
    raise SystemExit(OLLAMA_MISSING)


def install_argv(system: str, which) -> list[str]:
    if system == "Darwin":
        if which("brew"):
            return ["brew", "install", "ollama"]
        return ["bash", "-c", DARWIN_ZIP]
    if system == "Windows":
        return ["powershell", "-NoProfile", "-Command", WINDOWS_INSTALL]
    return list(LINUX_INSTALL)


def recommended_model(
    config: dict,
    memory_gb,
    server_name: str,
    model: str | None = None,
    quality: str | None = None,
) -> str:
    preference = {}
    if quality:
        preference["quality"] = quality
    if model:
        preference["preferredModel"] = model
    rung, _fits, _recommended = server.resolve_choice(config, memory_gb, preference or None)
    if server_name == "mlx":
        return str(rung["mlxTag"])
    return str(rung["tag"])


def _wait_until(predicate, timeout: float, sleep, interval: float = OLLAMA_POLL_INTERVAL) -> bool:
    waited = 0.0
    while waited < timeout:
        if predicate():
            return True
        sleep(interval)
        waited += interval
    return bool(predicate())


def start_ollama(host, binary: str) -> None:
    if host.system() == "Darwin":
        if host.try_run(["open", "-a", "Ollama"]) == 0 and _wait_until(
            lambda: host.healthy(OLLAMA_HEALTH_URL),
            OLLAMA_POLL_SECONDS,
            host.sleep,
        ):
            return
        sys.stdout.write("setup: Ollama.app did not open the server; starting the binary\n")
        sys.stdout.flush()
    host.spawn([binary, "serve"])
    if not _wait_until(lambda: host.healthy(OLLAMA_HEALTH_URL), OLLAMA_POLL_SECONDS, host.sleep):
        raise SystemExit(
            f"ollama did not start. {binary} serve did not answer http://127.0.0.1:11434 "
            f"within {OLLAMA_POLL_SECONDS:g}s."
        )


def _server_binary(host):
    try:
        return find_ollama(host.which, host.is_file, host.home_dir(), host.localappdata())
    except SystemExit:
        return None


def _model_installed(host, tag: str) -> bool:
    names = host.installed_models() if hasattr(host, "installed_models") else []
    for name in names:
        if not isinstance(name, str):
            continue
        if name == tag or name.split("@", 1)[0] == tag:
            return True
    return False


def perform_set_model(home: Path, model: str, quality: str | None = None) -> None:
    config = server.load_config()
    if server.rung_by_tag(config, model) is None:
        raise SystemExit(f"unknown model {model}")
    server.save_model_choice(server.user_config_path(home, config), model, quality)
    sys.stdout.write(f"saved model choice {model}\n")


def perform_setup(
    server_name: str,
    home: Path,
    host: LiveSetup | None = None,
    model: str | None = None,
    quality: str | None = None,
) -> str:
    if server_name not in ("ollama", "mlx"):
        raise SystemExit("setup server must be ollama or mlx")
    host = host or LiveSetup()
    system = host.system()
    if server_name == "mlx" and not (system == "Darwin" and host.arch() in ("arm64", "aarch64")):
        raise SystemExit("MLX setup is only for Apple Silicon")
    config = server.load_config()
    model_tag = recommended_model(config, host.memory_gb(), server_name, model, quality)
    if model or quality:
        server.save_model_choice(server.user_config_path(home, config), model, quality)
        binary = _server_binary(host)
        if binary and server_name != "mlx":
            if not host.healthy(OLLAMA_HEALTH_URL):
                start_ollama(host, binary)
            if _model_installed(host, model_tag):
                sys.stdout.write(f"setup: {model_tag} is already installed; saved the choice\n")
                sys.stdout.flush()
                return model_tag
            sys.stdout.write(f"setup: server already installed; pulling {model_tag}\n")
            sys.stdout.flush()
            host.run([binary, "pull", model_tag])
            return _verify_tier(home, host)
    sys.stdout.write(f"setup: installing {server_name} for {system}\n")
    sys.stdout.flush()
    if server_name == "mlx":
        host.run([sys.executable, "-m", "pip", "install", "mlx-lm"])
        sys.stdout.write(f"setup: pulling {model_tag}\n")
        sys.stdout.flush()
        code = "from huggingface_hub import snapshot_download; snapshot_download(%r)" % model_tag
        host.run([sys.executable, "-c", code])
        if not host.healthy("http://127.0.0.1:8080/v1/models"):
            host.spawn([sys.executable, "-m", "mlx_lm.server", "--host", "127.0.0.1", "--port", "8080"])
            _wait_until(lambda: host.healthy("http://127.0.0.1:8080/v1/models"), OLLAMA_POLL_SECONDS, host.sleep)
    else:
        host.run(install_argv(system, host.which))
        binary = find_ollama(host.which, host.is_file, host.home_dir(), host.localappdata())
        if not host.healthy(OLLAMA_HEALTH_URL):
            start_ollama(host, binary)
        sys.stdout.write(f"setup: pulling {model_tag}\n")
        sys.stdout.flush()
        host.run([binary, "pull", model_tag])
    return _verify_tier(home, host)


def perform_pull(
    home: Path,
    host: LiveSetup | None = None,
    discover=None,
    model: str | None = None,
    quality: str | None = None,
) -> None:
    host = host or LiveSetup()
    config = server.load_config()
    found = (discover or server.discover_backend)(config)
    if found is None:
        raise SystemExit("no local backend found; run python3 install.py --setup only after the user says yes")
    if model or quality:
        server.save_model_choice(server.user_config_path(home, config), model, quality)
    model = recommended_model(config, host.memory_gb(), found.name, model, quality)
    sys.stdout.write(f"setup: pulling {model} on {found.name}\n")
    sys.stdout.flush()
    if found.name == "mlx":
        code = "from huggingface_hub import snapshot_download; snapshot_download(%r)" % model
        host.run([sys.executable, "-c", code])
        return
    if found.name != "ollama" and found.api != "ollama":
        raise SystemExit(f"{found.name} is running; load {model} in that app")
    binary = find_ollama(host.which, host.is_file, host.home_dir(), host.localappdata())
    host.run([binary, "pull", model])


def _verify_tier(home: Path, host: LiveSetup) -> str:
    port = int(server.load_config()["port"])
    url = f"http://127.0.0.1:{port}/tier"
    sys.stdout.write("setup: verifying /tier\n")
    sys.stdout.flush()
    status, body = host.fetch_tier(url)
    if status == 0:
        script = runtime_dir(home) / "server.py"
        if not script.is_file():
            script = Path(__file__).resolve().parent / "server.py"
        host.spawn([sys.executable, str(script)])
        for _ in range(20):
            host.sleep(0.25)
            status, body = host.fetch_tier(url)
            if status != 0:
                break
    sys.stdout.write(f"setup: /tier {status}\n")
    if body:
        sys.stdout.write(body + "\n")
    sys.stdout.flush()
    if status != 200:
        raise SystemExit(f"/tier returned {status}")
    return body


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
    parser.add_argument(
        "--setup",
        nargs="?",
        const="ollama",
        default=None,
        help="install ollama (default) or mlx, then pull the recommended model. Only after the user says yes.",
    )
    parser.add_argument("--pull", action="store_true", help="pull the recommended model; do not install a server")
    parser.add_argument("--model", help="model tag to pull during --setup or --pull, and save as preferredModel")
    parser.add_argument("--set-model", help="save preferredModel only; do not install or pull")
    parser.add_argument("--quality", choices=("fast", "balanced", "best"), help="fast, balanced, or best")
    parser.add_argument("--never", action="store_true", help="save the never-ask flag")
    args = parser.parse_args(argv)
    supplied = list(sys.argv[1:] if argv is None else argv)
    home = default_home()
    if args.set_model:
        if args.path or _option_given(supplied, "--repo") or args.setup is not None or args.pull or args.never or args.model:
            raise SystemExit("pass --set-model by itself")
        perform_set_model(home, args.set_model, args.quality)
        return
    maintenance = [name for name, enabled in (("setup", args.setup is not None), ("pull", args.pull), ("never", args.never)) if enabled]
    if maintenance:
        if args.path or _option_given(supplied, "--repo"):
            raise SystemExit("pass either an install source or a setup command")
        if len(maintenance) > 1:
            raise SystemExit("pass only one of --never, --pull, or --setup")
        if (args.model or args.quality) and not (args.setup is not None or args.pull):
            raise SystemExit("pass --model or --quality with --setup or --pull")
        if args.never:
            path = server.user_config_path(home)
            server.write_never_ask(path)
            sys.stdout.write(f"saved never-ask flag to {path}\n")
            return
        if args.pull:
            perform_pull(home, model=args.model, quality=args.quality)
            return
        perform_setup(str(args.setup), home, model=args.model, quality=args.quality)
        return
    if args.model or args.quality:
        raise SystemExit("pass --model or --quality with --setup or --pull")
    dest = install(
        args.repo,
        args.path,
        home,
        repo_given=_option_given(supplied, "--repo"),
    )
    sys.stdout.write(next_commands(dest, home))


if __name__ == "__main__":
    main()
