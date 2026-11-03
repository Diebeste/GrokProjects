#!/usr/bin/env python3
"""Loopback helper for a local OpenAI-compatible model.

The host assigns a bounded slice. This process can read listed files, draft
text, and run an allow-listed test command in a temp copy of those files.
It does not write the real repository, install a backend, or pull a model.
"""

from __future__ import annotations

import glob
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"

NVIDIA_COMMAND = [
    "nvidia-smi",
    "--query-gpu=memory.total",
    "--format=csv,noheader,nounits",
]
DARWIN_RAM_COMMAND = ["sysctl", "-n", "hw.memsize"]
WINDOWS_RAM_COMMAND = [
    "powershell",
    "-NoProfile",
    "-Command",
    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
]
AMD_VRAM_GLOB = "/sys/class/drm/card*/device/mem_info_vram_total"

MAX_BODY = 64 * 1024
GENERATE_TIMEOUT = 1800
PROBE_TIMEOUT = 1.5
SYSTEM_PROMPT = (
    "You generate one code slice and nothing else. "
    "You have no tools, no filesystem, and no network. "
    "Do not run commands. Output only the code the host will apply, with no markdown fences."
)
SUMMARY_PROMPT = (
    "You summarize or transform only the text you are given. "
    "You have no tools and no network. Return only the requested result."
)
_PARAM_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")
_FILE_BLOCK = re.compile(r"<<<FILE ([^>\n]+)>>>\n(.*?)<<<END>>>", re.DOTALL)
_SECRET_NAMES = {".env", "id_rsa", "credentials.json", "credentials.yaml"}
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}
_ENV_KEEP = (
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "SYSTEMROOT",
    "COMSPEC",
    "WINDIR",
    "PATHEXT",
)
_OS_HINTS = {
    "Darwin": "macOS: install Ollama.app or run `brew install ollama`",
    "Windows": "Windows: run the official Ollama installer from https://ollama.com/download/OllamaSetup.exe",
    "Linux": "Linux: run the official installer at https://ollama.com/install.sh",
}
_LOG_KEYS = ("ts", "kind", "attempt", "ok", "timedOut", "tokensSaved", "model", "backend")


class JobError(Exception):
    pass


class ServiceError(Exception):
    def __init__(self, message: str, status: int = 503) -> None:
        super().__init__(message)
        self.status = status
        self.hint = None
        self.setup = None
        self.extra = None


@dataclass(frozen=True)
class InstalledModel:
    name: str
    billion: float | None


@dataclass(frozen=True)
class Probe:
    kind: str
    value: float | None = None


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    api: str


@dataclass
class CommandResult:
    code: int | None
    stdout: str
    stderr: str
    timed_out: bool


def assert_loopback_bind(host: str) -> None:
    if host != "127.0.0.1":
        raise SystemExit("refusing to bind anything except 127.0.0.1")


def assert_loopback_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise SystemExit("backend host must be http://127.0.0.1")


def user_config_path(home: Path) -> Path:
    return home / ".local-offload" / "config"


def log_path_for(home: Path) -> Path:
    return home / ".local-offload" / "log.jsonl"


def _validate_rung(rung: dict) -> None:
    if not isinstance(rung, dict):
        raise SystemExit("invalid config: rung")
    tag = rung.get("tag")
    if not isinstance(tag, str) or tag.count(":") != 1 or tag.startswith(":") or tag.endswith(":"):
        raise SystemExit("invalid config: model tag")
    instruction = rung.get("instruction")
    if not isinstance(instruction, str) or "\n" in instruction or instruction.strip() == "":
        raise SystemExit("invalid config: instruction")
    for key in ("nominalBillion", "minBillion", "maxBillion", "downloadGb", "mlxDownloadGb"):
        if not isinstance(rung.get(key), (int, float)):
            raise SystemExit(f"invalid config: {key}")
    if rung["minBillion"] > rung["maxBillion"]:
        raise SystemExit("invalid config: parameter band")
    if not isinstance(rung.get("maxTokens"), int) or rung["maxTokens"] < 1:
        raise SystemExit("invalid config: maxTokens")
    mlx_tag = rung.get("mlxTag")
    if not isinstance(mlx_tag, str) or mlx_tag.strip() == "":
        raise SystemExit("invalid config: mlxTag")


def _validate_prefixes(config: dict) -> None:
    prefixes = config.get("allowedTestPrefixes")
    if not isinstance(prefixes, list) or not prefixes:
        raise SystemExit("invalid config: allowedTestPrefixes")
    for prefix in prefixes:
        if not isinstance(prefix, list) or not prefix:
            raise SystemExit("invalid config: allowedTestPrefixes")
        for part in prefix:
            if not isinstance(part, str) or part == "":
                raise SystemExit("invalid config: allowedTestPrefixes")


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise SystemExit("invalid config")
    if config.get("bindHost") != "127.0.0.1":
        raise SystemExit("invalid config: bindHost is 127.0.0.1")
    if config.get("fallbackRung") != "7B":
        raise SystemExit("invalid config: fallback rung is 7B")
    if config.get("lowBelowGb") != 8 or config.get("highAboveGb") != 16:
        raise SystemExit("invalid config: memory boundaries are 8 and 16 GB")
    base_url = config.get("baseUrl", "")
    if not isinstance(base_url, str):
        raise SystemExit("invalid config: baseUrl")
    if base_url.strip():
        assert_loopback_url(base_url.strip())
    backends = config.get("backends")
    if not isinstance(backends, list) or not backends:
        raise SystemExit("invalid config: backends")
    names = []
    for item in backends:
        if not isinstance(item, dict):
            raise SystemExit("invalid config: backend")
        name = item.get("name")
        url = item.get("baseUrl")
        if not isinstance(name, str) or name.strip() == "":
            raise SystemExit("invalid config: backend name")
        if not isinstance(url, str):
            raise SystemExit("invalid config: backend baseUrl")
        assert_loopback_url(url)
        names.append(name)
    if names[:4] != ["ollama", "lmstudio", "llamacpp", "mlx"]:
        raise SystemExit("invalid config: backends are ollama, lmstudio, llamacpp, and mlx")
    port = config.get("port")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        raise SystemExit("invalid config: port")
    for key in ("maxAttempts", "attemptTimeoutSec", "overallTimeoutSec", "localReadChars"):
        value = config.get(key)
        if not isinstance(value, int) or value < 1:
            raise SystemExit(f"invalid config: {key}")
    limits = config.get("limits")
    if not isinstance(limits, dict):
        raise SystemExit("invalid config: limits")
    for key in ("instructionChars", "testChars", "contextChars"):
        value = limits.get(key)
        if not isinstance(value, int) or value < 1:
            raise SystemExit(f"invalid config: limits.{key}")
    _validate_prefixes(config)
    rungs = config.get("rungs")
    if not isinstance(rungs, list):
        raise SystemExit("invalid config: rungs")
    found: dict[str, dict] = {}
    for rung in rungs:
        _validate_rung(rung)
        found[str(rung.get("parameterClass"))] = rung
    if set(found) != {"7B", "14B", "32B"}:
        raise SystemExit("invalid config: rungs are 7B, 14B, and 32B")
    slices = {"7B": "function", "14B": "file", "32B": "multifile"}
    for klass, slice_limit in slices.items():
        if found[klass].get("slice") != slice_limit:
            raise SystemExit("invalid config: slice limits")


def load_config(path: Path | None = None) -> dict:
    config_path = CONFIG_PATH if path is None else Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid config: {exc}") from exc
    validate_config(config)
    return config


def parse_billion(text: str) -> float | None:
    if not text:
        return None
    match = _PARAM_SIZE.search(text)
    if not match:
        return None
    return float(match.group(1))


def rung_by_class(config: dict, name: str) -> dict:
    for rung in config["rungs"]:
        if rung["parameterClass"] == name:
            return rung
    raise ServiceError("ladder is missing a rung")


def memory_failed(memory_gb: float | None) -> bool:
    if memory_gb is None:
        return True
    if isinstance(memory_gb, float) and math.isnan(memory_gb):
        return True
    return memory_gb < 0


def select_rung(memory_gb: float | None, config: dict) -> dict:
    if memory_failed(memory_gb):
        return rung_by_class(config, config["fallbackRung"])
    assert memory_gb is not None
    if memory_gb < config["lowBelowGb"]:
        return rung_by_class(config, "7B")
    if memory_gb <= config["highAboveGb"]:
        return rung_by_class(config, "14B")
    return rung_by_class(config, "32B")


def in_band(billion: float | None, rung: dict) -> bool:
    if billion is None:
        return False
    return rung["minBillion"] <= billion <= rung["maxBillion"]


def pick_model(rung: dict, installed: list[InstalledModel]) -> tuple[str, bool]:
    fitting = [model for model in installed if in_band(model.billion, rung)]
    pinned = rung["tag"]
    for model in fitting:
        if model.name == pinned:
            return pinned, False
    if not fitting:
        sized = [model for model in installed if model.billion is not None]
        if installed and not sized:
            return installed[0].name, False
        return pinned, True
    fitting.sort(key=lambda model: (abs(model.billion - rung["nominalBillion"]), model.name))
    return fitting[0].name, False


def format_gb(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def ask_sentence(server_name: str, model: str, download_gb: float) -> str:
    return (
        f"No local model found. Install {server_name} + {model} "
        f"(~{format_gb(download_gb)} GB)? [yes / not now / never]"
    )


def os_hint(system: str) -> str:
    return _OS_HINTS.get(system, "Install Ollama for this OS from https://ollama.com/download")


def display_commands(system: str, model: str) -> dict[str, str]:
    if system == "Darwin":
        install = "brew install ollama"
    elif system == "Windows":
        install = "download and run https://ollama.com/download/OllamaSetup.exe"
    else:
        install = "curl -fsSL https://ollama.com/install.sh | sh"
    return {
        "install": install,
        "pull": f"ollama pull {model}",
        "setup": "python3 install.py --setup",
        "never": "python3 install.py --never",
    }


def read_never_ask(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    if not isinstance(data, dict):
        return False
    return data.get("neverAsk") is True


def write_never_ask(path: Path) -> None:
    data: dict = {}
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(current, dict):
            data = current
    except (OSError, json.JSONDecodeError, UnicodeError):
        data = {}
    data["neverAsk"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_setup(
    config: dict,
    system: str,
    arch: str,
    memory_gb: float | None,
    never_ask: bool,
) -> dict:
    rung = select_rung(memory_gb, config)
    recommended = {
        "server": "ollama",
        "model": rung["tag"],
        "downloadGb": rung["downloadGb"],
        "parameterClass": rung["parameterClass"],
        "slice": rung["slice"],
    }
    alternatives = []
    if system == "Darwin" and arch in ("arm64", "aarch64"):
        alternatives.append(
            {
                "server": "mlx",
                "model": rung["mlxTag"],
                "downloadGb": rung["mlxDownloadGb"],
                "note": "optional on Apple Silicon",
                "commands": {"setup": "python3 install.py --setup mlx"},
            }
        )
    return {
        "os": system,
        "arch": arch,
        "memoryGb": memory_gb,
        "recommended": recommended,
        "alternatives": alternatives,
        "hint": os_hint(system),
        "commands": display_commands(system, rung["tag"]),
        "ask": None if never_ask else ask_sentence("ollama", rung["tag"], rung["downloadGb"]),
        "neverAsk": never_ask,
    }


def build_prompt(slice_limit: str, instruction: str, test: str, context: str, local: str = "") -> str:
    parts = [f"Slice limit: {slice_limit}.", "Instruction:", instruction]
    if test:
        parts.extend(["Test the host will run:", test])
    if local:
        parts.extend(["Local file text:", local])
    if context:
        parts.extend(["Context pasted by the host:", context])
    return "\n".join(parts)


def validate_job(body: dict, config: dict) -> tuple[str, str, str]:
    if not isinstance(body, dict):
        raise JobError("job must be a JSON object")
    limits = config["limits"]
    instruction = body.get("instruction")
    if not isinstance(instruction, str) or instruction.strip() == "":
        raise JobError("instruction is required")
    if len(instruction) > limits["instructionChars"]:
        raise JobError(f"instruction exceeds {limits['instructionChars']} characters")
    test = body.get("test")
    context = body.get("context")
    if test is None:
        test = ""
    if context is None:
        context = ""
    if not isinstance(test, str):
        raise JobError("test must be a string")
    if not isinstance(context, str):
        raise JobError("context must be a string")
    if len(test) > limits["testChars"]:
        raise JobError(f"test exceeds {limits['testChars']} characters")
    if len(context) > limits["contextChars"]:
        raise JobError(f"context exceeds {limits['contextChars']} characters")
    return instruction, test, context


def endpoint_list(config: dict) -> list[tuple[str, str]]:
    items = []
    base = config.get("baseUrl") or ""
    if isinstance(base, str) and base.strip():
        items.append(("custom", base.strip()))
    for item in config["backends"]:
        items.append((str(item["name"]), str(item["baseUrl"])))
    return items


def _openai_models(body: dict) -> bool:
    return isinstance(body, dict) and isinstance(body.get("data"), list)


def _ollama_tags(body: dict) -> bool:
    return isinstance(body, dict) and isinstance(body.get("models"), list)


def probe_backends(endpoints: list[tuple[str, str]], fetch) -> Backend | None:
    tried = set()
    for name, base in endpoints:
        base = base.rstrip("/")
        if not base or base in tried:
            continue
        tried.add(base)
        try:
            assert_loopback_url(base)
        except SystemExit:
            continue
        status, body = fetch("GET", base + "/v1/models", None, PROBE_TIMEOUT)
        if status == 200 and _openai_models(body):
            return Backend(name, base, "openai")
        status, body = fetch("GET", base + "/api/tags", None, PROBE_TIMEOUT)
        if status == 200 and _ollama_tags(body):
            return Backend(name if name != "custom" else "ollama", base, "ollama")
    return None


def decode_ollama_body(raw: bytes) -> dict:
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
        if not isinstance(parsed, dict):
            raise ServiceError("backend returned invalid JSON")
    if not isinstance(parsed, dict):
        raise ServiceError("backend returned invalid JSON")
    return parsed


def real_fetch(method: str, url: str, payload: dict | None = None, timeout: float = PROBE_TIMEOUT):
    data = None
    headers = {"Accept": "application/json", "User-Agent": "local-offload"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        finally:
            exc.close()
        return exc.code, {}
    except urllib.error.URLError:
        return 0, {}
    if not raw:
        return status, {}
    try:
        return status, decode_ollama_body(raw)
    except ServiceError:
        return status, {}


def openai_chat_payload(model: str, prompt: str, max_tokens: int, summary: bool = False) -> dict:
    system = SUMMARY_PROMPT if summary else SYSTEM_PROMPT
    return {
        "model": model,
        "stream": False,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }


def ollama_chat_payload(model: str, prompt: str, max_tokens: int, summary: bool = False) -> dict:
    system = SUMMARY_PROMPT if summary else SYSTEM_PROMPT
    return {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.1, "num_predict": max_tokens},
    }


def chat_payload(model: str, prompt: str, max_tokens: int) -> dict:
    return openai_chat_payload(model, prompt, max_tokens)


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


class BackendClient:
    def __init__(self, backend: Backend, fetch=None) -> None:
        assert_loopback_url(backend.base_url)
        self.name = backend.name
        self.api = backend.api
        self.base_url = backend.base_url.rstrip("/")
        self._fetch = fetch or real_fetch

    def list_models(self) -> list[InstalledModel]:
        if self.api == "openai":
            data = self._get("/v1/models", 10)
            found = []
            for item in data.get("data") or []:
                if not isinstance(item, dict):
                    continue
                name = item.get("id") or ""
                if not isinstance(name, str) or name == "":
                    continue
                found.append(InstalledModel(name=name, billion=parse_billion(name)))
            return found
        data = self._get("/api/tags", 10)
        found = []
        for item in data.get("models") or []:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model") or ""
            if not isinstance(name, str) or name == "":
                continue
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            billion = parse_billion(str(details.get("parameter_size") or ""))
            if billion is None:
                billion = parse_billion(name)
            found.append(InstalledModel(name=name, billion=billion))
        return found

    def generate(self, model: str, prompt: str, max_tokens: int, summary: bool = False) -> str:
        if self.api == "openai":
            data = self._post(
                "/v1/chat/completions",
                openai_chat_payload(model, prompt, max_tokens, summary),
                GENERATE_TIMEOUT,
                "generation failed",
                502,
            )
            choices = data.get("choices") if isinstance(data.get("choices"), list) else []
            message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
            content = _message_text(message.get("content") if isinstance(message, dict) else "")
        else:
            data = self._post(
                "/api/chat",
                ollama_chat_payload(model, prompt, max_tokens, summary),
                GENERATE_TIMEOUT,
                "generation failed",
                502,
            )
            if data.get("error"):
                raise ServiceError("generation failed", status=502)
            message = data.get("message") if isinstance(data.get("message"), dict) else {}
            content = message.get("content") if isinstance(message.get("content"), str) else ""
        if not isinstance(content, str) or content == "":
            raise ServiceError("empty generation", status=502)
        return content

    def _get(self, path: str, timeout: float) -> dict:
        return self._request("GET", path, None, timeout, "backend is not running", 503)

    def _post(self, path: str, payload: dict, timeout: float | None, failure: str, status: int = 503) -> dict:
        return self._request("POST", path, payload, timeout, failure, status)

    def _request(self, method: str, path: str, payload: dict | None, timeout: float | None, failure: str, status: int) -> dict:
        code, body = self._fetch(method, self.base_url + path, payload, timeout if timeout is not None else GENERATE_TIMEOUT)
        if code == 0:
            raise ServiceError("backend is not running")
        if code >= 400:
            sys.stderr.write(f"local-offload: {self.name} {path} returned {code}\n")
            raise ServiceError(failure, status=status)
        if not isinstance(body, dict):
            raise ServiceError(failure, status=status)
        return body


def discover_backend(config: dict, fetch=None) -> Backend | None:
    return probe_backends(endpoint_list(config), fetch or real_fetch)


class HostInfo:
    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )

    def read(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8", errors="replace")

    def glob(self, pattern: str) -> list[str]:
        return sorted(glob.glob(pattern))

    def system(self) -> str:
        return platform.system()


def _nvidia_vram(host: HostInfo) -> Probe:
    if host.which("nvidia-smi") is None:
        return Probe("absent")
    try:
        completed = host.run(NVIDIA_COMMAND)
    except (OSError, subprocess.TimeoutExpired):
        return Probe("failed")
    if completed.returncode != 0:
        return Probe("failed")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return Probe("failed")
    total_mib = 0
    for line in lines:
        field = line.split(",")[0].strip()
        if not field.isdigit():
            return Probe("failed")
        total_mib += int(field)
    if total_mib <= 0:
        return Probe("failed")
    return Probe("value", total_mib / 1024)


def _amd_vram(host: HostInfo) -> Probe:
    if host.system() != "Linux":
        return Probe("absent")
    paths = host.glob(AMD_VRAM_GLOB)
    if not paths:
        return Probe("absent")
    total_bytes = 0
    for path in paths:
        try:
            text = host.read(path).strip()
        except OSError:
            return Probe("failed")
        if not text.isdigit():
            return Probe("failed")
        total_bytes += int(text)
    if total_bytes <= 0:
        return Probe("failed")
    return Probe("value", total_bytes / (1024**3))


def _system_ram_gb(host: HostInfo) -> float | None:
    system = host.system()
    try:
        if system == "Linux":
            text = host.read("/proc/meminfo")
            match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
            if not match:
                return None
            return int(match.group(1)) / (1024**2)
        if system == "Darwin":
            completed = host.run(DARWIN_RAM_COMMAND)
            if completed.returncode != 0:
                return None
            return int(completed.stdout.strip()) / (1024**3)
        if system == "Windows":
            completed = host.run(WINDOWS_RAM_COMMAND)
            if completed.returncode != 0:
                return None
            return int(completed.stdout.strip()) / (1024**3)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    return None


def read_memory_gb(host: HostInfo | None = None) -> float | None:
    host = host or HostInfo()
    nvidia = _nvidia_vram(host)
    if nvidia.kind == "failed":
        return None
    if nvidia.kind == "value":
        return nvidia.value
    amd = _amd_vram(host)
    if amd.kind == "failed":
        return None
    if amd.kind == "value":
        return amd.value
    return _system_ram_gb(host)


def _dangerous_arg(part: str) -> bool:
    if any(ch in part for ch in ("|", ";", "&", "`", "$", "\n", "\r", "<", ">")):
        return True
    return "://" in part


def command_allowed(argv, prefixes: list[list[str]]) -> bool:
    if not isinstance(argv, list) or not argv:
        return False
    if not all(isinstance(part, str) and part != "" for part in argv):
        return False
    if any(_dangerous_arg(part) for part in argv):
        return False
    name = Path(argv[0]).name
    for prefix in prefixes:
        if not prefix:
            continue
        if [name] + list(argv[1 : len(prefix)]) == list(prefix):
            return True
    return False


def scrub_env(source: dict | None = None) -> dict[str, str]:
    current = os.environ if source is None else source
    env = {}
    for key in _ENV_KEEP:
        value = current.get(key)
        if isinstance(value, str):
            env[key] = value
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    return env


def _secret_name(name: str) -> bool:
    if name in _SECRET_NAMES or name.startswith(".env") or name.endswith(".pem"):
        return True
    return name in {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}


def safe_relative(rel: str) -> Path:
    if not isinstance(rel, str) or rel.strip() == "":
        raise JobError("file path must be relative")
    if rel.startswith("/") or rel.startswith("~") or "\\" in rel:
        raise JobError("file path must be relative")
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        raise JobError("file path must stay inside the repo")
    if _secret_name(path.name):
        raise JobError("refusing secret file")
    return path


def resolve_inside(repo: Path, rel: str) -> Path:
    relative = safe_relative(rel)
    root = repo.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise JobError("file path must stay inside the repo")
    return path


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def append_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {key: record.get(key) for key in _LOG_KEYS}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


def strip_fence(text: str) -> str:
    stripped = text.strip()
    if not (stripped.startswith("```") and stripped.endswith("```")):
        return text
    lines = stripped.splitlines()
    if len(lines) < 2 or not lines[0].startswith("```") or not lines[-1].startswith("```"):
        return text
    body = "\n".join(lines[1:-1])
    if text.endswith("\n"):
        body += "\n"
    return body


def parse_model_files(text: str, targets: list[str]) -> dict[str, str]:
    found = {}
    for match in _FILE_BLOCK.finditer(text):
        found[match.group(1).strip()] = match.group(2)
    if not found and len(targets) == 1:
        return {targets[0]: strip_fence(text)}
    missing = [rel for rel in targets if rel not in found]
    if missing:
        raise JobError("model output missing a target file")
    return {rel: found[rel] for rel in targets}


def _skip_dir_name(name: str) -> bool:
    return name in _SKIP_DIRS or _secret_name(name)


def stage_files(repo: Path, files: list[str], scratch: Path) -> None:
    for rel in files:
        src = resolve_inside(repo, rel)
        if not src.is_file() or src.is_symlink():
            raise JobError(f"missing file {rel}")
        dest = scratch / safe_relative(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def stage_repo(repo: Path, scratch: Path) -> None:
    root = repo.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        kept = []
        for name in dirnames:
            path = current / name
            if path.is_symlink() or _skip_dir_name(name):
                continue
            kept.append(name)
        dirnames[:] = kept
        rel_dir = current.relative_to(root)
        for name in filenames:
            src = current / name
            if src.is_symlink() or _skip_dir_name(name):
                continue
            dest = scratch / rel_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


def read_local_text(repo: Path, files: list[str], limit: int) -> str:
    chunks = []
    total = 0
    for rel in files:
        src = resolve_inside(repo, rel)
        if not src.is_file() or src.is_symlink():
            raise JobError(f"missing file {rel}")
        text = src.read_text(encoding="utf-8", errors="replace")
        remaining = limit - total
        if remaining <= 0:
            chunks.append(f"File {rel}: [truncated]")
            break
        clipped = text[:remaining]
        total += len(clipped)
        note = "" if len(clipped) == len(text) else "\n[truncated]"
        chunks.append(f"File {rel}:\n{clipped}{note}")
    return "\n\n".join(chunks)


def run_command(argv: list[str], cwd: Path, timeout: float, env: dict[str, str]) -> CommandResult:
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        return CommandResult(None, stdout, stderr, True)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr, False)


def _string_list(value, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise JobError(f"{label} must be a list of paths")
    return list(value)


class Service:
    def __init__(
        self,
        config: dict,
        memory_gb,
        backend=None,
        discover=None,
        runner=None,
        log_path: Path | None = None,
        system: str | None = None,
        arch: str | None = None,
        home: Path | None = None,
        clock=None,
    ) -> None:
        self.config = config
        self.memory_gb = memory_gb
        self.backend = backend
        self.discover = discover
        self.runner = runner or run_command
        self.log_path = log_path
        self.system = system
        self.arch = arch
        self.home = home
        self.clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._tier: dict | None = None

    def _log(self, record: dict) -> None:
        if self.log_path is None:
            return
        record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        append_log(self.log_path, record)

    def _memory(self) -> float | None:
        try:
            return self.memory_gb()
        except Exception as exc:
            sys.stderr.write(f"local-offload: memory read failed: {exc}\n")
            return None

    def _missing(self) -> ServiceError:
        system = self.system if self.system else platform.system()
        arch = self.arch if self.arch else platform.machine()
        home = self.home if self.home is not None else Path.home()
        setup = build_setup(
            self.config,
            system,
            arch,
            self._memory(),
            read_never_ask(user_config_path(home)),
        )
        error = ServiceError("no local backend found", status=503)
        error.hint = setup["hint"]
        error.setup = setup
        return error

    def _ensure_backend(self):
        if self.backend is not None:
            return self.backend
        found = self.discover() if self.discover is not None else discover_backend(self.config)
        if found is None:
            raise self._missing()
        if isinstance(found, Backend):
            found = BackendClient(found)
        self.backend = found
        return found

    def tier(self) -> dict:
        with self._lock:
            if self._tier is not None:
                return dict(self._tier)
            backend = self._ensure_backend()
            rung = select_rung(self._memory(), self.config)
            name, needs_pull = pick_model(rung, backend.list_models())
            payload = {
                "model": name,
                "parameterClass": rung["parameterClass"],
                "slice": rung["slice"],
                "instruction": rung["instruction"],
                "backend": backend.name,
                "api": backend.api,
                "needsPull": needs_pull,
            }
            if needs_pull:
                payload["pull"] = "python3 install.py --pull"
                return payload
            sys.stderr.write(
                f"local-offload: {backend.name} {name} {rung['parameterClass']} slice={rung['slice']}\n"
            )
            self._tier = payload
            return dict(payload)

    def _require_model(self) -> dict:
        payload = self.tier()
        if payload.get("needsPull"):
            error = ServiceError("model is not installed", status=503)
            error.extra = {"pull": "python3 install.py --pull", "model": payload["model"]}
            raise error
        return payload

    def _repo(self, body: dict) -> Path:
        repo = body.get("repo")
        if not isinstance(repo, str) or repo.strip() == "":
            raise JobError("repo is required")
        path = Path(repo).expanduser().resolve()
        if not path.is_dir():
            raise JobError("repo is not a directory")
        return path

    def _file_context(self, body: dict) -> str:
        files = _string_list(body.get("files"), "files")
        if not files:
            return ""
        return read_local_text(self._repo(body), files, int(self.config["localReadChars"]))

    def job(self, body: dict):
        instruction, test, context = validate_job(body, self.config)
        if body.get("testCommand") is not None:
            return self.test_loop(body, instruction, test, context)
        tier = self._require_model()
        local = self._file_context(body)
        rung = rung_by_class(self.config, tier["parameterClass"])
        prompt = build_prompt(tier["slice"], instruction, test, context, local)
        summary = bool(body.get("files")) and not test
        text = self.backend.generate(tier["model"], prompt, rung["maxTokens"], summary=summary)
        saved = estimate_tokens(prompt + text)
        self._log(
            {
                "kind": "generate",
                "attempt": 1,
                "ok": True,
                "timedOut": False,
                "tokensSaved": saved,
                "model": tier["model"],
                "backend": tier["backend"],
            }
        )
        return text

    def test_loop(self, body: dict, instruction: str, test: str, context: str) -> dict:
        argv = body.get("testCommand")
        if not command_allowed(argv, self.config["allowedTestPrefixes"]):
            raise JobError("test command is not allowed")
        repo = self._repo(body)
        files = _string_list(body.get("files"), "files")
        targets = _string_list(body.get("targets"), "targets")
        if not targets:
            if len(files) == 1:
                targets = list(files)
            else:
                raise JobError("targets are required")
        for rel in files:
            src = resolve_inside(repo, rel)
            if src.is_symlink() or not src.is_file():
                raise JobError(f"missing file {rel}")
        for rel in targets:
            safe_relative(rel)
        return self._run_loop(repo, files, targets, argv, instruction, test, context)

    def _run_loop(self, repo, files, targets, argv, instruction, test, context) -> dict:
        max_attempts = int(self.config["maxAttempts"])
        attempt_timeout = float(self.config["attemptTimeoutSec"])
        overall = float(self.config["overallTimeoutSec"])
        started = self.clock()
        scratch = Path(tempfile.mkdtemp(prefix="local-offload-"))
        failure = ""
        last_files: dict[str, str] = {}
        last_output = ""
        timed_out = False
        attempts = 0
        saved_total = 0
        tier = self._require_model()
        rung = rung_by_class(self.config, tier["parameterClass"])
        try:
            if files:
                to_copy = list(files)
                for rel in targets:
                    if rel in to_copy:
                        continue
                    src = resolve_inside(repo, rel)
                    if src.is_file() and not src.is_symlink():
                        to_copy.append(rel)
                stage_files(repo, to_copy, scratch)
            else:
                stage_repo(repo, scratch)
            for attempt in range(1, max_attempts + 1):
                now = self.clock()
                if now - started >= overall:
                    break
                attempts = attempt
                remaining = overall - (now - started)
                timeout = min(attempt_timeout, remaining)
                prompt = self._loop_prompt(
                    tier["slice"], instruction, test, context, targets, scratch, failure
                )
                text = self.backend.generate(tier["model"], prompt, rung["maxTokens"])
                saved = estimate_tokens(prompt + text)
                saved_total += saved
                try:
                    written = parse_model_files(text, targets)
                except JobError:
                    written = None
                if written is None:
                    last_output = "model output did not include the target files"
                    timed_out = False
                    failure = last_output
                    self._log(self._attempt_record(attempt, False, False, saved, tier))
                    continue
                for rel, content in written.items():
                    dest = scratch / safe_relative(rel)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                last_files = written
                if timeout <= 0:
                    timed_out = True
                    last_output = "overall time limit reached"
                    self._log(self._attempt_record(attempt, False, True, saved, tier))
                    break
                result = self.runner(list(argv), scratch, timeout, scrub_env())
                last_output = ((result.stdout or "") + (result.stderr or ""))[-8000:]
                timed_out = bool(result.timed_out)
                ok = result.code == 0 and not result.timed_out
                self._log(self._attempt_record(attempt, ok, timed_out, saved, tier))
                if ok:
                    return {
                        "ok": True,
                        "attempts": attempt,
                        "files": last_files,
                        "testOutput": last_output,
                        "report": None,
                        "timedOut": False,
                        "tokensSaved": saved_total,
                    }
                if result.timed_out:
                    failure = f"test timed out after {timeout:g} seconds\n{last_output}"
                else:
                    failure = last_output or "test failed"
            report = f"failed after {attempts} attempts"
            if timed_out:
                report += "; last attempt timed out"
            return {
                "ok": False,
                "attempts": attempts,
                "files": last_files,
                "testOutput": last_output,
                "report": report,
                "timedOut": timed_out,
                "tokensSaved": saved_total,
            }
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _attempt_record(self, attempt: int, ok: bool, timed_out: bool, saved: int, tier: dict) -> dict:
        return {
            "kind": "attempt",
            "attempt": attempt,
            "ok": ok,
            "timedOut": timed_out,
            "tokensSaved": saved,
            "model": tier["model"],
            "backend": tier["backend"],
        }

    def _loop_prompt(self, slice_limit, instruction, test, context, targets, scratch, failure) -> str:
        parts = [build_prompt(slice_limit, instruction, test, context)]
        for rel in targets:
            path = scratch / safe_relative(rel)
            current = ""
            if path.is_file():
                current = path.read_text(encoding="utf-8", errors="replace")
            parts.append(f"Current {rel}:\n{current}")
        if len(targets) == 1:
            parts.append(f"Output only the full new contents of {targets[0]}. No markdown fence.")
        else:
            parts.append("Output each target as <<<FILE path>>> then the contents then <<<END>>>.")
        if failure:
            parts.append("Previous test output:\n" + failure[-4000:])
        return "\n".join(parts)


class LoopbackServer(ThreadingHTTPServer):
    address_family = socket.AF_INET
    allow_reuse_address = True
    daemon_threads = True


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "local-offload"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def __init__(self, *args, **kwargs) -> None:
            self.service = service
            super().__init__(*args, **kwargs)

        def do_GET(self) -> None:
            if self._path() != "/tier":
                self._json(404, {"error": "not found"})
                return
            try:
                payload = self.service.tier()
            except ServiceError as exc:
                sys.stderr.write(f"local-offload: {exc}\n")
                self._error(exc)
                return
            self._json(200, payload)

        def do_POST(self) -> None:
            if self._path() != "/job":
                self._json(404, {"error": "not found"})
                return
            body = self._read_json()
            if body is None:
                return
            try:
                result = self.service.job(body)
            except JobError as exc:
                self._json(400, {"error": str(exc)})
                return
            except ServiceError as exc:
                sys.stderr.write(f"local-offload: {exc}\n")
                self._error(exc)
                return
            if isinstance(result, dict):
                self._json(200, result)
                return
            self._text(200, result)

        def _path(self) -> str:
            return urllib.parse.urlsplit(self.path).path

        def _error(self, exc: ServiceError) -> None:
            payload = {"error": str(exc)}
            if exc.hint:
                payload["hint"] = exc.hint
            if exc.setup is not None:
                payload["setup"] = exc.setup
            if exc.extra:
                payload.update(exc.extra)
            self._json(exc.status, payload)

        def _read_json(self) -> dict | None:
            header = self.headers.get("Content-Length")
            if header is None:
                self._json(411, {"error": "content-length is required"})
                return None
            try:
                size = int(header)
            except ValueError:
                self._json(400, {"error": "bad content-length"})
                return None
            if size < 0:
                self._json(400, {"error": "bad content-length"})
                return None
            if size > MAX_BODY:
                self.close_connection = True
                if size <= 1_000_000:
                    self.rfile.read(size)
                self._json(400, {"error": "job is too large"})
                return None
            raw = self.rfile.read(size)
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json(400, {"error": "invalid JSON"})
                return None
            if not isinstance(body, dict):
                self._json(400, {"error": "job must be a JSON object"})
                return None
            return body

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

        def _text(self, status: int, text: str) -> None:
            self._send(status, "text/plain; charset=utf-8", text.encode("utf-8"))

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.close_connection = True
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("local-offload " + (fmt % args) + "\n")

    return Handler


def warmup(service: Service) -> None:
    try:
        service.tier()
    except ServiceError as exc:
        sys.stderr.write(f"local-offload: {exc}\n")


def main() -> None:
    config = load_config()
    assert_loopback_bind(config["bindHost"])
    service = Service(
        config,
        lambda: read_memory_gb(HostInfo()),
        log_path=log_path_for(Path.home()),
    )
    try:
        httpd = LoopbackServer((config["bindHost"], int(config["port"])), make_handler(service))
    except OSError as exc:
        raise SystemExit(f"cannot listen on {config['bindHost']}:{config['port']}: {exc}") from exc
    threading.Thread(target=warmup, args=(service,), daemon=True).start()
    sys.stderr.write(f"local-offload listening on http://{config['bindHost']}:{config['port']}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
