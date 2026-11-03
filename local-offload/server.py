#!/usr/bin/env python3
"""Loopback code-slice generator.

The host agent sizes one slice, POSTs it here, and applies the text.
This process does not read a repository, apply a patch, or run tests.
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
INSTALL_COMMAND = [
    "bash",
    "-c",
    "curl -fsSL --retry 3 --max-time 180 https://ollama.com/install.sh | sh",
]
OLLAMA_CANDIDATES = ("/usr/local/bin/ollama", "/usr/bin/ollama", "/bin/ollama")

MAX_BODY = 64 * 1024
GENERATE_TIMEOUT = 1800
SYSTEM_PROMPT = (
    "You generate one code slice and nothing else. "
    "You have no tools, no filesystem, and no network. "
    "Do not run commands. Output only the code the host will apply, with no markdown fences."
)
_PARAM_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")


class JobError(Exception):
    pass


class ServiceError(Exception):
    def __init__(self, message: str, status: int = 503) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class InstalledModel:
    name: str
    billion: float | None


@dataclass(frozen=True)
class Probe:
    kind: str
    value: float | None = None


def assert_loopback_bind(host: str) -> None:
    if host != "127.0.0.1":
        raise SystemExit("refusing to bind anything except 127.0.0.1")


def assert_loopback_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise SystemExit("ollama host must be http://127.0.0.1")


def _validate_rung(rung: dict) -> None:
    if not isinstance(rung, dict):
        raise SystemExit("invalid config: rung")
    tag = rung.get("tag")
    if not isinstance(tag, str) or tag.count(":") != 1 or tag.startswith(":") or tag.endswith(":"):
        raise SystemExit("invalid config: model tag")
    instruction = rung.get("instruction")
    if not isinstance(instruction, str) or "\n" in instruction or instruction.strip() == "":
        raise SystemExit("invalid config: instruction")
    for key in ("nominalBillion", "minBillion", "maxBillion"):
        if not isinstance(rung.get(key), (int, float)):
            raise SystemExit(f"invalid config: {key}")
    if rung["minBillion"] > rung["maxBillion"]:
        raise SystemExit("invalid config: parameter band")
    if not isinstance(rung.get("maxTokens"), int) or rung["maxTokens"] < 1:
        raise SystemExit("invalid config: maxTokens")


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise SystemExit("invalid config")
    if config.get("bindHost") != "127.0.0.1":
        raise SystemExit("invalid config: bindHost is 127.0.0.1")
    if config.get("fallbackRung") != "7B":
        raise SystemExit("invalid config: fallback rung is 7B")
    if config.get("lowBelowGb") != 8 or config.get("highAboveGb") != 16:
        raise SystemExit("invalid config: memory boundaries are 8 and 16 GB")
    assert_loopback_url(str(config.get("ollamaHost", "")))
    port = config.get("port")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        raise SystemExit("invalid config: port")
    limits = config.get("limits")
    if not isinstance(limits, dict):
        raise SystemExit("invalid config: limits")
    for key in ("instructionChars", "testChars", "contextChars"):
        value = limits.get(key)
        if not isinstance(value, int) or value < 1:
            raise SystemExit(f"invalid config: limits.{key}")
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
        return pinned, True
    fitting.sort(key=lambda model: (abs(model.billion - rung["nominalBillion"]), model.name))
    return fitting[0].name, False


def build_prompt(slice_limit: str, instruction: str, test: str, context: str) -> str:
    parts = [f"Slice limit: {slice_limit}.", "Instruction:", instruction]
    if test:
        parts.extend(["Test the host will run:", test])
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


def ensure_ollama(find, install, start, healthy, sleep=time.sleep, attempts: int = 60):
    if healthy():
        return find()
    binary = find()
    if binary is None:
        install()
        binary = find()
        if not binary:
            raise ServiceError("ollama install failed")
    if healthy():
        return binary
    start(binary)
    for _ in range(attempts):
        if healthy():
            return binary
        sleep(0.5)
    raise ServiceError("ollama is not running")


def find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    for candidate in OLLAMA_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def install_ollama() -> None:
    completed = subprocess.run(INSTALL_COMMAND, check=False)
    if completed.returncode != 0:
        raise ServiceError("ollama install failed")


def start_ollama(binary: str) -> None:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    subprocess.Popen(
        [binary, "serve"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=None,
        start_new_session=True,
    )


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
            raise ServiceError("ollama returned invalid JSON")
    if not isinstance(parsed, dict):
        raise ServiceError("ollama returned invalid JSON")
    return parsed


def chat_payload(model: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.1, "num_predict": max_tokens},
    }


class OllamaClient:
    def __init__(self, host: str) -> None:
        assert_loopback_url(host)
        self.host = host.rstrip("/")

    def healthy(self) -> bool:
        try:
            self._get("/api/tags", 2)
        except ServiceError:
            return False
        return True

    def list_models(self) -> list[InstalledModel]:
        data = self._get("/api/tags", 10)
        found: list[InstalledModel] = []
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

    def pull(self, name: str) -> None:
        data = self._post(
            "/api/pull",
            {"name": name, "stream": False},
            None,
            "model pull failed",
        )
        if data.get("error"):
            raise ServiceError("model pull failed")

    def generate(self, model: str, prompt: str, max_tokens: int) -> str:
        data = self._post(
            "/api/chat",
            chat_payload(model, prompt, max_tokens),
            GENERATE_TIMEOUT,
            "generation failed",
            502,
        )
        if data.get("error"):
            raise ServiceError("generation failed", status=502)
        message = data.get("message") if isinstance(data.get("message"), dict) else {}
        content = message.get("content")
        if not isinstance(content, str) or content == "":
            raise ServiceError("empty generation", status=502)
        return content

    def _get(self, path: str, timeout: float) -> dict:
        return self._request("GET", path, None, timeout, "ollama is not running", 503)

    def _post(self, path: str, payload: dict, timeout: float | None, failure: str, status: int = 503) -> dict:
        return self._request("POST", path, payload, timeout, failure, status)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None,
        timeout: float | None,
        failure: str,
        status: int,
    ) -> dict:
        data = None
        headers = {"Accept": "application/json", "User-Agent": "local-offload"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.host + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            try:
                exc.read()
            finally:
                exc.close()
            sys.stderr.write(f"local-offload: ollama {path} returned {exc.code}\n")
            raise ServiceError(failure, status=status) from exc
        except urllib.error.URLError as exc:
            raise ServiceError("ollama is not running") from exc
        return decode_ollama_body(raw)


class Service:
    def __init__(self, config: dict, memory_gb, ollama, ensure) -> None:
        self.config = config
        self.memory_gb = memory_gb
        self.ollama = ollama
        self.ensure = ensure
        self._lock = threading.Lock()
        self._tier: dict | None = None

    def tier(self) -> dict:
        with self._lock:
            if self._tier is not None:
                return dict(self._tier)
            self.ensure()
            try:
                memory = self.memory_gb()
            except Exception as exc:
                sys.stderr.write(f"local-offload: memory read failed: {exc}\n")
                memory = None
            rung = select_rung(memory, self.config)
            name, needs_pull = pick_model(rung, self.ollama.list_models())
            if needs_pull:
                sys.stderr.write(f"local-offload: pulling {name}\n")
                self.ollama.pull(name)
            self._tier = {
                "model": name,
                "parameterClass": rung["parameterClass"],
                "slice": rung["slice"],
                "instruction": rung["instruction"],
            }
            sys.stderr.write(
                f"local-offload: {name} {rung['parameterClass']} slice={rung['slice']}\n"
            )
            return dict(self._tier)

    def job(self, body: dict) -> str:
        instruction, test, context = validate_job(body, self.config)
        tier = self.tier()
        rung = rung_by_class(self.config, tier["parameterClass"])
        prompt = build_prompt(tier["slice"], instruction, test, context)
        return self.ollama.generate(tier["model"], prompt, rung["maxTokens"])


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
                self._json(exc.status, {"error": str(exc)})
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
                text = self.service.job(body)
            except JobError as exc:
                self._json(400, {"error": str(exc)})
                return
            except ServiceError as exc:
                sys.stderr.write(f"local-offload: {exc}\n")
                self._json(exc.status, {"error": str(exc)})
                return
            self._text(200, text)

        def _path(self) -> str:
            return urllib.parse.urlsplit(self.path).path

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
    assert_loopback_url(config["ollamaHost"])
    ollama = OllamaClient(config["ollamaHost"])
    service = Service(
        config,
        lambda: read_memory_gb(HostInfo()),
        ollama,
        lambda: ensure_ollama(find_ollama, install_ollama, start_ollama, ollama.healthy),
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
