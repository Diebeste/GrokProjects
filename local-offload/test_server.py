#!/usr/bin/env python3
"""Ladder and HTTP tests. No live model required."""

from __future__ import annotations

import ast
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import install as project_install
import server
from server import (
    AMD_VRAM_GLOB,
    CONFIG_PATH,
    DARWIN_RAM_COMMAND,
    NVIDIA_COMMAND,
    WINDOWS_RAM_COMMAND,
    Backend,
    BackendClient,
    CommandResult,
    InstalledModel,
    JobError,
    LoopbackServer,
    Service,
    ServiceError,
    ask_sentence,
    assert_loopback_bind,
    assert_loopback_url,
    build_setup,
    chat_payload,
    command_allowed,
    decode_ollama_body,
    endpoint_list,
    load_config,
    make_handler,
    ollama_chat_payload,
    parse_billion,
    pick_model,
    probe_backends,
    read_memory_gb,
    read_never_ask,
    select_rung,
    user_config_path,
    write_never_ask,
)


class Cmd:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeHost:
    def __init__(self, which=None, runs=None, files=None, globs=None, system: str = "Linux") -> None:
        self._which = which or {}
        self._runs = runs or {}
        self._files = files or {}
        self._globs = globs or {}
        self._system = system

    def which(self, name: str):
        return self._which.get(name)

    def run(self, args):
        key = tuple(args)
        if key not in self._runs:
            raise AssertionError(f"unexpected command: {list(args)}")
        value = self._runs[key]
        if isinstance(value, BaseException):
            raise value
        return value

    def read(self, path: str) -> str:
        if path not in self._files:
            raise OSError(path)
        value = self._files[path]
        if isinstance(value, BaseException):
            raise value
        return value

    def glob(self, pattern: str):
        return list(self._globs.get(pattern, []))

    def system(self) -> str:
        return self._system


class FakeOllama:
    def __init__(self, models, text: str = "def add(a, b):\n    return a + b\n") -> None:
        self.models = list(models)
        self.text = text
        self.pulled = []
        self.generated = []
        self.error = None
        self.name = "ollama"
        self.api = "openai"
        self.base_url = "http://127.0.0.1:11434"

    def list_models(self):
        return list(self.models)

    def pull(self, name: str) -> None:
        self.pulled.append(name)
        raise AssertionError("pull is not automatic")

    def generate(self, model: str, prompt: str, max_tokens: int, summary: bool = False) -> str:
        self.generated.append(
            {"model": model, "prompt": prompt, "max_tokens": max_tokens, "summary": summary}
        )
        if self.error:
            raise self.error
        return self.text


def nvidia_host(stdout: str, ram_kib: int = 33554432) -> FakeHost:
    return FakeHost(
        which={"nvidia-smi": "/usr/bin/nvidia-smi"},
        runs={tuple(NVIDIA_COMMAND): Cmd(0, stdout)},
        files={"/proc/meminfo": f"MemTotal:       {ram_kib} kB\n"},
    )


class MemoryTests(unittest.TestCase):
    def test_nvidia_vram_wins_over_a_larger_ram_figure(self) -> None:
        cfg = load_config()
        cases = [
            ("4096\n", "7B"),
            ("8191\n", "7B"),
            ("8192\n", "14B"),
            ("16384\n", "14B"),
            ("16385\n", "32B"),
            ("4096\n4096\n", "14B"),
        ]
        for stdout, klass in cases:
            with self.subTest(stdout=stdout):
                memory = read_memory_gb(nvidia_host(stdout))
                self.assertEqual(select_rung(memory, cfg)["parameterClass"], klass)

    def test_nvidia_failure_does_not_fall_through_to_ram(self) -> None:
        host = FakeHost(
            which={"nvidia-smi": "/usr/bin/nvidia-smi"},
            runs={tuple(NVIDIA_COMMAND): Cmd(1, "", "no devices")},
            files={"/proc/meminfo": "MemTotal:       33554432 kB\n"},
        )
        self.assertIsNone(read_memory_gb(host))

    def test_nvidia_non_numeric_output_is_a_failed_read(self) -> None:
        host = nvidia_host("[N/A]\n")
        self.assertIsNone(read_memory_gb(host))

    def test_nvidia_timeout_is_a_failed_read(self) -> None:
        host = FakeHost(
            which={"nvidia-smi": "/usr/bin/nvidia-smi"},
            runs={tuple(NVIDIA_COMMAND): subprocess.TimeoutExpired(NVIDIA_COMMAND, 10)},
            files={"/proc/meminfo": "MemTotal:       33554432 kB\n"},
        )
        self.assertIsNone(read_memory_gb(host))

    def test_amd_vram_is_used_when_nvidia_is_absent(self) -> None:
        path = "/sys/class/drm/card0/device/mem_info_vram_total"
        host = FakeHost(
            files={
                path: "8589934592\n",
                "/proc/meminfo": "MemTotal:       33554432 kB\n",
            },
            globs={AMD_VRAM_GLOB: [path]},
        )
        memory = read_memory_gb(host)
        self.assertAlmostEqual(memory, 8.0)
        self.assertEqual(select_rung(memory, load_config())["parameterClass"], "14B")

    def test_amd_read_failure_does_not_fall_through_to_ram(self) -> None:
        path = "/sys/class/drm/card0/device/mem_info_vram_total"
        host = FakeHost(
            files={"/proc/meminfo": "MemTotal:       33554432 kB\n"},
            globs={AMD_VRAM_GLOB: [path]},
        )
        self.assertIsNone(read_memory_gb(host))

    def test_linux_ram_boundaries(self) -> None:
        cfg = load_config()
        cases = [
            (4194304, "7B"),
            (8388608, "14B"),
            (16777216, "14B"),
            (16777217, "32B"),
            (16387280, "14B"),
        ]
        for kib, klass in cases:
            with self.subTest(kib=kib):
                host = FakeHost(files={"/proc/meminfo": f"MemTotal:       {kib} kB\n"})
                rung = select_rung(read_memory_gb(host), cfg)
                self.assertEqual(rung["parameterClass"], klass)

    def test_missing_linux_ram_is_a_failed_read(self) -> None:
        self.assertIsNone(read_memory_gb(FakeHost()))
        host = FakeHost(files={"/proc/meminfo": "MemFree: 100 kB\n"})
        self.assertIsNone(read_memory_gb(host))

    def test_darwin_ram(self) -> None:
        host = FakeHost(
            system="Darwin",
            runs={tuple(DARWIN_RAM_COMMAND): Cmd(0, "17179869184\n")},
        )
        self.assertAlmostEqual(read_memory_gb(host), 16.0)

    def test_windows_ram_uses_cim_total(self) -> None:
        host = FakeHost(
            system="Windows",
            runs={tuple(WINDOWS_RAM_COMMAND): Cmd(0, "17179869184\n")},
        )
        self.assertAlmostEqual(read_memory_gb(host), 16.0)
        self.assertNotIn("dxdiag", " ".join(WINDOWS_RAM_COMMAND).lower())

    def test_windows_ram_failure(self) -> None:
        host = FakeHost(
            system="Windows",
            runs={tuple(WINDOWS_RAM_COMMAND): Cmd(1, "")},
        )
        self.assertIsNone(read_memory_gb(host))

    def test_unknown_system_without_a_gpu_fails_closed(self) -> None:
        self.assertIsNone(read_memory_gb(FakeHost(system="FreeBSD")))

    def test_source_does_not_call_dxdiag(self) -> None:
        text = Path(server.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("dxdiag", text)


class LadderTests(unittest.TestCase):
    def test_rung_boundaries(self) -> None:
        cfg = load_config()
        cases = [
            (0, "7B", "function"),
            (7.999, "7B", "function"),
            (8, "14B", "file"),
            (16, "14B", "file"),
            (16.001, "32B", "multifile"),
            (None, "7B", "function"),
            (float("nan"), "7B", "function"),
            (-0.1, "7B", "function"),
        ]
        for memory, klass, slice_limit in cases:
            with self.subTest(memory=memory):
                rung = select_rung(memory, cfg)
                self.assertEqual(rung["parameterClass"], klass)
                self.assertEqual(rung["slice"], slice_limit)

    def test_parameter_bands_and_pinned_fallback(self) -> None:
        cfg = load_config()
        cases = [
            (1, 6.0, False, None),
            (1, 5.99, True, "qwen2.5-coder:7b"),
            (1, 9.0, False, None),
            (1, 9.01, True, "qwen2.5-coder:7b"),
            (10, 12.0, False, None),
            (10, 16.0, False, None),
            (10, 11.9, True, "qwen2.5-coder:14b"),
            (10, 16.01, True, "qwen2.5-coder:14b"),
            (40, 30.0, False, None),
            (40, 36.0, False, None),
            (40, 29.9, True, "qwen2.5-coder:32b"),
            (40, 70.0, True, "qwen2.5-coder:32b"),
        ]
        for memory, billion, needs_pull, pinned in cases:
            with self.subTest(memory=memory, billion=billion):
                name, pull = pick_model(select_rung(memory, cfg), [InstalledModel("custom", billion)])
                self.assertEqual(pull, needs_pull)
                self.assertEqual(name, pinned if needs_pull else "custom")

    def test_prefers_pinned_tag_when_it_is_installed(self) -> None:
        installed = [
            InstalledModel("other:14b", 14.0),
            InstalledModel("qwen2.5-coder:14b", 14.8),
        ]
        name, pull = pick_model(select_rung(10, load_config()), installed)
        self.assertEqual((name, pull), ("qwen2.5-coder:14b", False))

    def test_tie_breaks_on_name(self) -> None:
        installed = [
            InstalledModel("m-model:15b", 15.0),
            InstalledModel("a-model:13b", 13.0),
        ]
        name, pull = pick_model(select_rung(10, load_config()), installed)
        self.assertEqual((name, pull), ("a-model:13b", False))

    def test_smaller_or_larger_installed_model_does_not_satisfy_the_rung(self) -> None:
        cfg = load_config()
        name, pull = pick_model(select_rung(12, cfg), [InstalledModel("qwen2.5-coder:7b", 7.6)])
        self.assertEqual((name, pull), ("qwen2.5-coder:14b", True))
        name, pull = pick_model(select_rung(4, cfg), [InstalledModel("qwen2.5-coder:32b", 32.8)])
        self.assertEqual((name, pull), ("qwen2.5-coder:7b", True))

    def test_failed_memory_stays_on_the_7b_tag(self) -> None:
        rung = select_rung(None, load_config())
        name, pull = pick_model(rung, [InstalledModel("qwen2.5-coder:32b", 32.8)])
        self.assertEqual(rung["slice"], "function")
        self.assertEqual((name, pull), ("qwen2.5-coder:7b", True))

    def test_parse_billion(self) -> None:
        self.assertEqual(parse_billion("7.6B"), 7.6)
        self.assertEqual(parse_billion("14.8B"), 14.8)
        self.assertEqual(parse_billion("32.8B"), 32.8)
        self.assertEqual(parse_billion("qwen2.5-coder:7b"), 7.0)
        self.assertEqual(parse_billion("qwen2.5-coder:14b"), 14.0)
        self.assertEqual(parse_billion("qwen2.5-coder:32b-instruct"), 32.0)
        self.assertIsNone(parse_billion("qwen2"))
        self.assertIsNone(parse_billion(""))

    def test_service_does_not_pull_when_the_model_is_missing(self) -> None:
        ollama = FakeOllama([])
        service = Service(load_config(), lambda: 20.0, ollama)
        payload = service.tier()
        self.assertEqual(payload["slice"], "multifile")
        self.assertEqual(payload["model"], "qwen2.5-coder:32b")
        self.assertTrue(payload["needsPull"])
        self.assertEqual(payload["pull"], "python3 install.py --pull")
        self.assertEqual(ollama.pulled, [])
        self.assertTrue(service.tier()["needsPull"])
        self.assertEqual(ollama.generated, [])

    def test_service_uses_an_installed_model_in_band(self) -> None:
        ollama = FakeOllama([InstalledModel("local-coder:14b", 14.2)])
        service = Service(load_config(), lambda: 12.0, ollama)
        payload = service.tier()
        self.assertEqual(payload["model"], "local-coder:14b")
        self.assertEqual(payload["slice"], "file")
        self.assertFalse(payload["needsPull"])
        self.assertEqual(ollama.pulled, [])

    def test_unsized_installed_model_is_used(self) -> None:
        ollama = FakeOllama([InstalledModel("local-model", None)])
        service = Service(load_config(), lambda: 4.0, ollama)
        payload = service.tier()
        self.assertEqual(payload["model"], "local-model")
        self.assertFalse(payload["needsPull"])

    def test_service_memory_failure_does_not_pull(self) -> None:
        ollama = FakeOllama([InstalledModel("qwen2.5-coder:32b", 32.8)])
        service = Service(load_config(), lambda: None, ollama)
        payload = service.tier()
        self.assertEqual(payload["parameterClass"], "7B")
        self.assertEqual(payload["slice"], "function")
        self.assertTrue(payload["needsPull"])
        self.assertEqual(ollama.pulled, [])

    def test_memory_exception_does_not_pull(self) -> None:
        def boom():
            raise RuntimeError("sysfs vanished")

        ollama = FakeOllama([InstalledModel("qwen2.5-coder:32b", 32.8)])
        service = Service(load_config(), boom, ollama)
        payload = service.tier()
        self.assertEqual(payload["parameterClass"], "7B")
        self.assertTrue(payload["needsPull"])
        self.assertEqual(ollama.pulled, [])

    def test_job_does_not_pull_a_missing_model(self) -> None:
        ollama = FakeOllama([])
        service = Service(load_config(), lambda: None, ollama)
        with self.assertRaises(ServiceError) as caught:
            service.job({"instruction": "write add"})
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(str(caught.exception), "model is not installed")
        self.assertEqual(ollama.pulled, [])
        self.assertEqual(ollama.generated, [])


class ProbeTests(unittest.TestCase):
    def test_native_ollama_is_only_the_fallback(self) -> None:
        calls = []

        def fetch(method, url, payload=None, timeout=2):
            calls.append(url)
            if url == "http://127.0.0.1:11434/v1/models":
                return 0, {}
            if url == "http://127.0.0.1:11434/api/tags":
                return 200, {"models": [{"name": "qwen2.5-coder:7b"}]}
            return 0, {}

        backend = probe_backends(endpoint_list(load_config()), fetch)
        self.assertEqual(backend.name, "ollama")
        self.assertEqual(backend.api, "ollama")
        self.assertEqual(calls[0], "http://127.0.0.1:11434/v1/models")
        self.assertEqual(calls[1], "http://127.0.0.1:11434/api/tags")

    def test_user_base_url_is_probed_first(self) -> None:
        cfg = load_config()
        cfg["baseUrl"] = "http://127.0.0.1:1234"

        def fetch(method, url, payload=None, timeout=2):
            if url == "http://127.0.0.1:1234/v1/models":
                return 200, {"data": [{"id": "local-model"}]}
            return 0, {}

        backend = probe_backends(endpoint_list(cfg), fetch)
        self.assertEqual((backend.name, backend.api, backend.base_url), ("custom", "openai", "http://127.0.0.1:1234"))

    def test_shared_port_is_probed_once(self) -> None:
        seen = []

        def fetch(method, url, payload=None, timeout=2):
            seen.append(url)
            if url == "http://127.0.0.1:8080/v1/models":
                return 200, {"data": [{"id": "local"}]}
            return 0, {}

        backend = probe_backends(endpoint_list(load_config()), fetch)
        self.assertEqual(backend.name, "llamacpp")
        self.assertEqual(backend.api, "openai")
        self.assertEqual(seen.count("http://127.0.0.1:8080/v1/models"), 1)

    def test_no_backend_when_every_probe_fails(self) -> None:
        def fetch(method, url, payload=None, timeout=2):
            return 0, {}

        self.assertIsNone(probe_backends(endpoint_list(load_config()), fetch))

    def test_server_does_not_install_or_pull(self) -> None:
        text = Path(server.__file__).read_text(encoding="utf-8")
        self.assertNotIn("def ensure_ollama", text)
        self.assertNotIn("def install_ollama", text)
        self.assertNotIn("INSTALL_COMMAND", text)
        self.assertNotIn("perform_setup", text)
        self.assertNotIn("/api/pull", text)


class BackendClientTests(unittest.TestCase):
    def test_chat_payload_has_no_tools(self) -> None:
        payload = chat_payload("qwen2.5-coder:7b", "write add", 128)
        self.assertEqual(payload["model"], "qwen2.5-coder:7b")
        self.assertIs(payload["stream"], False)
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["max_tokens"], 128)
        self.assertIn("no tools", payload["messages"][0]["content"])
        native = ollama_chat_payload("qwen2.5-coder:7b", "write add", 128)
        self.assertEqual(native["options"]["num_predict"], 128)
        self.assertNotIn("tools", native)
        self.assertNotIn("tools", native["options"])

    def test_openai_generate_posts_chat_completions(self) -> None:
        seen = {}

        def fetch(method, url, payload=None, timeout=2):
            seen["method"] = method
            seen["url"] = url
            seen["payload"] = payload
            return 200, {"choices": [{"message": {"content": "abc"}}]}

        client = BackendClient(Backend("lmstudio", "http://127.0.0.1:1234", "openai"), fetch)
        self.assertEqual(client.generate("local", "write add", 32), "abc")
        self.assertEqual(seen["url"], "http://127.0.0.1:1234/v1/chat/completions")
        self.assertNotIn("tools", seen["payload"])
        self.assertEqual(seen["payload"]["messages"][1]["content"], "write add")
        self.assertEqual(seen["payload"]["max_tokens"], 32)

    def test_ollama_native_generate_is_the_fallback(self) -> None:
        seen = {}

        def fetch(method, url, payload=None, timeout=2):
            seen["url"] = url
            seen["payload"] = payload
            return 200, {"message": {"content": "abc"}}

        client = BackendClient(Backend("ollama", "http://127.0.0.1:11434", "ollama"), fetch)
        self.assertEqual(client.generate("qwen2.5-coder:7b", "write add", 32), "abc")
        self.assertEqual(seen["url"], "http://127.0.0.1:11434/api/chat")
        self.assertNotIn("tools", seen["payload"])
        self.assertEqual(seen["payload"]["options"]["num_predict"], 32)

    def test_empty_generation_is_502(self) -> None:
        def fetch(method, url, payload=None, timeout=2):
            return 200, {"choices": [{"message": {"content": ""}}]}

        client = BackendClient(Backend("lmstudio", "http://127.0.0.1:1234", "openai"), fetch)
        with self.assertRaises(ServiceError) as caught:
            client.generate("m", "p", 8)
        self.assertEqual(caught.exception.status, 502)
        self.assertEqual(str(caught.exception), "empty generation")

    def test_list_models_prefers_parameter_size(self) -> None:
        def fetch(method, url, payload=None, timeout=2):
            self.assertEqual(url, "http://127.0.0.1:11434/api/tags")
            return 200, {
                "models": [
                    {"name": "custom:latest", "details": {"parameter_size": "14.8B"}},
                    {"name": "qwen2.5-coder:7b", "details": {}},
                    {"name": "", "details": {}},
                    "skip",
                ]
            }

        client = BackendClient(Backend("ollama", "http://127.0.0.1:11434", "ollama"), fetch)
        models = client.list_models()
        self.assertEqual(
            [(model.name, model.billion) for model in models],
            [("custom:latest", 14.8), ("qwen2.5-coder:7b", 7.0)],
        )

    def test_decode_ollama_body(self) -> None:
        pretty = b'{\n  "message": {"content": "hi"}\n}\n'
        self.assertEqual(decode_ollama_body(pretty)["message"]["content"], "hi")
        ndjson = b'{"status":"pulling"}\n{"status":"success"}\n'
        self.assertEqual(decode_ollama_body(ndjson)["status"], "success")
        self.assertEqual(decode_ollama_body(b""), {})
        with self.assertRaises(ServiceError):
            decode_ollama_body(b"not-json")

    def test_refuses_non_loopback(self) -> None:
        assert_loopback_bind("127.0.0.1")
        assert_loopback_url("http://127.0.0.1:11434")
        with self.assertRaises(SystemExit):
            assert_loopback_bind("0.0.0.0")
        with self.assertRaises(SystemExit):
            assert_loopback_url("http://localhost:11434")
        with self.assertRaises(SystemExit):
            assert_loopback_url("http://10.0.0.5:11434")
        with self.assertRaises(SystemExit):
            BackendClient(Backend("ollama", "http://10.0.0.5:11434", "ollama"))


class HttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ollama = FakeOllama([InstalledModel("qwen2.5-coder:7b", 7.6)])
        self.service = Service(load_config(), lambda: 4.0, self.ollama)
        self.httpd = LoopbackServer(("127.0.0.1", 0), make_handler(self.service))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self._wait_until_listening()

    def _wait_until_listening(self) -> None:
        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("server did not listen")

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
        last = None
        for _ in range(20):
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                conn.request(method, path, body=body, headers=headers or {})
                resp = conn.getresponse()
                data = resp.read()
                status = resp.status
                ctype = resp.getheader("Content-Type")
                conn.close()
                return status, ctype, data
            except (ConnectionRefusedError, OSError) as exc:
                last = exc
                time.sleep(0.02)
        raise AssertionError(last)

    def post(self, payload: dict):
        return self.request(
            "POST",
            "/job",
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json"},
        )

    def test_listens_on_loopback(self) -> None:
        self.assertEqual(self.httpd.server_address[0], "127.0.0.1")

    def test_get_tier_shape(self) -> None:
        status, ctype, data = self.request("GET", "/tier")
        self.assertEqual(status, 200)
        self.assertIn("application/json", ctype)
        payload = json.loads(data)
        for key in ("model", "parameterClass", "slice", "instruction", "backend", "api", "needsPull"):
            self.assertIn(key, payload)
        self.assertEqual(payload["model"], "qwen2.5-coder:7b")
        self.assertEqual(payload["parameterClass"], "7B")
        self.assertEqual(payload["slice"], "function")
        self.assertFalse(payload["needsPull"])
        self.assertNotIn("\n", payload["instruction"])
        self.assertIn("one short function", payload["instruction"])
        status, _, _ = self.request("GET", "/tier?x=1")
        self.assertEqual(status, 200)

    def test_post_job_returns_only_generated_text(self) -> None:
        self.ollama.text = "def add(a, b):\n    return a + b\n"
        status, ctype, data = self.post(
            {
                "instruction": "Write add.",
                "test": "add(1, 2) is 3",
                "context": "existing helper",
                "path": "/etc/passwd",
                "tools": [{"name": "bash"}],
            }
        )
        self.assertEqual(status, 200)
        self.assertTrue(ctype.startswith("text/plain"))
        self.assertEqual(data, self.ollama.text.encode("utf-8"))
        prompt = self.ollama.generated[0]["prompt"]
        self.assertIn("Slice limit: function.", prompt)
        self.assertIn("Write add.", prompt)
        self.assertIn("add(1, 2) is 3", prompt)
        self.assertIn("existing helper", prompt)
        self.assertNotIn("/etc/passwd", prompt)
        self.assertNotIn("bash", prompt)
        self.assertEqual(self.ollama.generated[0]["model"], "qwen2.5-coder:7b")
        self.assertEqual(self.ollama.generated[0]["max_tokens"], 1024)
        self.assertEqual(self.ollama.pulled, [])

    def test_null_test_and_context_are_omitted(self) -> None:
        status, ctype, _ = self.post({"instruction": "write add", "test": None, "context": None})
        self.assertEqual(status, 200)
        self.assertTrue(ctype.startswith("text/plain"))
        prompt = self.ollama.generated[0]["prompt"]
        self.assertNotIn("Test the host will run:", prompt)
        self.assertNotIn("Context pasted by the host:", prompt)

    def test_context_limit(self) -> None:
        context = "c" * 12000
        status, _, _ = self.post({"instruction": "write add", "context": context})
        self.assertEqual(status, 200)
        status, _, data = self.post({"instruction": "write add", "context": context + "x"})
        self.assertEqual(status, 400)
        self.assertIn(b"context exceeds 12000 characters", data)
        self.assertEqual(len(self.ollama.generated), 1)

    def test_instruction_is_required_and_bounded(self) -> None:
        status, _, data = self.post({})
        self.assertEqual(status, 400)
        self.assertIn(b"instruction is required", data)
        status, _, data = self.post({"instruction": "   "})
        self.assertEqual(status, 400)
        self.assertIn(b"instruction is required", data)
        status, _, data = self.post({"instruction": "x" * 4001})
        self.assertEqual(status, 400)
        self.assertIn(b"instruction exceeds 4000 characters", data)
        status, _, data = self.post({"instruction": "write add", "test": 1})
        self.assertEqual(status, 400)
        self.assertIn(b"test must be a string", data)
        self.assertEqual(self.ollama.generated, [])
        self.assertEqual(self.ollama.pulled, [])

    def test_oversize_body(self) -> None:
        raw = b"{" + b"x" * 70000 + b"}"
        status, _, data = self.request(
            "POST",
            "/job",
            raw,
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertIn(b"job is too large", data)
        self.assertEqual(self.ollama.generated, [])

    def test_invalid_json_and_non_object(self) -> None:
        status, _, data = self.request("POST", "/job", b"{", {"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertIn(b"invalid JSON", data)
        status, _, data = self.request("POST", "/job", b"[]", {"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertIn(b"job must be a JSON object", data)

    def test_post_requires_content_length(self) -> None:
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        sock.sendall(b"POST /job HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        sock.close()
        self.assertTrue(data.startswith(b"HTTP/1.1 411 "))
        self.assertIn(b"content-length is required", data)

    def test_unknown_routes(self) -> None:
        status, _, data = self.request("GET", "/health")
        self.assertEqual(status, 404)
        self.assertIn(b"not found", data)
        status, _, _ = self.request("POST", "/tier", b"{}", {"Content-Type": "application/json"})
        self.assertEqual(status, 404)

    def test_tier_503_includes_setup_and_does_not_install(self) -> None:
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.service.backend = None
        self.service.discover = lambda: None
        self.service.home = Path(home.name)
        self.service.system = "Darwin"
        self.service.arch = "arm64"
        self.service.memory_gb = lambda: 16.0
        self.service._tier = None
        status, ctype, data = self.request("GET", "/tier")
        self.assertEqual(status, 503)
        self.assertIn("application/json", ctype)
        payload = json.loads(data)
        self.assertEqual(payload["error"], "no local backend found")
        self.assertIn("Ollama.app", payload["hint"])
        setup = payload["setup"]
        self.assertEqual(setup["os"], "Darwin")
        self.assertEqual(setup["arch"], "arm64")
        self.assertEqual(setup["memoryGb"], 16)
        self.assertEqual(setup["recommended"]["server"], "ollama")
        self.assertEqual(setup["recommended"]["model"], "qwen2.5-coder:14b")
        self.assertEqual(setup["recommended"]["downloadGb"], 9)
        self.assertEqual(setup["ask"], ask_sentence("ollama", "qwen2.5-coder:14b", 9))
        self.assertFalse(setup["neverAsk"])
        self.assertEqual(setup["alternatives"][0]["server"], "mlx")
        self.assertIn("python3 install.py --setup", setup["commands"]["setup"])
        self.assertFalse(user_config_path(Path(home.name)).exists())
        self.assertEqual(self.ollama.pulled, [])
        self.assertEqual(self.ollama.generated, [])

    def test_never_ask_suppresses_the_question(self) -> None:
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        write_never_ask(user_config_path(Path(home.name)))
        self.service.backend = None
        self.service.discover = lambda: None
        self.service.home = Path(home.name)
        self.service.system = "Linux"
        self.service.arch = "x86_64"
        self.service.memory_gb = lambda: 4.0
        self.service._tier = None
        status, _, data = self.request("GET", "/tier")
        self.assertEqual(status, 503)
        setup = json.loads(data)["setup"]
        self.assertTrue(setup["neverAsk"])
        self.assertIsNone(setup["ask"])
        self.assertIn("install.sh", setup["hint"])
        self.assertEqual(setup["recommended"]["model"], "qwen2.5-coder:7b")
        self.assertEqual(setup["alternatives"], [])
        self.assertTrue(read_never_ask(user_config_path(Path(home.name))))

    def test_generation_failure_is_502(self) -> None:
        self.ollama.error = ServiceError("generation failed", 502)
        status, _, data = self.post({"instruction": "write add"})
        self.assertEqual(status, 502)
        self.assertIn(b"generation failed", data)


class ContractTests(unittest.TestCase):
    def test_config_pins_the_three_rungs(self) -> None:
        cfg = load_config()
        self.assertEqual(
            [rung["tag"] for rung in cfg["rungs"]],
            ["qwen2.5-coder:7b", "qwen2.5-coder:14b", "qwen2.5-coder:32b"],
        )
        self.assertEqual([rung["slice"] for rung in cfg["rungs"]], ["function", "file", "multifile"])
        self.assertEqual([rung["parameterClass"] for rung in cfg["rungs"]], ["7B", "14B", "32B"])
        self.assertEqual(cfg["bindHost"], "127.0.0.1")
        self.assertEqual(cfg["port"], 8765)
        self.assertEqual(
            [item["baseUrl"] for item in cfg["backends"]],
            [
                "http://127.0.0.1:11434",
                "http://127.0.0.1:1234",
                "http://127.0.0.1:8080",
                "http://127.0.0.1:8080",
            ],
        )
        self.assertEqual(cfg["maxAttempts"], 5)
        self.assertEqual(cfg["fallbackRung"], "7B")
        self.assertEqual(cfg["lowBelowGb"], 8)
        self.assertEqual(cfg["highAboveGb"], 16)
        for rung in cfg["rungs"]:
            self.assertNotIn("\n", rung["instruction"])

    def test_skill_matches_the_service_contract(self) -> None:
        skill = (CONFIG_PATH.parent / "SKILL.md").read_text(encoding="utf-8").lower()
        cfg = load_config()
        self.assertIn(f"127.0.0.1:{cfg['port']}", skill)
        for rung in cfg["rungs"]:
            self.assertIn(rung["slice"], skill)
        self.assertIn("no local model found", skill)
        self.assertIn("not now", skill)
        self.assertIn("never", skill)
        self.assertIn("boilerplate", skill)
        self.assertIn("summar", skill)
        self.assertIn("python3 install.py --setup", skill)
        self.assertIn("do the work yourself", skill)
        self.assertIn("open-ended design", skill)

    def test_readme_explains_run_and_install(self) -> None:
        readme = (CONFIG_PATH.parent / "README.md").read_text(encoding="utf-8")
        self.assertIn("python3 server.py", readme)
        self.assertIn("https://ollama.com/install.sh", readme)
        self.assertIn("127.0.0.1:8765", readme)
        self.assertIn("~/.cursor/skills/local-offload/SKILL.md", readme)
        self.assertIn("~/.codex/skills/local-offload/SKILL.md", readme)

    def test_codex_skill_contract(self) -> None:
        path = CONFIG_PATH.parent / "codex" / "SKILL.md"
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\n"))
        front, body = text.split("---", 2)[1:]
        fields = {}
        for line in front.strip().splitlines():
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip('"')
        self.assertEqual(fields["name"], "local-offload")
        self.assertGreater(len(fields["description"]), 0)
        self.assertNotIn("\n", fields["description"])
        lowered = body.lower()
        self.assertIn("get http://127.0.0.1:8765/tier", lowered)
        self.assertIn("post", lowered)
        self.assertIn("/job", lowered)
        for slice_limit in ("function", "file", "multifile"):
            self.assertIn(slice_limit, lowered)
        self.assertIn("no local model found", lowered)
        self.assertIn("not now", lowered)
        self.assertIn("never", lowered)
        self.assertIn("boilerplate", lowered)
        self.assertIn("do the work yourself", lowered)
        self.assertIn("open-ended design", lowered)
        cursor = (CONFIG_PATH.parent / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("name: local-offload", cursor)
        self.assertNotEqual(text, cursor)


class InstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.checkout = self._checkout(self.root / "src")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _git(self, *args: str, cwd: Path) -> None:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "user.name=Local Offload",
                "-c",
                "user.email=local-offload@example.com",
                *args,
            ],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)

    def _checkout(self, repo: Path) -> Path:
        repo.mkdir()
        bundle = CONFIG_PATH.parent
        for rel in project_install.PAYLOAD:
            target = repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((bundle / rel).read_bytes())
        self._git("init", "-b", "main", cwd=repo)
        self._git("add", ".", cwd=repo)
        self._git("commit", "-m", "initial", cwd=repo)
        return repo

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["HOME"] = str(self.home)
        env["GIT_TERMINAL_PROMPT"] = "0"
        return subprocess.run(
            [sys.executable, str(CONFIG_PATH.parent / "install.py"), *args],
            cwd=self.home,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_default_directory_is_under_home(self) -> None:
        self.assertEqual(project_install.runtime_dir(Path.home()), Path.home() / ".local-offload")

    def test_path_copies_payload_and_skills(self) -> None:
        completed = self._run("--path", str(self.checkout))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        dest = self.home / ".local-offload"
        for rel in project_install.PAYLOAD:
            self.assertEqual((dest / rel).read_bytes(), (self.checkout / rel).read_bytes(), rel)
        self.assertEqual(
            (self.home / ".cursor/skills/local-offload/SKILL.md").read_bytes(),
            (self.checkout / "SKILL.md").read_bytes(),
        )
        self.assertEqual(
            (self.home / ".codex/skills/local-offload/SKILL.md").read_bytes(),
            (self.checkout / "codex/SKILL.md").read_bytes(),
        )
        self.assertIn(f"cd {dest}", completed.stdout)
        self.assertIn("python3 server.py", completed.stdout)
        self.assertIn("curl -sS http://127.0.0.1:8765/tier", completed.stdout)
        self.assertIn("curl -sS http://127.0.0.1:8765/job", completed.stdout)
        self.assertNotIn("/cursor/stores", completed.stdout)

    def test_repo_clones_then_pulls(self) -> None:
        first = self._run("--repo", str(self.checkout))
        self.assertEqual(first.returncode, 0, first.stderr)
        readme = self.checkout / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8") + "\nupdated\n", encoding="utf-8")
        self._git("add", "README.md", cwd=self.checkout)
        self._git("commit", "-m", "update readme", cwd=self.checkout)
        second = self._run("--repo", str(self.checkout))
        self.assertEqual(second.returncode, 0, second.stderr)
        installed = (self.home / ".local-offload" / "README.md").read_text(encoding="utf-8")
        self.assertIn("updated", installed)

    def test_rejects_a_path_that_is_not_a_checkout(self) -> None:
        loose = self.root / "loose"
        loose.mkdir()
        completed = self._run("--path", str(loose))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("not a git checkout", completed.stderr)
        self.assertFalse((self.home / ".local-offload" / "server.py").exists())

    def test_installer_source_has_no_store_path(self) -> None:
        text = (CONFIG_PATH.parent / "install.py").read_text(encoding="utf-8")
        self.assertNotIn("/cursor/stores", text)

    def test_skills_install_once_from_a_checkout_or_url(self) -> None:
        for rel in ("SKILL.md", "codex/SKILL.md", "README.md"):
            text = (CONFIG_PATH.parent / rel).read_text(encoding="utf-8")
            self.assertIn("install once", text.lower())
            self.assertIn("python3 install.py --path", text)
            self.assertIn("python3 install.py --repo", text)
            self.assertIn("~/.local-offload", text)
            self.assertIn("python3 server.py", text)

    def test_never_flag_is_saved(self) -> None:
        completed = self._run("--never")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        path = self.home / ".local-offload" / "config"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIs(data["neverAsk"], True)
        self.assertIn("never-ask", completed.stdout)
        self.assertFalse((self.home / ".local-offload" / "server.py").exists())

    def test_setup_commands_match_the_os(self) -> None:
        linux = project_install.install_argv("Linux", lambda name: None)
        self.assertIn("https://ollama.com/install.sh", " ".join(linux))
        darwin = project_install.install_argv("Darwin", lambda name: "/opt/homebrew/bin/brew")
        self.assertEqual(darwin, ["brew", "install", "ollama"])
        app = project_install.install_argv("Darwin", lambda name: None)
        self.assertIn("Ollama-darwin.zip", " ".join(app))
        windows = " ".join(project_install.install_argv("Windows", lambda name: None))
        self.assertIn("OllamaSetup.exe", windows)
        self.assertIn("https://ollama.com/download/OllamaSetup.exe", windows)

    def test_pull_refuses_without_a_backend(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            project_install.perform_pull(self.home, discover=lambda config: None)
        self.assertIn("no local backend", str(caught.exception))

    def test_setup_uses_the_fake_runner(self) -> None:
        host = _FakeSetup("Linux", "x86_64")
        body = project_install.perform_setup("ollama", self.home, host)
        binary = "/usr/local/bin/ollama"
        self.assertIn("https://ollama.com/install.sh", " ".join(host.commands[0]))
        self.assertEqual(host.spawned, [[binary, "serve"]])
        self.assertEqual(host.commands[-1][:2], [binary, "pull"])
        self.assertIn("qwen2.5-coder", host.commands[-1][-1])
        self.assertNotIn(["open", "-a", "Ollama"], host.commands)
        self.assertIn("qwen2.5-coder", body)

    def test_macos_opens_the_app_and_pulls_with_the_bundle_binary(self) -> None:
        app = "/Applications/Ollama.app/Contents/Resources/ollama"
        host = _FakeSetup("Darwin", "arm64", ollama_path=None, files=[app], healthy_after_open=True)
        project_install.perform_setup("ollama", self.home, host)
        self.assertIn("Ollama-darwin.zip", " ".join(host.commands[0]))
        self.assertEqual(host.commands[1], ["open", "-a", "Ollama"])
        self.assertEqual(host.commands[-1][:2], [app, "pull"])
        self.assertEqual(host.spawned, [])

    def test_macos_falls_back_to_the_bundle_binary_when_open_fails(self) -> None:
        app = "/Applications/Ollama.app/Contents/Resources/ollama"
        host = _FakeSetup("Darwin", "arm64", ollama_path=None, files=[app], open_code=1)
        project_install.perform_setup("ollama", self.home, host)
        self.assertEqual(host.commands[1], ["open", "-a", "Ollama"])
        self.assertEqual(host.spawned, [[app, "serve"]])
        self.assertEqual(host.commands[-1][:2], [app, "pull"])

    def test_setup_errors_when_the_binary_is_missing(self) -> None:
        host = _FakeSetup("Darwin", "arm64", ollama_path=None)
        with self.assertRaises(SystemExit) as caught:
            project_install.perform_setup("ollama", self.home, host)
        self.assertIn("ollama was not found", str(caught.exception))
        self.assertIn("Ollama.app", str(caught.exception))
        self.assertEqual(host.spawned, [])

    def test_mlx_setup_is_apple_silicon_only(self) -> None:
        host = _FakeSetup("Darwin", "arm64")
        project_install.perform_setup("mlx", self.home, host)
        flat = " ".join(" ".join(cmd) for cmd in host.commands)
        self.assertIn("mlx-lm", flat)
        self.assertIn("mlx-community", flat)
        with self.assertRaises(SystemExit):
            project_install.perform_setup("mlx", self.home, _FakeSetup("Linux", "x86_64"))


class _FakeSetup:
    def __init__(
        self,
        system: str,
        arch: str,
        ollama_path: str | None = "/usr/local/bin/ollama",
        files=None,
        open_code: int = 0,
        healthy_after_open: bool = False,
        localappdata: str | None = None,
        home: str = "/Users/tester",
    ) -> None:
        self._system = system
        self._arch = arch
        self._ollama_path = ollama_path
        self._files = {Path(path) for path in (files or [])}
        self._open_code = open_code
        self._healthy_after_open = healthy_after_open
        self._localappdata = localappdata
        self._home = Path(home)
        self.commands = []
        self.spawned = []
        self.opened = False

    def system(self) -> str:
        return self._system

    def arch(self) -> str:
        return self._arch

    def which(self, name: str):
        if name == "ollama":
            return self._ollama_path
        return None

    def is_file(self, path: Path) -> bool:
        return Path(path) in self._files

    def home_dir(self) -> Path:
        return self._home

    def localappdata(self) -> str | None:
        return self._localappdata

    def run(self, argv: list) -> None:
        self.commands.append(list(argv))

    def try_run(self, argv: list) -> int:
        self.commands.append(list(argv))
        if list(argv)[:3] == ["open", "-a", "Ollama"] and self._open_code == 0:
            self.opened = True
        return self._open_code

    def spawn(self, argv: list) -> None:
        self.spawned.append(list(argv))

    def memory_gb(self):
        return 16.0

    def healthy(self, url: str) -> bool:
        if self.opened and self._healthy_after_open:
            return True
        return bool(self.spawned)

    def sleep(self, seconds: float) -> None:
        return None

    def fetch_tier(self, url: str):
        return 200, '{"model":"qwen2.5-coder:14b"}'


class _NamedBackend:
    def __init__(self, name: str, api: str) -> None:
        self.name = name
        self.api = api


class OllamaBinaryTests(unittest.TestCase):
    def test_path_wins_over_later_files(self) -> None:
        seen = []

        def is_file(path: Path) -> bool:
            seen.append(Path(path))
            return True

        found = project_install.find_ollama(
            lambda name: "/opt/homebrew/bin/ollama" if name == "ollama" else None,
            is_file,
            Path("/Users/brand"),
            "C:/Users/brand/AppData/Local",
        )
        self.assertEqual(found, "/opt/homebrew/bin/ollama")
        self.assertEqual(seen, [])

    def test_search_order_and_each_hit(self) -> None:
        home = Path("/Users/brand")
        local = "C:/Users/brand/AppData/Local"
        expected = project_install.ollama_search_paths(home, local)
        self.assertEqual(
            [path.name for path in expected],
            ["ollama", "ollama", "ollama", "ollama", "ollama.exe"],
        )
        self.assertEqual(expected[0], Path("/Applications/Ollama.app/Contents/Resources/ollama"))
        self.assertEqual(
            expected[1],
            home / "Applications" / "Ollama.app" / "Contents" / "Resources" / "ollama",
        )
        self.assertEqual(expected[2], Path("/usr/local/bin/ollama"))
        self.assertEqual(expected[3], Path("/opt/homebrew/bin/ollama"))
        self.assertEqual(expected[4], Path(local) / "Programs" / "Ollama" / "ollama.exe")
        for chosen in expected:
            def is_file(path: Path, chosen: Path = chosen) -> bool:
                return Path(path) == chosen

            found = project_install.find_ollama(lambda name: None, is_file, home, local)
            self.assertEqual(found, str(chosen))

    def test_missing_binary_names_every_location(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            project_install.find_ollama(
                lambda name: None,
                lambda path: False,
                Path("/Users/brand"),
                None,
            )
        message = str(caught.exception)
        self.assertIn("ollama was not found", message)
        self.assertIn("/Applications/Ollama.app/Contents/Resources/ollama", message)
        self.assertIn("~/Applications/Ollama.app/Contents/Resources/ollama", message)
        self.assertIn("/usr/local/bin/ollama", message)
        self.assertIn("/opt/homebrew/bin/ollama", message)
        self.assertIn("ollama.exe", message)

    def test_windows_pull_uses_localappdata_when_path_is_stale(self) -> None:
        local = "C:/Users/brand/AppData/Local"
        binary = Path(local) / "Programs" / "Ollama" / "ollama.exe"
        host = _FakeSetup("Windows", "AMD64", ollama_path=None, files=[binary], localappdata=local)
        project_install.perform_pull(
            Path("/tmp/unused"),
            host,
            discover=lambda config: _NamedBackend("ollama", "ollama"),
        )
        self.assertEqual(host.commands, [[str(binary), "pull", "qwen2.5-coder:14b"]])
        self.assertEqual(host.spawned, [])


class SetupPayloadTests(unittest.TestCase):
    def test_ask_sentence_and_platform_variants(self) -> None:
        cfg = load_config()
        small = build_setup(cfg, "Linux", "x86_64", 4, False)
        self.assertTrue(small["ask"].startswith("No local model found. Install ollama + qwen2.5-coder:7b "))
        self.assertIn("(~4.7 GB)", small["ask"])
        self.assertTrue(small["ask"].endswith("[yes / not now / never]"))
        self.assertEqual(small["ask"], ask_sentence("ollama", "qwen2.5-coder:7b", 4.7))
        mac = build_setup(cfg, "Darwin", "arm64", 32, False)
        self.assertEqual(mac["recommended"]["model"], "qwen2.5-coder:32b")
        self.assertEqual(mac["recommended"]["downloadGb"], 19)
        self.assertEqual(mac["alternatives"][0]["server"], "mlx")
        self.assertEqual(mac["commands"]["setup"], "python3 install.py --setup")
        self.assertEqual(build_setup(cfg, "Darwin", "x86_64", 4, False)["alternatives"], [])
        windows = build_setup(cfg, "Windows", "AMD64", None, False)
        self.assertIn("Ollama installer", windows["hint"])
        self.assertEqual(windows["recommended"]["parameterClass"], "7B")
        self.assertIsNone(windows["memoryGb"])
        silent = build_setup(cfg, "Linux", "x86_64", 4, True)
        self.assertIsNone(silent["ask"])
        self.assertTrue(silent["neverAsk"])

    def test_never_flag_round_trip_keeps_other_keys(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = user_config_path(Path(tmp.name))
        path.parent.mkdir(parents=True)
        path.write_text('{"keep": 1}\n', encoding="utf-8")
        write_never_ask(path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertIs(saved["neverAsk"], True)
        self.assertEqual(saved["keep"], 1)
        self.assertTrue(read_never_ask(path))
        self.assertFalse(read_never_ask(path.parent / "missing"))


class LoopTests(unittest.TestCase):
    def _repo(self, root: Path) -> Path:
        repo = root / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "add.py").write_text("def add(a, b):\n    return 0\n", encoding="utf-8")
        (repo / "tests").mkdir()
        (repo / "tests" / "test_add.py").write_text("def test_add():\n    assert True\n", encoding="utf-8")
        return repo

    def test_fake_model_passes_on_attempt_three(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        repo = self._repo(root)
        original = (repo / "src" / "add.py").read_text(encoding="utf-8")
        log = root / "log.jsonl"
        ollama = FakeOllama(
            [InstalledModel("qwen2.5-coder:7b", 7.6)],
            "def add(a, b):\n    return a + b\n",
        )
        calls = {"n": 0}

        def runner(argv, cwd, timeout, env):
            calls["n"] += 1
            calls["env"] = env
            self.assertEqual(argv[:3], ["python3", "-m", "pytest"])
            self.assertIn("return a + b", (cwd / "src" / "add.py").read_text(encoding="utf-8"))
            self.assertEqual((repo / "src" / "add.py").read_text(encoding="utf-8"), original)
            if calls["n"] < 3:
                return CommandResult(1, "", "fail %s" % calls["n"], False)
            return CommandResult(0, "passed", "", False)

        os.environ["SECRET_TOKEN"] = "super-secret"
        self.addCleanup(lambda: os.environ.pop("SECRET_TOKEN", None))
        service = Service(load_config(), lambda: 4.0, ollama, runner=runner, log_path=log)
        result = service.job(
            {
                "instruction": "Make add return the sum.",
                "files": ["src/add.py", "tests/test_add.py"],
                "targets": ["src/add.py"],
                "repo": str(repo),
                "testCommand": ["python3", "-m", "pytest", "tests/test_add.py"],
            }
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 3)
        self.assertFalse(result["timedOut"])
        self.assertIn("return a + b", result["files"]["src/add.py"])
        self.assertIn("passed", result["testOutput"])
        self.assertEqual((repo / "src" / "add.py").read_text(encoding="utf-8"), original)
        self.assertNotIn("SECRET_TOKEN", calls["env"])
        self.assertNotIn("super-secret", " ".join(calls["env"].values()))
        prompts = [item["prompt"] for item in ollama.generated]
        self.assertEqual(len(prompts), 3)
        self.assertIn("fail 1", prompts[1])
        self.assertIn("fail 2", prompts[2])
        lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([line["attempt"] for line in lines], [1, 2, 3])
        self.assertEqual([line["ok"] for line in lines], [False, False, True])
        self.assertIsInstance(lines[-1]["tokensSaved"], int)
        logged = log.read_text(encoding="utf-8")
        self.assertNotIn("return a + b", logged)
        self.assertNotIn("super-secret", logged)

    def test_command_timeout_returns_a_failure_report(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        repo = root / "repo"
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "add.py").write_text("def add(a, b):\n    return 0\n", encoding="utf-8")
        (repo / "test_slow.py").write_text(
            "import time\nimport unittest\n\n"
            "class Slow(unittest.TestCase):\n"
            "    def test_slow(self):\n"
            "        time.sleep(30)\n",
            encoding="utf-8",
        )
        cfg = load_config()
        cfg["maxAttempts"] = 1
        cfg["attemptTimeoutSec"] = 1
        cfg["overallTimeoutSec"] = 10
        ollama = FakeOllama([InstalledModel("qwen2.5-coder:7b", 7.6)], "def add(a, b):\n    return 1\n")
        service = Service(cfg, lambda: 4.0, ollama, log_path=root / "log.jsonl")
        started = time.monotonic()
        result = service.job(
            {
                "instruction": "Make add return the sum.",
                "files": ["src/add.py", "test_slow.py"],
                "targets": ["src/add.py"],
                "repo": str(repo),
                "testCommand": ["python3", "-m", "unittest", "test_slow"],
            }
        )
        self.assertLess(time.monotonic() - started, 8)
        self.assertFalse(result["ok"])
        self.assertTrue(result["timedOut"])
        self.assertEqual(result["attempts"], 1)
        self.assertIn("timed out", result["report"])
        self.assertIn("return 0", (repo / "src" / "add.py").read_text(encoding="utf-8"))
        record = json.loads((root / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(record["timedOut"])
        self.assertFalse(record["ok"])

    def test_rejects_shell_network_and_secret_files(self) -> None:
        cfg = load_config()
        prefixes = cfg["allowedTestPrefixes"]
        self.assertTrue(command_allowed(["python3", "-m", "pytest", "tests"], prefixes))
        self.assertTrue(command_allowed(["/usr/bin/python3", "-m", "unittest", "t"], prefixes))
        self.assertTrue(command_allowed(["npm", "test"], prefixes))
        self.assertTrue(command_allowed(["go", "test", "./..."], prefixes))
        self.assertFalse(command_allowed(["bash", "-c", "curl http://evil"], prefixes))
        self.assertFalse(command_allowed("python3 -m pytest", prefixes))
        self.assertFalse(command_allowed(["python3", "-m", "pytest", "http://evil"], prefixes))
        ollama = FakeOllama([InstalledModel("qwen2.5-coder:7b", 7.0)])
        service = Service(cfg, lambda: 4.0, ollama)
        with self.assertRaises(JobError):
            service.job(
                {
                    "instruction": "write add",
                    "repo": "/tmp",
                    "targets": ["a.py"],
                    "testCommand": ["bash", "-c", "echo hi"],
                }
            )
        self.assertEqual(ollama.generated, [])
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "repo"
        repo.mkdir()
        (repo / ".env").write_text("TOKEN=1\n", encoding="utf-8")
        with self.assertRaises(JobError) as caught:
            service.job({"instruction": "Summarize this.", "files": [".env"], "repo": str(repo)})
        self.assertIn("secret", str(caught.exception))
        self.assertEqual(ollama.generated, [])

    def test_summary_reads_a_large_file_locally(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "repo"
        (repo / "logs").mkdir(parents=True)
        blob = "ERROR line\n" * 3000
        self.assertGreater(len(blob), 12000)
        (repo / "logs" / "app.log").write_text(blob, encoding="utf-8")
        ollama = FakeOllama([InstalledModel("qwen2.5-coder:7b", 7.0)], "short summary")
        service = Service(load_config(), lambda: 4.0, ollama)
        text = service.job(
            {
                "instruction": "Summarize this log.",
                "files": ["logs/app.log"],
                "repo": str(repo),
            }
        )
        self.assertEqual(text, "short summary")
        self.assertIn("ERROR line", ollama.generated[0]["prompt"])
        self.assertTrue(ollama.generated[0]["summary"])
        self.assertEqual((repo / "logs" / "app.log").read_text(encoding="utf-8"), blob)


class Python39Tests(unittest.TestCase):
    def test_sources_avoid_newer_syntax(self) -> None:
        banned = tuple(
            getattr(ast, name)
            for name in ("Match", "TryStar", "TypeAlias")
            if hasattr(ast, name)
        )
        for name in ("server.py", "install.py", "test_server.py"):
            path = CONFIG_PATH.parent / name
            text = path.read_text(encoding="utf-8")
            self.assertIn("from __future__ import annotations\n", text)
            tree = ast.parse(text)
            for node in ast.walk(tree):
                self.assertNotIsInstance(node, banned)
        for name in ("server.py", "install.py"):
            text = (CONFIG_PATH.parent / name).read_text(encoding="utf-8")
            self.assertNotIn("import tomllib", text)
            self.assertNotIn("except*", text)


if __name__ == "__main__":
    unittest.main()
