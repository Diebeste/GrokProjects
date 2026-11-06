# Local offload

Loopback service that spends the user's local hardware on bounded work: boilerplate, test scaffolding, docstrings, mechanical edits, format conversions, and summaries of large files or logs. The Cursor or Codex host keeps open-ended design, debugging, and anything security-sensitive. It applies the result and reruns the test once.

The service talks to any local OpenAI-compatible server. It probes `GET /v1/models`, then calls `POST /v1/chat/completions`. Ollama's native `/api/tags` and `/api/chat` are used only when `/v1/models` is missing.

## Install

`install.py` installs this skill from git. The default `--repo` is `https://github.com/Diebeste/GrokProjects`, and the skill lives in that repo's `local-offload` subfolder. Pass `--path` to copy a git checkout the host already has. It does not look up a store path. Install once, then start the service, then call `/tier`.

```bash
python3 install.py
python3 install.py --repo https://github.com/Diebeste/GrokProjects
python3 install.py --path /path/to/local-offload
```

With no arguments it clones `https://github.com/Diebeste/GrokProjects` and copies the `local-offload` subfolder. On the MacBook, to copy a checkout you already have (not from `~`):

```bash
python3 install.py --path /path/to/local-offload
```

It writes the service, config, tests, README, installer, and both skills to `~/.local-offload` (`/Users/macbookdesarai/.local-offload` on this Mac). It also copies the skills to `~/.cursor/skills/local-offload/SKILL.md` and `~/.codex/skills/local-offload/SKILL.md`. The installer prints the next commands.

The service never installs a backend and never pulls a model. Those commands run only when you pass them, and only after the user says yes:

```bash
python3 install.py --setup
python3 install.py --setup mlx
python3 install.py --pull
python3 install.py --never
```

## Run

From `~/.local-offload`, not from `~`:

```bash
cd ~/.local-offload
python3 server.py
```

The process listens on `http://127.0.0.1:8765`. There is no API key. It binds to `127.0.0.1` only. Python 3.9 is enough (`from __future__ import annotations` is set; the code does not use 3.10-only syntax).

On startup it probes, in order:

| Name | Base URL |
| --- | --- |
| `baseUrl` in `config.json`, when set | user URL |
| Ollama | `http://127.0.0.1:11434` |
| LM Studio | `http://127.0.0.1:1234` |
| llama.cpp server | `http://127.0.0.1:8080` |
| MLX (`mlx_lm.server`) | `http://127.0.0.1:8080` |

A URL is probed once. llama.cpp and MLX share port 8080, so the first configured name wins. Put the server you actually run earlier in `backends`, or set `baseUrl`.

```bash
curl -sS http://127.0.0.1:8765/tier
curl -sS http://127.0.0.1:8765/job \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"Write a Python function add(a, b) that returns the sum.","test":"add(1, 2) is 3"}'
```

GET /tier returns JSON with the selected model, parameter class, slice (function, file, or multifile), instruction, backend, API type, needsPull, and modelOptions. Each installed option includes its model name, supported slice, parameter class, and known size. POST /job accepts the exact installed model name in an optional model field; the service validates it and applies that model’s slice limit. Without testCommand, the endpoint returns generated text only.

## When no backend is running

`GET /tier` returns 503. It does not run an installer. The body has a one-line `hint` and a `setup` object: detected OS, arch, and memory; the recommended server (Ollama); the model tag and download size for this hardware; the exact commands; and `ask`. On Apple Silicon, `setup.alternatives` includes MLX.

The host asks the user the `setup.ask` string:

`No local model found. Install <server> + <model> (~X GB)? [yes / not now / never]`

- yes: `python3 install.py --setup`. macOS uses `brew install ollama` when Homebrew is present, otherwise the official Ollama.app zip. Windows runs the official `OllamaSetup.exe` installer. Linux runs `https://ollama.com/install.sh`. The command prints progress, pulls the recommended model, and verifies `GET /tier`. `python3 install.py --setup mlx` is the Apple Silicon alternative. The `ollama` binary is the first of: `PATH`, `/Applications/Ollama.app/Contents/Resources/ollama`, `~/Applications/Ollama.app/Contents/Resources/ollama`, `/usr/local/bin/ollama`, `/opt/homebrew/bin/ollama`, and on Windows `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`. `pull` uses that full path. On macOS the server is started with `open -a Ollama`, then `127.0.0.1:11434` is polled; if the app does not answer, setup starts that binary with `serve`.
- not now: the host does this task itself and asks again next session. Nothing is saved.
- never: `python3 install.py --never` writes `{"neverAsk": true}` to `~/.local-offload/config`. Later 503 responses set `setup.ask` to null and `setup.neverAsk` to true, and the host stops asking.

Never install without the user's yes. `python3 install.py --pull` only downloads the recommended model, and only when a backend is already up.

## Memory ladder

The service reads discrete GPU VRAM when it can: `nvidia-smi` total memory, or Linux AMD `mem_info_vram_total`. Otherwise it reads system RAM (`/proc/meminfo`, `sysctl hw.memsize`, or Windows CIM `TotalPhysicalMemory`). Figures are 1024-based GB, so 8192 MiB is 8 GB. A failed read uses the 7B tag and stops.

The figure is discrete GPU VRAM when that is available, otherwise total RAM. A rung is comfortable only when memory clears `comfortGb` in `config.json` (7B at 8 GB, 14B at 24 GB, 32B at 48 GB). Between `tightGb` and that line the model can load but may swap, so it is offered as a choice and is not the default. On a 16 GB machine the recommended default is `qwen2.5-coder:7b`; `qwen2.5-coder:14b` is the better-quality option and may swap. Override the thresholds in `comfortGb` and `tightGb`. A saved `preferredModel` or `quality` (`fast`, `balanced`, `best`) in the state config wins over that default.

| Memory | Recommended | Also offered |
| --- | --- | --- |
| under 24 GB | 7B function, `qwen2.5-coder:7b` (4.7 GB) | 14B when RAM is at least 12 GB |
| 24–48 GB | 14B file, `qwen2.5-coder:14b` (9 GB) | 32B as a tight fit from 24 GB |
| 48 GB and above | 32B multifile, `qwen2.5-coder:32b` (19 GB) | smaller rungs as the faster option |

Tags and sizes are pinned in `config.json`. If an installed model has a parameter count in that rung's band, the service uses it. If several fit, the pinned tag wins when it is installed; otherwise the closest parameter count is used. If none fit, `/tier` sets `needsPull` and does not download anything. A backend that lists models with no parameter size in the id is used as-is.

`/tier` is probed again on every call, so the API (`openai` or Ollama native) matches the backend as it is now. `POST /job` returns `model`, `backend`, `durationMs`, `loadMs` when known, `tokensPerSec` when known, `attempts`, and `tokensSaved` in JSON, and the same values as response headers for `text/plain`. The function and file tiers accept exactly one target. Only the multifile tier accepts several. `pytest` is used only when `importlib.util.find_spec("pytest")` succeeds; otherwise a unittest-style file falls back to `python3 -m unittest`, and any other pytest command returns a clear error. `/tier` lists `testRunners`.

State files `log.jsonl` and `config` (including `neverAsk`, `preferredModel`, and `quality`) live in `LOCAL_OFFLOAD_HOME`, or `stateDir` from `config.json`, or `~/.local-offload`.

`python3 install.py --set-model qwen2.5-coder:14b` only saves `preferredModel`. It does not install a server or pull a model. `python3 install.py --setup --model <tag>` or `--quality fast|balanced|best` also saves the choice, and it installs or pulls only when that server or that model is missing. When the hardware fits more than one model, `setup.ask` lists them, with `[recommended]` on the default.

## Local test loop

Send `repo`, `files`, `targets`, and `testCommand` to run the loop. The service copies the listed files (or the repo when `files` is omitted) into a temp scratch directory, writes the model's code there, and runs the command with a timeout. Failure output goes back to the model. It retries up to `maxAttempts` (default 5) and stops at `overallTimeoutSec`. The response is JSON: passing `files` plus `testOutput`, or a failure `report`. The real repo is never written. The host applies `files` only when `ok` is true and reruns the test once.

`testCommand` is an argv array and must start with an allow-listed prefix: `python3 -m pytest`, `python3 -m unittest`, `python -m pytest`, `python -m unittest`, `npm test`, or `go test`. The process gets a scrubbed environment (no secrets copied in) and the arguments cannot contain shell operators or URLs.

Each attempt is appended to `~/.local-offload/log.jsonl` with the attempt number and `tokensSaved` (about 4 characters per token of the local prompt and reply). The log does not store the prompt or the file text.

## Verified local model run

A live smoke test sent a bounded add-function fix through the local service to Mellum 2.1 Q5_K_M. The model returned a patch that passed the scratch unittest on the first attempt. On the tested RTX 4070 Laptop GPU, Ollama reported a 39% CPU / 61% GPU split; generation was about 40 seconds for that task. This is one observed run, not a general speed guarantee.

## Skills

Cursor follows `SKILL.md`. Copy that file to `~/.cursor/skills/local-offload/SKILL.md`.

Codex follows `codex/SKILL.md`. Install it at `~/.codex/skills/local-offload/SKILL.md`:

```bash
mkdir -p ~/.codex/skills/local-offload
cp codex/SKILL.md ~/.codex/skills/local-offload/SKILL.md
```

Invoke it as `$local-offload`, or let Codex select it when a bounded slice should be drafted locally.

## Tests

```bash
python3 -m unittest test_server.py
```

The unit tests cover the ladder, backend probing, the no-backend 503 and its setup payload, the never-ask flag, the scratch retry loop, and the HTTP handlers. They do not need a live model.
