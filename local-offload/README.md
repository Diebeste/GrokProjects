# Local offload

Loopback service that drafts one code slice with Ollama. The Cursor host assigns the slice, applies the text, and runs the test it defined. The service does not read a repository, apply a patch, or run a test.

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

## Run

From `~/.local-offload`, not from `~`:

```bash
cd ~/.local-offload
python3 server.py
```

The process listens on `http://127.0.0.1:8765`. There is no API key. It binds to `127.0.0.1` only.

If `ollama` is missing from `PATH`, the first `GET /tier` runs the official installer at `https://ollama.com/install.sh` and then starts `ollama serve` on `127.0.0.1:11434`. The installer needs root, outbound network, and `zstd` (the Linux archive is zstd-compressed). If install or startup fails, `GET /tier` returns 503 and the host does the work.

The first `GET /tier` also pulls the pinned model for this machine when no installed model fits the rung. That download can take several minutes.

```bash
curl -sS http://127.0.0.1:8765/tier
curl -sS http://127.0.0.1:8765/job \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"Write a Python function add(a, b) that returns the sum.","test":"add(1, 2) is 3"}'
```

`GET /tier` returns JSON: `model`, `parameterClass`, `slice` (`function`, `file`, or `multifile`), and a one-line `instruction`. `POST /job` returns the generated text only.

## Memory ladder

The service reads discrete GPU VRAM when it can: `nvidia-smi` total memory, or Linux AMD `mem_info_vram_total`. Otherwise it reads system RAM (`/proc/meminfo`, `sysctl hw.memsize`, or Windows CIM `TotalPhysicalMemory`). Figures are 1024-based GB, so 8192 MiB is 8 GB. A failed read uses the 7B tag and stops.

| Memory | Class | Slice | Pinned tag |
| --- | --- | --- | --- |
| under 8 GB | 7B | function | `qwen2.5-coder:7b` |
| 8–16 GB | 14B | file | `qwen2.5-coder:14b` |
| above 16 GB | 32B | multifile | `qwen2.5-coder:32b` |

Tags are pinned in `config.json`. If an installed model has a parameter count in that rung's band, the service uses it. If several fit, the pinned tag wins when it is installed; otherwise the closest parameter count is used. If none fit, the service pulls the pinned tag.

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

The unit tests cover the ladder and the HTTP handlers without a live model.
