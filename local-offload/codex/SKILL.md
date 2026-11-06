---
name: local-offload
description: "Offload boilerplate, scaffolding, mechanical edits, and summaries to the local service at 127.0.0.1:8765. Use when Codex already knows the slice, will apply the text, and will rerun the test once. If no local backend is running, ask before any install. Keep open-ended design, debugging, and security-sensitive work."
---

# Local offload

You are Codex. You send one bounded slice to the local model, apply what comes back, and rerun the test you defined. You keep the repository, credentials, and files. Call the service with `curl` against loopback. There is no API key.

## What you offload

Send work small models do well:

- boilerplate and repetitive glue
- test scaffolding
- docstrings and comments
- mechanical renames and refactors
- format conversions
- summaries of large files or logs

For a summary, pass the file path in `files`. The local model reads the big text. You read only the summary. Do not paste the file into your own context.

## What you keep

Keep open-ended design, debugging, and anything security-sensitive. Do not send secrets, tokens, or `.env` files.

## If /tier is not 200

`GET http://127.0.0.1:8765/tier`

```bash
curl -sS --max-time 30 http://127.0.0.1:8765/tier
```

If the status is 503 and `setup.neverAsk` is true, do the work yourself and do not ask.

If `setup.ask` is set, ask the user that exact string. It looks like:

`No local model found. Install <server> + <model> (~X GB)? [yes / not now / never]`

On Apple Silicon, mention MLX only when it is listed in `setup.alternatives`. Still ask `setup.ask`.

- yes: from `~/.local-offload`, run `python3 install.py --setup` only when no server is installed. If the server and the chosen model are already installed, run `python3 install.py --set-model <tag>` and do not pull again. Pass `--quality fast` or `--quality best` with `--setup` only when that model still needs a pull. The `[recommended]` line is the default. Use `python3 install.py --setup mlx` only when the user chose MLX and it is not installed. Never install without the user's yes.
- not now: do this task yourself. Ask again next session. Do not write a flag.
- never: `python3 install.py --never`, which saves `~/.local-offload/config`, then do the work yourself.

If the connection is refused, install once when `~/.local-offload/server.py` is missing. Do not install a second time when that file is already there.

From a git checkout the host already has:

```bash
python3 install.py --path /path/to/local-offload
```

When a repo URL exists:

```bash
python3 install.py --repo <git-url>
```

Then start the service and request `GET http://127.0.0.1:8765/tier` again:

```bash
cd ~/.local-offload
python3 server.py
```

If that file was already installed, skip the installer and only start it. If `/tier` is still not 200 after the user says not now or never, do the work yourself.

## Size the job

Read `slice`, `model`, `parameterClass`, `instruction`, and `needsPull`.

| `slice` | What you send |
| --- | --- |
| `function` | One short function |
| `file` | One file |
| `multifile` | One multi-file change |

GET /tier also lists installed models in modelOptions, with each model's supported slice and parameter class. The default may need a pull even when another local model is installed.

If the user requests a different installed model, or needsPull is true while modelOptions is non-empty, offer those exact model names and ask which one to use. For a selected option, add its exact name as the "model" field in the POST /job JSON. The service validates that the model is installed and applies the matching slice limit. If the user chooses an uninstalled recommendation, ask before pulling it; never download a model without the user's agreement.

If `needsPull` is true, ask before `python3 install.py --pull`. Do not pull by yourself.

If the task is larger than that slice, keep the rest or do the whole task yourself.

## POST /job

Summary, without a test command:

```bash
curl -sS --max-time 300 http://127.0.0.1:8765/job \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"Summarize this log in one short paragraph.","files":["logs/app.log"],"repo":"/absolute/path/to/repo"}'
```

A 200 `text/plain` body is the summary or the draft. Read a summary as-is.

Local test loop:

```bash
curl -sS --max-time 600 http://127.0.0.1:8765/job \
  -H 'Content-Type: application/json' \
  -d '{"instruction":"Make add return the sum.","files":["src/add.py","tests/test_add.py"],"targets":["src/add.py"],"repo":"/absolute/path/to/repo","testCommand":["python3","-m","pytest","tests/test_add.py"]}'
```

`testCommand` is an argv array: `pytest`, `unittest`, `npm test`, or `go test`. No shell string, no network URL, no secrets.

The service copies the files into a scratch dir, runs the test, and retries up to 5 times inside an overall time limit. It never writes the real repo. Log lines go to `~/.local-offload/log.jsonl`.

If the JSON `ok` field is true, write `files` into the repo and rerun that test command once. If `ok` is false, do the work yourself.

You get one local loop per task.
