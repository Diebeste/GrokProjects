---
name: local-offload
description: "Offload boilerplate, scaffolding, mechanical edits, and summaries to the local service at 127.0.0.1:8765. Use when a small local model can do the slice and you will apply the result. If no local backend is running, ask before any install. Keep open-ended design, debugging, and security-sensitive work on the host."
---

# Local offload

You spend fewer paid tokens by sending bounded work to the user's local model. You keep the repository, credentials, and the final apply. The service does not write the real repo.

## What you offload

Send work small models do well:

- boilerplate and repetitive glue
- test scaffolding
- docstrings and comments
- mechanical renames and refactors
- format conversions
- summaries of large files or logs

For a summary, pass the file path. The local model reads the big text. You read only the summary it returns. Do not paste the file into the host context.

## What you keep

Keep open-ended design, debugging, and anything security-sensitive on the host. Do not send secrets, tokens, or `.env` files.

## When no local model is running

Request `GET http://127.0.0.1:8765/tier`.

If the status is 503 and `setup.neverAsk` is true, do the work yourself and do not ask.

If `setup.ask` is a string, ask the user that exact question. It has this form:

`No local model found. Install <server> + <model> (~X GB)? [yes / not now / never]`

On Apple Silicon, `setup.alternatives` may list MLX. You may mention that option. The question you ask is still `setup.ask`.

- yes: from `~/.local-offload`, run `python3 install.py --setup` only when no server is installed. If `setup.choices` lists more than one model, ask which note they want. When the server and model are already installed, save the choice with `python3 install.py --set-model <tag>` (or `--quality fast|balanced|best` on `--setup` only if a pull is still required). Use `python3 install.py --setup mlx` when they chose MLX and MLX is not installed. Never install without the user's yes. The `[recommended]` line is the default when they do not pick.
- not now: do this task yourself. Do not save a flag. Ask again next session.
- never: run `python3 install.py --never`. That writes `~/.local-offload/config` so later sessions do not ask. Then do the work yourself.

If the connection is refused, install once when `~/.local-offload/server.py` is missing, start it, and request `/tier` again.

From a git checkout the host already has:

```bash
python3 install.py --path /path/to/local-offload
```

When a repo URL exists:

```bash
python3 install.py --repo <git-url>
```

Then:

```bash
cd ~/.local-offload
python3 server.py
```

If `~/.local-offload/server.py` was already installed, skip the installer and only start it. If `/tier` is still not 200 after the user declines setup, do the work yourself.

## Size the slice

Read `slice`, `model`, `parameterClass`, `instruction`, and `needsPull` from the JSON body.

| `slice` | What you send |
| --- | --- |
| `function` | One short function |
| `file` | One file |
| `multifile` | One multi-file change |

GET /tier also lists installed models in modelOptions, with each model's supported slice and parameter class. The default may need a pull even when another local model is installed.

If the user requests a different installed model, or needsPull is true while modelOptions is non-empty, offer those exact model names and ask which one to use. For a selected option, add its exact name as the "model" field in the POST /job JSON. The service validates that the model is installed and applies the matching slice limit. If the user chooses an uninstalled recommendation, ask before pulling it; never download a model without the user's agreement.

If `needsPull` is true, the backend is up and the model is missing. Run `python3 install.py --pull` only after the user agrees, then request `/tier` again. Do not pull on your own.

If the task is larger than that slice, keep the rest and send one fitting slice, or do the whole task yourself.

## POST the job

`POST http://127.0.0.1:8765/job` with `Content-Type: application/json`.

Summary or one draft, with no test command:

```json
{
  "instruction": "Summarize this log in one short paragraph.",
  "files": ["logs/app.log"],
  "repo": "/absolute/path/to/repo"
}
```

A 200 response is `text/plain`. Apply a code draft yourself. For a summary, read that text and do not open the whole file.

Local test loop, when the model should iterate on a test:

```json
{
  "instruction": "Make add return the sum of its arguments.",
  "files": ["src/add.py", "tests/test_add.py"],
  "targets": ["src/add.py"],
  "repo": "/absolute/path/to/repo",
  "testCommand": ["python3", "-m", "pytest", "tests/test_add.py"]
}
```

`testCommand` must be an argv array in the allow-list (`pytest`, `unittest`, `npm test`, `go test`). Do not send a shell string, network URLs, or secrets.

The service copies those files into a temp scratch dir, writes the model output there, runs the test with a timeout, and retries up to the configured cap (default 5) within an overall time limit. It never writes the real repo.

A 200 JSON body has `ok`, `attempts`, `files`, `testOutput`, and `report`. If `ok` is true, write `files` into the repo yourself and rerun the test command once. If `ok` is false, or any other status, do the work yourself.

You get one local loop per task. Do not start a second loop after a failure report.
