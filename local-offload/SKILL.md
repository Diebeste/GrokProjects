---
name: local-offload
description: "Draft one bounded code slice on the local Ollama service at 127.0.0.1:8765. Use when the host already knows the slice (one short function, one file, or one multi-file change) and will apply the text and run a test the host defined. If that service is unreachable, do the work on the host."
---

# Local offload

You assign a bounded slice to the local service, apply the text it returns, and run the test you defined. You keep the repository, credentials, and files. You keep open-ended design.

The service generates text only. It has no tools and no filesystem.

## When the service is down

Request `GET http://127.0.0.1:8765/tier`. Allow several minutes on the first call. That call can install Ollama and download the model.

If the connection is refused, the request times out, or the status is not 200, install once when `~/.local-offload/server.py` is missing. Do not install a second time when that file is already there.

From a git checkout the host already has:

```bash
python3 install.py --path /path/to/local-offload
```

When a repo URL exists:

```bash
python3 install.py --repo <git-url>
```

Then start the service and request `/tier` again:

```bash
cd ~/.local-offload
python3 server.py
```

If `~/.local-offload/server.py` was already installed, skip the installer and only start it, then request `/tier` again. If that request is still not 200, do the work yourself.

## Size the slice

Read `slice`, `model`, `parameterClass`, and `instruction` from the JSON body.

| `slice` | What you send |
| --- | --- |
| `function` | One short function |
| `file` | One file |
| `multifile` | One multi-file change |

If the task is larger than that slice, keep the rest and send one fitting slice, or do the whole task yourself. Follow the `instruction` line from `/tier`.

Define the test you will run before you POST. You run that test. The service does not run it.

## POST the job

`POST http://127.0.0.1:8765/job` with `Content-Type: application/json`:

```json
{
  "instruction": "Write a Python function add(a, b) that returns the sum of its arguments.",
  "test": "add(1, 2) is 3 and add(-1, 1) is 0",
  "context": "optional small excerpt you paste"
}
```

`instruction` is required. `test` is the test description you already defined. `context` is a small excerpt you paste. Leave secrets out. There is no API key.

A 200 response is `text/plain`: the generated slice and nothing else. If a single markdown fence wraps it, apply the fenced body. Apply it in the repo yourself, then run your test.

Any other status means you do the work yourself.

## One retry

If your test fails, POST `/job` once more. Keep the same slice. Put a short failure excerpt in `context`. If the second result fails the test, finish the work yourself. Do not send a third job.

You get one retry.
