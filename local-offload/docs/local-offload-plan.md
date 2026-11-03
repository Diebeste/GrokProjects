# Local offload, v1

Cursor on your machine offloads a bounded coding job to a local model, then checks the result itself.

## Decisions

- The host agent assigns the work and reviews what comes back.
- The local service only generates the slice. It does not see the repo, credentials, or files.
- The host applies the result and runs the test it already defined. One retry, then the host takes the work back.
- Slice size follows a memory ladder. Under 8 GB: one short function. 8–16 GB: one file. Above that: a multi-file change, still with a host check.
- Open-ended design and the final answer stay on the host.
- One localhost endpoint. No API key. If Cursor cannot reach it, the host does the work itself.
- Ollama is the runtime. Install and start it when it is missing.
- Pick the model from a short fixed ladder using VRAM, or RAM when there is no GPU. A failed memory read pulls a 7B-class code model and stops. The ladder is config, not a search for the newest model.
- Cursor only. Claude Code, Codex, and Grok wait until this loop is reliable.

## Done when

A Cursor skill reads the tier from localhost, sends one bounded coding job, and accepts or rejects it by running the host's test.
