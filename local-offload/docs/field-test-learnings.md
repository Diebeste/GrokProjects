# Field Testing & Model Selection Insights

This document captures empirical learnings from field testing the `token-maxing` / `local-offload` skill against real-world local models, hardware rungs, and domain tasks.

---

## 1. Thinking Model Token Starvation

### The Phenomenon
When testing **`qwen3-vl:8b`** (Q4_K_M, 8.2B multimodal vision-language model with native thinking capability enabled by default), the service failed with:
```json
{"error": "empty generation"} // HTTP 502 Bad Gateway
```

### Root Cause Analysis
1. **Chain-of-Thought Generation**: Models with `thinking: true` (e.g., Qwen3-VL, DeepSeek-R1) write an internal reasoning monologue before emitting any answer in `message.content`.
2. **Token Ceiling Depletion**: In bounded offloading, the 7B/8B rung specifies `maxTokens: 1024`. Because `qwen3-vl:8b` generated over 1,000 tokens of internal reasoning debating prompt semantics (e.g. "ignoring spaces and case" vs "ignoring non-alphanumeric"), the generation hit `finish_reason: length` while `message.content` was still `""` (empty string).
3. **Severe Latency Overhead**: Even when increasing the limit to 2,048 tokens, the model consumed all 2,048 tokens strictly on reasoning, taking **~4.5 minutes (270 seconds)** on Apple Silicon before returning an empty string.
4. **Contrast with Direct Code Models**: By comparison, **`qwen2.5-coder:14b`** and **`qwen2.5-coder:7b`** emitted exact, fence-free Python implementations in **2.8 seconds** using only 65 tokens.

### Key Takeaway
> **Thinking models are counter-productive for bounded, mechanical offloading.**
> Offloading loops require deterministic, fast, low-latency completions. If a model with `thinking` capability must be used, the backend request should explicitly disable thinking (e.g., `options: {"thinking": false}`) or allocate a significantly expanded token buffer.

---

## 2. Drafter GGUFs vs. Standalone Models

### The Phenomenon
When attempting to pull and execute `hf.co/sirfyyn/DeepSeek-V4.1-Flash-1017-DSpark-FLE-GGUF:latest` in Ollama, generation aborted with:
```
Error: 500 Internal Server Error: llama-server process has terminated: exit status 1: 
llama_init_from_model: failed to initialize the context: dflash requires ctx_other to be set
```

### Root Cause Analysis
1. **Speculative Decoding Architecture**: The repository contains only the **DSpark speculative-decoding drafter** (Multi-Token Prediction / MTP block with Markov and confidence heads) for DeepSeek-V4.1-Flash, not a standalone model.
2. **Missing Target Context (`ctx_other`)**: In `llama.cpp` and `llama-server`, the `dflash` architecture requires the primary target model's context to be attached as `ctx_other`. It cannot initialize independent context memory.

### Key Takeaway
> **Verify that community GGUF repositories provide complete base models rather than speculative drafters or adapters before pulling.** Standalone local offloading requires a self-contained transformer graph.

---

## 3. Domain Task Performance Matrix

Field testing evaluated the skill across three distinct software engineering domains:

| Domain | Task Tested | Result | Observations |
| :--- | :--- | :--- | :--- |
| **App Development** | E-Commerce `OrderService` (discounts, tax, validation) | **100% Pass** (Attempt 1, 74s on 14B) | Excellent at adhering to docstrings, input validation (`ValueError`), and business math. Highly recommended. |
| **Reverse Engineering** | Binary Telemetry Packet Parser (`struct.unpack`, bitwise flags, checksum) | **Partial / Guardrail Verified** (5 attempts, safe fallback) | Accurately generated `struct.unpack('<2sHHIfBBH')` and bitwise masks. Demonstrated that the 5-attempt retry cap protects against broken test harnesses without workspace corruption. |
| **Game Development** | Pure Math & Algorithms (A*, procedural noise, state machines) | **Suitable for Logic Only** | Pure algorithms excel; game engine code (Unreal/Unity) is limited because C++/C# build systems are outside default test runners. |

---

## 4. Architectural Recommendations for Next Iterations

1. **Thinking Suppression Option**:
   In `server.py`, when sending requests to Ollama for models advertising `thinking` capability:
   ```python
   # Disable thinking by default for bounded mechanical slices
   payload["options"] = {"thinking": False}
   ```
2. **Diagnostic Error Messages**:
   When `choices[0].message.content` is empty but `choices[0].message.reasoning` is populated with `finish_reason == "length"`, return a clear diagnostic:
   ```
   "model exhausted token limit during internal reasoning; use a non-thinking model or disable thinking"
   ```
   rather than a generic `empty generation` error.
3. **Runner Whitelist Extensibility**:
   Allow configuring additional test prefixes in `config.json` (such as `cargo test` for Rust or `make test`) for native systems and game developers.