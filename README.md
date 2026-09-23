<a name="top"></a>

> [!NOTE]
> **This fork adds native support for [Kev](https://github.com/jaredpalmer/kev) System One decision models.** A new `kev-rs` binary loads a Kev checkpoint (Qwen3.5 backbone + pointer head) through the mistral.rs engine and exposes the TypeSafe-compatible `POST /v1/systemone` endpoint. Prefill-only, batched across question branches, with the shared state prefix (attention KV + Gated DeltaNet recurrent state) cached once per state. Runs on CPU, Metal (Apple Silicon) and CUDA. No Python at inference time.
>
> **Pick a model.** Ready-to-serve exports (merged weights + pointer head, Apache-2.0) are on the Hub; `kev-rs` downloads them on first use (`HF_HOME` cache).
>
> | Hub id | Source run | Download | RAM | Notes |
> |---|---|---|---|---|
> | `espetro/kev-0.8b-mistralrs` | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | 2.9 GB | ~4 GB | fp32; lightest, laptop-friendly |
> | `espetro/kev-4b-mistralrs` | [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) | 16 GB | ~18 GB | fp32; middle ground |
> | `espetro/kev-9b-mistralrs` | [jaredpalmer/kev-9b](https://huggingface.co/jaredpalmer/kev-9b) | 17 GB | ~20 GB | bf16; most capable |
>
> **1. Install** (every fork release ships `mistralrs` + `kev-rs` in one archive: Metal, Linux CPU x86_64/aarch64, Windows CPU, consumer CUDA sm86/89/120 on Linux). The installer and `mise` always resolve the newest fork release, prereleases included, so nothing here pins a version:
>
> ```sh
> sh -c "$(curl -fsSL https://raw.githubusercontent.com/espetro/mistral.rs/kev/install.sh)"   # picks Metal / CUDA / CPU for this machine
> # or: mise use -g "github:espetro/mistral.rs[prerelease=true,matching=mistralrs-metal]@latest"   # matching=mistralrs-cpu / mistralrs-cuda128-sm89
> # pin instead: MISTRALRS_INSTALL_TAG=<tag from the releases page> sh -c "$(curl -fsSL .../install.sh)"
> # from source: cargo build --release -p kev-rs --features kev-rs/metal   (kev-rs/cuda, or no feature for CPU)
> ```
>
> Archives for every platform are on the [releases page](https://github.com/espetro/mistral.rs/releases).
>
> **2. Hello world.** Serve the lightest model and ask one question:
>
> ```sh
> kev-rs serve --checkpoint espetro/kev-0.8b-mistralrs --run jaredpalmer/kev-0.8b     # http://127.0.0.1:8009
> curl localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
>   "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
>   "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
>                                "criteria": {"returns": "Exchanges, refunds", "shipping": "Delays, lost packages", "billing": "Charges, invoices"}}}
> }'
> # {"model":"kev-latest","answers":{"department":{"type":"choice","choice":"shipping","confidence":0.34,"probabilities":{...}}},"usage":{...},"latency_ms":...}
> ```
>
> Swap in `espetro/kev-4b-mistralrs` / `jaredpalmer/kev-4b` for the 4B model, or `espetro/kev-9b-mistralrs` / `jaredpalmer/kev-9b` for the 9B. `--isq 8` quantizes the checkpoint in-situ at load (roughly halves memory again, at some accuracy cost), and `--paged` enables PagedAttention for the attention layers.
>
> **3. Use the API: single, batch, parallel.**
>
> - *Single*: one `state` + one question, as above.
> - *Batch*: put several questions in one request. They are independent branches over the same state; the state prefix is prefilled once (and cached across requests), then all branches are scheduled together in the engine:
>
>   ```sh
>   curl localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
>     "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
>     "questions": {
>       "department": {"type": "choice", "instructions": "Which team should handle this?",
>                      "criteria": {"returns": "Exchanges, refunds", "shipping": "Delays, lost packages", "billing": "Charges, invoices"}},
>       "escalate":   {"type": "noul", "instructions": "Should a human agent take over right away?"},
>       "frustration":{"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}
>     }
>   }'
>   ```
>
> - *Parallel*: the server is async; concurrent HTTP requests are batched by the mistral.rs scheduler like any other mistral.rs traffic (measured on an 8-core CPU box with the 0.8B model: 8 concurrent copies of the request above finished in 2.0 s wall-clock vs 0.6 s for one). E.g. from a shell: `seq 1 8 | xargs -P 8 -I{} curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d @req.json`. Watch `GET /v1/models` -> `prefix_cache.hits` grow when requests share a state.
>
> `POST /v1/systemone/permute` (score a choice question under every option order) and `POST /v1/systemone/separate` (score each question alone, without the others in context) are also served; bearer auth is enabled by setting `KEV_API_KEY`; the TypeSafe SDK works unchanged: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="kev-latest")`.
>
> **4. UI.** Kev's playground (Next.js, in the Kev repo) talks only to the `/v1/*` routes, so point it at `kev-rs`:
>
> ```sh
> git clone https://github.com/jaredpalmer/kev.git && cd kev/playground && npm install
> KEV_API=http://127.0.0.1:8009 npm run dev -- -p 3001   # open http://localhost:3001
> ```
>
> The same archive also has the stock `mistralrs` CLI for ordinary chat/completions: `mistralrs serve -m Qwen/Qwen3.5-0.8B` (OpenAI API on :1234, web UI at `/ui`), `mistralrs run -m <model>` for an interactive session.
>
> **5. Docker** (Linux CPU, amd64 + arm64; image built by the release workflow with both binaries):
>
> ```sh
> docker run --rm -p 8009:8009 -v hf-cache:/data --entrypoint kev-rs ghcr.io/espetro/mistral.rs:cpu-kev \
>   serve --host 0.0.0.0 --checkpoint espetro/kev-0.8b-mistralrs --run jaredpalmer/kev-0.8b
> # local export instead of a Hub id: -v ~/kev-0.8b:/ckpt:ro ... --checkpoint /ckpt
> # stock server: docker run --rm -p 1234:1234 -v hf-cache:/data ghcr.io/espetro/mistral.rs:cpu-kev serve -m Qwen/Qwen3.5-0.8B
> ```
>
> `cpu-kev` tracks the newest fork release (also tagged `cpu-<version>`). The container is CPU-only: Docker on macOS cannot reach Metal, so on Apple Silicon use the native binary from step 1; for NVIDIA use the CUDA archive on the host (no fork CUDA image yet).
>
> **6. Export a checkpoint yourself** (for your own Kev runs, a different precision, or to rebuild the Hub exports). From a Kev checkout, with Python + torch:
>
> ```sh
> git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
> uv run --extra serve python /path/to/mistral.rs/kev-rs/scripts/export_checkpoint.py --run jaredpalmer/kev-9b --out ~/kev-9b --dtype bf16
> kev-rs serve --checkpoint ~/kev-9b --run jaredpalmer/kev-9b
> ```
>
> The exporter loads the base and merges the LoRA in `--dtype` (fp32/bf16/fp16); a bf16 9B export peaks around 20 GB of RAM, fp32 around 40 GB. The base weights come from the Hub. Upload the directory with `hf upload <you>/kev-9b-mistralrs ~/kev-9b .` and `--checkpoint <you>/kev-9b-mistralrs` works everywhere.
>
> **Hosted / browser.** No in-browser inference: `kev-rs` is a native binary (Candle CPU/Metal/CUDA), there is no WASM target, and even the 0.8B export is 2.9 GB of fp32 weights. What works today is a hosted API plus a browser UI: the Linux CPU archive or the Docker image runs anywhere a container or shell is available (a Hugging Face Space with a Docker SDK, a Kaggle/Colab notebook, a VPS), and Kev's playground or the TypeSafe SDK talks to it over HTTP. These hosted paths have not been exercised from this fork yet; the Linux binary, the installer, the Hub download and the Docker image have.
>
> **Status.** CPU parity with Kev's PyTorch reference on the 0.8B fixtures is max |dp| 0.00011, 0 argmax flips (34 questions); the bf16 9B export shows max |dp| 0.023, 0/40 flips, and `--isq 8` shows 1/40 flips on the same set. Kev's `tests/test_api.py` passes 10/10 against `kev-rs`. Metal and CUDA binaries are built by CI but their runtime parity and speed are not yet measured; `date_facts` and `option_isolation` checkpoints are not supported. Details in [kev-rs/README.md](kev-rs/README.md); the engine change is a generic `return_hidden_states` prefill mode in `mistralrs-core` (intended for upstream). The release workflow's CUDA-on-free-runner leg, the `cpu-kev` image tag and the installer overrides are **fork-only workarounds** and will not be proposed upstream. Everything else is stock upstream mistral.rs.

<!--
<h1 align="center">
  mistral.rs
</h1>
-->

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="res/banner-dark.gif">
    <source media="(prefers-color-scheme: light)" srcset="res/banner-light.gif">
    <img src="res/banner-dark.png" alt="mistral.rs - Fast, flexible LLM inference." width="100%" style="max-width: 800px;">
  </picture>
</div>

<p align="center">
  | <a href="https://docs.mistralrs.dev/"><b>Documentation</b></a> | <a href="https://docs.mistralrs.dev/quickstart/"><b>Quickstart</b></a> | <a href="https://docs.mistralrs.dev/reference/supported-models/"><b>Supported models</b></a> | <a href="https://crates.io/crates/mistralrs"><b>Rust SDK</b></a> | <a href="https://docs.mistralrs.dev/guides/python/getting-started/"><b>Python SDK</b></a> | <a href="https://discord.gg/SZrecqK8qw"><b>Discord</b></a> |
</p>

<p align="center">
  <a href="https://github.com/EricLBuehler/mistral.rs/stargazers">
    <img src="https://img.shields.io/github/stars/EricLBuehler/mistral.rs?style=social&label=Star" alt="GitHub stars">
  </a>
</p>

## Latest

- **Muse Glimmer 30B**: native text, image, and video inference with ATEM tool calling, reasoning controls, LoRA, ISQ/UQFF, and companion-projector GGUF loading. [Model notes](https://docs.mistralrs.dev/guides/models/model-family-notes/#muse-glimmer)
- **GGUF loading**: load a local file with `-f`, or select a published artifact with `--quant`. Tokenizer, configuration, and multimodal projector files are discovered when the available metadata identifies them unambiguously. [Guide](https://docs.mistralrs.dev/guides/models/run-gguf/)
- **OpenAI-compatible Skills**: upload `/v1/skills` bundles and reference them from Responses requests for reusable procedures, helper scripts, and local data. [Guide](https://docs.mistralrs.dev/guides/agents/skills/)
- **OpenAI-compatible file inputs**: upload `/v1/files`, attach Responses `input_file` or Chat `file` parts, and mount request files into shell/code sessions. [Guide](https://docs.mistralrs.dev/guides/agents/file-inputs/)
- **DiffusionGemma**: block-diffusion text generation. Fully integrated: paged attention, prefix caching, ISQ, multimodal, and tool calling. [Guide](https://docs.mistralrs.dev/guides/models/use-block-diffusion/)
- **Anthropic Messages API**: `mistralrs serve` now exposes Anthropic-compatible `/v1/messages` and `/v1/messages/count_tokens` endpoints alongside the OpenAI-compatible `/v1` API. [Guide](https://docs.mistralrs.dev/guides/serve/anthropic-messages-api/)
- **v0.8.2 CUDA performance**: paged-attention and MoE optimizations deliver strong results on GB10, B200, and H100 SXM. [Benchmarks](#benchmarks)
- **Agentic runtime**: web search, local Python code execution, shell execution, OpenAI-compatible Skills, session management, and custom tool hooks. [Guide](https://docs.mistralrs.dev/guides/agents/)
- **Gemma 4**: full multimodal: text, image, video, and audio input. [Supported models](https://docs.mistralrs.dev/reference/supported-models/) | [Video setup](https://docs.mistralrs.dev/guides/models/video-setup/)

## Benchmarks

<details>
<summary><b>v0.8.2 CUDA benchmarks</b></summary>

Mean tokens per second across prompt lengths and decode depths from 128 to 16384 tokens. Decode uses 256 generated tokens. See the full [v0.8.2 report](releases/v0.8.2/report.md) for commands, model revisions, host metadata, and appendix tables.

**Q8 prefill TPS: mistral.rs UQFF q8 vs llama.cpp GGUF Q8_0**

| Model | Hardware | mistral.rs | llama.cpp |
|---|---|---:|---:|
| Gemma 4 E4B | GB10 | 7395.7 | 3973.7 |
| Gemma 4 E4B | B200 | 27705.6 | 11992.4 |
| Gemma 4 E4B | H100 SXM | 26220.6 | 11702.1 |
| Gemma 4 26B-A4B | GB10 | 2947.0 | 2178.5 |
| Gemma 4 26B-A4B | B200 | 12725.3 | 8503.4 |
| Gemma 4 26B-A4B | H100 SXM | 12362.3 | 8055.1 |

**Q8 decode TPS: mistral.rs UQFF q8 vs llama.cpp GGUF Q8_0**

| Model | Hardware | mistral.rs | llama.cpp |
|---|---|---:|---:|
| Gemma 4 E4B | GB10 | 44.1 | 40.5 |
| Gemma 4 E4B | B200 | 241.4 | 194.4 |
| Gemma 4 E4B | H100 SXM | 223.1 | 183.0 |
| Gemma 4 26B-A4B | GB10 | 46.8 | 46.4 |
| Gemma 4 26B-A4B | B200 | 210.9 | 192.2 |
| Gemma 4 26B-A4B | H100 SXM | 199.8 | 183.9 |

**BF16 prefill TPS: mistral.rs BF16 vs vLLM BF16**

| Model | Hardware | mistral.rs | vLLM |
|---|---|---:|---:|
| Gemma 4 E4B | GB10 | 5838.9 | 5812.9 |
| Gemma 4 E4B | B200 | 43547.8 | 39431.2 |
| Gemma 4 E4B | H100 SXM | 35852.2 | 39293.7 |
| Gemma 4 26B-A4B | GB10 | 592.2 | 3878.6 |
| Gemma 4 26B-A4B | B200 | 3467.3 | 28532.8 |
| Gemma 4 26B-A4B | H100 SXM | 2766.0 | 26295.9 |

**BF16 decode TPS: mistral.rs BF16 vs vLLM BF16**

| Model | Hardware | mistral.rs | vLLM |
|---|---|---:|---:|
| Gemma 4 E4B | GB10 | 25.1 | 18.8 |
| Gemma 4 E4B | B200 | 202.6 | 196.2 |
| Gemma 4 E4B | H100 SXM | 174.4 | 153.0 |
| Gemma 4 26B-A4B | GB10 | 26.9 | 23.2 |
| Gemma 4 26B-A4B | B200 | 159.6 | 220.2 |
| Gemma 4 26B-A4B | H100 SXM | 138.7 | 148.0 |

</details>

## Why mistral.rs?

- **Automatic model loading**: Architecture, weight format, and chat template are detected for supported Hugging Face models and GGUF files, with flags available for explicit selection.
- **True multimodality**: Text, vision, video, and audio, speech generation, image generation, and embeddings in one engine.
- **Quantization selection**: `--quant` selects a matching artifact from GGUF repositories. For other Hugging Face repositories, it uses a prebuilt UQFF when available and otherwise applies ISQ. [Docs](https://docs.mistralrs.dev/guides/quantization/quantize-a-model/)
- **OpenAI + Anthropic compatible serving**: The same `mistralrs serve` process exposes OpenAI-compatible `/v1` endpoints and Anthropic-compatible Messages endpoints.
- **Prometheus metrics**: `mistralrs serve` exposes a `/metrics` endpoint in Prometheus format, recording per-request counts and latency labeled by method, route, and status. [Docs](https://docs.mistralrs.dev/reference/http-api/)
- **Built-in web UI**: Served at `/ui` by default. Shows reasoning, code execution, plots, and files inline. Edit any message and the new branch runs with its own Python state. Pass `--no-ui` to disable.
- **Hardware-aware**: `mistralrs tune` recommends quantization and device mapping from the model config and your detected hardware.
- **Flexible SDKs**: Python package and Rust crate to build your projects.
- **Native agentic support**: built-in [agentic loop](https://docs.mistralrs.dev/guides/agents/) with web search, local Python code execution, shell execution, OpenAI-compatible Skills, session management, and custom tool hooks.

## Quick Start

### Install

**Linux/macOS:**
```bash
curl -fsSL https://mistralrs.dev/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://mistralrs.dev/install.ps1 | iex
```

Downloads a self-contained prebuilt binary for your platform (Metal on Apple Silicon; per-GPU CUDA or CPU on Linux; CPU on Windows), falling back to a source build if none matches. Standard acceleration needs no Rust or CUDA toolkit. Optional cuTile acceleration requires NVIDIA's separately installed `tileiras` tool.

[Manual installation, accelerator details & other platforms](https://docs.mistralrs.dev/quickstart/)

### Run Your First Model

```bash
# Interactive chat
mistralrs run -m Qwen/Qwen3-4B

# One-shot prompt (no interactive session)
mistralrs run -m Qwen/Qwen3-4B -i "What is the capital of France?"

# One-shot with an image
mistralrs run -m google/gemma-4-E4B-it --image photo.jpg -i "Describe this image"

# Run a local GGUF or select a published 4-bit GGUF
mistralrs run -f /path/to/model.gguf
mistralrs run -m unsloth/Qwen3.5-4B-GGUF --quant 4

# Agentic REPL: search + code execution + shell from the terminal
mistralrs run --agent -m Qwen/Qwen3-4B

# Start an API server with the built-in web UI
mistralrs serve -m google/gemma-4-E4B-it
```

For the server command, visit `http://localhost:1234/ui` for the web chat interface. OpenAI-compatible clients use `http://localhost:1234/v1`; Anthropic-compatible clients use `http://localhost:1234`.

### The `mistralrs` CLI

The CLI uses the same `run`, `serve`, and `bench` commands for model repositories, local directories, and GGUF files.

- **Auto-detection**: Automatically detects model architecture, quantization format, and chat template
- **All-in-one**: Single binary for chat, server, benchmarks, and web UI (`run`, `serve`, `bench`)
- **Hardware-aware tuning**: `mistralrs tune` recommends quantization and device mapping for your model and hardware
- **Model formats**: Hugging Face checkpoints, [GGUF files](https://docs.mistralrs.dev/guides/models/run-gguf/), and [UQFF quantizations](https://docs.mistralrs.dev/reference/uqff-format/)

```bash
# Recommend settings for your hardware and emit a config file
mistralrs tune -m Qwen/Qwen3-4B --emit-config config.toml

# Run using the generated config
mistralrs from-config -f config.toml

# Diagnose system issues (CUDA, Metal, Hugging Face connectivity)
mistralrs doctor
```

[Full CLI documentation](https://docs.mistralrs.dev/reference/cli/)

<details open>
  <summary><b>UI Demo</b></summary>
  <br>
  <img src="https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/res/ui.gif" alt="UI Demo" />
</details>

## What Makes It Fast

**Performance**
- Continuous batching support by default on all devices.
- CUDA with FlashAttention V2/V3, Metal, and [multi-GPU/distributed inference](https://docs.mistralrs.dev/guides/perf/distributed-inference/)
- [PagedAttention](https://docs.mistralrs.dev/guides/perf/paged-attention/) for high throughput continuous batching on CUDA or Apple Silicon, prefix caching (including multimodal)

**Quantization** ([full docs](https://docs.mistralrs.dev/reference/quantization-types/))
- [In-situ quantization (ISQ)](https://docs.mistralrs.dev/guides/quantization/quantize-a-model/) for Hugging Face models
- [GGUF](https://docs.mistralrs.dev/reference/gguf-support/) (2-8 bit), GPTQ, AWQ, HQQ, FP8, BNB support
- ⭐ [Per-layer topology](https://docs.mistralrs.dev/guides/perf/topology/): Fine-tune quantization per layer for optimal quality/speed
- ⭐ Auto-select fastest quant method for your hardware

**Flexibility**
- [LoRA & X-LoRA](https://docs.mistralrs.dev/guides/customize/lora-adapters/) with per-request LoRA selection and X-LoRA adapter mixing
- AnyMoE: Create mixture-of-experts on any base model
- [Multiple models](https://docs.mistralrs.dev/guides/serve/multiple-models/): Load/unload at runtime

**Agentic Features**
- Integrated [tool calling](https://docs.mistralrs.dev/guides/agents/tool-calling-basics/) with grammar enforcement and strict schema mode
- ⭐ Server-side [agentic loop](https://docs.mistralrs.dev/guides/agents/tool-calling-basics/): auto-execute tools and feed results back
- ⭐ [Python code execution](https://docs.mistralrs.dev/guides/agents/enable-code-execution/): persistent Jupyter-like sessions with matplotlib capture and multimodal feedback
- ⭐ [Shell execution](https://docs.mistralrs.dev/guides/agents/enable-shell/): persistent command-line sessions with sandboxing and approval controls
- ⭐ [OpenAI-compatible Skills](https://docs.mistralrs.dev/guides/agents/skills/): uploaded skill bundles for Responses API agents
- ⭐ [OpenAI-compatible file inputs](https://docs.mistralrs.dev/guides/agents/file-inputs/): `/v1/files`, Responses `input_file`, Chat `file`, and workdir mounts
- ⭐ [Web search integration](https://docs.mistralrs.dev/guides/agents/web-search/) with embedding-based ranking
- ⭐ [Tool dispatch URL](https://docs.mistralrs.dev/guides/agents/tool-calling-basics/): POST tool calls to your own endpoint
- ⭐ [MCP client](https://docs.mistralrs.dev/guides/agents/connect-mcp-server/): Connect to external tools via Process, HTTP, or WebSocket
- Python/Rust [tool callbacks](https://docs.mistralrs.dev/guides/agents/tool-calling-basics/) for custom execution

[Full feature documentation](https://docs.mistralrs.dev/)

## Supported Models

Text, multimodal, speech, image generation, and embedding models across 45+ architectures. The **[supported models reference](https://docs.mistralrs.dev/reference/supported-models/)** is the single source of truth: it explains how to check whether your model's `config.json` is supported, lists every architecture with copy-paste run commands, and is generated directly from the engine's loader registry so it never drifts.

[Supported models reference](https://docs.mistralrs.dev/reference/supported-models/) | [Request a new model](https://github.com/EricLBuehler/mistral.rs/issues/156)

## Python SDK

```bash
pip install mistralrs
```

In-process inference from Python: load a model with `Runner` and send OpenAI-shaped requests, no server required. Accelerator-specific wheels (CUDA, Metal, MKL, Accelerate) are listed in the getting-started guide.

[Get started](https://docs.mistralrs.dev/guides/python/getting-started/) | [API reference](https://docs.mistralrs.dev/reference/python/) | [Examples](examples/python)

## Rust SDK

```bash
cargo add mistralrs
```

Embed the engine in a Rust application with the high-level `mistralrs` crate.

[Get started](https://docs.mistralrs.dev/guides/rust/getting-started/) | [docs.rs](https://docs.rs/mistralrs) | [Crate](https://crates.io/crates/mistralrs) | [Examples](mistralrs/examples)

## Docker

Prebuilt CPU and CUDA images are published to GHCR. Pull commands, tags, and Kubernetes notes are in the [Docker guide](https://docs.mistralrs.dev/guides/deploy/docker/).

## Documentation

For complete documentation, see the **[Documentation](https://docs.mistralrs.dev/)**.

**Quick Links:**
- [Quickstart](https://docs.mistralrs.dev/quickstart/) - Install, first run, first serve
- [CLI Reference](https://docs.mistralrs.dev/reference/cli/) - All commands and options
- [Anthropic Messages API](https://docs.mistralrs.dev/guides/serve/anthropic-messages-api/) - Anthropic-compatible Messages, streaming, tool use, and token counting
- [HTTP API](https://docs.mistralrs.dev/reference/http-api/) - OpenAI-compatible and Anthropic-compatible endpoints
- [Quantization](https://docs.mistralrs.dev/reference/quantization-types/) - ISQ, GGUF, GPTQ, and more
- [Multi-GPU and Distributed](https://docs.mistralrs.dev/guides/perf/distributed-inference/) - NCCL TP, P2P layer mapping, multi-node, and ring
- [MCP Integration](https://docs.mistralrs.dev/guides/agents/connect-mcp-server/) - MCP integration documentation
- [Troubleshooting](https://docs.mistralrs.dev/reference/troubleshooting/) - Common issues and solutions
- [Environment variables](https://docs.mistralrs.dev/reference/environment-variables/) - Environment variables for configuration

## Citation

If you use mistral.rs in your research, please cite:

```bibtex
@misc{mistralrs,
  author = {Buehler, Eric},
  title = {{mistral.rs}: Fast, flexible {LLM} inference},
  year = {2024},
  url = {https://github.com/EricLBuehler/mistral.rs}
}
```

Citation metadata is available in [CITATION.cff](CITATION.cff).

## Contributing

Contributions welcome! Please [open an issue](https://github.com/EricLBuehler/mistral.rs/issues) to discuss new features or report bugs. If you want to add a new model, please contact us via an issue and we can coordinate.

## Credits

This project would not be possible without the excellent work at [Candle](https://github.com/huggingface/candle). Thank you to all [contributors](https://github.com/EricLBuehler/mistral.rs/graphs/contributors)!

mistral.rs is not affiliated with Mistral AI.

<p align="right">
  <a href="#top">Back to Top</a>
</p>
