# kev-rs

Kev pointer-head serving on the mistral.rs hidden-states path. `kev-rs serve`
exposes the TypeSafe `/v1/systemone` API: a shared "state" prompt prefix is
cached once, then each question branch runs a prefill-only forward that returns
the post-final-norm hidden states at the `<|box_end|>` and `<|fim_suffix|>`
delimiter rows for the pointer head.

The top-level [README](../README.md) has the end-to-end quick start (install, models,
hello world, batch/parallel requests, playground UI, Docker). This file covers the crate.

## Install

Fork releases ship `mistralrs` and `kev-rs` in the same archive for Metal, Linux CPU
x86_64/aarch64, Windows CPU and Linux CUDA 12.8 (sm86/89/120, compiled without a GPU in
the `nvidia/cuda` devel container). `install.sh` and `mise` resolve the newest fork
release, prereleases included:

```sh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/espetro/mistral.rs/kev/install.sh)"
mise use -g "github:espetro/mistral.rs[prerelease=true,matching=mistralrs-metal]@latest"
MISTRALRS_INSTALL_TAG=<tag> ./install.sh            # pin a release from github.com/espetro/mistral.rs/releases
cargo build --release -p kev-rs --features kev-rs/metal   # or kev-rs/cuda; no feature for CPU
```

The CUDA release leg, the `cpu-kev` Docker tag and the installer overrides are fork-only
workarounds, not upstream material.

## Checkpoints

`--checkpoint` takes either a directory written by `scripts/export_checkpoint.py`
(`model/` merged HF weights + tokenizer, `head.safetensors`, `kev.json`) or a Hub repo
holding that layout as `owner/repo[@revision]`, downloaded into the hf-hub cache
(`HF_HOME`, private repos via `HF_TOKEN`).

| Hub id | Source | Size |
|---|---|---|
| `espetro/kev-0.8b-mistralrs` | `jaredpalmer/kev-0.8b` | 2.9 GB (fp32); bf16 export 1.5 GB |
| `espetro/kev-4b-mistralrs` | `jaredpalmer/kev-4b` | 16 GB (fp32); bf16 export 8 GB |
| `espetro/kev-9b-mistralrs` | `jaredpalmer/kev-9b` | 17 GB (bf16) |

Export any other run or precision from a Kev checkout:

```sh
uv run --extra serve python /path/to/mistral.rs/kev-rs/scripts/export_checkpoint.py --run jaredpalmer/kev-9b --out ~/kev-9b --dtype bf16
```

The exporter loads the base and merges the LoRA in `--dtype` (fp32/bf16/fp16; default
fp32). `option_isolation` checkpoints are rejected.

## Usage

```sh
kev-rs serve --checkpoint <dir | owner/repo> [--run jaredpalmer/kev-0.8b] [--host 127.0.0.1] [--port 8009] [--paged] [--dtype auto|f32|bf16|f16] [--isq <bits-or-type>] [--imatrix <file>] [--calibration-file <txt>] [--topology <yaml>]
kev-rs parity --checkpoint <dir | owner/repo> --reference <reference.json> [same load flags]
kev-rs encode-check --checkpoint <dir | owner/repo> --records <records.jsonl> --reference <reference.json>
```

`--isq` takes a bit width (2-8) or a type (`q8_0`, `afq8`, `q4k`, ...). `--topology`
selects ISQ per tensor with a YAML mapping layer ranges or `/regex/` to `{isq, device}`;
for Qwen3.5 hybrids the recurrent path is `linear_attn.*`, attention is `self_attn.*`.
`--imatrix` applies a llama.cpp `.imatrix` (or collected `.cimatrix`) to ISQ;
`--calibration-file` collects the imatrix in-process from a plain-text corpus at load
(this needs `--isq` and runs every 1024-token chunk through the backbone first).

Routes: `POST /v1/systemone`, `POST /v1/systemone/permute`, `POST /v1/systemone/separate`,
`GET /v1/models` (includes prefix-cache hit/miss counters). Set `KEV_API_KEY` to require a
bearer token; `KEV_PREFIX_CACHE` sizes the state-prefix cache. Inside a container bind with
`--host 0.0.0.0`:

```sh
docker run --rm -p 8009:8009 -v hf-cache:/data --entrypoint kev-rs ghcr.io/espetro/mistral.rs:cpu-kev \
  serve --host 0.0.0.0 --checkpoint espetro/kev-0.8b-mistralrs --run jaredpalmer/kev-0.8b
```

## Quantization

All variants below were parity-checked against the bf16 torch reference
(`scripts/reference_probs.py --dtype bf16 --no-merge` on the 30-record smoke fixture,
40 questions):

| Checkpoint | Variant | Flips | max \|dp\| | mean \|dp\| |
|---|---|---|---|---|
| 0.8B | bf16 | 0/40 | 0.018 | 0.0005 |
| 0.8B | `--isq 8` | 0/40 | 0.042 | 0.0009 |
| 0.8B | `--isq 4` | 2/40 | 0.107 | 0.0040 |
| 0.8B | `--isq 8` + imatrix | 0/40 | 0.042 | 0.0009 |
| 0.8B | `--isq 4` + imatrix | 6/40 | 0.512 | 0.012 |
| 0.8B | topology gdn8/attn6 | 0/40 | 0.025 | 0.0009 |
| 0.8B | topology gdn8/attn4 | 4/40 | 0.118 | 0.0034 |
| 4B | bf16 | 0/40 | 0.055 | 0.0007 |
| 4B | `--isq 8` | 0/40 | 0.029 | 0.0007 |
| 9B | bf16 | 0/40 | 0.023 | 0.0003 |
| 9B | `--isq 8` | 1/40 | 0.048 | 0.0009 |
| 9B | topology gdn8/attn6 | 1/40 | 0.046 | 0.0009 |

Verdicts, measured on this box:

- **bf16 export** is the recommended checkpoint for every size: halves file size and
  RAM vs fp32 at no measurable quality cost (0 flips on all three models).
- **`--isq 8`** halves memory again at load with at most 1 flip; ~4x faster per record
  on CPU than bf16 for the 0.8B.
- **Topology mixed ISQ** is the best size/quality point: `linear_attn` (the GDN
  recurrent path) at `afq8`, `self_attn` and `mlp` at `afq6` matches full `--isq 8`
  parity numbers on both 0.8B and 9B at ~25% less weight. Dropping attn/mlp to `afq4`
  loses too much (4/40).

  ```yaml
  '/linear_attn\./':
    isq: afq8
  '/self_attn\./':
    isq: afq6
  '/mlp\./':
    isq: afq6
  ```
- **imatrix calibration** now runs on hybrid models (upstream fix: the calibration
  forward allocates a temporary recurrent slot) but adds nothing here: `--isq 8` +
  imatrix is bit-identical to plain `--isq 8`, and `--isq 4` + imatrix is *worse* than
  plain `--isq 4`, which already keeps sensitive tensors at q6k.
- **GGUF / Unsloth dynamic quants are not applicable**: the GGUF pipelines refuse
  `return_hidden_states` and no quantized implementation of the Qwen3.5 hybrid arch
  exists; topology YAML covers the same per-tensor mixing on the normal pipeline.

## Status

CPU parity vs Kev's PyTorch reference (0.8B fixtures): max |dp| 0.00011, 0 argmax flips
over 34 questions; bf16 exports pass 0/40 flips on all three sizes (see Quantization).
Kev's `tests/test_api.py` passes 10/10 against `kev-rs`. Metal and CUDA
builds exist but their runtime parity and speed are unmeasured. `date_facts` and
`option_isolation` checkpoints are unsupported. There is no WASM/browser build; hosted use
means running the Linux binary or Docker image next to a UI (Kev playground, TypeSafe SDK).
