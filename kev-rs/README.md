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
| `espetro/kev-0.8b-mistralrs` | `jaredpalmer/kev-0.8b` | 2.9 GB |
| `espetro/kev-4b-mistralrs` | `jaredpalmer/kev-4b` | 16 GB |

Export any other run (e.g. `jaredpalmer/kev-9b`, 36 GB fp32) from a Kev checkout:

```sh
uv run --extra serve python /path/to/mistral.rs/kev-rs/scripts/export_checkpoint.py --run jaredpalmer/kev-9b --out ~/kev-9b
```

The exporter merges the LoRA in fp32 and checks every merged tensor bit-for-bit against
Kev's own loader (via digests, so only one fp32 copy is resident). `option_isolation`
checkpoints are rejected.

## Usage

```sh
kev-rs serve --checkpoint <dir | owner/repo> [--run jaredpalmer/kev-0.8b] [--host 127.0.0.1] [--port 8009] [--paged] [--dtype f32|bf16]
kev-rs parity --checkpoint <dir | owner/repo> --reference <reference.json>
kev-rs encode-check --checkpoint <dir | owner/repo> --records <records.jsonl> --reference <reference.json>
```

Routes: `POST /v1/systemone`, `POST /v1/systemone/permute`, `POST /v1/systemone/separate`,
`GET /v1/models` (includes prefix-cache hit/miss counters). Set `KEV_API_KEY` to require a
bearer token; `KEV_PREFIX_CACHE` sizes the state-prefix cache. Inside a container bind with
`--host 0.0.0.0`:

```sh
docker run --rm -p 8009:8009 -v hf-cache:/data --entrypoint kev-rs ghcr.io/espetro/mistral.rs:cpu-kev \
  serve --host 0.0.0.0 --checkpoint espetro/kev-0.8b-mistralrs --run jaredpalmer/kev-0.8b
```

## Status

CPU parity vs Kev's PyTorch reference (0.8B fixtures): max |dp| 0.00011, 0 argmax flips
over 34 questions; Kev's `tests/test_api.py` passes 10/10 against `kev-rs`. Metal and CUDA
builds exist but their runtime parity and speed are unmeasured. `date_facts` and
`option_isolation` checkpoints are unsupported. There is no WASM/browser build; hosted use
means running the Linux binary or Docker image next to a UI (Kev playground, TypeSafe SDK).
