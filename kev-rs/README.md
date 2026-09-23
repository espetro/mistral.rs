# kev-rs

Kev pointer-head serving on the mistral.rs hidden-states path. `kev-rs serve`
exposes the TypeSafe `/v1/systemone` API: a shared "state" prompt prefix is
cached once, then each question branch runs a prefill-only forward that returns
the post-final-norm hidden states at the `<|box_end|>` and `<|fim_suffix|>`
delimiter rows for the pointer head.

## Kev fork

This fork's release workflow builds GitHub prereleases on GitHub-hosted runners
(Metal, Linux CPU x86_64/aarch64, Windows CPU, Linux CUDA 12.8 for sm86/89/120
compiled without a GPU in the `nvidia/cuda` devel container; the CUDA leg and the
installer overrides are fork-only workarounds, not upstream material). Binaries
ship `mistralrs` and `kev-rs` in the same archive. To install a fork prerelease:

```bash
MISTRALRS_INSTALL_TAG=v0.9.4-pre.1 ./install.sh
```

## Usage

Export a checkpoint first (`scripts/export_checkpoint.py`, run from the kev
repo), then:

```bash
kev-rs serve --checkpoint <export dir> [--port 8009] [--paged] [--dtype f32|bf16]
kev-rs parity --checkpoint <export dir> --reference <reference.json>
kev-rs encode-check --checkpoint <export dir> --records <records.jsonl> --reference <reference.json>
```
