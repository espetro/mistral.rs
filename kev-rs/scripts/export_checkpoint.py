"""Export a Kev checkpoint into a plain HF backbone dir + pointer head for kev-rs.

Run from the kev repo: uv run --extra serve python <path> --run jaredpalmer/kev-0.8b --out <dir> [--dtype bf16]
Writes <out>/model (merged HF checkpoint + tokenizer), <out>/head.safetensors, <out>/kev.json.

--dtype controls the saved backbone precision. bf16/fp16 keep a single low-precision
copy resident the whole run: the base loads straight in the target dtype and the LoRA
delta (still computed in fp32 per tensor by PEFT) is added in place, so a 9B export
peaks around one model in that dtype instead of two fp32 copies.
"""
import argparse
import gc
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import SPECIAL, pad_id
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def load_lm_head(meta, dtype):
    """Only lm_head.weight from the base checkpoint, for untied-embedding models."""
    try:
        index = json.loads(
            Path(hf_hub_download(meta.base, "model.safetensors.index.json", revision=meta.base_revision)).read_text()
        )
        shard = index["weight_map"]["lm_head.weight"]
    except Exception:
        shard = "model.safetensors"
    path = hf_hub_download(meta.base, shard, revision=meta.base_revision)
    with safe_open(path, framework="pt") as f:
        return f.get_tensor("lm_head.weight").to(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp32", choices=sorted(DTYPES))
    a = ap.parse_args()
    dtype = DTYPES[a.dtype]

    ck = Checkpoint(a.run)
    meta = ck.meta
    if meta.option_isolation:
        raise ValueError("option_isolation checkpoints are not supported by kev-rs")
    if ck.adapter_config().get("trainable_token_indices"):
        raise ValueError("checkpoints with trained token embeddings are not supported by kev-rs")

    # merge=False so the backbone stays in `dtype` from the start; merge_and_unload then
    # applies each LoRA delta in fp32 and stores back in `dtype`, which matches merging
    # in fp32 and casting afterwards.
    tok, model = ck.load("cpu", LoadOptions(dtype=dtype, merge=False, backend="torch"))
    head = {k: v.contiguous() for k, v in model.head.state_dict().items()}
    temperature = model.head.temperature
    backbone = model.lm.merge_and_unload()
    hidden_size = backbone.config.hidden_size
    del model
    gc.collect()

    # The DecisionModel keeps only the text backbone (AutoModelForCausalLM(...).model),
    # so attach it to a weight-free ForCausalLM shell for save_pretrained to write a
    # complete HF checkpoint.
    config = AutoConfig.from_pretrained(meta.base, revision=meta.base_revision)
    with torch.device("meta"):
        merged = AutoModelForCausalLM.from_config(config)
    merged.model = backbone
    if getattr(config, "tie_word_embeddings", False):
        merged.tie_weights()
    else:
        merged.lm_head.weight = torch.nn.Parameter(load_lm_head(meta, dtype))

    out = Path(a.out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out / "model", safe_serialization=True)
    tok.save_pretrained(out / "model")

    save_file({k: head[k] for k in ("q.weight", "q.bias", "k.weight", "k.bias")}, out / "head.safetensors")

    kev_json = {
        "base": meta.base,
        "base_revision": meta.base_revision,
        "head_dim": meta.head_dim,
        "temperature": temperature,
        "hidden_size": hidden_size,
        "dtype": a.dtype,
        "special_tokens": {name: tok.convert_tokens_to_ids(name) for name in SPECIAL},
        "pad_id": pad_id(tok),
        "option_isolation": meta.option_isolation,
    }
    (out / "kev.json").write_text(json.dumps(kev_json, indent=2) + "\n")

    arch = getattr(merged.config, "architectures", None)
    print(f"exported {a.run} -> {out} ({a.dtype})")
    print(f"arch={arch} hidden_size={kev_json['hidden_size']} head_dim={meta.head_dim} "
          f"temperature={temperature}")
    print(f"special_tokens={kev_json['special_tokens']} pad_id={kev_json['pad_id']}")


if __name__ == "__main__":
    main()
