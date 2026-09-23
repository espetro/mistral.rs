"""Export a Kev checkpoint into a plain HF backbone dir + pointer head for kev-rs.

Run from the kev repo: uv run --extra serve python <path> --run jaredpalmer/kev-0.8b --out <dir>
Writes <out>/model (merged HF checkpoint + tokenizer), <out>/head.safetensors, <out>/kev.json.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import SPECIAL, pad_id
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ck = Checkpoint(a.run)
    meta = ck.meta
    if meta.option_isolation:
        raise ValueError("option_isolation checkpoints are not supported by kev-rs")

    tok, model = ck.load("cpu", LoadOptions(dtype=None, merge=True, backend="torch"))
    hidden_size = model.lm.config.hidden_size
    head = {k: v.contiguous() for k, v in model.head.state_dict().items()}
    temperature = model.head.temperature

    # The DecisionModel keeps only the text backbone (AutoModelForCausalLM(...).model).
    # mistral.rs wants a complete HF checkpoint, so rebuild the full merged model the
    # same way (fp32 base + adapter merged in fp32) and verify it against model.lm.
    # Only digests of the reference survive so two fp32 copies never coexist (4B/9B on 32 GB boxes).
    ref = {name: digest(t) for name, t in model.lm.state_dict().items()}
    del model
    gc.collect()

    full = AutoModelForCausalLM.from_pretrained(meta.base, revision=meta.base_revision, dtype=torch.float32)
    # the adapter was saved with names base_model.model.layers.* (peft wrapped the inner
    # text model), so apply it to full.model, not to the whole ForCausalLM.
    merged_inner = PeftModel.from_pretrained(full.model, ck.path).merge_and_unload()
    full.model = merged_inner
    merged = full

    got = merged.model.state_dict()
    if set(ref) != set(got):
        raise ValueError(f"state_dict key mismatch: {sorted(set(ref) ^ set(got))[:5]}")
    for name, t in got.items():
        if digest(t) != ref[name]:
            raise ValueError(f"merged weight {name} differs from the loaded checkpoint's")

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
        "special_tokens": {name: tok.convert_tokens_to_ids(name) for name in SPECIAL},
        "pad_id": pad_id(tok),
        "option_isolation": meta.option_isolation,
    }
    (out / "kev.json").write_text(json.dumps(kev_json, indent=2) + "\n")

    arch = getattr(merged.config, "architectures", None)
    print(f"exported {a.run} -> {out}")
    print(f"arch={arch} hidden_size={kev_json['hidden_size']} head_dim={meta.head_dim} "
          f"temperature={temperature}")
    print(f"special_tokens={kev_json['special_tokens']} pad_id={kev_json['pad_id']}")


if __name__ == "__main__":
    main()
