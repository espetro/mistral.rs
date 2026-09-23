"""Export a Kev checkpoint into a plain HF backbone dir + pointer head for kev-rs.

Run from the kev repo: uv run --extra serve python <path> --run jaredpalmer/kev-0.8b --out <dir>
Writes <out>/model (merged HF checkpoint + tokenizer), <out>/head.safetensors, <out>/kev.json.
"""
import argparse
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from peft import PeftModel

from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import SPECIAL, pad_id


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

    # The DecisionModel keeps only the text backbone (AutoModelForCausalLM(...).model).
    # mistral.rs wants a complete HF checkpoint, so rebuild the full merged model the
    # same way (fp32 base + adapter merged in fp32) and verify it against model.lm.
    full = AutoModelForCausalLM.from_pretrained(meta.base, revision=meta.base_revision, dtype=torch.float32)
    # the adapter was saved with names base_model.model.layers.* (peft wrapped the inner
    # text model), so apply it to full.model, not to the whole ForCausalLM.
    merged_inner = PeftModel.from_pretrained(full.model, ck.path).merge_and_unload()
    full.model = merged_inner
    merged = full

    ref = model.lm.state_dict()
    got = merged.model.state_dict()
    if set(ref) != set(got):
        raise ValueError(f"state_dict key mismatch: {sorted(set(ref) ^ set(got))[:5]}")
    for name, t in ref.items():
        if not torch.equal(t, got[name]):
            raise ValueError(f"merged weight {name} differs from the loaded checkpoint's")

    out = Path(a.out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out / "model", safe_serialization=True)
    tok.save_pretrained(out / "model")

    head = model.head.state_dict()
    save_file(
        {
            "q.weight": head["q.weight"].contiguous(),
            "q.bias": head["q.bias"].contiguous(),
            "k.weight": head["k.weight"].contiguous(),
            "k.bias": head["k.bias"].contiguous(),
        },
        out / "head.safetensors",
    )

    kev_json = {
        "base": meta.base,
        "base_revision": meta.base_revision,
        "head_dim": meta.head_dim,
        "temperature": model.head.temperature,
        "hidden_size": model.lm.config.hidden_size,
        "special_tokens": {name: tok.convert_tokens_to_ids(name) for name in SPECIAL},
        "pad_id": pad_id(tok),
        "option_isolation": meta.option_isolation,
    }
    (out / "kev.json").write_text(json.dumps(kev_json, indent=2) + "\n")

    arch = getattr(merged.config, "architectures", None)
    print(f"exported {a.run} -> {out}")
    print(f"arch={arch} hidden_size={kev_json['hidden_size']} head_dim={meta.head_dim} "
          f"temperature={model.head.temperature}")
    print(f"special_tokens={kev_json['special_tokens']} pad_id={kev_json['pad_id']}")


if __name__ == "__main__":
    main()
