"""Torch reference probabilities for kev-rs parity testing.

Run from the kev repo:
uv run --extra serve python <path> --run jaredpalmer/kev-0.8b --records <jsonl> --out <json> [--limit N]

Each output entry carries the packed ids, the rows_of split (state_len + branch ids
with readout offsets), and the fp32 probability vectors kev.probs returns.
"""
import argparse
import json

import torch

from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import load_records, materialize
from kev.model import rows_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    records = load_records(a.records)
    # most questions per record first, so a small limit still exercises batches of branches
    records = sorted(records, key=lambda r: -len(r["questions"]))
    if a.limit:
        records = records[: a.limit]

    ck = Checkpoint(a.run)
    tok, model = ck.load("cpu", LoadOptions(dtype=None, merge=True, backend="torch"))

    out = []
    for i, req in enumerate(records):
        rec = materialize(req)
        enc = model.encode(tok, rec)
        state_ids, _, rows = rows_of(enc)
        ps = model.probs(enc)
        out.append(
            {
                "ids": enc["ids"],
                "state_len": len(state_ids),
                "decide_idx": enc["decide_idx"],
                "opt_idx": enc["opt_idx"],
                "branches": rows,
                "labels": enc["labels"],
                "probs": [p.tolist() for p in ps],
            }
        )
        print(f"{i + 1}/{len(records)}: {len(enc['ids'])} tokens, {len(ps)} questions", flush=True)

    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(out)} records -> {a.out}")


if __name__ == "__main__":
    main()
