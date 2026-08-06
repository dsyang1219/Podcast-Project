"""BERT arm, deliverable 3 — pool sentence embeddings to chunk vectors.

    python -m nlp.pool [--target-words 300 500 800]

Reads sentence_embeddings_<N>.npy (+ its meta) and sentences_<N>.csv, groups
rows by chunk_id (preserving each chunk's first-appearance order — sentences
for a given chunk are already contiguous since sentences_<N>.csv is written
in chunk order), and pools to one vector per chunk:

  length-weighted mean (primary): chunk vector = sum(w_i * v_i) / sum(w_i),
  w_i = sentence's n_raw_words. A one-word "Yeah." doesn't count the same as
  a 40-word argument.

  plain mean (robustness variant): unweighted average.

Output chunk_embeddings_<N>_<pool>.npy, row order joinable to
chunks_<N>_bert.csv via the chunk_id list saved in the sidecar meta.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline import config as pipeline_config
from .embed import verify_pairing


def pool_chunk(target: int) -> None:
    csv_path = pipeline_config.OUTPUT_DIR / f"sentences_{target}.csv"
    npy_path = pipeline_config.OUTPUT_DIR / f"sentence_embeddings_{target}.npy"
    meta_path = pipeline_config.OUTPUT_DIR / f"sentence_embeddings_{target}.meta.json"

    verify_pairing(csv_path, meta_path)  # refuses a stale/mismatched embeddings file

    df = pd.read_csv(csv_path, keep_default_na=False)
    embeddings = np.load(npy_path)
    assert len(df) == embeddings.shape[0], (
        f"row count mismatch: {csv_path.name} has {len(df)} rows, "
        f"{npy_path.name} has {embeddings.shape[0]}"
    )

    chunk_ids = []
    weighted_vecs = []
    plain_vecs = []
    for chunk_id, group in df.groupby("chunk_id", sort=False):
        idx = group.index.to_numpy()
        vecs = embeddings[idx]
        weights = group["n_raw_words"].to_numpy(dtype=np.float64)
        if weights.sum() == 0:
            # every member sentence is literally empty text (n_raw_words=0
            # for all) — fall back to an unweighted mean rather than divide
            # by zero; this is a degenerate edge case, not the common path.
            weighted = vecs.mean(axis=0)
        else:
            weighted = (vecs * weights[:, None]).sum(axis=0) / weights.sum()
        plain = vecs.mean(axis=0)
        chunk_ids.append(chunk_id)
        weighted_vecs.append(weighted)
        plain_vecs.append(plain)

    weighted_arr = np.stack(weighted_vecs).astype(np.float32)
    plain_arr = np.stack(plain_vecs).astype(np.float32)

    base_meta = {
        "source_sentence_embeddings": npy_path.name,
        "source_sentences_csv": csv_path.name,
        "n_chunks": len(chunk_ids),
        "dim": weighted_arr.shape[1],
        "chunk_id_order": list(map(str, chunk_ids)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    for pool_name, arr in [("weighted", weighted_arr), ("plain", plain_arr)]:
        out_path = pipeline_config.OUTPUT_DIR / f"chunk_embeddings_{target}_{pool_name}.npy"
        meta_out_path = pipeline_config.OUTPUT_DIR / f"chunk_embeddings_{target}_{pool_name}.meta.json"
        np.save(out_path, arr)
        meta = dict(base_meta, pool=pool_name,
                    weight_field=("n_raw_words" if pool_name == "weighted" else None))
        meta_out_path.write_text(json.dumps(meta, indent=2))
        print(f"[pool target={target} pool={pool_name}] {arr.shape} -> {out_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-words", type=int, nargs="+", default=[300, 500, 800])
    args = ap.parse_args()
    for target in args.target_words:
        pool_chunk(target)


if __name__ == "__main__":
    main()
