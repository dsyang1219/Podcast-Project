"""BERT arm, deliverable 2 — sentence embedding.

    python -m nlp.embed [--target-words 300 500 800] [--force]
                         [--model NAME] [--prefix STR]

Reads data/output/sentences_<N>.csv (see nlp/emit_sentences.py), embeds every
sentence's raw `text` with a sentence-transformers model, and writes:

    sentence_embeddings_<N>.npy        float32, row order == input CSV
    sentence_embeddings_<N>.meta.json  model, dim, prefix, n_rows, csv hash

The meta's `input_csv_sha256` is a hash of the exact CSV bytes the embeddings
were built from. Any downstream consumer (nlp/pool.py, nlp/embed_rank.py)
must call `verify_pairing` before trusting a (csv, npy) pair — this is what
stops a stale embeddings file from being silently used against a CSV that has
since been regenerated with different chunking/cleaning.

Model choice: defaults (nlp/config.py EMBED_MODEL/EMBED_PREFIX) to a
prefix-free 768d model (all-mpnet-base-v2) specifically to avoid the E5/BGE
class of silent error, where forgetting the required "passage: " (E5) or
document convention (BGE) on even one input leaves it off-distribution with
no error raised. Pass --model/--prefix to run an E5/BGE model as a robustness
check; if you do, the prefix MUST be supplied and applied to every sentence
uniformly — there is no per-row default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config as pipeline_config
from . import config as nlp_config


def compute_csv_hash(csv_path: Path) -> str:
    return hashlib.sha256(csv_path.read_bytes()).hexdigest()


def load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())


def verify_pairing(csv_path: Path, meta_path: Path) -> dict:
    """Raise if `meta_path` was not built from the exact bytes of `csv_path`.

    Returns the loaded meta dict on success, so callers get both the check
    and the metadata (model name, dim, chunk/sentence ordering assumptions)
    in one call.
    """
    if not meta_path.exists():
        raise FileNotFoundError(f"no embeddings meta at {meta_path}")
    meta = load_meta(meta_path)
    actual_hash = compute_csv_hash(csv_path)
    if meta["input_csv_sha256"] != actual_hash:
        raise ValueError(
            f"embeddings/CSV mismatch: {meta_path.name} was built from a CSV "
            f"with sha256={meta['input_csv_sha256'][:12]}..., but {csv_path.name} "
            f"currently hashes to {actual_hash[:12]}... — regenerate embeddings "
            f"(python -m nlp.embed --force) before using them with this CSV."
        )
    return meta


def _resolve_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _needs_recompute(target: int, csv_hash: str, model_name: str, prefix: str, force: bool) -> bool:
    npy_path = pipeline_config.OUTPUT_DIR / f"sentence_embeddings_{target}.npy"
    meta_path = pipeline_config.OUTPUT_DIR / f"sentence_embeddings_{target}.meta.json"
    if force or not (npy_path.exists() and meta_path.exists()):
        return True
    existing = load_meta(meta_path)
    up_to_date = (existing.get("input_csv_sha256") == csv_hash
                  and existing.get("model") == model_name
                  and existing.get("prefix") == prefix)
    if up_to_date:
        print(f"[embed target={target}] up to date (hash/model/prefix match) "
              f"-> {npy_path.name} (skipped; use --force to recompute)")
    return not up_to_date


def embed_sentences(target: int, model_name: str, prefix: str, force: bool,
                     batch_size: int = nlp_config.EMBED_BATCH_SIZE,
                     fp16: bool = nlp_config.EMBED_FP16) -> None:
    """Single-target convenience wrapper (no cross-target dedup)."""
    embed_targets([target], model_name, prefix, force, batch_size, fp16)


def embed_targets(targets: list[int], model_name: str, prefix: str, force: bool,
                   batch_size: int = nlp_config.EMBED_BATCH_SIZE,
                   fp16: bool = nlp_config.EMBED_FP16) -> None:
    """Embed sentences for multiple target sizes, deduplicated across them.

    sentences_<N>.csv for different N largely overlap: Stage 1 cleaning runs
    ONCE per episode (see nlp/run_chunks.py) and only the chunk-boundary
    grouping differs per target size, so the same sentence text commonly
    appears in all three CSVs (plus repeats *within* a CSV -- short
    backchannel lines like "Yeah." recur constantly in podcast transcripts).
    Since a sentence-embedding model embeds each sentence independently of
    its neighbors, the same exact text always yields the same vector --
    so we embed each unique (prefixed) text once and fan the result back out
    per target, rather than paying GPU cost 3x for ~3x redundant text.

    This is purely an internal compute-sharing optimization: each target's
    on-disk npy/meta contract (row order == its own CSV, own content hash)
    is unchanged, so verify_pairing() and every downstream consumer are
    unaffected.
    """
    dfs, csv_hashes = {}, {}
    for target in targets:
        csv_path = pipeline_config.OUTPUT_DIR / f"sentences_{target}.csv"
        dfs[target] = pd.read_csv(csv_path, keep_default_na=False)
        csv_hashes[target] = compute_csv_hash(csv_path)

    to_compute = [t for t in targets
                  if _needs_recompute(t, csv_hashes[t], model_name, prefix, force)]
    if not to_compute:
        return

    unique_texts: dict[str, int] = {}
    for t in to_compute:
        for txt in dfs[t]["text"]:
            if txt not in unique_texts:
                unique_texts[txt] = len(unique_texts)
    texts_list = list(unique_texts.keys())
    n_total_rows = sum(len(dfs[t]) for t in to_compute)
    print(f"[embed] {len(to_compute)} target(s) to compute, {n_total_rows} total rows, "
          f"{len(texts_list)} unique sentence texts "
          f"({100 * (1 - len(texts_list) / n_total_rows):.1f}% redundancy avoided)")

    prefixed = [prefix + t for t in texts_list]

    from sentence_transformers import SentenceTransformer
    device = _resolve_device()
    model = SentenceTransformer(model_name, device=device)
    use_fp16 = fp16 and device == "cuda"
    if use_fp16:
        model = model.half()
    print(f"[embed] model={model_name} prefix={prefix!r} device={device} "
          f"fp16={use_fp16} batch_size={batch_size}")

    # Embeddings are computed in fp16 (if enabled) for throughput but always
    # cast to float32 below before saving -- storage precision is unaffected.
    unique_embeddings = model.encode(
        prefixed, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    ).astype(np.float32)

    for t in to_compute:
        npy_path = pipeline_config.OUTPUT_DIR / f"sentence_embeddings_{t}.npy"
        meta_path = pipeline_config.OUTPUT_DIR / f"sentence_embeddings_{t}.meta.json"
        idx = [unique_texts[txt] for txt in dfs[t]["text"]]
        embeddings = unique_embeddings[idx]  # row order == this target's own CSV

        np.save(npy_path, embeddings)
        meta = {
            "model": model_name,
            "prefix": prefix,
            "dim": embeddings.shape[1],
            "n_rows": embeddings.shape[0],
            "input_csv": f"sentences_{t}.csv",
            "input_csv_sha256": csv_hashes[t],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[embed target={t}] {embeddings.shape} -> {npy_path.name}, {meta_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-words", type=int, nargs="+", default=[300, 500, 800])
    ap.add_argument("--model", default=nlp_config.EMBED_MODEL)
    ap.add_argument("--prefix", default=nlp_config.EMBED_PREFIX)
    ap.add_argument("--batch-size", type=int, default=nlp_config.EMBED_BATCH_SIZE)
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=nlp_config.EMBED_FP16)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    embed_targets(args.target_words, args.model, args.prefix, args.force,
                  batch_size=args.batch_size, fp16=args.fp16)


if __name__ == "__main__":
    main()
