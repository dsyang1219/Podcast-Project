"""BERT arm, deliverable 4 -- dimensionality test via PCA + Horn's parallel analysis.

    python -m nlp.embed_rank [--target-words 500] [--chunk-b 199] [--sentence-b 19]

Runs at TWO units:
  - chunk level (primary -- comparable to the CA arm's chunk-level unit):
    all 5 corrected variants (a=centered, b=centered+standardized,
    c1/c2/c3=all-but-top-{1,2,3}), each WITH and WITHOUT show-residualization.
  - sentence level (robustness -- much higher N): the centered-only variant
    only, WITH and WITHOUT residualization, smaller B and randomized SVD
    capped at the top 50 components (exact full-spectrum SVD on millions of
    rows is computationally impractical and unnecessary -- parallel analysis
    only needs enough components to find where observed drops below null).

Anisotropy correction is mandatory (Ethayarajh 2019); "raw" (uncorrected) is
intentionally NOT included here -- the acceptance-4 checkpoint already
established raw PC1 correlates with sentence length/frequency and correction
is required before interpreting components. What that checkpoint also found
(replicated at full-corpus scale) is that no single corrected variant is
uniformly best, and nothing crosses the |r|>0.5 danger threshold at full
scale -- so per that review, Horn's runs across every corrected variant
rather than a designated winner, and the finding to look at is whether the
RETAINED-DIMENSION COUNT agrees across variants. `centered` is used as the
headline "primary" variant for the deliverable-5 convergence table, but that
is a reporting choice, not a claim that it's the only valid one.

Show-level nuisance control: optionally subtract each show's own mean vector
before the anisotropy-variant transform and PCA (matching the CA arm's
show-residualization intent). Reported with and without so the effect is
visible.

NOTE ON PROVENANCE: no existing show-residualization / permutation-null
implementation was found elsewhere in this repo to mirror -- nlp/pilot/,
nlp/compare_lda_tlda.py, and nlp/sweep_k.py implement a different LDA-vs-TLDA
effective-dimensionality diagnostic (participation-ratio d_eff via
correspondence analysis, no permutation test). This module implements the
show-residualization / within-show permutation-null semantics directly from
the task brief, not by copying an existing "Arm 1" implementation.

Null: Horn's parallel analysis via WITHIN-SHOW permutation -- each embedding
dimension (column) is independently shuffled, but only within a show's own
row-block, which destroys cross-dimension correlation while preserving each
show's marginal per-dimension distribution and between-show structure. A
component is retained if its observed eigenvalue exceeds the 95th percentile
of the permuted-null eigenvalue at that same rank, counting contiguously
from the top (standard parallel-analysis stopping rule).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.utils.extmath import randomized_svd

from pipeline import config as pipeline_config

# Permutations are embarrassingly parallel (independent draws) -- at
# chunk-level B=199 sequential took an estimated ~1.7h; spreading across
# cores is the same lever used for Stage 1 cleaning. Left a few cores free
# on this shared machine rather than claiming all of them.
N_JOBS = max(1, min(14, (os.cpu_count() or 1) - 2))

N_COMPONENTS = 300
ARTIFACT_N_CHECK = 5
ARTIFACT_R_FLAG = 0.5

# Sentence level is a robustness pass, not required to use every sentence --
# subsampling keeps N far above chunk-level (the primary unit) while keeping
# memory/time bounded. A first attempt at full N=3.68M repeatedly allocated
# fresh 11GB arrays per permutation (transform + permute), which drove this
# machine to <1GB free and made a single B=2 smoke test take 50+ CPU-minutes.
SENTENCE_SAMPLE_SIZE = 500_000

CHUNK_VARIANTS = [("centered", 0), ("standardized", 0),
                  ("all_but_top", 1), ("all_but_top", 2), ("all_but_top", 3)]
SENTENCE_VARIANTS = [("centered", 0)]  # robustness pass: primary variant only


def residualize_by_show(X: np.ndarray, show_ids: np.ndarray) -> np.ndarray:
    Xr = X.copy()
    for show in np.unique(show_ids):
        idx = np.where(show_ids == show)[0]
        Xr[idx] -= X[idx].mean(axis=0, keepdims=True)
    return Xr


def permute_within_show(X: np.ndarray, show_ids: np.ndarray, rng: np.random.Generator,
                         out: np.ndarray | None = None) -> np.ndarray:
    """Independently permute each column, separately within each show's rows.

    Accepts a reusable `out` buffer -- called once per permutation in the
    Horn's null loop, and a fresh full-size allocation every call is exactly
    the pattern that caused the memory blowup this was refactored to avoid.
    """
    Xp = out if out is not None else np.empty_like(X)
    for show in np.unique(show_ids):
        idx = np.where(show_ids == show)[0]
        block = X[idx]
        perm = np.argsort(rng.random(block.shape), axis=0)
        Xp[idx] = np.take_along_axis(block, perm, axis=0)
    return Xp


def transform_variant(X: np.ndarray, variant: str, k: int, n_iter: int, seed: int,
                       out: np.ndarray | None = None) -> np.ndarray:
    if out is not None:
        np.subtract(X, X.mean(axis=0, keepdims=True), out=out)
        Xc = out
    else:
        Xc = X - X.mean(axis=0, keepdims=True)
    if variant == "centered":
        return Xc
    if variant == "standardized":
        std = Xc.std(axis=0, keepdims=True)
        # A few embedding dims are near-constant (measured: 2 exactly 0,
        # 2 more at ~1e-7-1e-8 against a typical std of ~0.01) -- dividing by
        # those inflates them ~1e5x and dominates the SVD with a numerical
        # artifact, not a real direction. `std == 0` alone missed the
        # near-zero ones; floor generously since anything this far below the
        # real range is degenerate, not a genuine low-variance signal dim.
        std[std < 1e-6] = 1.0
        Xc /= std  # in place: Xc is already a private copy (`out` or the fresh subtraction above)
        return Xc
    if variant == "all_but_top":
        U, S, Vt = randomized_svd(Xc, n_components=max(k, 3), random_state=seed, n_iter=n_iter)
        top_dirs = Vt[:k]
        Xc -= (Xc @ top_dirs.T) @ top_dirs  # in place, same reasoning as above
        return Xc
    raise ValueError(variant)


def variant_label(variant: str, k: int) -> str:
    if variant == "all_but_top":
        return f"all_but_top_{k}"
    return variant


def top_eigen(X: np.ndarray, n_components: int, n_iter: int, seed: int):
    U, S, Vt = randomized_svd(X, n_components=n_components, random_state=seed, n_iter=n_iter)
    return U, S, S ** 2


def artifact_correlations(U: np.ndarray, S: np.ndarray, lengths: np.ndarray, log_freqs: np.ndarray) -> list[dict]:
    out = []
    for i in range(min(ARTIFACT_N_CHECK, U.shape[1])):
        scores = U[:, i] * S[i]
        r_len, _ = pearsonr(scores, lengths)
        r_freq, _ = pearsonr(scores, log_freqs)
        out.append({
            "component": i + 1,
            "r_length": float(r_len),
            "r_log_freq": float(r_freq),
            "flagged": bool(abs(r_len) > ARTIFACT_R_FLAG or abs(r_freq) > ARTIFACT_R_FLAG),
        })
    return out


def _one_permutation(X_analysis: np.ndarray, show_ids: np.ndarray, variant: str, k: int,
                      n_iter: int, seed: int) -> np.ndarray:
    """One Horn's-null draw: permute within show, transform, top eigenvalues.

    Module-level (not a closure/method) so joblib's loky backend can pickle
    and dispatch it to worker processes.
    """
    rng = np.random.default_rng(seed)
    Xp = permute_within_show(X_analysis, show_ids, rng)
    Xpt = transform_variant(Xp, variant, k, n_iter, seed)
    _, _, eigs = top_eigen(Xpt, N_COMPONENTS, n_iter, seed)
    return eigs


def run_config(X: np.ndarray, show_ids: np.ndarray, lengths: np.ndarray, log_freqs: np.ndarray,
               variant: str, k: int, residualize: bool, B: int, n_iter: int, seed: int) -> dict:
    label = variant_label(variant, k)
    t0 = time.time()
    X_analysis = residualize_by_show(X, show_ids) if residualize else X

    Xt = transform_variant(X_analysis, variant, k, n_iter, seed)
    U, S, observed_eigs = top_eigen(Xt, N_COMPONENTS, n_iter, seed)
    artifacts = artifact_correlations(U, S, lengths, log_freqs)

    # Permutations run across processes (joblib, N_JOBS workers) -- each is
    # an independent draw, so this is embarrassingly parallel. joblib
    # memory-maps large numpy arguments (X_analysis, show_ids) once rather
    # than re-pickling per task, so this doesn't repeat the earlier memory
    # blowup from allocating a fresh full-size array per permutation.
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_one_permutation)(X_analysis, show_ids, variant, k, n_iter, seed + 1 + b)
        for b in range(B)
    )
    null_eigs = np.stack(results)

    null_95th = np.percentile(null_eigs, 95, axis=0)
    retained = 0
    for i in range(N_COMPONENTS):
        if observed_eigs[i] > null_95th[i]:
            retained += 1
        else:
            break

    elapsed = time.time() - t0
    print(f"  [{label} residualized={residualize}] retained={retained} "
          f"observed[0:3]={observed_eigs[:3].round(1)} null95[0:3]={null_95th[:3].round(1)} "
          f"({elapsed:.1f}s, B={B})")

    return {
        "variant": label,
        "residualized": residualize,
        "B": B,
        "n_components_tested": N_COMPONENTS,
        "retained_dimensions": retained,
        "observed_eigenvalues": observed_eigs.tolist(),
        "null_95th_percentile": null_95th.tolist(),
        "artifact_correlations": artifacts,
        "elapsed_seconds": round(elapsed, 1),
    }


def load_freq_table(texts) -> dict:
    import re
    word_re = re.compile(r"[a-z0-9']+")
    freq: dict[str, int] = {}
    for t in texts:
        for w in word_re.findall(t.lower()):
            freq[w] = freq.get(w, 0) + 1
    return freq


def avg_log_freq(texts, freq: dict) -> np.ndarray:
    import re
    word_re = re.compile(r"[a-z0-9']+")
    out = np.empty(len(texts), dtype=np.float64)
    for i, t in enumerate(texts):
        words = word_re.findall(t.lower())
        out[i] = np.mean([np.log(freq.get(w, 1)) for w in words]) if words else 0.0
    return out


def run_target(target: int, chunk_b: int, sentence_b: int, n_iter: int, seed: int) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    report: dict = {
        "target": target, "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_components_tested": N_COMPONENTS, "artifact_r_flag": ARTIFACT_R_FLAG,
        "primary_variant_for_convergence_table": "centered",
        "chunk_level": [], "sentence_level": [],
    }

    print(f"\n=== target={target}: CHUNK level (primary) ===")
    chunks_bert = pd.read_csv(OUT / f"chunks_{target}_bert.csv",
                               usecols=["chunk_id", "collection_id", "bert_text", "n_content_words"])
    meta = json.loads((OUT / f"chunk_embeddings_{target}_weighted.meta.json").read_text())
    chunk_id_order = meta["chunk_id_order"]
    X_chunk = np.load(OUT / f"chunk_embeddings_{target}_weighted.npy")

    by_id = chunks_bert.set_index("chunk_id")
    show_ids = by_id.loc[chunk_id_order, "collection_id"].to_numpy()
    bert_texts = by_id.loc[chunk_id_order, "bert_text"].tolist()

    freq = load_freq_table(bert_texts)
    lengths = np.array([len(t.split()) for t in bert_texts], dtype=np.float64)
    log_freqs = avg_log_freq(bert_texts, freq)

    for variant, k in CHUNK_VARIANTS:
        for residualize in (False, True):
            cfg = run_config(X_chunk, show_ids, lengths, log_freqs,
                              variant, k, residualize, chunk_b, n_iter, seed)
            report["chunk_level"].append(cfg)

    del X_chunk

    print(f"\n=== target={target}: SENTENCE level (robustness) ===")
    sents = pd.read_csv(OUT / f"sentences_{target}.csv", keep_default_na=False,
                         usecols=["chunk_id", "collection_id", "n_raw_words", "text"])
    n_total = len(sents)
    if n_total > SENTENCE_SAMPLE_SIZE:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=SENTENCE_SAMPLE_SIZE, replace=False))
        sents = sents.iloc[idx].reset_index(drop=True)
        # mmap so only the sampled rows are actually read off disk/into RAM --
        # the full array (11GB+ at full corpus scale) is never materialized.
        X_sent_mmap = np.load(OUT / f"sentence_embeddings_{target}.npy", mmap_mode="r")
        X_sent = np.asarray(X_sent_mmap[idx])
        print(f"[sentence-level] subsampled {SENTENCE_SAMPLE_SIZE}/{n_total} sentences "
              f"(robustness pass -- still {SENTENCE_SAMPLE_SIZE // 1000}k >> "
              f"{report['chunk_level'] and 'chunk-level N' or ''})")
    else:
        X_sent = np.load(OUT / f"sentence_embeddings_{target}.npy")
    show_ids_sent = sents["collection_id"].to_numpy()
    sent_lengths = sents["n_raw_words"].to_numpy(dtype=np.float64)
    sent_freq = load_freq_table(sents["text"].tolist())
    sent_log_freqs = avg_log_freq(sents["text"].tolist(), sent_freq)

    for variant, k in SENTENCE_VARIANTS:
        for residualize in (False, True):
            cfg = run_config(X_sent, show_ids_sent, sent_lengths, sent_log_freqs,
                              variant, k, residualize, sentence_b, n_iter, seed)
            report["sentence_level"].append(cfg)

    report_path = OUT / f"embed_rank_report_{target}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[report] -> {report_path.name}")

    make_scree_plots(report, target, OUT)


def make_scree_plots(report: dict, target: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for unit, configs in [("chunk_level", report["chunk_level"]),
                          ("sentence_level", report["sentence_level"])]:
        if not configs:
            continue
        for residualize in (False, True):
            subset = [c for c in configs if c["residualized"] == residualize]
            if not subset:
                continue
            fig, ax = plt.subplots(figsize=(9, 6))
            x = np.arange(1, N_COMPONENTS + 1)
            for cfg in subset:
                ax.plot(x, cfg["observed_eigenvalues"], label=f"{cfg['variant']} observed")
                ax.plot(x, cfg["null_95th_percentile"], "--", alpha=0.5,
                        label=f"{cfg['variant']} null 95th")
            ax.set_xlabel("component rank")
            ax.set_ylabel("eigenvalue")
            ax.set_title(f"{unit} target={target} residualized={residualize}\n"
                         f"observed vs. within-show-permutation null (all variants overlaid)")
            ax.legend(fontsize=7, ncol=2)
            ax.set_xlim(1, N_COMPONENTS)
            fig.tight_layout()
            fig_path = out_dir / f"scree_{unit}_{target}_resid{int(residualize)}.png"
            fig.savefig(fig_path, dpi=120)
            plt.close(fig)
            print(f"[scree] -> {fig_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-words", type=int, nargs="+", default=[500])
    ap.add_argument("--chunk-b", type=int, default=199)
    ap.add_argument("--sentence-b", type=int, default=19)
    ap.add_argument("--n-iter", type=int, default=2, help="randomized_svd power iterations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for target in args.target_words:
        run_target(target, args.chunk_b, args.sentence_b, args.n_iter, args.seed)


if __name__ == "__main__":
    main()
