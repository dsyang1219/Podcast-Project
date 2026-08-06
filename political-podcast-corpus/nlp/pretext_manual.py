"""Part B, standard preText output (Denny & Spirling 2018, Political
Analysis 26(2), 168-189) -- Python reimplementation.

Denny & Spirling's own `preText` R package requires `topicmodels`, which
requires the GNU Scientific Library (libgsl-dev) at the OS level; not
installed here, and installing it needs sudo the agent doesn't have (see
project history for the R attempt in r/). The task brief explicitly allows
"preText OR an equivalent manual sweep" -- this is that sweep, following
the same design as the paper:

  1. Enumerate the 2^6 = 64 factorial combinations of six binary
     preprocessing choices: punctuation removal, number removal,
     lowercasing, stemming (Porter), stopword removal, n-gram (bigram)
     inclusion. (Matches the paper's canonical six; this project's
     min/max-docfreq and disfluency/spoken-stopword-list toggles are
     varied separately in the curated 4-regime substantive comparison --
     bundling them into this factorial would make it 2^8=256 and dilute
     the specific "does this classic-D&S choice move things" question.)
  2. Build a document-term matrix per specification over a SAMPLE of
     passages (preText's own examples/paper use samples of a few hundred
     documents, not a full corpus -- this is an O(n^2) pairwise-distance
     method).
  3. Compute pairwise cosine document distances per specification.
  4. Score each specification's "unusualness": how different its
     document-distance structure is from the consensus across all other
     specifications. THIS PROJECT'S OPERATIONALIZATION (documented
     explicitly since it is a reimplementation, not the original package's
     internals): preText_score(s) = 1 - mean_{s' != s} corr(dist_vec(s),
     dist_vec(s')), i.e. one minus the average correlation of that
     specification's pairwise-distance vector with every other
     specification's pairwise-distance vector. A specification whose
     notion of "which documents are similar" diverges from the consensus
     gets a HIGH score (an unusual/risky choice combination); a
     specification close to consensus gets a LOW score. This is the same
     spirit as Denny & Spirling's score (identifying outlier
     specifications) even though the exact estimator differs from their
     unreleased-here R implementation.
  5. Regress preText_score on the six binary choice indicators (OLS, via
     numpy since statsmodels is not installed) -- this answers "which
     preprocessing choices drive the most variation," the standard
     citable preText output.

Usage:
    .venv/bin/python -m nlp.pretext_manual
"""
from __future__ import annotations

import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from .porter_stemmer import PorterStemmer
from .preprocess_regimes import extended_stopwords

CHUNKS_PATH = Path("data/output/chunks_500_nolemma.csv")
OUT_DIR = Path("data/output/regimes")
SAMPLE_N = 300
SEED = 0

CHOICE_NAMES = ["remove_punct", "remove_numbers", "lowercase", "stem", "remove_stopwords", "use_ngrams"]

_WORD_KEEP_PUNCT_RE = re.compile(r"[a-zA-Z0-9]+|[^\w\s]")  # words/numbers OR single punctuation marks
_WORD_ONLY_RE = re.compile(r"[a-zA-Z]+")
_WORD_NUM_RE = re.compile(r"[a-zA-Z0-9]+")


def stratified_sample(n: int = SAMPLE_N, seed: int = SEED) -> pd.DataFrame:
    raw = pd.read_csv(CHUNKS_PATH, usecols=["chunk_id", "collection_id", "text"],
                       dtype={"chunk_id": str, "collection_id": str})
    raw["text"] = raw["text"].fillna("")
    raw = raw[raw["text"].str.strip() != ""]
    rng = np.random.default_rng(seed)
    per_show_cap = max(1, int(np.ceil(n / raw["collection_id"].nunique())) + 2)
    parts = []
    for _, g in raw.groupby("collection_id"):
        k = min(len(g), per_show_cap)
        parts.append(g.sample(n=k, random_state=seed))
    sample = pd.concat(parts)
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    return sample.reset_index(drop=True)


def build_spec_tokens(texts: list[str], remove_punct: bool, remove_numbers: bool, lowercase: bool,
                       stem: bool, remove_stopwords: bool, use_ngrams: bool) -> list[list[str]]:
    stemmer = PorterStemmer() if stem else None
    sw = extended_stopwords() if remove_stopwords else None

    token_lists = []
    for t in texts:
        s = t.lower() if lowercase else t
        if remove_punct and remove_numbers:
            toks = _WORD_ONLY_RE.findall(s)
        elif remove_punct and not remove_numbers:
            toks = _WORD_NUM_RE.findall(s)
        elif not remove_punct and remove_numbers:
            toks = [w for w in re.findall(r"\S+", s) if not any(c.isdigit() for c in w)]
        else:
            toks = re.findall(r"\S+", s)
        token_lists.append(toks)

    if remove_stopwords:
        cmp_sw = {w.lower() for w in sw} if lowercase else sw
        token_lists = [[t for t in toks if t.lower() not in cmp_sw] for toks in token_lists]

    if stem:
        token_lists = [[stemmer.stem(t) for t in toks] for toks in token_lists]

    if use_ngrams:
        unigram_counts: dict = {}
        bigram_counts: dict = {}
        for toks in token_lists:
            for w in toks:
                unigram_counts[w] = unigram_counts.get(w, 0) + 1
            for i in range(len(toks) - 1):
                bg = (toks[i], toks[i + 1])
                bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
        total = sum(unigram_counts.values()) or 1
        candidates = {bg: c for bg, c in bigram_counts.items() if c >= 5}
        scored = sorted(candidates, key=lambda bg: candidates[bg] * total / (
            unigram_counts[bg[0]] * unigram_counts[bg[1]]), reverse=True)
        top = set(scored[:100])
        new_lists = []
        for toks in token_lists:
            merged, i = [], 0
            while i < len(toks):
                if i < len(toks) - 1 and (toks[i], toks[i + 1]) in top:
                    merged.append(f"{toks[i]}_{toks[i + 1]}")
                    i += 2
                else:
                    merged.append(toks[i])
                    i += 1
            new_lists.append(merged)
        token_lists = new_lists

    return token_lists


def tokens_to_matrix(token_lists: list[list[str]]) -> np.ndarray:
    vocab: dict = {}
    for toks in token_lists:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    mat = np.zeros((len(token_lists), len(vocab)), dtype=np.float64)
    for i, toks in enumerate(token_lists):
        for t in toks:
            mat[i, vocab[t]] += 1
    return mat


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[pretext] sampling passages (stratified by show)...")
    sample = stratified_sample()
    print(f"[pretext] {len(sample)} passages sampled from {sample['collection_id'].nunique()} shows")
    texts = sample["text"].tolist()

    specs = list(itertools.product([True, False], repeat=len(CHOICE_NAMES)))
    print(f"[pretext] {len(specs)} factorial specifications "
          f"(2^{len(CHOICE_NAMES)} over {CHOICE_NAMES})")

    n_docs = len(texts)
    triu_idx = np.triu_indices(n_docs, k=1)
    dist_vectors = {}
    spec_records = []

    for spec_i, choices in enumerate(specs):
        kwargs = dict(zip(CHOICE_NAMES, choices))
        token_lists = build_spec_tokens(texts, **kwargs)
        mat = tokens_to_matrix(token_lists)
        if mat.shape[1] == 0:
            dist = np.ones(len(triu_idx[0]))
        else:
            d = cosine_distances(mat)
            dist = d[triu_idx]
        dist_vectors[spec_i] = dist
        rec = {f"choice_{k}": int(v) for k, v in kwargs.items()}
        rec["spec_id"] = spec_i
        rec["vocab_size"] = mat.shape[1]
        spec_records.append(rec)
        if (spec_i + 1) % 16 == 0:
            print(f"[pretext] built {spec_i + 1}/{len(specs)} specifications...")

    print("[pretext] scoring specifications (correlation-to-consensus)...")
    n_specs = len(specs)
    corr_matrix = np.zeros((n_specs, n_specs))
    stacked = np.array([dist_vectors[i] for i in range(n_specs)])
    for i in range(n_specs):
        for j in range(n_specs):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue
            a, b = stacked[i], stacked[j]
            if np.std(a) == 0 or np.std(b) == 0:
                corr_matrix[i, j] = 0.0
            else:
                corr_matrix[i, j] = np.corrcoef(a, b)[0, 1]

    scores = []
    for i in range(n_specs):
        others = [corr_matrix[i, j] for j in range(n_specs) if j != i]
        scores.append(1 - np.mean(others))

    for rec, score in zip(spec_records, scores):
        rec["preText_score"] = float(score)

    scores_df = pd.DataFrame(spec_records)
    scores_df.to_csv(OUT_DIR / "pretext_scores.csv", index=False)
    print(f"[pretext] scores written -> {OUT_DIR / 'pretext_scores.csv'}")

    # OLS regression of preText_score on the 6 binary choice indicators
    # (plain numpy least squares + standard errors, no statsmodels installed)
    X_cols = [f"choice_{c}" for c in CHOICE_NAMES]
    X = scores_df[X_cols].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    y = scores_df["preText_score"].to_numpy(dtype=float)

    beta, residuals, rank, sv = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    resid = y - y_hat
    n, p = X_design.shape
    dof = max(n - p, 1)
    sigma2 = float(np.sum(resid ** 2) / dof)
    XtX_inv = np.linalg.pinv(X_design.T @ X_design)
    se = np.sqrt(np.clip(np.diag(sigma2 * XtX_inv), 0, None))
    t_stats = beta / np.where(se == 0, np.nan, se)
    from scipy import stats as sstats
    p_values = 2 * (1 - sstats.t.cdf(np.abs(t_stats), df=dof))

    reg_table = pd.DataFrame({
        "term": ["intercept"] + X_cols,
        "coef": beta,
        "std_err": se,
        "t_stat": t_stats,
        "p_value": p_values,
    })
    reg_table.to_csv(OUT_DIR / "pretext_regression_table.csv", index=False)
    print(f"[pretext] regression table written -> {OUT_DIR / 'pretext_regression_table.csv'}")
    print(reg_table.to_string(index=False))

    ranked = scores_df.sort_values("preText_score", ascending=False)
    top5 = ranked.head(5)[["spec_id", "preText_score"] + X_cols]
    bottom5 = ranked.tail(5)[["spec_id", "preText_score"] + X_cols]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "manual Python reimplementation of Denny & Spirling (2018) preText, "
                  "not the original R package (see module docstring for why + exact score definition)",
        "n_passages_sampled": len(sample),
        "n_shows_sampled": int(sample["collection_id"].nunique()),
        "n_specifications": n_specs,
        "choice_names": CHOICE_NAMES,
        "mean_score": float(np.mean(scores)),
        "sd_score": float(np.std(scores)),
        "regression_r2": float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)),
        "most_influential_choice": reg_table.iloc[1:].reindex(
            reg_table.iloc[1:]["coef"].abs().sort_values(ascending=False).index
        )["term"].iloc[0],
        "top5_most_unusual_specs": top5.to_dict(orient="records"),
        "top5_least_unusual_specs": bottom5.to_dict(orient="records"),
    }
    (OUT_DIR / "pretext_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[pretext] summary -> {OUT_DIR / 'pretext_summary.json'}")
    print(f"[pretext] most influential preprocessing choice on score variation: "
          f"{summary['most_influential_choice']}")
    print("[pretext] DONE")


if __name__ == "__main__":
    run()
