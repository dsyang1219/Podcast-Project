"""Is ideology's low variance an artifact of the LDA step? Test it on raw words.

The narrow question
-------------------
The topic-PCA found ideology recoverable but LOW-VARIANCE: a ~1.5%-variance
direction that would rank ~14th in the scree (K=75, ridge, LOSO r2=0.478). A
reviewer-style objection is that the LDA step compressed the ideological
structure away BEFORE the PCA looked for it -- 38,008 vocabulary terms squeezed
into 75 topics could plausibly discard exactly the axis of interest.

This skips LDA entirely and runs the same dimensionality analysis directly on
the show-level word matrix. Either ideology is still low-variance on raw words
(LDA is not responsible), or it is not (LDA was hiding it).

HONEST EXPECTATION, STATED UP FRONT
-----------------------------------
The likely result is that ideology remains LOW-VARIANCE on raw words -- that
discourse varies more by subject matter than by ideological position regardless
of representation. If that is what happens it CONFIRMS and STRENGTHENS the
existing finding rather than adding a new one. If instead ideology turns out to
be high-variance at the word level, that is a genuine surprise that qualifies
the topic-arm result and would mean the representation needs reconsidering.
Whichever occurs is reported; nothing is tuned toward either.

Comparability
-------------
The matrix is built from tokens_moderate_adclean.csv -- the SAME token stream
the topic arm fed to LDA (moderate regime, ad-span-cleaned, docfreq-pruned to
38,008 terms). So the only thing that differs between the two arms is whether
LDA sits in the middle. Same 204 shows, same targets, same Horn's rule.

Weighting and centering
-----------------------
TF-IDF then L2 row normalization -- the standard text-SVD normalization. Word
counts are NOT CLR-transformed: CLR is the right treatment for topic
proportions, whose simplex constraint is a modelling artifact, but applying it
to raw term counts would take logs of a extremely sparse matrix dominated by
zeros and is not what the LSA literature does.

Two decompositions are reported because they answer subtly different questions:

  - CENTERED PCA is the primary number, because the topic arm used centered PCA.
    A rank comparison against "rank ~14 on topics" is only fair if both sides
    center.
  - UNCENTERED truncated SVD (classical LSA) is reported alongside. Its first
    component is well known to capture the corpus centroid -- the "mean
    document" direction -- rather than a contrast, so ideology's rank under LSA
    is inflated by one near-artifactual component. Stated rather than hidden.

    .venv/bin/python -m nlp.adspan_word_svd [--b 200] [--components 30]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import RidgeCV
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .adspan_phase_c import OUT_DIR, paths_for
from .adspan_target_robustness import build_targets
from .adspan_topic_pca import fisher_ci

N_TOP_WORDS = 12
TOPIC_ARM_REFERENCE = {
    "k": 75,
    "unsupervised_ideology_rank": 5,
    "unsupervised_pct_var": 5.13,
    "unsupervised_abs_r": 0.281,
    "supervised_ridge_pct_var": 1.51,
    "supervised_ridge_insert_rank": 14,
    "supervised_ridge_r2_loso": 0.478,
    "n_components_total": 74,
}


# ------------------------------------------------------------- matrix ----
def build_show_dtm() -> tuple[pd.Index, np.ndarray, list[str]]:
    """204 shows x vocabulary counts, from the topic arm's own token stream."""
    P = paths_for("clean")
    df = pd.read_csv(P["tokens"], dtype={"collection_id": str})
    df["clean_text"] = df["clean_text"].fillna("")
    docs = df.groupby("collection_id")["clean_text"].apply(" ".join)
    vec = CountVectorizer(analyzer=str.split, lowercase=False, min_df=1)
    counts = vec.fit_transform(docs.to_numpy())
    return docs.index, counts, vec.get_feature_names_out().tolist()


def tfidf_l2(counts) -> np.ndarray:
    """TF-IDF then L2 row-normalize. Dense is fine at 204 x ~38k (~62 MB)."""
    X = TfidfTransformer(sublinear_tf=True).fit_transform(counts)
    return np.asarray(normalize(X, norm="l2").todense())


# ---------------------------------------------------------------- SVD ----
def eig_scores_loadings(X: np.ndarray, n_comp: int, center: bool):
    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    eig = (s ** 2) / (Xc.shape[0] - 1)
    return eig[:n_comp], U[:, :n_comp] * s[:n_comp], Vt[:n_comp].T, float(eig.sum())


def gram_eigs(X: np.ndarray, n_comp: int, center: bool) -> np.ndarray:
    """Top eigenvalues via the 204x204 Gram matrix.

    With n=204 rows and ~38k columns the covariance eigenvalues equal those of
    X Xt / (n-1), which is 204x204 -- vastly cheaper than an SVD of the full
    matrix and exact, not approximate. This is what makes a 200-draw Horn's null
    tractable on a word matrix at all.
    """
    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    g = Xc @ Xc.T / (Xc.shape[0] - 1)
    w = np.linalg.eigvalsh(g)[::-1]
    return np.clip(w[:n_comp], 0, None)


def horns_dtm(counts, n_comp: int, b: int, seed: int) -> dict[str, dict]:
    """Permute each term column across shows, then re-apply TF-IDF + L2.

    Permute-then-transform, mirroring the topic arm. Column permutation leaves
    each term's document frequency (and therefore its idf) unchanged while
    destroying between-term correlation, which is exactly the null wanted.

    Both centerings are drawn from the SAME permutation loop -- the permutation
    is the expensive step (38k columns per draw) and it is identical for both,
    so running them separately would double the cost for no extra information.
    """
    rng = np.random.default_rng(seed)
    dense_counts = np.asarray(counts.todense())
    X_obs = tfidf_l2(counts)
    observed = {c: gram_eigs(X_obs, n_comp, c) for c in (True, False)}

    null = {True: np.empty((b, n_comp)), False: np.empty((b, n_comp))}
    perm = np.empty_like(dense_counts)
    for i in range(b):
        for j in range(dense_counts.shape[1]):
            perm[:, j] = rng.permutation(dense_counts[:, j])
        Xp = tfidf_l2(csr_matrix(perm))
        for c in (True, False):
            null[c][i] = gram_eigs(Xp, n_comp, c)

    out = {}
    for c, label in ((True, "centered_pca"), (False, "uncentered_lsa")):
        null95 = np.percentile(null[c], 95, axis=0)
        retained = 0
        for i in range(n_comp):
            if observed[c][i] > null95[i]:
                retained += 1
            else:
                break
        out[label] = {"b": b, "retained": int(retained),
                      "n_tested": n_comp,
                      "hit_ceiling": retained == n_comp,
                      "observed": observed[c].tolist(),
                      "null_95th": null95.tolist()}
    return out


# --------------------------------------------------------- supervised ----
def ridge_direction(X: np.ndarray, y: np.ndarray):
    mu = X.mean(axis=0)
    r = RidgeCV(alphas=np.logspace(-3, 5, 40))
    r.fit(X - mu, y - y.mean())
    w = r.coef_.ravel()
    n = np.linalg.norm(w)
    return (w / n if n > 0 else w), mu


def supervised_cell(X: np.ndarray, y: np.ndarray, eig: np.ndarray,
                    total_var: float, vocab: list[str]) -> dict:
    w, mu = ridge_direction(X, y)
    t = (X - mu) @ w
    var_dir = float(t.var(ddof=1))

    preds = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), dtype=bool)
        m[i] = False
        wi, mui = ridge_direction(X[m], y[m])
        preds[i] = float((X[i] - mui) @ wi)
    r_cv = float(np.corrcoef(preds, y)[0, 1])

    order = np.argsort(w)
    return {
        "r_in_sample": float(np.corrcoef(t, y)[0, 1]),
        "r_loso_cv": r_cv,
        "r2_loso_cv": r_cv ** 2,
        "pct_var": float(var_dir / total_var * 100),
        "would_insert_at_rank": int((eig > var_dir).sum() + 1),
        "top_words_positive": [vocab[i] for i in order[::-1][:N_TOP_WORDS]],
        "top_words_negative": [vocab[i] for i in order[:N_TOP_WORDS]],
        "note": "SUPERVISED -- recoverability, not dominance; not a component",
    }


# --------------------------------------------------------------- main ----
def locate_ideology(show_index: pd.Index, scores: np.ndarray, eig: np.ndarray,
                    total_var: float, n_report: int,
                    targets: dict[str, pd.Series]) -> dict:
    out = {}
    for name, y_series in targets.items():
        common = show_index.intersection(y_series.index)
        pos = show_index.get_indexer(common)
        y = y_series.loc[common].to_numpy(dtype=float)
        rows = []
        for c in range(n_report):
            x = scores[pos, c]
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
            rows.append({"component": c + 1, "r": r, "abs_r": abs(r),
                         "ci_95": fisher_ci(r, len(y)),
                         "pct_var": float(eig[c] / total_var * 100)})
        best = max(rows, key=lambda d: d["abs_r"])
        out[name] = {"n": int(len(y)), "per_component": rows,
                     "top_component": best["component"], "top_abs_r": best["abs_r"],
                     "top_ci": best["ci_95"], "top_pct_var": best["pct_var"],
                     "c1_abs_r": rows[0]["abs_r"],
                     "on_c1_or_c2": best["component"] <= 2}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b", type=int, default=200)
    ap.add_argument("--components", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    targets = build_targets()
    show_index, counts, vocab = build_show_dtm()
    print(f"[dtm] {counts.shape[0]} shows x {counts.shape[1]} terms "
          f"({counts.nnz / np.prod(counts.shape) * 100:.1f}% dense), "
          f"{int(counts.sum()):,} tokens")
    print(f"[dtm] preprocessing inherited from the topic arm's "
          f"tokens_moderate_adclean.csv -- no LDA in this path")

    X = tfidf_l2(counts)
    n_comp = min(args.components, X.shape[0] - 1)
    # Horn's is tested over EVERY available component, not just the ones
    # reported: retention counts contiguously from the top, so capping the test
    # at the reporting depth would silently report the cap as the answer.
    n_horn = X.shape[0] - 1
    horns_all = horns_dtm(counts, n_horn, args.b, args.seed)

    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "n_shows": int(X.shape[0]), "vocab_size": int(X.shape[1]),
              "weighting": "TF-IDF (sublinear tf) + L2 row norm; no CLR",
              "expectation_stated_up_front": (
                  "ideology expected to remain LOW-VARIANCE on raw words, which "
                  "would confirm LDA is not responsible"),
              "topic_arm_reference": TOPIC_ARM_REFERENCE,
              "arms": {}}

    for center, label in ((True, "centered_pca"), (False, "uncentered_lsa")):
        eig, scores, loadings, total_var = eig_scores_loadings(X, n_comp, center)
        horn = horns_all[label]
        ideo = locate_ideology(show_index, scores, eig, total_var, n_comp, targets)

        comps = []
        for c in range(min(6, n_comp)):
            o = np.argsort(loadings[:, c])
            comps.append({"component": c + 1,
                          "pct_var": float(eig[c] / total_var * 100),
                          "top_words_positive": [vocab[i] for i in o[::-1][:N_TOP_WORDS]],
                          "top_words_negative": [vocab[i] for i in o[:N_TOP_WORDS]]})

        sup = {}
        for name in ("extended_all", "host_only"):
            common = show_index.intersection(targets[name].index)
            pos = show_index.get_indexer(common)
            y = targets[name].loc[common].to_numpy(dtype=float)
            sup[name] = supervised_cell(X[pos], y, eig, total_var, vocab)

        report["arms"][label] = {
            "centered": center, "n_components": n_comp,
            "horns": horn,
            "pct_var_per_component": (eig / total_var * 100).tolist(),
            "cum_pct_var": (np.cumsum(eig) / total_var * 100).tolist(),
            "ideology": ideo, "components": comps, "supervised": sup,
        }

        ceiling = "  [HIT CEILING -- true count may be higher]" if horn["hit_ceiling"] else ""
        print(f"\n{'=' * 74}\n[{label}] Horn's retained {horn['retained']} of "
              f"{horn['n_tested']} tested (B={args.b}){ceiling}\n{'=' * 74}")
        for c in range(min(8, n_comp)):
            m = "  <-retained" if c < horn["retained"] else ""
            print(f"  C{c+1:<3} eig={horn['observed'][c]:.5f} null95={horn['null_95th'][c]:.5f} "
                  f"{eig[c]/total_var*100:5.1f}%  cum {np.cumsum(eig)[c]/total_var*100:5.1f}%{m}")
        for name, res in ideo.items():
            print(f"  {name:<13} n={res['n']:<4} strongest C{res['top_component']} "
                  f"|r|={res['top_abs_r']:.3f} CI{np.round(res['top_ci'],3).tolist()} "
                  f"var={res['top_pct_var']:.2f}%  C1 |r|={res['c1_abs_r']:.3f}")
        for name, s in sup.items():
            print(f"  [supervised] {name:<13} r_LOSO={s['r_loso_cv']:+.3f} "
                  f"r2={s['r2_loso_cv']:.3f}  var={s['pct_var']:.2f}%  "
                  f"would-insert #{s['would_insert_at_rank']}")

    out = OUT_DIR / "word_svd_dimensionality.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
