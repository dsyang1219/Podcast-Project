"""Characterize the dominant dimensions of the embedding space (inverse of the
ideology probe): the ideology axis (nlp/embed_ideology.py) sits at PC22,
~2% of variance -- a minor, secondary structure. This asks what DOES organize
the other ~98%: for each of the top chunk-level PCs, what does it correlate
with (quantitative, Step 1) and what do its extreme chunks sound like when
read blind (qualitative, Step 2), then how much of each PC's spread is a
stable show-identity trait vs. an episode-to-episode trait (Step 3), before
naming (Step 4).

    python -m nlp.embed_dimensions [--target-words 500] [--n-pcs 10]

Unit: chunk-level, primary anisotropy variant (centered-only) -- same matrix
nlp/embed_rank.py's Horn's test and nlp/embed_ideology.py's probe used. PC
scores are recomputed here via the identical `chunk_pc_scores` helper
(variant="centered", k=0, seed=0, n_iter=4) so PC ranks line up exactly with
the ideology report's PC22 finding.

Feature provenance (Step 0), all DERIVED from existing metadata -- nothing
re-embedded or re-pooled:
  - episode_id = md5(audio_url)[:16] (matches nlp/run_chunks.py's chunk_id
    prefix and sample_episodes.py's original episode_id construction) joins
    chunks_{target}_bert.csv -> data/output/episodes.csv.gz for pub_date,
    duration_sec, episode_title, episode_description. Verified: 100% of
    chunk-level episode_ids resolve via this hash (7615/7615 distinct
    episodes matched before any row was written into this module).
  - collection_id joins to corpus.csv (show-level production-cadence stats;
    primary_genre/apple_category/apple_subcategory/country are CONSTANT
    across this corpus by sampling design -- reported, not correlated),
    lean_validation.csv (brookings_partisan_leaning, partial coverage), and
    the DIME CFscore lookup nlp/embed_ideology.py already validated
    (avg_host_cfscore, partial coverage) -- included here only as a
    consistency check against the established PC22 finding, not a new probe.

This is descriptive only: no supervised fitting, no re-embedding, no LDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from pipeline import config as pipeline_config
from .embed_ideology import chunk_pc_scores, load_cfscore

TARGET = 500
VARIANT, VARIANT_K = "centered", 0
DEFAULT_N_PCS = 10
NAMING_PCS = 5
N_EXTREME_PER_POLE = 15

GUEST_RE = re.compile(r"\b(?:with|interview|feat\.?|featuring|guest|joins)\b", re.I)
QA_RE = re.compile(r"\b(?:q&a|mailbag|ask me|listener|questions)\b", re.I)
LIVE_RE = re.compile(r"\b(?:live|special|bonus|emergency)\b", re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")

LEAN_MAP = {"More Conservative": 1.0, "More Liberal": -1.0, "Moderate": 0.0, "Unknown": np.nan}

# Features tested for correlation against every top PC. Grouped so Step 1 can
# report per-hypothesis "best PC" (or "no PC found") rather than a flat list.
FEATURE_GROUPS = {
    "length_format": ["log_duration_sec", "n_raw_words", "n_content_words",
                       "title_length_words", "desc_length_words"],
    "temporal": ["recency_days", "pub_hour", "is_weekend"],
    "interview_vs_monologue": ["guest_cue"],
    "episode_type": ["qa_cue", "live_cue"],
    "show_production": ["episode_count", "episodes_with_audio", "hours_available",
                         "avg_episode_minutes", "episodes_per_week"],
    "ideology_consistency_check": ["lean_numeric", "avg_host_cfscore"],
}
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
MEANINGFUL_R = 0.3


def strip_html(s) -> str:
    if not isinstance(s, str):
        return ""
    return HTML_TAG_RE.sub(" ", s)


def episode_id_from_audio_url(url) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16] if isinstance(url, str) and url else ""


def load_episode_metadata() -> pd.DataFrame:
    eps = pd.read_csv(pipeline_config.OUTPUT_DIR / "episodes.csv.gz")
    eps["episode_id"] = eps["audio_url"].map(episode_id_from_audio_url)
    # A handful of audio_urls collide/duplicate across the 153k raw episode
    # rows (7 of 153202) -- keep first, doesn't touch any matched chunk.
    eps = eps.drop_duplicates(subset="episode_id", keep="first")
    return eps.set_index("episode_id")


def load_show_metadata() -> dict:
    """Returns {'features': df indexed by collection_id, 'coverage_report': dict}.

    Reports which corpus.csv fields are usable vs. constant-by-design, so
    Step 0's coverage table is honest about what CAN'T be tested here.
    """
    corpus = pd.read_csv(pipeline_config.OUTPUT_DIR / "corpus.csv")
    coverage_report = {}
    for col in ["primary_genre", "apple_category", "apple_subcategory", "country", "rss_language"]:
        nunique = corpus[col].nunique()
        coverage_report[col] = (
            f"constant ({corpus[col].iloc[0]!r}, {nunique} distinct value(s) across "
            f"{len(corpus)} shows) -- excluded from correlation, corpus is single-genre by design"
            if nunique <= 1 else
            f"{nunique} distinct values, top={corpus[col].value_counts().idxmax()!r} "
            f"({corpus[col].value_counts().max()}/{len(corpus)}) -- not a numeric feature, reported only"
        )

    numeric_show_cols = ["episode_count", "episodes_with_audio", "hours_available",
                          "avg_episode_minutes", "episodes_per_week"]
    show = corpus.set_index("collection_id")[numeric_show_cols].copy()

    lean = pd.read_csv(pipeline_config.OUTPUT_DIR / "lean_validation.csv",
                        usecols=["collection_id", "brookings_partisan_leaning"])
    lean = lean.drop_duplicates(subset="collection_id").set_index("collection_id")
    show["brookings_partisan_leaning"] = lean["brookings_partisan_leaning"]
    show["lean_numeric"] = show["brookings_partisan_leaning"].map(LEAN_MAP)
    coverage_report["brookings_partisan_leaning"] = (
        f"{show['brookings_partisan_leaning'].notna().sum()}/{len(show)} shows matched to a "
        f"Brookings lean label (partial coverage by design -- fuzzy-join, see lean_validation.csv)"
    )

    cf = load_cfscore()  # index: show_id as str, genuine coverage in {full, partial} only
    cf = cf.copy()
    cf.index = cf.index.astype(np.int64)
    show["avg_host_cfscore"] = cf["avg_host_cfscore"]
    coverage_report["avg_host_cfscore"] = (
        f"{show['avg_host_cfscore'].notna().sum()}/{len(show)} shows -- same genuine-only DIME "
        f"CFscore lookup nlp/embed_ideology.py uses (no 0.0-placeholder rows); included here only "
        f"as a consistency check against the established PC22 finding, not a new ideology probe"
    )
    coverage_report["no_follower_count_field"] = (
        "no follower/subscriber-count field exists anywhere in the corpus outputs -- "
        "track_count (episode count) and hours_available are the closest production-scale proxies"
    )
    return {"features": show, "coverage_report": coverage_report}


def build_feature_table(target: int) -> tuple[pd.DataFrame, dict]:
    OUT = pipeline_config.OUTPUT_DIR
    chunks = pd.read_csv(OUT / f"chunks_{target}_bert.csv")
    meta = json.loads((OUT / f"chunk_embeddings_{target}_weighted.meta.json").read_text())
    chunk_id_order = meta["chunk_id_order"]
    chunks = chunks.set_index("chunk_id").loc[chunk_id_order].reset_index()

    chunks["n_raw_words"] = chunks["bert_text"].str.split().apply(len)

    eps = load_episode_metadata()
    ep_cols = ["episode_title", "episode_description", "pub_date", "duration_sec", "duration_source"]
    n_before = len(chunks)
    chunks = chunks.join(eps[ep_cols], on="episode_id")
    n_ep_matched = chunks["pub_date"].notna().sum()

    chunks["pub_dt"] = pd.to_datetime(chunks["pub_date"], utc=True, errors="coerce")
    corpus_start = chunks["pub_dt"].min()
    chunks["recency_days"] = (chunks["pub_dt"] - corpus_start).dt.total_seconds() / 86400.0
    chunks["pub_hour"] = chunks["pub_dt"].dt.hour.astype(float)
    chunks["pub_dow"] = chunks["pub_dt"].dt.dayofweek.astype(float)  # Monday=0 .. Sunday=6
    chunks["pub_dow_name"] = chunks["pub_dt"].dt.day_name()
    chunks["is_weekend"] = chunks["pub_dow"].isin([5.0, 6.0]).astype(float)
    chunks.loc[chunks["pub_dt"].isna(), "is_weekend"] = np.nan

    chunks["log_duration_sec"] = np.log1p(chunks["duration_sec"])

    title = chunks["episode_title"].fillna("")
    desc = chunks["episode_description"].map(strip_html)
    combined = title.str.cat(desc, sep=" ")
    chunks["guest_cue"] = combined.str.contains(GUEST_RE).astype(float)
    chunks["qa_cue"] = combined.str.contains(QA_RE).astype(float)
    chunks["live_cue"] = combined.str.contains(LIVE_RE).astype(float)
    chunks["title_length_words"] = title.str.split().apply(len).astype(float)
    chunks["desc_length_words"] = desc.str.split().apply(len).astype(float)
    # cue flags are undefined (not "0") where there's no title/description text at all
    no_text = chunks["episode_title"].isna() & chunks["episode_description"].isna()
    chunks.loc[no_text, ["guest_cue", "qa_cue", "live_cue"]] = np.nan

    show_meta = load_show_metadata()
    show_cols = ["episode_count", "episodes_with_audio", "hours_available",
                 "avg_episode_minutes", "episodes_per_week", "lean_numeric", "avg_host_cfscore"]
    chunks = chunks.join(show_meta["features"][show_cols], on="collection_id")

    join_report = {
        "n_chunks": int(n_before),
        "n_chunks_episode_matched": int(n_ep_matched),
        "n_distinct_episode_ids": int(chunks["episode_id"].nunique()),
        "n_distinct_shows": int(chunks["collection_id"].nunique()),
        "corpus_csv_join": show_meta["coverage_report"],
    }
    return chunks, join_report


def feature_coverage_table(df: pd.DataFrame, features: list[str]) -> list[dict]:
    n_total = len(df)
    rows = []
    for f in features:
        n_nonnull = int(df[f].notna().sum())
        rows.append({
            "feature": f, "n_nonnull": n_nonnull, "n_total": n_total,
            "pct_coverage": round(100 * n_nonnull / n_total, 1),
        })
    return rows


def compute_pc_scores(target: int, n_components: int) -> pd.DataFrame:
    OUT = pipeline_config.OUTPUT_DIR
    meta = json.loads((OUT / f"chunk_embeddings_{target}_weighted.meta.json").read_text())
    X = np.load(OUT / f"chunk_embeddings_{target}_weighted.npy")
    scores = chunk_pc_scores(X, VARIANT, VARIANT_K, n_components)
    cols = [f"PC{i + 1}" for i in range(n_components)]
    df = pd.DataFrame(scores, columns=cols)
    df["chunk_id"] = meta["chunk_id_order"]
    return df


def eta_squared(values: pd.Series, groups: pd.Series) -> tuple[float, int]:
    """Between-group SS / total SS. Used for day-of-week (Step 1, categorical
    -- weekday isn't an ordered quantity, so Pearson/Spearman on its numeric
    encoding is a weak proxy) and for show-id (Step 3)."""
    mask = values.notna() & groups.notna()
    v, g = values[mask].to_numpy(), groups[mask].to_numpy()
    if len(v) < 2:
        return float("nan"), 0
    grand_mean = v.mean()
    ss_tot = np.sum((v - grand_mean) ** 2)
    if ss_tot == 0:
        return float("nan"), len(v)
    ss_between = 0.0
    for grp in np.unique(g):
        idx = g == grp
        ss_between += idx.sum() * (v[idx].mean() - grand_mean) ** 2
    return float(ss_between / ss_tot), len(v)


def step1_correlations(df: pd.DataFrame, pc_cols: list[str]) -> dict:
    matrix = []
    for pc in pc_cols:
        pc_num = int(pc[2:])
        for feat in ALL_FEATURES:
            sub = df[[pc, feat]].dropna()
            n = len(sub)
            if n < 10 or sub[feat].nunique() < 2:
                matrix.append({"pc": pc_num, "feature": feat, "n": n,
                                "pearson_r": None, "pearson_p": None,
                                "spearman_r": None, "spearman_p": None})
                continue
            r_p, p_p = pearsonr(sub[pc], sub[feat])
            r_s, p_s = spearmanr(sub[pc], sub[feat])
            matrix.append({
                "pc": pc_num, "feature": feat, "n": n,
                "pearson_r": float(r_p), "pearson_p": float(p_p),
                "spearman_r": float(r_s), "spearman_p": float(p_s),
            })

    dow_eta = {}
    for pc in pc_cols:
        eta, n = eta_squared(df[pc], df["pub_dow_name"])
        dow_eta[pc] = {"eta_squared": eta, "n": n}

    top3_per_pc = {}
    flagged = []
    for pc in pc_cols:
        pc_num = int(pc[2:])
        rows = [r for r in matrix if r["pc"] == pc_num and r["pearson_r"] is not None]
        rows_sorted = sorted(rows, key=lambda r: max(abs(r["pearson_r"]), abs(r["spearman_r"])), reverse=True)
        top3_per_pc[pc] = rows_sorted[:3]
        for r in rows_sorted:
            if abs(r["pearson_r"]) > MEANINGFUL_R or abs(r["spearman_r"]) > MEANINGFUL_R:
                flagged.append(r)

    hypothesis_tests = {}
    for group_name, feats in FEATURE_GROUPS.items():
        candidates = [r for r in matrix if r["feature"] in feats and r["pearson_r"] is not None]
        if not candidates:
            hypothesis_tests[group_name] = {"verdict": "no data"}
            continue
        best = max(candidates, key=lambda r: max(abs(r["pearson_r"]), abs(r["spearman_r"])))
        best_abs = max(abs(best["pearson_r"]), abs(best["spearman_r"]))
        hypothesis_tests[group_name] = {
            "best_match": best,
            "found_home": best_abs > MEANINGFUL_R,
            "verdict": (f"PC{best['pc']} (feature={best['feature']}, "
                        f"pearson_r={best['pearson_r']:.3f}, spearman_r={best['spearman_r']:.3f})"
                        if best_abs > MEANINGFUL_R else
                        f"no top PC corresponds to {group_name} "
                        f"(best: PC{best['pc']}/{best['feature']}, "
                        f"|r|<= {MEANINGFUL_R}, pearson_r={best['pearson_r']:.3f})"),
        }

    return {
        "pc_feature_matrix": matrix,
        "top3_per_pc": top3_per_pc,
        "flagged_meaningful": flagged,
        "meaningful_threshold": MEANINGFUL_R,
        "day_of_week_eta_squared": dow_eta,
        "hypothesis_tests": hypothesis_tests,
    }


# Step 4 reconciliation corrected PC1 and PC4's interpretation (see
# embedding_dimensions_report_500.json:step4_naming) -- PC1 is a detector
# with a residual (not bipolar) negative pole, PC4 conflates topic with a
# length/ad-adjacency confound. Figures carry that correction so a reader
# skimming the image alone doesn't get the pre-Step-4 bipolar-dimension
# story for either.
PC_LABEL_OVERRIDES = {
    1: "PC1\n(IR-interview\ndetector)",
    4: "PC4\n(topic+length\nconfound)",
}
PC_LABEL_COLORS = {1: "#8a4b00", 4: "#8a4b00"}  # flag the two corrected columns


def make_heatmap(matrix: list[dict], pc_cols: list[str], out_dir: Path,
                  pc_labels: dict[int, str] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc_labels = pc_labels or {}
    pcs = [int(pc[2:]) for pc in pc_cols]
    grid = np.full((len(ALL_FEATURES), len(pcs)), np.nan)
    by_key = {(r["pc"], r["feature"]): r["pearson_r"] for r in matrix}
    for i, feat in enumerate(ALL_FEATURES):
        for j, pc in enumerate(pcs):
            v = by_key.get((pc, feat))
            grid[i, j] = v if v is not None else np.nan

    fig, ax = plt.subplots(figsize=(1.1 * len(pcs) + 2, 0.45 * len(ALL_FEATURES) + 2))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(pcs)))
    ax.set_xticklabels([pc_labels.get(p, f"PC{p}") for p in pcs])
    for tick, p in zip(ax.get_xticklabels(), pcs):
        if p in PC_LABEL_COLORS:
            tick.set_color(PC_LABEL_COLORS[p])
            tick.set_fontweight("bold")
    ax.set_yticks(range(len(ALL_FEATURES)))
    ax.set_yticklabels(ALL_FEATURES)
    for i in range(len(ALL_FEATURES)):
        for j in range(len(pcs)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(v) < 0.35 else "white")
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"PC x feature correlation (chunk-level, target={TARGET}, variant={VARIANT})\n"
                 f"PC1 reclassified as a detector (residual negative pole); "
                 f"PC4 carries a topic+length+ad-adjacency confound -- see step4_naming")
    fig.tight_layout()
    fig_path = out_dir / f"pc_feature_heatmap_{TARGET}.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"[heatmap] -> {fig_path.name}")


def step2_blind_extremes(df: pd.DataFrame, pc_cols: list[str], out_dir: Path) -> None:
    for pc in pc_cols:
        sorted_df = df.sort_values(pc)
        low = sorted_df.head(N_EXTREME_PER_POLE).copy()
        low["pole"] = "low"
        high = sorted_df.tail(N_EXTREME_PER_POLE).sort_values(pc, ascending=False).copy()
        high["pole"] = "high"
        extremes = pd.concat([high, low])[
            ["pole", pc, "chunk_id", "show_name", "episode_title", "bert_text"]
        ].rename(columns={pc: "pc_score"})
        out_path = out_dir / f"pc_extremes_{pc[2:]}.csv"
        extremes.to_csv(out_path, index=False)
        print(f"[extremes] -> {out_path.name} ({len(extremes)} rows, NO correlation results attached)")


def step3_variance(df: pd.DataFrame, pc_cols: list[str]) -> dict:
    """Between/within-show variance decomposition (eta^2 = SS_between/SS_total)
    on collection_id groups, plus concentration diagnostics: eta^2 alone can't
    distinguish "ranks all N shows on a continuum" from "three shows sit at
    the extreme, everyone else is bunched in the middle" -- both can produce
    the same eta^2. So this also reports, per PC: how many DISTINCT shows
    supply the top/bottom 15 extreme chunks (out of a corpus of ~204 shows --
    few distinct shows at fixed N=30 extreme chunks is itself a concentration
    signal), and the full show-level mean-PC-score distribution's outlier
    shows (|z|>2), with what fraction of that PC's total between-show SS
    those outlier shows alone account for.
    """
    n_shows = df["collection_id"].nunique()
    results = {}
    for pc in pc_cols:
        eta, n = eta_squared(df[pc], df["collection_id"])

        show_means = df.groupby("collection_id")[pc].mean()
        show_ns = df.groupby("collection_id")[pc].size()
        show_names = df.groupby("collection_id")["show_name"].first()
        grand_mean = df[pc].mean()
        z = (show_means - show_means.mean()) / show_means.std()
        outliers = z[z.abs() > 2].index
        ss_between_total = float((show_ns * (show_means - grand_mean) ** 2).sum())
        ss_between_outliers = float((show_ns[outliers] * (show_means[outliers] - grand_mean) ** 2).sum())
        outlier_share = ss_between_outliers / ss_between_total if ss_between_total > 0 else float("nan")

        sorted_df = df.sort_values(pc)
        n_shows_in_low15 = sorted_df.head(N_EXTREME_PER_POLE)["collection_id"].nunique()
        n_shows_in_high15 = sorted_df.tail(N_EXTREME_PER_POLE)["collection_id"].nunique()

        top_shows = show_means.sort_values(ascending=False).head(5)
        bottom_shows = show_means.sort_values().head(5)

        results[pc] = {
            "between_show_eta_squared": eta,
            "within_show_fraction": 1 - eta if not np.isnan(eta) else None,
            "n_chunks": n,
            "n_shows_total": int(n_shows),
            "n_distinct_shows_in_top15_extreme_chunks": int(n_shows_in_high15),
            "n_distinct_shows_in_bottom15_extreme_chunks": int(n_shows_in_low15),
            "n_outlier_shows_zgt2": int(len(outliers)),
            "outlier_shows_share_of_between_ss": round(outlier_share, 3) if not np.isnan(outlier_share) else None,
            "top5_shows_by_mean": [
                {"show": show_names[sid], "mean_pc": round(float(m), 4), "n_chunks": int(show_ns[sid])}
                for sid, m in top_shows.items()
            ],
            "bottom5_shows_by_mean": [
                {"show": show_names[sid], "mean_pc": round(float(m), 4), "n_chunks": int(show_ns[sid])}
                for sid, m in bottom_shows.items()
            ],
        }
    return results


def run_step3(target: int) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    cache_path = OUT / f"embed_dimensions_feature_table_{target}.pkl"
    print(f"[step3] loading cached feature table {cache_path.name} ...")
    df = pd.read_pickle(cache_path)
    pc_cols = [c for c in df.columns if re.fullmatch(r"PC\d+", c)]
    pc_cols = sorted(pc_cols, key=lambda c: int(c[2:]))

    results = step3_variance(df, pc_cols)
    for pc, r in results.items():
        print(f"\n=== {pc} ===")
        print(f"  between-show eta^2 = {r['between_show_eta_squared']:.3f} "
              f"(within-show fraction = {r['within_show_fraction']:.3f})")
        print(f"  distinct shows in top15/bottom15 extreme chunks: "
              f"{r['n_distinct_shows_in_top15_extreme_chunks']}/{r['n_distinct_shows_in_bottom15_extreme_chunks']} "
              f"(out of {r['n_shows_total']} shows in corpus)")
        print(f"  outlier shows (|z|>2 on show-mean): {r['n_outlier_shows_zgt2']}, "
              f"accounting for {r['outlier_shows_share_of_between_ss']} of total between-show SS")
        print(f"  top5 shows by mean: {[(s['show'], s['mean_pc']) for s in r['top5_shows_by_mean']]}")
        print(f"  bottom5 shows by mean: {[(s['show'], s['mean_pc']) for s in r['bottom5_shows_by_mean']]}")

    report_path = OUT / f"embedding_dimensions_report_{target}.json"
    report = json.loads(report_path.read_text())
    report["step3_between_within_variance"] = results
    report["note"] = ("Step 3 done. Step 4 (naming/reconciliation) intentionally NOT run yet -- "
                       "paused for human review together.")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] updated -> {report_path.name}")


def run(target: int, n_pcs: int) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    print(f"[step0] building feature table (target={target}) ...")
    df, join_report = build_feature_table(target)
    print(f"[step0] {json.dumps(join_report, indent=2, default=str)}")

    coverage = feature_coverage_table(df, ALL_FEATURES)
    print("[step0] feature coverage:")
    for row in coverage:
        print(f"  {row['feature']:<24} {row['n_nonnull']:>6}/{row['n_total']} ({row['pct_coverage']}%)")

    print(f"\n[pcs] computing top {n_pcs} chunk-level PCs (variant={VARIANT}) ...")
    pc_df = compute_pc_scores(target, n_pcs)
    pc_cols = [c for c in pc_df.columns if c != "chunk_id"]
    df = df.merge(pc_df, on="chunk_id", how="inner")
    print(f"[pcs] merged, N={len(df)}")

    print("\n[step1] correlations ...")
    step1 = step1_correlations(df, pc_cols)
    for pc in pc_cols:
        top = step1["top3_per_pc"][pc]
        summary = ", ".join(f"{r['feature']}(r={r['pearson_r']:.2f})" for r in top)
        print(f"  {pc}: top3 = {summary}")
    print("\n[step1] hypothesis tests:")
    for name, res in step1["hypothesis_tests"].items():
        print(f"  {name}: {res['verdict']}")

    make_heatmap(step1["pc_feature_matrix"], pc_cols, OUT, pc_labels=PC_LABEL_OVERRIDES)

    print(f"\n[step2] emitting blind extremes for PC1..PC{NAMING_PCS} ...")
    step2_blind_extremes(df, pc_cols[:NAMING_PCS], OUT)

    report = {
        "target": target, "variant": VARIANT, "n_pcs": n_pcs,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "join_report": join_report,
        "feature_coverage": coverage,
        "step1_correlations": {k: v for k, v in step1.items() if k != "pc_feature_matrix"},
        "step1_pc_feature_matrix": step1["pc_feature_matrix"],
        "note": ("Steps 3 (between/within-show variance) and 4 (naming) intentionally "
                 "NOT run yet -- paused after Steps 0/1/2 per task brief for human review "
                 "of the heatmap and blind extremes before reconciliation."),
    }
    report_path = OUT / f"embedding_dimensions_report_{target}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path.name}")

    # stash for the (not-yet-run) Step 3/4 continuation without recomputing
    df.to_pickle(OUT / f"embed_dimensions_feature_table_{target}.pkl")
    print(f"[cache] -> embed_dimensions_feature_table_{target}.pkl (for Step 3/4 continuation)")


def run_relabel(target: int) -> None:
    """Regenerate the heatmap figure from the already-computed report, with
    the Step 4 corrected PC1/PC4 labels -- no re-run of PCA or correlations,
    just re-rendering the same numbers with corrected annotations."""
    OUT = pipeline_config.OUTPUT_DIR
    report = json.loads((OUT / f"embedding_dimensions_report_{target}.json").read_text())
    matrix = report["step1_pc_feature_matrix"]
    pc_cols = sorted({f"PC{r['pc']}" for r in matrix}, key=lambda c: int(c[2:]))
    make_heatmap(matrix, pc_cols, OUT, pc_labels=PC_LABEL_OVERRIDES)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-words", type=int, default=TARGET)
    ap.add_argument("--n-pcs", type=int, default=DEFAULT_N_PCS)
    ap.add_argument("--step3", action="store_true",
                     help="run only Step 3 (between/within-show variance) from the cached feature table")
    ap.add_argument("--relabel-only", action="store_true",
                     help="regenerate the heatmap figure only, with corrected Step 4 PC1/PC4 labels, "
                          "from the existing report (no re-run of PCA or correlations)")
    args = ap.parse_args()
    if args.relabel_only:
        run_relabel(args.target_words)
    elif args.step3:
        run_step3(args.target_words)
    else:
        run(args.target_words, args.n_pcs)


if __name__ == "__main__":
    main()
