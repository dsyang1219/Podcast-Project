"""
Episode sampler for the political podcast corpus.

Draws a random sample of episodes per show until a target transcript-hours
budget H is met, within a fixed lookback window.

Two modes:
  1. build_sample()            -> the real corpus sample at H hours
  2. build_saturation_ladder() -> nested samples at 5/10/15/20/25/30/40h
     for the saturation check. NESTED is the key property: the 10h sample is
     a superset of the 5h sample, etc. So d_eff(10h) vs d_eff(5h) differ only
     because you ADDED data, not because you drew a different random sample.

Reads the pipeline's episode table (data/output/episodes.csv.gz), which has:
    collection_id, show_name, episode_title, episode_description,
    pub_date, audio_url, audio_bytes, duration_sec, duration_source

ADAPTATIONS from the original standalone script (both matter for correctness):
  * Grouping key is the Apple collection_id, NOT the show name. The frame has
    a name collision ("Week In Review" -> two distinct feeds); grouping on the
    string would silently merge them. `show` is carried through as a label.
  * Per-show seed uses a stable hashlib digest, not builtin hash(). Python
    salts hash() of strings per process (PYTHONHASHSEED), so the original
    abs(hash((seed, show))) would reshuffle on every run and SEED would not
    actually freeze the draw. hashlib makes the frozen seed real.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- config -----------------------------
EPISODES_CSV  = "data/output/episodes.csv.gz"   # RSS-parsed episode table
OUT_DIR       = Path("data/output/sample_out")
WINDOW_MONTHS = 24
H_HOURS       = 25                 # chosen from the survival curve
SEED          = 20260713           # freeze this. never change it after the run starts.
LADDER        = [5, 10, 15, 20, 25, 30, 40]

# minimum episode length to include — drops promos, trailers, "we'll be back
# Monday" filler, which are pure noise in a topic model
MIN_EP_MIN    = 5
# ------------------------------------------------------------------


def _stable_seed(seed, key) -> int:
    """Reproducible 32-bit seed from (global seed, per-show key).

    Independent across shows, so adding/removing a show does not reshuffle
    anyone else, and identical across runs (unlike builtin hash()).
    """
    digest = hashlib.md5(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def load_episodes(path=EPISODES_CSV):
    df = pd.read_csv(path)

    # map pipeline columns -> the sampler's schema
    df = df.rename(columns={"show_name": "show", "collection_id": "show_id"})
    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce", utc=True)
    df["duration_hr"] = df["duration_sec"] / 3600.0
    # stable per-episode id (no guid in the feed table); natural key is the
    # audio URL, hashed so drop_duplicates and manifests stay compact
    df["episode_id"] = df["audio_url"].fillna("").map(
        lambda u: hashlib.md5(u.encode()).hexdigest()[:16] if u else "")

    before = len(df)
    df = df[df["pub_date"].notna()]
    df = df[df["duration_sec"].notna() & (df["duration_sec"] > 0)]
    df = df[df["duration_sec"] >= MIN_EP_MIN * 60]
    df = df[df["audio_url"].notna() & (df["audio_url"] != "")]   # must be downloadable
    df = df.drop_duplicates(subset=["show_id", "episode_id"])
    print(f"[load] {before:,} rows -> {len(df):,} usable episodes "
          f"({df['show_id'].nunique()} shows)")
    return df


def apply_window(df, months=WINDOW_MONTHS):
    """Window relative to the LATEST episode in the corpus, not today.

    Using 'today' would penalize shows whose feed lags a few weeks. Anchoring
    to the corpus max keeps every show measured against the same clock.
    """
    anchor = df["pub_date"].max()
    cutoff = anchor - pd.DateOffset(months=months)
    out = df[df["pub_date"] >= cutoff].copy()
    print(f"[window] anchor={anchor.date()}  cutoff={cutoff.date()}  "
          f"-> {len(out):,} episodes, {out['show_id'].nunique()} shows")
    return out


def _draw_until(g, target_hours, rng):
    """Shuffle a show's episodes, accumulate until target_hours is crossed.

    Keeps the episode that crosses the line (slight overshoot accepted).
    Returns the selected rows, in draw order, or None if the show can't reach
    the budget.
    """
    idx = rng.permutation(len(g))
    g = g.iloc[idx]
    cum = g["duration_hr"].cumsum()
    hit = np.searchsorted(cum.values, target_hours, side="left")
    if hit >= len(g):
        return None
    return g.iloc[: hit + 1]


def build_sample(df, H=H_HOURS, seed=SEED):
    """The actual corpus sample."""
    kept, dropped = [], []
    for show_id, g in df.groupby("show_id", sort=True):
        show = g["show"].iloc[0]
        rng = np.random.default_rng(_stable_seed(seed, show_id))
        sel = _draw_until(g, H, rng)
        if sel is None:
            dropped.append({
                "show_id": show_id, "show": show,
                "hours_available": round(g["duration_hr"].sum(), 2),
                "n_episodes": len(g),
                "reason": f"< {H}h in window",
            })
            continue
        sel = sel.copy()
        sel["draw_order"] = range(len(sel))
        kept.append(sel)

    sample = pd.concat(kept, ignore_index=True)
    drops = pd.DataFrame(dropped)

    print(f"\n[sample] H={H}h  ->  {sample['show_id'].nunique()} shows kept, "
          f"{len(drops)} dropped")
    print(f"[sample] {len(sample):,} episodes, "
          f"{sample['duration_hr'].sum():,.0f} total hours")
    return sample, drops


def build_saturation_ladder(df, ladder=LADDER, seed=SEED):
    """Nested samples at each rung.

    Because we shuffle ONCE per show and take longer prefixes, the samples are
    nested by construction: rung k is a superset of rung k-1.
    """
    rows = []
    for show_id, g in df.groupby("show_id", sort=True):
        rng = np.random.default_rng(_stable_seed(seed, show_id))
        idx = rng.permutation(len(g))
        gs = g.iloc[idx].copy()
        cum = gs["duration_hr"].cumsum().values

        for H in ladder:
            hit = np.searchsorted(cum, H, side="left")
            if hit >= len(gs):
                continue                 # this show can't reach this rung
            sel = gs.iloc[: hit + 1].copy()
            sel["rung_h"] = H
            rows.append(sel)

    lad = pd.concat(rows, ignore_index=True)

    print("\n[ladder] episodes per rung:")
    summ = (lad.groupby("rung_h")
               .agg(n_shows=("show_id", "nunique"),
                    n_eps=("episode_id", "count"),
                    hours=("duration_hr", "sum"))
               .round(1))
    print(summ.to_string())
    return lad


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_episodes()
    win = apply_window(df)

    sample, drops = build_sample(win)
    cols = ["show_id", "show", "episode_id", "draw_order", "pub_date",
            "episode_title", "duration_sec", "duration_hr", "duration_source",
            "audio_url"]
    sample[cols].to_csv(OUT_DIR / f"sample_manifest_H{H_HOURS}.csv", index=False)
    drops.to_csv(OUT_DIR / "excluded_shows.csv", index=False)

    ladder = build_saturation_ladder(win)
    ladder_cols = ["rung_h", "show_id", "show", "episode_id", "pub_date",
                   "episode_title", "duration_sec", "duration_hr",
                   "duration_source", "audio_url"]
    ladder[ladder_cols].to_csv(OUT_DIR / "saturation_ladder.csv", index=False)

    # per-show summary — sanity check that hours are actually matched
    per_show = (sample.groupby(["show_id", "show"])
                      .agg(n_eps=("episode_id", "count"),
                           hours=("duration_hr", "sum"),
                           med_ep_min=("duration_sec", lambda s: round(s.median()/60, 1)))
                      .reset_index()
                      .sort_values("n_eps", ascending=False)
                      .round(2))
    per_show.to_csv(OUT_DIR / "per_show_summary.csv", index=False)

    print("\n[check] hours per show should be ~H with small overshoot:")
    print(f"  min={per_show.hours.min():.1f}  median={per_show.hours.median():.1f}  "
          f"max={per_show.hours.max():.1f}")
    print("\n[check] episode counts SHOULD vary wildly (that's the point):")
    print(f"  min={per_show.n_eps.min()}  median={per_show.n_eps.median():.0f}  "
          f"max={per_show.n_eps.max()}")
    print(f"\nwrote -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
