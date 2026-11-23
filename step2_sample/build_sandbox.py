#!/usr/bin/env python3
"""Build the sandbox: a self-contained slice of the corpus for exploration.

The slice is the k = 5 ladder design: the pre-2018 census plus ladder bands
1-5 (five random episodes per show-quarter) for the 204 frame shows. It is a
stratified random sample of the full corpus by construction, so distributions
computed on it are unbiased estimates for the frame.

Writes sandbox/
  README.md                 what is in it, distributions, how to use it
  episodes.csv              one row per episode (design, dates, lengths, words, speaker summary)
  shows.csv                 the 204 shows with their sandbox counts
  transcripts/<show>/<episode_id>.json, .txt   (the audio-aligned v2 transcript when one exists)
  diarization/<show>/<episode_id>.json
  speakers/episode_hosts.csv, episode_speakers.csv, show_hosts.csv
  measures/mfte_episode_rates.csv, mfte_imperative_episode_counts.csv, show_level_outcomes.csv
  figures/*.png             distribution plots

    .venv/bin/python step2_sample/build_sandbox.py [--bands 5] [--out sandbox]
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=5)
    ap.add_argument("--out", default="sandbox")
    ap.add_argument("--no-copy", action="store_true", help="tables and figures only")
    args = ap.parse_args()
    out = ROOT / args.out
    for d in ["transcripts", "diarization", "speakers", "measures", "figures"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(ROOT / "data/output/corpus_episode_table.csv", dtype={"show_id": str, "episode_id": str}, parse_dates=["pub_date"])
    cs = pd.read_csv(ROOT / "data/output/corpus_shows.csv", dtype=str)
    kept = set(cs[cs.excluded_reason.isna()].show_id)
    sel = t[t.show_id.isin(kept) & t.band.notna() & (t.band <= args.bands)].copy()
    sel["stratum"] = np.where(sel.band == 0, "census", "ladder")
    sel["band"] = sel.band.astype(int)

    # speaker summary per episode
    eh = pd.read_csv(ROOT / "data/output/diarization/episode_hosts.csv", dtype={"show_id": str, "episode_id": str})
    sel = sel.merge(eh[["episode_id", "n_speakers", "host_share", "n_hosts", "host_status", "aligned"]], on="episode_id", how="left")
    # which transcript is shipped: v2 when it exists
    sel["transcript_source"] = ["archive" if (ROOT / f"data/transcripts_v2/{s}/{e}.json").exists() else "original"
                                for s, e in zip(sel.show_id, sel.episode_id)]
    sel = sel.sort_values(["show_id", "pub_date"])
    cols = ["show_id", "episode_id", "title", "pub_date", "stratum", "band", "quarter", "duration", "n_segments", "n_words",
            "model", "transcript_source", "n_speakers", "n_hosts", "host_share", "host_status", "aligned"]
    sel[cols].to_csv(out / "episodes.csv", index=False)

    shows = pd.read_csv(ROOT / "data/output/corpus_show_list.csv", dtype={"show_id": str})
    g = sel.groupby("show_id").agg(sandbox_episodes=("episode_id", "size"), sandbox_hours=("duration", lambda x: round(x.sum() / 3600, 1)))
    shows = shows.merge(g, on="show_id", how="left").fillna({"sandbox_episodes": 0, "sandbox_hours": 0})
    shows.to_csv(out / "shows.csv", index=False)

    # tables
    for name, path, key in [("episode_hosts.csv", "data/output/diarization/episode_hosts.csv", "episode_id"),
                            ("episode_speakers.csv", "data/output/diarization/episode_speakers.csv", "episode_id")]:
        d = pd.read_csv(ROOT / path, dtype=str, low_memory=False)
        d[d[key].isin(set(sel.episode_id))].to_csv(out / "speakers" / name, index=False)
    sh = pd.read_csv(ROOT / "data/output/diarization/show_hosts.csv", dtype=str)
    sh[sh.show_id.isin(kept)].to_csv(out / "speakers" / "show_hosts.csv", index=False)
    for name, path in [("mfte_episode_rates.csv", "step7_audience/inputs/mfte_episode_rates.csv"),
                       ("mfte_imperative_episode_counts.csv", "step7_audience/inputs/mfte_imperative_episode_counts.csv")]:
        d = pd.read_csv(ROOT / path, dtype=str)
        d[d.episode_id.isin(set(sel.episode_id))].to_csv(out / "measures" / name, index=False)
    shutil.copy(ROOT / "data/output/show_level_outcomes.csv", out / "measures" / "show_level_outcomes.csv")

    # files
    n_t = n_d = 0
    if not args.no_copy:
        for s, e in zip(sel.show_id, sel.episode_id):
            (out / "transcripts" / s).mkdir(exist_ok=True); (out / "diarization" / s).mkdir(exist_ok=True)
            v2 = ROOT / f"data/transcripts_v2/{s}/{e}.json"
            src = v2 if v2.exists() else next(iter(sorted((ROOT / f"data/transcripts/{s}").glob(f"{e}_*.json"))), None)
            if src is not None:
                dst = out / "transcripts" / s / f"{e}.json"
                if not dst.exists():
                    shutil.copy(src, dst)
                    rec = json.loads(src.read_text())
                    (out / "transcripts" / s / f"{e}.txt").write_text("\n".join(x["text"] for x in rec["segments"]) + "\n")
                n_t += 1
            dj = ROOT / f"data/diarization/{s}/{e}.json"
            if dj.exists():
                dst = out / "diarization" / s / f"{e}.json"
                if not dst.exists():
                    shutil.copy(dj, dst)
                n_d += 1

    # distributions
    hours = sel.duration.sum() / 3600
    L = sel.duration / 60
    yr = sel.groupby(sel.pub_date.dt.year).agg(episodes=("episode_id", "size"), hours=("duration", lambda x: x.sum() / 3600))
    per_show = sel.groupby("show_id").size()
    stats = {
        "shows": int(sel.show_id.nunique()), "episodes": int(len(sel)), "hours": round(hours), "words_M": round(sel.n_words.sum() / 1e6, 1),
        "census": int((sel.band == 0).sum()), "ladder": int((sel.band > 0).sum()),
        "length_median_min": round(L.median(), 1), "length_p10": round(L.quantile(.1), 1), "length_p90": round(L.quantile(.9), 1),
        "per_show_median": int(per_show.median()), "per_show_min": int(per_show.min()), "per_show_max": int(per_show.max()),
        "diarized": int(sel.n_speakers.notna().sum()), "speakers_median": float(sel.n_speakers.median()),
        "host_status": sel.host_status.value_counts().to_dict(), "aligned_share": round(float(sel.aligned.mean()), 3),
        "transcript_source": sel.transcript_source.value_counts().to_dict(),
        "years": {int(k): (int(v.episodes), int(round(v.hours))) for k, v in yr.iterrows()},
        "files": {"transcripts": n_t, "diarization": n_d},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))

    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 10})
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
    ax[0, 0].hist(L.clip(upper=180), bins=36, color="#2A78D6"); ax[0, 0].set_xlabel("episode length (min, capped at 180)"); ax[0, 0].set_ylabel("episodes")
    yr.episodes.plot.bar(ax=ax[0, 1], color="#2A78D6"); ax[0, 1].set_xlabel("publication year"); ax[0, 1].set_ylabel("episodes"); ax[0, 1].tick_params(axis="x", labelrotation=90, labelsize=8)
    ax[1, 0].hist(per_show, bins=30, color="#E34948"); ax[1, 0].set_xlabel("episodes per show"); ax[1, 0].set_ylabel("shows")
    ns = sel.n_speakers.dropna().clip(upper=20); ax[1, 1].hist(ns, bins=np.arange(0.5, 21.5, 1), color="#EB6834"); ax[1, 1].set_xlabel("speakers per episode (capped at 20)"); ax[1, 1].set_ylabel("episodes")
    for a in ax.flat: a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Sandbox: census + ladder bands 1–{args.bands}, {len(sel):,} episodes, {hours:,.0f} hours", y=0.995)
    fig.tight_layout(); fig.savefig(out / "figures" / "distributions.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    hs = sel.host_share.dropna(); ax[0].hist(hs, bins=25, color="#2A78D6"); ax[0].set_xlabel("host share of speech (host_found episodes)"); ax[0].set_ylabel("episodes")
    wpm = (sel.n_words / (sel.duration / 60))[sel.duration > 60]; ax[1].hist(wpm.clip(100, 230), bins=30, color="#E34948"); ax[1].set_xlabel("words per minute"); ax[1].set_ylabel("episodes")
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "figures" / "speech.png", dpi=150); plt.close(fig)

    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
