#!/usr/bin/env python3
"""Identify the host(s) of each show from the diarization output.

Two independent routes, then agreement:

1. Cross-episode recurrence of the voice (primary). Within a show, cluster the
   per-episode speaker centroids across episodes (cosine distance, average
   linkage). A guest's voice appears in one episode; a host's appears in nearly
   all of them. A global cluster is a host if it recurs in at least
   --min-recurrence of the show's diarized episodes and holds at least
   --min-share of the show's speech. Up to --max-hosts hosts (co-hosted shows).
   Precedent: the podcast-corpus method of building a per-show host centroid
   from the dominant cluster across episodes (arXiv:2206.03400).

2. Self-introduction by name (check). Host names come from
   data/output/host_lookup.csv (used for the DIME match) and the feed's author
   field when it is a person's name. Within the first --intro-seconds of an
   episode, a segment matching "I'm <first name>", "this is <first name>",
   "my name is <first name>" marks its speaker as a host. This is SPoRC's
   name heuristic (Litterer, Jurgens & Card 2025) restricted to self-reference.

Writes
  data/output/diarization/episode_speakers.csv   one row per (episode, speaker)
  data/output/diarization/show_hosts.csv          one row per show with host clusters,
                                                   recurrence, share and name agreement
  data/output/diarization/episode_hosts.csv       one row per episode with host_status:
                                                   host_found / host_absent (stable-host show,
                                                   guest-hosted or special) / no_fixed_host
                                                   (rotating-presenter feed)
  data/transcripts_host/<show_id>/<episode_id>.txt (with --write-host-text) host-only text,
                                                   purity-filtered, for the MFTE rebuild

    .venv/bin/python step4_diarize/identify_hosts.py --write-host-text
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAR_DIR = PROJECT_ROOT / "data/diarization"
OUT_DIR = PROJECT_ROOT / "data/output/diarization"
HOST_TEXT_DIR = PROJECT_ROOT / "data/transcripts_host"
HOST_LOOKUP = PROJECT_ROOT / "data/output/host_lookup.csv"
CORPUS_SHOWS = PROJECT_ROOT / "data/output/corpus_shows.csv"
CORPUS = PROJECT_ROOT / "data/output/corpus.csv"

ORG_WORDS = re.compile(r"\b(networks?|media|studios?|podcasts?|pod|news|radio|show|the|inc|llc|tv|press|"
                       r"productions?|group|company|foundation|institute|project|report|daily|weekly|review|"
                       r"america|american|nation|hour|brief|dispatch|times|post|journal|live|today|tonight|"
                       r"morning|politics|political|democracy|republic|liberty|freedom|patriot|truth)\b", re.I)
PERSON_NAME = re.compile(r"^[A-Z][a-z'\-\.]+(\s(?:[A-Z]\.?\s)?[A-Z][a-z'\-\.]+){1,2}$")  # Title Case tokens, optional middle initial


# --------------------------------------------------------------------------- names

def person_names(raw: str | float) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = re.split(r"\s*(?:,|&|\band\b|\bwith\b|/|\+)\s*", raw)
    names = []
    for p in parts:
        p = re.sub(r"\(.*?\)", "", p).strip()
        if p and PERSON_NAME.match(p) and not ORG_WORDS.search(p):
            names.append(p)
    return names


def load_host_names() -> dict[str, list[str]]:
    """show_id -> list of host full names from host_lookup.csv and the feed author."""
    shows = pd.read_csv(CORPUS_SHOWS, dtype=str)
    name_to_id = {}
    for _, r in shows.iterrows():
        for col in ("show_name", "show_label"):
            if isinstance(r.get(col), str):
                name_to_id[r[col].strip().lower()] = r["show_id"]
    show_titles: dict[str, set[str]] = defaultdict(set)
    for _, r in shows.iterrows():
        for col in ("show_name", "show_label"):
            if isinstance(r.get(col), str):
                show_titles[r["show_id"]].add(r[col].strip().lower())
    out: dict[str, list[str]] = defaultdict(list)
    if HOST_LOOKUP.exists():
        for _, r in pd.read_csv(HOST_LOOKUP, dtype=str).iterrows():
            sid = name_to_id.get(str(r["show"]).strip().lower())
            if sid:
                out[sid].extend(person_names(r.get("extracted_host")))
    if CORPUS.exists():
        c = pd.read_csv(CORPUS, dtype=str)
        for _, r in c.iterrows():
            for col in ("rss_author", "rss_owner_name"):
                out[r["collection_id"]].extend(person_names(r.get(col)))
    # a feed author equal to the show's own title is a brand, not a person
    return {k: sorted({n for n in v if n.lower() not in show_titles.get(k, set())})
            for k, v in out.items() if v}


def intro_regex(names: list[str]) -> re.Pattern | None:
    firsts = sorted({n.split()[0] for n in names if len(n.split()[0]) > 2})
    if not firsts:
        return None
    alt = "|".join(re.escape(f) for f in firsts)
    return re.compile(rf"\b(?:I'?m|I am|this is|my name is|it'?s)\s+(?:{alt})\b", re.I)


# --------------------------------------------------------------------------- load

def is_aligned(rec: dict) -> bool:
    """Whisper timestamps match the diarized audio. False when the archived
    audio differs in length from the transcribed file (dynamic ad insertion),
    in which case the segment-level speaker labels are unreliable until the
    episode is re-transcribed (retranscribe.py)."""
    p = rec.get("provenance", {})
    if "aligned" in p:
        return bool(p["aligned"])
    td = p.get("transcript_duration")
    return td is not None and abs(float(rec["duration"]) - float(td)) <= 0.5


def load_show(show_dir: Path, variant: str) -> list[dict]:
    recs = []
    for p in sorted(show_dir.glob("*.json")):
        parts = p.name.split(".")
        v = parts[1] if len(parts) == 3 else "default"
        if v != variant:
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        rec["_aligned"] = is_aligned(rec)
        recs.append(rec)
    return recs


# --------------------------------------------------------------------------- core

def cluster_show(recs: list[dict], distance: float) -> tuple[dict[tuple[str, str], int], int]:
    """Map (episode_id, speaker) -> global cluster id across the show's episodes."""
    keys, vecs = [], []
    for r in recs:
        for spk, info in r["speakers"].items():
            if "centroid" in info:
                v = np.asarray(info["centroid"], dtype=np.float64)
                n = np.linalg.norm(v)
                if n > 0:
                    keys.append((r["episode_id"], spk))
                    vecs.append(v / n)
    if not keys:
        return {}, 0
    if len(keys) == 1:
        return {keys[0]: 0}, 1
    X = np.vstack(vecs)
    Z = linkage(pdist(X, metric="cosine"), method="average")
    labels = fcluster(Z, t=distance, criterion="distance") - 1
    return dict(zip(keys, labels)), int(labels.max()) + 1


def identify(recs: list[dict], names: list[str], args) -> tuple[list[dict], dict]:
    show_id = recs[0]["show_id"]
    n_ep = len(recs)
    assign, n_clusters = cluster_show(recs, args.distance)

    # recurrence and share per global cluster
    ep_of: dict[int, set] = defaultdict(set)
    dur_of: dict[int, float] = defaultdict(float)
    total = 0.0
    for r in recs:
        for spk, info in r["speakers"].items():
            g = assign.get((r["episode_id"], spk))
            total += info["duration"]
            if g is not None:
                ep_of[g].add(r["episode_id"])
                dur_of[g] += info["duration"]
    recurrence = {g: len(eps) / n_ep for g, eps in ep_of.items()}
    share = {g: d / total for g, d in dur_of.items()} if total else {}

    ranked = sorted(recurrence, key=lambda g: (recurrence[g], share.get(g, 0)), reverse=True)
    hosts = [g for g in ranked if recurrence[g] >= args.min_recurrence and share.get(g, 0) >= args.min_share]
    low_confidence = n_ep < args.min_episodes
    if low_confidence or not hosts:
        # too few episodes for recurrence to mean anything: fall back to the
        # most-recurrent cluster, or per-episode dominant speaker if none recurs
        hosts = ranked[: 1] if ranked else []
    hosts = hosts[: args.max_hosts]
    host_set = set(hosts)

    # continuous "hostness": cosine similarity of each speaker centroid to the
    # show's host voice (duration-weighted mean of the unit centroids in the
    # host clusters). Catches a host whose voice split into two clusters across
    # recording setups, and gives a score instead of only a flag.
    vec = {}
    for r in recs:
        for spk, info in r["speakers"].items():
            if "centroid" in info:
                v = np.asarray(info["centroid"], dtype=np.float64)
                n_ = np.linalg.norm(v)
                if n_ > 0:
                    vec[(r["episode_id"], spk)] = (v / n_, info["duration"])
    host_vecs = [(v, d) for k, (v, d) in vec.items() if assign.get(k) in host_set]
    if host_vecs:
        hc = sum(v * d for v, d in host_vecs) / max(1e-9, sum(d for _, d in host_vecs))
        hc /= max(1e-9, np.linalg.norm(hc))
        host_sim = {k: float(v @ hc) for k, (v, _) in vec.items()}
    else:
        host_sim = {}

    # name route
    rx = intro_regex(names)
    name_hits: dict[tuple[str, str], int] = defaultdict(int)
    if rx is not None:
        for r in recs:
            if not r.get("_aligned", True):
                continue
            for s in r["segments"]:
                if s["start"] > args.intro_seconds:
                    break
                # purity guard: the intro line and the guest's first answer often
                # share one Whisper segment; only credit a self-introduction to a
                # speaker who actually dominates that segment
                if (s.get("speaker", "UNK") != "UNK" and s.get("purity", 1.0) >= args.name_min_purity
                        and rx.search(s.get("text", ""))):
                    name_hits[(r["episode_id"], s["speaker"])] += 1

    top_rec = recurrence[ranked[0]] if ranked else 0.0
    fixed_host = (not low_confidence) and top_rec >= args.stable_recurrence

    rows = []
    agree = disagree = 0
    for r in recs:
        eid = r["episode_id"]
        for spk, info in r["speakers"].items():
            g = assign.get((eid, spk))
            is_host_centroid = g in host_set
            if low_confidence and g is None:
                is_host_centroid = info["share"] == max(i["share"] for i in r["speakers"].values())
            sim = host_sim.get((eid, spk))
            if args.sim_threshold and sim is not None and sim >= args.sim_threshold:
                is_host_centroid = True
            is_host_name = name_hits.get((eid, spk), 0) > 0
            if is_host_name:
                agree += int(is_host_centroid)
                disagree += int(not is_host_centroid)
            is_host_final = is_host_centroid or (is_host_name and low_confidence)
            rec_g = recurrence.get(g, 0.0) if g is not None else 0.0
            role = "host" if is_host_final else ("regular" if rec_g >= args.regular_recurrence else "guest")
            rows.append({
                "show_id": show_id, "episode_id": eid, "speaker": spk, "role": role,
                "duration": info["duration"], "share": info["share"],
                "global_cluster": g if g is not None else "",
                "recurrence": round(recurrence.get(g, 0.0), 3) if g is not None else "",
                "host_sim": round(sim, 4) if sim is not None else "",
                "is_host_centroid": int(is_host_centroid),
                "is_host_name": int(is_host_name),
                "is_host": int(is_host_final),
                "low_confidence": int(low_confidence),
                "show_fixed_host": int(fixed_host),
                "aligned": int(r.get("_aligned", True)),
            })
    n_regular = sum(1 for g_ in recurrence if g_ not in host_set and recurrence[g_] >= args.regular_recurrence)
    summary = {
        "show_id": show_id, "n_episodes": n_ep, "n_global_clusters": n_clusters,
        "fixed_host": int(fixed_host), "n_hosts": len(hosts), "n_regular_voices": n_regular,
        "host_recurrence": ";".join(f"{recurrence[g]:.2f}" for g in hosts),
        "host_share": ";".join(f"{share.get(g, 0):.2f}" for g in hosts),
        "host_names_known": ";".join(names),
        "name_intro_episodes": len({k[0] for k in name_hits}),
        "name_agree": agree, "name_disagree": disagree,
        "low_confidence": int(low_confidence),
        "n_unaligned": sum(not r.get("_aligned", True) for r in recs),
    }
    return rows, summary


def write_host_text(recs: list[dict], rows: list[dict], min_purity: float) -> int:
    host_of = {(r["episode_id"], r["speaker"]) for r in rows if r["is_host"]}
    n = 0
    for r in recs:
        if not r.get("_aligned", True):
            continue
        d = HOST_TEXT_DIR / r["show_id"]
        d.mkdir(parents=True, exist_ok=True)
        lines = [s["text"].strip() for s in r["segments"]
                 if (r["episode_id"], s.get("speaker")) in host_of and s.get("purity", 0) >= min_purity]
        if not lines:
            continue  # host absent or show without a fixed host: no host-only text
        (d / f"{r['episode_id']}.txt").write_text("\n".join(lines) + "\n")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="default", help="which diarize.py output variant to read")
    ap.add_argument("--distance", type=float, default=0.35, help="cosine distance cut for voices across episodes")
    ap.add_argument("--min-recurrence", type=float, default=0.5)
    ap.add_argument("--min-share", type=float, default=0.10)
    ap.add_argument("--max-hosts", type=int, default=3)
    ap.add_argument("--min-episodes", type=int, default=4, help="below this, recurrence is not trusted")
    ap.add_argument("--intro-seconds", type=float, default=300.0)
    ap.add_argument("--min-purity", type=float, default=0.8, help="segment purity needed for host-only text")
    ap.add_argument("--name-min-purity", type=float, default=0.8,
                    help="a self-introduction counts only in segments this pure for the labelled speaker")
    ap.add_argument("--sim-threshold", type=float, default=0.6,
                    help="also call a speaker host when its cosine similarity to the show's host voice is at least this "
                         "(name-identified hosts sit at 0.93, other speakers at 0.13 with a 99th percentile of 0.52); 0 disables")
    ap.add_argument("--stable-recurrence", type=float, default=0.7,
                    help="a show has a fixed host only if its top voice recurs in at least this share of episodes")
    ap.add_argument("--regular-recurrence", type=float, default=0.1,
                    help="a non-host voice recurring in at least this share of a show's episodes is a 'regular', else a 'guest'")
    ap.add_argument("--write-host-text", action="store_true")
    ap.add_argument("--shows", default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names_by_show = load_host_names()
    shows = set(args.shows.split(",")) if args.shows else None

    all_rows, summaries = [], []
    for show_dir in sorted(DIAR_DIR.iterdir()) if DIAR_DIR.exists() else []:
        if not show_dir.is_dir() or (shows and show_dir.name not in shows):
            continue
        recs = load_show(show_dir, args.variant)
        if not recs:
            continue
        rows, summary = identify(recs, names_by_show.get(show_dir.name, []), args)
        if args.write_host_text:
            summary["host_text_files"] = write_host_text(recs, rows, args.min_purity)
        all_rows.extend(rows)
        summaries.append(summary)

    if not all_rows:
        print("no diarization output found under", DIAR_DIR)
        return
    ep = pd.DataFrame(all_rows)
    sh = pd.DataFrame(summaries)
    ep.to_csv(OUT_DIR / "episode_speakers.csv", index=False)
    sh.to_csv(OUT_DIR / "show_hosts.csv", index=False)

    # one row per episode: did we find the host, and if not, why not
    g = ep.groupby(["show_id", "episode_id"])
    eph = g.agg(n_speakers=("speaker", "size"), host_share=("share", lambda x: float(x[ep.loc[x.index, "is_host"] == 1].sum())),
                n_hosts=("is_host", "sum"), n_regular=("role", lambda x: int((x == "regular").sum())),
                show_fixed_host=("show_fixed_host", "max"), aligned=("aligned", "max")).reset_index()
    eph["host_status"] = np.where(eph.n_hosts > 0, "host_found",
                                  np.where(eph.show_fixed_host == 1, "host_absent", "no_fixed_host"))
    eph.to_csv(OUT_DIR / "episode_hosts.csv", index=False)
    print("episode host status:", eph.host_status.value_counts().to_dict())
    print(f"shows with a fixed host: {int(sh.fixed_host.sum())} of {len(sh)}; speaker roles: {ep.role.value_counts().to_dict()}")

    n_shows = len(sh)
    host_share = ep[ep.is_host == 1].groupby(["show_id", "episode_id"])["share"].sum()
    named = sh[sh.name_intro_episodes > 0]
    print(f"{n_shows} shows, {len(ep.episode_id.unique())} episodes, {len(ep)} speaker rows")
    print(f"hosts per show: {sh.n_hosts.value_counts().sort_index().to_dict()}; low-confidence shows: {int(sh.low_confidence.sum())}")
    print(f"host speaking share per episode: median {host_share.median():.2f}, IQR {host_share.quantile(.25):.2f}-{host_share.quantile(.75):.2f}")
    if len(named):
        a, d = int(named.name_agree.sum()), int(named.name_disagree.sum())
        print(f"name check: {len(named)} shows with a self-introduction; voice and name agree on {a} of {a + d} speakers ({a / max(1, a + d):.0%})")
    print("wrote", OUT_DIR / "episode_speakers.csv", "and", OUT_DIR / "show_hosts.csv")


if __name__ == "__main__":
    main()
