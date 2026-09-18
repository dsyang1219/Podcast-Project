#!/usr/bin/env python3
"""Priority-ladder sampler for the corpus expansion.

Emits ONE manifest whose row order IS the priority order, so the run can be
stopped at any point and what has been transcribed is a coherent, balanced
corpus rather than a half-finished sweep.

Ladder
------
  P0            every episode published before --census-before (default 2018-01-01)
  P1..Pk        k episodes per show per quarter from that date onward

P0 is a census because the pre-2018 tail is thin (~6.2k episodes, ~5.1k hours)
and irreplaceable: RSS feeds only ever get shorter, so the 2016 cycle cannot be
re-densified later. Everything after is stratified per show per quarter, which
keeps show-time aligned — a recency cap would give a daily show a 5-month
window and a weekly show a 3-year one, confounding exactly the temporal
comparison this corpus exists to support.

Nesting
-------
Ranks come from a frozen hash of (SEED, episode_id), not a shuffle, so they are
stable regardless of input row order or pandas version. The k=8 draw is a strict
subset of k=30: raising --max-k later only appends rows, it never reshuffles
what was already transcribed. Freeze episodes.csv.gz alongside the manifest —
re-pulling feeds adds episodes and would change the per-cell ranks.

Row order
---------
Within a priority band, rows are round-robined across shows. This matters a lot:
download_audio.py paces requests at 1/second per show_id, so a show-grouped
manifest pins every worker to the same show and caps the whole downloader at
~1 file/second no matter the bandwidth. Interleaving lets all workers run
against distinct hosts concurrently.

Usage
-----
    python sample_ladder.py                      # k=30 ladder, default paths
    python sample_ladder.py --max-k 12           # shallower
    python sample_ladder.py --budget-gpu-days 30 --rtf 107.7
"""
import argparse
import csv
import hashlib
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repository root (this script lives one folder down)
EPISODES = PROJECT_ROOT / "data/output/episodes.csv.gz"
TRANSCRIPTS = PROJECT_ROOT / "data/transcripts"
OUT_DIR = PROJECT_ROOT / "data/output/sample_out"
# The frozen show frame. Before this existed the corpus was defined as "whichever
# numeric directories happen to sit under data/transcripts/", so any operation
# that added or moved one silently redefined the corpus for the next run — the
# same freezing problem the docstring already flags for episodes.csv.gz, but on
# the show axis and undocumented. Writing the ids down makes the frame reviewable
# in version control and reproducible off a bare checkout with no transcripts.
CORPUS_SHOWS = PROJECT_ROOT / "data/output/corpus_shows.csv"

SEED = "ladder-20260713"

MANIFEST_COLS = ["show_id", "show", "episode_id", "draw_order", "pub_date",
                 "episode_title", "duration_sec", "duration_hr", "duration_source",
                 "audio_url", "priority", "quarter"]


def stable_rank_key(episode_id: str) -> str:
    """Frozen per-episode sort key. hashlib, not builtin hash() — Python salts
    hash() of strings per process, which would reshuffle the draw every run and
    silently break the nesting guarantee."""
    return hashlib.md5(f"{SEED}:{episode_id}".encode()).hexdigest()


def load_corpus_shows() -> tuple[set[int], str]:
    """Return the frozen show frame, falling back to a directory scan.

    The fallback exists only to bootstrap the file on a machine that already has
    transcripts; it warns, because a scan silently redefines the corpus whenever
    a show directory is added, moved, or cleaned up (see data/transcripts_orphaned).
    Once CORPUS_SHOWS is committed the frame is reproducible from a bare checkout.
    """
    if CORPUS_SHOWS.exists():
        with open(CORPUS_SHOWS, newline="") as f:
            # A row with excluded_reason set stays in the file as a record of a
            # deliberate decision, but is not part of the corpus. Keeping it
            # visible is the point: a show dropped by deleting its row is
            # indistinguishable from one that was never considered.
            ids = {int(r["show_id"]) for r in csv.DictReader(f)
                   if r["show_id"].strip() and not (r.get("excluded_reason") or "").strip()}
        return ids, CORPUS_SHOWS.name

    ids = {int(p.name) for p in TRANSCRIPTS.iterdir()
           if p.is_dir() and p.name.isdigit()}
    print(f"WARNING: {CORPUS_SHOWS.name} missing — falling back to scanning "
          f"{TRANSCRIPTS.name}/ ({len(ids)} shows). Run --freeze-shows to pin "
          f"this frame before it drifts.")
    return ids, "directory scan (UNPINNED)"


def write_corpus_shows(show_ids: set[int], excluded: dict[int, str] | None = None) -> None:
    """Pin the frame, annotated with chart rank and name so a reviewer can see
    what the corpus is without cross-referencing three other files."""
    excluded = dict(excluded or {})
    chart = {}
    for p in sorted(Path(PROJECT_ROOT / "data/output").glob("raw_chart_*.csv")):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                chart[int(r["collection_id"])] = (r.get("rank", ""), r.get("show_name", ""),
                                                  r.get("publisher", ""))
    # show_name is not unique — two distinct feeds both publish as "Week In Review".
    # Anything that groups by name silently merges them into one show, which would
    # corrupt exactly the per-show temporal comparison this corpus exists for. Emit
    # a show_label that is unique by construction, so downstream can group on it.
    from collections import Counter
    dupes = {n for n, c in Counter(chart.get(s, ("", "", ""))[1] for s in show_ids).items()
             if n and c > 1}

    # Carry forward any exclusion decisions already recorded, so rewriting the
    # file never silently readmits a show someone deliberately dropped.
    prior: dict[int, str] = {}
    if CORPUS_SHOWS.exists():
        with open(CORPUS_SHOWS, newline="") as f:
            for r in csv.DictReader(f):
                reason = (r.get("excluded_reason") or "").strip()
                if reason:
                    prior[int(r["show_id"])] = reason
    for sid, reason in excluded.items():
        prior[sid] = reason

    CORPUS_SHOWS.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_SHOWS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["show_id", "chart_rank", "show_name", "show_label",
                    "publisher", "excluded_reason"])
        for sid in sorted(set(show_ids) | set(prior),
                          key=lambda s: int(chart.get(s, ("9999",))[0] or 9999)):
            rank, name, pub = chart.get(sid, ("", "", ""))
            label = f"{name} ({pub})" if name in dupes else name
            w.writerow([sid, rank, name, label, pub, prior.get(sid, "")])
    if prior:
        print(f"  ({len(prior)} show(s) recorded as deliberately excluded)")
    if dupes:
        print(f"note: {len(dupes)} show name(s) are shared by multiple feeds "
              f"({', '.join(sorted(dupes))}) — disambiguated in show_label")
    print(f"wrote {CORPUS_SHOWS} ({len(show_ids)} shows)")


def chart_frame() -> tuple[dict[int, dict], set[int]]:
    """The documented frame: newest chart pull, plus the ids filter.py rejected."""
    charts = sorted((PROJECT_ROOT / "data/output").glob("raw_chart_*.csv"))
    chart: dict[int, dict] = {}
    if charts:
        with open(charts[-1], newline="") as f:
            chart = {int(r["collection_id"]): r for r in csv.DictReader(f)}
    excl: set[int] = set()
    p = PROJECT_ROOT / "data/output/exclusions.csv"
    if p.exists():
        with open(p, newline="") as f:
            excl = {int(r["collection_id"]) for r in csv.DictReader(f)}
    return chart, excl


def audit_shows() -> list[int]:
    """Reconcile the corpus against chart minus exclusions, and return the gap.

    The corpus was historically defined by directory presence, which encodes
    "a previous run happened to transcribe this show" — not any stated criterion.
    Shows that pass every documented rule but were never transcribed therefore
    drop out silently, with no exclusion row and no recorded reason.
    """
    chart, excl = chart_frame()
    corpus, src = load_corpus_shows()
    eligible = set(chart) - excl
    deliberate: dict[int, str] = {}
    if CORPUS_SHOWS.exists():
        with open(CORPUS_SHOWS, newline="") as f:
            for r in csv.DictReader(f):
                reason = (r.get("excluded_reason") or "").strip()
                if reason:
                    deliberate[int(r["show_id"])] = reason

    missing = sorted(eligible - corpus - set(deliberate),
                     key=lambda c: int(chart[c]["rank"]))
    extra = sorted(corpus - eligible)

    print(f"chart shows        : {len(chart)}")
    print(f"excluded (R1-R6)   : {len(excl)}")
    print(f"eligible           : {len(eligible)}")
    print(f"in corpus ({src}): {len(corpus)}")
    if deliberate:
        print(f"deliberately excluded: {len(deliberate)} "
              f"(recorded in {CORPUS_SHOWS.name}, not counted as a gap)")
    print()
    if missing:
        print(f"{len(missing)} eligible show(s) NOT in the corpus — no recorded reason:")
        for c in missing:
            print(f"  rank {chart[c]['rank']:>4}  {c}  {chart[c]['show_name'][:50]}")
    if extra:
        print(f"\n{len(extra)} corpus show(s) not in the current eligible set "
              f"(older chart pull, or hand-added):")
        for c in extra:
            print(f"  {c}")
    if not missing and not extra:
        print("corpus matches the documented frame exactly.")
    return missing


def load_episodes(corpus_show_ids: set[int]) -> pd.DataFrame:
    eps = pd.read_csv(EPISODES, compression="gzip")
    eps = eps[eps.collection_id.isin(corpus_show_ids)].copy()
    eps["pub_date"] = pd.to_datetime(eps.pub_date, utc=True, errors="coerce",
                                     format="mixed")
    # two feeds carry an epoch-zero pub_date; they read as 56-year-old episodes
    bad = eps.pub_date.isna() | (eps.pub_date < "2000-01-01")
    if bad.any():
        print(f"dropping {bad.sum()} episodes with unusable pub_date")
        eps = eps[~bad]
    eps = eps[eps.audio_url.notna() & (eps.audio_url != "")].copy()
    eps["episode_id"] = eps.audio_url.map(
        lambda u: hashlib.md5(u.encode()).hexdigest()[:16])
    eps = eps.drop_duplicates(subset=["collection_id", "episode_id"])
    return eps


def transcribed_ids() -> set[str]:
    done = set()
    if not TRANSCRIPTS.exists():
        return done
    for show_dir in TRANSCRIPTS.iterdir():
        if show_dir.is_dir():
            for f in os.listdir(show_dir):
                if f.endswith(".json"):
                    done.add(f.split("_", 1)[0])
    return done


def interleave_by_show(df: pd.DataFrame) -> pd.DataFrame:
    """Round-robin rows across shows so consecutive rows hit different hosts."""
    df = df.copy()
    df["_slot"] = df.groupby("show_id").cumcount()
    return df.sort_values(["_slot", "show_id"]).drop(columns="_slot")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-k", type=int, default=30,
                    help="episodes per show per quarter for the post-census era")
    ap.add_argument("--min-date", default=None,
                    help="drop every episode published before this date (e.g. "
                         "2018-01-01). Unlike --max-k this is not nesting-safe: "
                         "excluded material may be gone from feeds if you widen "
                         "the range later.")
    ap.add_argument("--census-before", default="2018-01-01",
                    help="take EVERY episode published before this date")
    ap.add_argument("--rtf", type=float, default=107.7,
                    help="measured realtime factor, for the cost estimate")
    ap.add_argument("--budget-gpu-days", type=float, default=None,
                    help="annotate where the ladder crosses this budget")
    ap.add_argument("--out", default=str(OUT_DIR / "sample_manifest_ladder.csv"))
    ap.add_argument("--keep-transcribed", action="store_true",
                    help="Keep already-transcribed episodes that fall past --max-k, "
                         "labelled stratum=supplementary. Adds no work (they are "
                         "done by definition) and never enters the manifest; it "
                         "only stops the corpus from discarding transcripts bought "
                         "under an earlier, wider sampling design.")
    ap.add_argument("--emit-corpus", nargs="?", const=str(OUT_DIR / "corpus_episodes.csv"),
                    default=None,
                    help="Also write the FULL selected corpus (every ladder row, "
                         "already-transcribed ones included) with a `done` column. "
                         "--out is only the work queue: it subtracts what is already "
                         "on disk, so counting progress against it undercounts "
                         "coverage by however much was transcribed before the "
                         "manifest was generated.")
    ap.add_argument("--freeze-shows", action="store_true",
                    help="write the current show frame to corpus_shows.csv and "
                         "exit, without regenerating the manifest")
    ap.add_argument("--audit-shows", action="store_true",
                    help="reconcile the frozen frame against chart minus "
                         "exclusions, report any gap, and exit")
    ap.add_argument("--add-missing-shows", action="store_true",
                    help="add eligible-but-absent shows to corpus_shows.csv and "
                         "exit. Changes the corpus definition — the manifest must "
                         "be regenerated afterwards for it to take effect.")
    args = ap.parse_args()

    if args.audit_shows:
        audit_shows()
        return

    if args.add_missing_shows:
        missing = audit_shows()
        if not missing:
            return
        corpus, _ = load_corpus_shows()
        write_corpus_shows(corpus | set(missing))
        print(f"\nadded {len(missing)} show(s). Regenerate the manifest to apply.")
        return

    if args.freeze_shows:
        # Pin whatever the corpus currently *is*, so freezing never silently
        # changes it. Reconciling against the filter output is a separate,
        # deliberate decision — see --audit-shows.
        ids = {int(p.name) for p in TRANSCRIPTS.iterdir()
               if p.is_dir() and p.name.isdigit()}
        write_corpus_shows(ids)
        return

    show_ids, frame_src = load_corpus_shows()
    print(f"corpus shows: {len(show_ids)} (from {frame_src})")

    eps = load_episodes(show_ids)
    done = transcribed_ids()
    eps["done"] = eps.episode_id.isin(done)
    print(f"episodes in frame: {len(eps):,}   already transcribed: {eps.done.sum():,}")

    if args.min_date:
        # Hard corpus boundary, applied before any ranking so the per-cell draws
        # are computed only over episodes that are actually in scope. Note this is
        # NOT reversible the way --max-k is: raising max-k later only appends rows,
        # but material excluded by date can vanish from RSS feeds in the meantime.
        floor = pd.Timestamp(args.min_date, tz="UTC")
        before = len(eps)
        eps = eps[eps.pub_date >= floor].copy()
        print(f"min-date {args.min_date}: dropped {before - len(eps):,} episodes "
              f"published earlier ({len(eps):,} remain)")

    split = pd.Timestamp(args.census_before, tz="UTC")
    eps["quarter"] = eps.pub_date.dt.tz_localize(None).dt.to_period("Q").astype(str)

    # --- priority assignment
    tail = eps[eps.pub_date < split].copy()
    tail["priority"] = 0

    head = eps[eps.pub_date >= split].copy()
    head["_key"] = head.episode_id.map(stable_rank_key)
    head = head.sort_values("_key")
    head["qrank"] = head.groupby(["collection_id", "quarter"]).cumcount()
    in_ladder = head.qrank < args.max_k
    # Episodes past the k cut that are ALREADY transcribed are real data bought with
    # real GPU time under an earlier, wider design (the 2006/k30 ladder and the H25
    # pilot). Keeping them costs nothing further — `done` is true by definition, so
    # they can never reach `todo` — but they are emphatically NOT part of the
    # probability sample: they are whatever an older design happened to draw. They
    # carry stratum="supplementary" so any per-quarter rate estimate can drop them,
    # while corpus-wide uses (search, training text, qualitative work) can keep them.
    if args.keep_transcribed:
        head = head[in_ladder | head.done].copy()
    else:
        head = head[in_ladder].copy()
    head["priority"] = head.qrank + 1
    head["stratum"] = head.qrank.lt(args.max_k).map({True: "ladder",
                                                     False: "supplementary"})

    tail = tail.copy()
    tail["stratum"] = "census"
    sel = pd.concat([tail, head.drop(columns=["_key", "qrank"])], ignore_index=True)

    # --- manifest shape
    sel = sel.rename(columns={"collection_id": "show_id", "show_name": "show"})
    sel["duration_hr"] = sel.duration_sec / 3600
    sel["draw_order"] = 0

    # The corpus is `sel`; the manifest is only the not-yet-done part of it. Emit
    # the whole thing on request so coverage can be measured against the corpus
    # rather than against the work queue — the two differ by every episode
    # transcribed before this manifest was generated, which is not a small number
    # once a run has been going for a while.
    if args.emit_corpus:
        corpus_path = Path(args.emit_corpus)
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        corpus = sel.sort_values(["priority", "pub_date"]).copy()
        corpus["draw_order"] = range(len(corpus))
        corpus[MANIFEST_COLS + ["stratum", "done"]].to_csv(corpus_path, index=False)
        print(f"corpus (incl. done): {corpus_path}  "
              f"{len(corpus):,} rows, {int(corpus.done.sum()):,} already transcribed "
              f"({corpus.done.mean() * 100:.1f}%)")
        for name, grp in corpus.groupby("stratum", sort=False):
            print(f"    {name:<14} {len(grp):>7,} rows  "
                  f"{int(grp.done.sum()):>7,} done ({grp.done.mean() * 100:4.1f}%)")

    todo = sel[~sel.done].copy()

    ordered = pd.concat(
        [interleave_by_show(todo[todo.priority == p].sort_values("pub_date"))
         for p in sorted(todo.priority.unique())],
        ignore_index=True)
    ordered["draw_order"] = range(len(ordered))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered[MANIFEST_COLS].to_csv(out, index=False)

    # --- report
    print(f"\nmanifest: {out}")
    print(f"rows (not yet transcribed): {len(ordered):,}")
    print(f"total selected incl. done : {len(sel):,}")

    hrs_per_day = args.rtf * 24
    print(f"\n{'band':<16} {'+eps':>8} {'cum eps':>9} {'+hours':>9} {'cum h':>9} {'cum GPU-days':>13}")
    cum_e = cum_h = 0
    marker_done = False
    for p in sorted(ordered.priority.unique()):
        band = ordered[ordered.priority == p]
        cum_e += len(band)
        cum_h += band.duration_sec.sum() / 3600
        label = "P0 census" if p == 0 else f"P{p} k={p}"
        days = cum_h / hrs_per_day
        flag = ""
        if args.budget_gpu_days and not marker_done and days > args.budget_gpu_days:
            flag = "  <-- budget"
            marker_done = True
        if p == 0 or p % 5 == 0 or p == ordered.priority.max() or flag:
            print(f"{label:<16} {len(band):>8,} {cum_e:>9,} "
                  f"{band.duration_sec.sum()/3600:>9,.0f} {cum_h:>9,.0f} {days:>13.1f}{flag}")

    q = ordered.groupby("quarter").size()
    print(f"\nquarters covered: {len(q)}  ({q.index.min()} .. {q.index.max()})")
    print(f"shows covered   : {ordered.show_id.nunique()}")

    runs = (ordered.show_id != ordered.show_id.shift()).cumsum()
    print(f"mean consecutive same-show run: {ordered.groupby(runs).size().mean():.2f} "
          f"(1.0 = fully interleaved; the old manifest was 36.6)")


if __name__ == "__main__":
    main()
