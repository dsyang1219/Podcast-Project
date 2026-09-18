#!/usr/bin/env python3
"""Shard the ladder manifest across N machines without breaking the sample.

    ./split_manifest.py --of 2                  # writes _shard0.csv and _shard1.csv

Row order in the ladder manifest IS priority order: row 0 is the highest-priority
episode, and the interleave already spreads shows evenly within each priority
band. So assigning row i to shard (i % N) gives every shard the same priority
mix and the same show mix as the whole -- stop any shard early and what it has
transcribed is still a valid prefix of the designed sample.

What NOT to do: splitting by show, by date, or by contiguous block. Any of those
makes a shard's contents non-exchangeable with the rest, so a run that stops
early (or a box that dies) leaves a corpus biased toward whatever that shard
happened to hold.

Each shard's draw_order is renumbered 0..n-1 so the downloader still walks it in
order; episode_id is untouched, which is what every downstream join uses.
"""
import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repository root (this script lives one folder down)
DEFAULT_MANIFEST = PROJECT_ROOT / "data/output/sample_out/sample_manifest_ladder.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--of", type=int, required=True, help="number of shards")
    ap.add_argument("--out-prefix", default=None,
                    help="default: <manifest stem>_shard<i>.csv beside the input")
    args = ap.parse_args()

    src = Path(args.manifest)
    rows = list(csv.DictReader(open(src)))
    if not rows:
        raise SystemExit(f"{src} is empty")
    if args.of < 2:
        raise SystemExit("--of must be at least 2")

    cols = list(rows[0].keys())
    prefix = args.out_prefix or str(src.with_suffix("")) + "_shard"

    shards: list[list[dict]] = [[] for _ in range(args.of)]
    for i, r in enumerate(rows):
        shards[i % args.of].append(r)

    print(f"{src.name}: {len(rows):,} rows -> {args.of} shards")
    for i, shard in enumerate(shards):
        for j, r in enumerate(shard):
            r["draw_order"] = j
        out = Path(f"{prefix}{i}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(shard)
        hrs = sum(float(r["duration_sec"] or 0) for r in shard) / 3600
        prio = sorted({int(r["priority"]) for r in shard})
        shows = len({r["show_id"] for r in shard})
        print(f"  {out.name}: {len(shard):,} rows  {hrs:,.0f} audio-hr  "
              f"{shows} shows  priorities {prio[0]}-{prio[-1]}")

    # Overlap is the one thing that would silently waste GPU-days, so assert it.
    seen: set[str] = set()
    for shard in shards:
        ids = {r["episode_id"] for r in shard}
        assert not (ids & seen), "shards overlap"
        seen |= ids
    assert len(seen) == len(rows), "shards lost rows"
    print(f"  verified: {len(seen):,} unique episode_ids, no overlap, none lost")


if __name__ == "__main__":
    main()
