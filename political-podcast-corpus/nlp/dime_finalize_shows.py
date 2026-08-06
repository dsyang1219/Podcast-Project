"""Finalize stage: aggregate the DIME guest-matching pickle
(data/output/_dime_guest_stage2_state.pkl, produced by nlp/match_guests_dime.py)
into a show-level CSV of guest ideology.

This is a standalone consumer of that pickle -- it does NOT re-run any of
the matching/disambiguation logic, it only aggregates the already-resolved
`resolution` dict up to one row per (collection_id, show_name).

    .venv/bin/python -m nlp.dime_finalize_shows

Schema (exact column order): collection_id, show_name, n_dime_guests,
n_appearances, mean_cfscore, low_confidence, weighted_mean_cfscore.

Logic
-----
1. Only appearances whose guest resolved to an actual DIME identity with a
   cfscore count (`resolution[key] == (method, winner, reason)` with
   `winner is not None`, i.e. method in {single_recipient, single_donor,
   description}). Guests resolved as unmatched / dropped_too_common /
   ambiguous_residual (winner is None) contribute nothing.
2. Group resolved appearances by (collection_id, show_name), then within a
   show's group, further group by distinct guest_key -- a guest who
   appeared N times on one show counts as 1 guest with N appearances, not
   N guests.
3. Per show:
   - n_dime_guests = number of distinct resolved guest_keys on that show
   - n_appearances = sum of each distinct guest's appearance count on that
     show (can exceed n_dime_guests)
   - mean_cfscore = simple mean of the DISTINCT guests' cfscores (equal
     weight per guest, regardless of appearance count)
   - weighted_mean_cfscore = mean of the same cfscores weighted by each
     guest's appearance count on that show
   - low_confidence = bool(n_dime_guests < 3)
4. Rows sorted ascending by collection_id (as a string).
"""
from __future__ import annotations

import csv
import pickle
from collections import defaultdict
from pathlib import Path


def _aggregate(unique: dict, appearances: list, resolution: dict) -> list[dict]:
    """Core aggregation logic, factored out so the self-test can call it
    directly on a small synthetic pickle-shaped dict without touching disk."""
    # (collection_id, show_name) -> guest_key -> appearance count on this show
    show_guest_counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for a in appearances:
        key = a["key"]
        res = resolution.get(key)
        if res is None:
            continue
        _method, winner, _reason = res
        if winner is None:
            continue  # unmatched / dropped_too_common / ambiguous_residual -- no cfscore
        show_id = (a["collection_id"], a["show_name"])
        show_guest_counts[show_id][key] += 1

    rows = []
    for (collection_id, show_name), guest_counts in show_guest_counts.items():
        cfscores = {}
        for key in guest_counts:
            _method, winner, _reason = resolution[key]
            cfscores[key] = float(winner["cfscore"])

        n_dime_guests = len(guest_counts)
        n_appearances = sum(guest_counts.values())

        mean_cfscore = sum(cfscores[k] for k in guest_counts) / n_dime_guests
        weighted_sum = sum(cfscores[k] * guest_counts[k] for k in guest_counts)
        weighted_mean_cfscore = weighted_sum / n_appearances

        rows.append({
            "collection_id": collection_id,
            "show_name": show_name,
            "n_dime_guests": n_dime_guests,
            "n_appearances": n_appearances,
            "mean_cfscore": mean_cfscore,
            "low_confidence": n_dime_guests < 3,
            "weighted_mean_cfscore": weighted_mean_cfscore,
        })

    rows.sort(key=lambda r: str(r["collection_id"]))
    return rows


def write_show_guest_ideology_csv(pickle_path: Path, out_path: Path) -> int:
    with pickle_path.open("rb") as f:
        state = pickle.load(f)

    unique = state["unique"]
    appearances = state["appearances"]
    resolution = state["resolution"]

    rows = _aggregate(unique, appearances, resolution)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["collection_id", "show_name", "n_dime_guests", "n_appearances",
                    "mean_cfscore", "low_confidence", "weighted_mean_cfscore"])
        for r in rows:
            w.writerow([r["collection_id"], r["show_name"], r["n_dime_guests"],
                        r["n_appearances"], r["mean_cfscore"], r["low_confidence"],
                        r["weighted_mean_cfscore"]])

    return len(rows)


# ------------------------------------------------------------- self-test ----

def _run_self_test() -> None:
    """Tiny synthetic in-memory fake pickle dict, exercising:
      - a show with a single guest appearing twice (n_appearances != n_dime_guests,
        mean == weighted_mean necessarily)
      - a show with 3 distinct guests with DIFFERENT appearance counts and
        DIFFERENT cfscores, so mean_cfscore and weighted_mean_cfscore
        provably diverge (catches a copy-pasted/unweighted bug)
      - guests that should be EXCLUDED entirely (unmatched / dropped /
        still-ambiguous -- winner is None)
    """
    appearances = [
        # Show A: guest "solo|guest|solo guest" appears twice.
        {"collection_id": "111", "show_name": "Show A", "key": "solo|guest|solo guest"},
        {"collection_id": "111", "show_name": "Show A", "key": "solo|guest|solo guest"},

        # Show B: three distinct guests, appearance counts 1, 2, 3 respectively.
        {"collection_id": "222", "show_name": "Show B", "key": "one|guest|one guest"},
        {"collection_id": "222", "show_name": "Show B", "key": "two|guest|two guest"},
        {"collection_id": "222", "show_name": "Show B", "key": "two|guest|two guest"},
        {"collection_id": "222", "show_name": "Show B", "key": "three|guest|three guest"},
        {"collection_id": "222", "show_name": "Show B", "key": "three|guest|three guest"},
        {"collection_id": "222", "show_name": "Show B", "key": "three|guest|three guest"},

        # Show B also has an unmatched guest -- must be excluded entirely,
        # and must not create a phantom show or affect Show B's counts.
        {"collection_id": "222", "show_name": "Show B", "key": "nobody|guest|nobody guest"},

        # Show C: only an unmatched guest -- must produce NO row at all.
        {"collection_id": "333", "show_name": "Show C", "key": "ghost|guest|ghost guest"},
    ]

    resolution = {
        "solo|guest|solo guest": ("single_recipient", {"cfscore": "-1.066"}, None),
        "one|guest|one guest": ("single_recipient", {"cfscore": "1.0"}, None),
        "two|guest|two guest": ("single_recipient", {"cfscore": "2.0"}, None),
        "three|guest|three guest": ("single_recipient", {"cfscore": "-3.0"}, None),
        "nobody|guest|nobody guest": ("unmatched", None, None),
        "ghost|guest|ghost guest": ("ambiguous_residual", None, None),
    }
    unique = {}  # not used by _aggregate

    rows = _aggregate(unique, appearances, resolution)
    by_show = {r["show_name"]: r for r in rows}

    assert len(rows) == 2, f"expected exactly 2 rows (Show C excluded), got {len(rows)}: {rows}"
    assert "Show C" not in by_show, "Show C had only an excluded guest -- must not appear"

    a = by_show["Show A"]
    assert a["n_dime_guests"] == 1
    assert a["n_appearances"] == 2
    assert a["mean_cfscore"] == -1.066
    assert a["weighted_mean_cfscore"] == -1.066, "single-guest show: mean must equal weighted_mean"
    assert a["low_confidence"] is True

    b = by_show["Show B"]
    assert b["n_dime_guests"] == 3, "the unmatched 4th guest must not be counted"
    assert b["n_appearances"] == 6  # 1 + 2 + 3, excluding the unmatched appearance
    # hand computation:
    # unweighted mean = (1.0 + 2.0 + (-3.0)) / 3 = 0.0 / 3 = 0.0
    expected_mean = (1.0 + 2.0 + (-3.0)) / 3
    assert abs(b["mean_cfscore"] - expected_mean) < 1e-12, (b["mean_cfscore"], expected_mean)
    assert expected_mean == 0.0
    # weighted mean = (1.0*1 + 2.0*2 + (-3.0)*3) / (1+2+3) = (1 + 4 - 9) / 6 = -4/6 = -0.6666...
    expected_weighted = (1.0 * 1 + 2.0 * 2 + (-3.0) * 3) / 6
    assert abs(b["weighted_mean_cfscore"] - expected_weighted) < 1e-12, (b["weighted_mean_cfscore"], expected_weighted)
    assert abs(expected_weighted - (-2 / 3)) < 1e-9
    # crucially: mean_cfscore != weighted_mean_cfscore here, proving the two
    # are computed by genuinely different formulas, not one copy-pasted as the other.
    assert abs(b["mean_cfscore"] - b["weighted_mean_cfscore"]) > 0.5, \
        "mean and weighted mean should clearly diverge for Show B"
    assert b["low_confidence"] is False  # n_dime_guests == 3, not < 3

    print("[self-test] PASS")
    print(f"  Show A: n_dime_guests={a['n_dime_guests']} n_appearances={a['n_appearances']} "
          f"mean={a['mean_cfscore']} weighted_mean={a['weighted_mean_cfscore']} "
          f"low_confidence={a['low_confidence']}")
    print(f"  Show B: n_dime_guests={b['n_dime_guests']} n_appearances={b['n_appearances']} "
          f"mean={b['mean_cfscore']} (hand-calc {expected_mean}) "
          f"weighted_mean={b['weighted_mean_cfscore']} (hand-calc {expected_weighted}) "
          f"low_confidence={b['low_confidence']}")
    print(f"  Show C correctly produced no row (only an excluded/unmatched guest).")


if __name__ == "__main__":
    _run_self_test()

    pickle_path = Path("data/output/_dime_guest_stage2_state.pkl")
    out_path = Path("data/output/show_guest_ideology.csv")

    n = write_show_guest_ideology_csv(pickle_path, out_path)
    print(f"\n[done] wrote {n} rows -> {out_path}")

    with out_path.open() as f:
        lines = [next(f, "") for _ in range(10)]
    print(f"\n[first 10 lines of {out_path}]")
    for line in lines:
        print(line, end="" if line.endswith("\n") else "\n")
