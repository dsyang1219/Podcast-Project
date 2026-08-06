"""Flatten the Stage 1-2 DIME-matching state (produced by
nlp/match_guests_dime.py, see its `run`/`reprocess` which pickle
`{"unique", "appearances", "candidates_by_key", "resolution"}` to
data/output/_dime_guest_stage2_state.pkl) into an appearance-level CSV.

One row per entry in `appearances` (i.e. one row per guest-appearance, NOT
per unique guest -- a guest on 3 episodes yields 3 rows), matching the exact
schema of the pre-existing small-scale-run guests_dime.csv:

    episode_id,collection_id,show_name,guest_name,guest_key,tier,
    n_candidates,resolution_method,dime_cfscore

    .venv/bin/python -m nlp.dime_finalize_appearances
"""
from __future__ import annotations

import csv
import pickle
from pathlib import Path

from pipeline import config as pipeline_config

OUT = pipeline_config.OUTPUT_DIR

FIELDNAMES = [
    "episode_id", "collection_id", "show_name", "guest_name", "guest_key",
    "tier", "n_candidates", "resolution_method", "dime_cfscore",
]


def write_guests_dime_csv(pickle_path: Path, out_path: Path) -> int:
    with pickle_path.open("rb") as f:
        state = pickle.load(f)

    appearances: list[dict] = state["appearances"]
    candidates_by_key: dict[str, list[dict]] = state["candidates_by_key"]
    resolution: dict[str, tuple] = state["resolution"]

    n_rows = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDNAMES)
        for a in appearances:
            key = a["key"]
            method, winner, _reason = resolution[key]
            n_candidates = len(candidates_by_key.get(key, []))
            cfscore = winner["cfscore"] if winner is not None else ""
            w.writerow([
                a["episode_id"], a["collection_id"], a["show_name"],
                a["guest_name"], key, a["tier"], n_candidates, method, cfscore,
            ])
            n_rows += 1

    return n_rows


# --------------------------------------------------------------- self-test ----

def _self_test() -> None:
    """Tiny synthetic pickle-shaped dict covering: unmatched, single_recipient
    (has a cfscore), and description (has a cfscore) -- verifies the row
    construction logic without touching real data."""
    fake_state = {
        "unique": {
            "house|niko|niko": {"display_name": "Niko House", "last": "house",
                                 "first_word": "niko", "first_full": "niko",
                                 "n_appearances": 1},
            "cruz|ted|ted": {"display_name": "Ted Cruz", "last": "cruz",
                              "first_word": "ted", "first_full": "ted",
                              "n_appearances": 2},
            "ben-ami|jeremy|jeremy": {"display_name": "Jeremy Ben-Ami", "last": "ben-ami",
                                       "first_word": "jeremy", "first_full": "jeremy",
                                       "n_appearances": 1},
        },
        "appearances": [
            {"episode_id": "ep1", "collection_id": "c1", "show_name": "Show A",
             "episode_title": "t1", "tier": "tier3_description", "guest_name": "Niko House",
             "descriptor": "", "topic": "", "context_text": "", "key": "house|niko|niko"},
            {"episode_id": "ep2", "collection_id": "c1", "show_name": "Show A",
             "episode_title": "t2", "tier": "tier1_rss", "guest_name": "Sen. Ted Cruz",
             "descriptor": "", "topic": "", "context_text": "", "key": "cruz|ted|ted"},
            {"episode_id": "ep3", "collection_id": "c2", "show_name": "Show B",
             "episode_title": "t3", "tier": "tier1_rss", "guest_name": "Ted Cruz",
             "descriptor": "", "topic": "", "context_text": "", "key": "cruz|ted|ted"},
            {"episode_id": "ep4", "collection_id": "c3", "show_name": "Show C",
             "episode_title": "t4", "tier": "tier2_title", "guest_name": "Jeremy Ben-Ami",
             "descriptor": "", "topic": "", "context_text": "", "key": "ben-ami|jeremy|jeremy"},
        ],
        "candidates_by_key": {
            "house|niko|niko": [],
            "cruz|ted|ted": [
                {"bonica_cid": "cand:1", "source": "recipient", "matched_name": "CRUZ, TED",
                 "cfscore": "0.92", "party": "200", "state": "TX", "seat": "federal:senate",
                 "occupation": "", "employer": ""},
            ],
            "ben-ami|jeremy|jeremy": [
                {"bonica_cid": "c1", "source": "contributor", "matched_name": "BEN-AMI, JEREMY",
                 "cfscore": "-0.85", "party": "", "state": "", "seat": "",
                 "occupation": "president", "employer": "j street"},
                {"bonica_cid": "c2", "source": "contributor", "matched_name": "BEN-AMI, JEREMY",
                 "cfscore": "-0.40", "party": "", "state": "", "seat": "",
                 "occupation": "retired", "employer": ""},
                {"bonica_cid": "c3", "source": "contributor", "matched_name": "BEN-AMI, JEREMY",
                 "cfscore": "0.10", "party": "", "state": "", "seat": "",
                 "occupation": "", "employer": ""},
                {"bonica_cid": "c4", "source": "contributor", "matched_name": "BEN-AMI, JEREMY",
                 "cfscore": "0.55", "party": "", "state": "", "seat": "",
                 "occupation": "", "employer": ""},
            ],
        },
        "resolution": {
            "house|niko|niko": ("unmatched", None, None),
            "cruz|ted|ted": ("single_recipient", {
                "bonica_cid": "cand:1", "source": "recipient", "matched_name": "CRUZ, TED",
                "cfscore": "0.92", "party": "200", "state": "TX", "seat": "federal:senate",
                "occupation": "", "employer": "",
            }, None),
            "ben-ami|jeremy|jeremy": ("description", {
                "bonica_cid": "c1", "source": "contributor", "matched_name": "BEN-AMI, JEREMY",
                "cfscore": "-0.85", "party": "", "state": "", "seat": "",
                "occupation": "president", "employer": "j street",
            }, None),
        },
    }

    tmp_dir = Path("/tmp/claude-1008/-home-dsyang/e560606f-1012-4b25-9b63-f3fb740b0318/scratchpad")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fake_pickle_path = tmp_dir / "_fake_dime_state.pkl"
    fake_out_path = tmp_dir / "_fake_guests_dime.csv"

    with fake_pickle_path.open("wb") as f:
        pickle.dump(fake_state, f)

    n = write_guests_dime_csv(fake_pickle_path, fake_out_path)
    assert n == 4, f"expected 4 rows, got {n}"

    with fake_out_path.open(newline="") as f:
        rows = list(csv.reader(f))

    expected_header = FIELDNAMES
    assert rows[0] == expected_header, f"header mismatch: {rows[0]}"

    expected_data = [
        ["ep1", "c1", "Show A", "Niko House", "house|niko|niko", "tier3_description", "0", "unmatched", ""],
        ["ep2", "c1", "Show A", "Sen. Ted Cruz", "cruz|ted|ted", "tier1_rss", "1", "single_recipient", "0.92"],
        ["ep3", "c2", "Show B", "Ted Cruz", "cruz|ted|ted", "tier1_rss", "1", "single_recipient", "0.92"],
        ["ep4", "c3", "Show C", "Jeremy Ben-Ami", "ben-ami|jeremy|jeremy", "tier2_title", "4", "description", "-0.85"],
    ]
    assert rows[1:] == expected_data, f"data rows mismatch:\ngot:      {rows[1:]}\nexpected: {expected_data}"

    print("[self-test] PASSED -- all 4 synthetic rows matched expected output exactly")


if __name__ == "__main__":
    _self_test()

    real_pickle = Path("data/output/_dime_guest_stage2_state.pkl")
    real_out = Path("data/output/guests_dime.csv")
    n_rows = write_guests_dime_csv(real_pickle, real_out)
    print(f"\n[done] wrote {n_rows} data rows -> {real_out}")

    with real_out.open(newline="") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(line.rstrip("\n"))
