"""Attach DIME CFscores to extracted guests (Stage 1-2 of the 3-stage
disambiguation cascade -- see task spec). Reuses the surname/first-word
streaming-match approach already proven on hosts (see the prior
match_dime.py: recipients.csv.gz for candidates, contributors.csv.gz for
donors, both keyed on bonica.cid).

    .venv/bin/python -m nlp.match_guests_dime run

Stage 1 -- normalize guest names (nlp.guest_common.normalize_name), stream
both DIME bulk files, bucket by n_dime_candidates (distinct bonica.cid):
0 (unmatched, not an error -- most pundits/journalists aren't donors),
1 (RESOLVED), 2+ (ambiguous -> Stage 2).

Stage 2 -- for 2+-candidate guests, disambiguate using episode
title+description+descriptor/topic text ALREADY IN HAND: match each
candidate's DIME attributes (state, party, seat for recipients;
occupation/employer for contributors) against that text. Unique match ->
RESOLVED (resolution=description). Otherwise stays ambiguous -> Stage 3
residual, written out for IRB-approved web search (NOT done by this script).

IRB scope note (2026-07-29): approval was extended by the researcher to
cover full-corpus-scale search volume, superseding the original "small-scale"
framing this script started with. Every search is still logged (guest,
context, query, result, decision) for the audit trail regardless of volume --
that requirement didn't change, only the volume ceiling did. The
DROP_THRESHOLD gate below is a separate, non-IRB concern: past a certain
same-name donor-pool size, a single search genuinely can't disambiguate a
specific individual without more identifying info than is available, so
those are dropped as documented exclusions rather than guessed, independent
of how much search budget exists.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from pipeline import config as pipeline_config
from .guest_common import load_episodes, normalize_name, nickname_normalize, strip_html_urls

OUT = pipeline_config.OUTPUT_DIR
GUESTS_RAW = OUT / "guests_raw.jsonl"
DIME_DIR = Path("/tmp/claude-1008/-home-dsyang/ce36ffa2-ffee-48aa-820c-664977c7c06e/scratchpad/dime")
RECIP_GZ = DIME_DIR / "dime_recipients_1979_2024.csv.gz"
CONTRIB_GZ = DIME_DIR / "dime_contributors_1979_2024.csv.gz"

STAGE3_RESIDUAL_PATH = OUT / "dime_stage3_residual.csv"

PARTY_MAP = {"100": "democrat", "200": "republican", "328": "independent"}
STATE_NAMES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia",
    "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa",
    "KS": "kansas", "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma",
    "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina",
    "SD": "south dakota", "TN": "tennessee", "TX": "texas", "UT": "utah", "VT": "vermont",
    "VA": "virginia", "WA": "washington", "WV": "west virginia", "WI": "wisconsin",
    "WY": "wyoming", "DC": "district of columbia",
}
SEAT_TOKENS = {
    "senate": ["senator", "senate"],
    "house": ["representative", "congressman", "congresswoman", "congress", "house"],
    "president": ["president", "presidential"],
    "governor": ["governor"],
}
STOPWORDS = {"self", "employed", "none", "retired", "not", "employed.", "n/a", "inc", "llc", "the", "and"}


def first_word_norm(raw: str) -> str:
    toks = normalize_name(raw).split()
    return nickname_normalize(toks[0]) if toks else ""


def name_tokens_norm(raw: str) -> set[str]:
    """All given-name tokens, nickname-normalized -- DIME sometimes stores a
    full given name with the common/nickname name buried in it (e.g. Ted
    Cruz's DIME fname is "rafael edward ted"), so matching must check all
    tokens, not just the first."""
    return {nickname_normalize(t) for t in normalize_name(raw).split()}


# ------------------------------------------------------------- guest load ----

def load_unique_guests() -> tuple[dict[str, dict], list[dict]]:
    """Returns (unique_guests: norm_key -> record, appearances: list of rows).
    A guest's identity key is its normalized full name (last-name block +
    nickname-normalized first word), matching the granularity DIME matching
    needs (one lookup per distinct name, not per appearance)."""
    eps = load_episodes()
    ep_by_id = eps.set_index("episode_id")[["episode_description"]].to_dict("index")

    appearances = []
    unique: dict[str, dict] = {}

    with GUESTS_RAW.open() as f:
        for line in f:
            row = json.loads(line)
            eid = row["episode_id"]
            desc_raw = ep_by_id.get(eid, {}).get("episode_description", "")
            desc = strip_html_urls(desc_raw) if isinstance(desc_raw, str) else ""
            for g in row.get("guests", []):
                raw_name = g.get("guest_canonical") or g["name"]
                norm = normalize_name(raw_name)
                tokens = norm.split()
                if len(tokens) < 2:
                    continue  # single-token names can't be surname-matched reliably
                last = tokens[-1]
                first_word = nickname_normalize(tokens[0])
                first_full = " ".join(tokens[:-1])
                key = f"{last}|{first_word}|{first_full}"

                appearance = {
                    "episode_id": eid, "collection_id": row["collection_id"],
                    "show_name": row["show_name"], "episode_title": row["episode_title"],
                    "tier": row["tier"], "guest_name": raw_name,
                    "descriptor": g.get("descriptor", ""), "topic": g.get("topic", ""),
                    "context_text": " ".join([row["episode_title"], desc,
                                               g.get("descriptor", ""), g.get("topic", "")]).lower(),
                }
                appearances.append({**appearance, "key": key})

                if key not in unique:
                    unique[key] = {"display_name": raw_name, "last": last,
                                   "first_word": first_word, "first_full": first_full,
                                   "n_appearances": 0}
                unique[key]["n_appearances"] += 1

    return unique, appearances


# --------------------------------------------------------- DIME streaming ----

def build_target_index(unique: dict[str, dict]) -> dict[str, list[str]]:
    idx = defaultdict(list)
    for key, u in unique.items():
        idx[u["last"]].append(key)
    return idx


def stream_recipients(target_surnames: set[str]) -> dict[str, dict[str, dict]]:
    """surname -> {bonica_cid -> most-recent-cycle recipient row dict}."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    with gzip.open(RECIP_GZ, "rt", newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {h.strip('"'): i for i, h in enumerate(header)}
        for row in r:
            lname_n = normalize_name(row[idx["lname"]])
            if lname_n not in target_surnames:
                continue
            fname_toks = name_tokens_norm(row[idx["ffname"]])  # full first name incl. middle/common names, e.g. "rafael edward ted"
            # bonica.cid is BLANK for ~40% of recipient rows (verified) -- using it
            # directly as the dedup key would collide unrelated candidates (e.g. Rep.
            # Wesley Hunt's rows all have blank cid and were getting overwritten by
            # whatever other blank-cid candidate had the highest cycle number).
            # Fall back to FEC.ID, then Cand.ID (always populated per the same check).
            raw_cid, fec_id, cand_id = row[idx["bonica.cid"]].strip(), row[idx["FEC.ID"]].strip(), row[idx["Cand.ID"]].strip()
            cid = raw_cid or (f"fec:{fec_id}" if fec_id else None) or f"cand:{cand_id}"
            cycle = row[idx["cycle"]]
            cur = out[lname_n].get(cid)
            try:
                is_newer = cur is None or int(cycle) > int(cur["cycle"])
            except ValueError:
                is_newer = cur is None
            if is_newer:
                out[lname_n][cid] = {
                    "bonica_cid": cid, "name": row[idx["name"]], "fname_tokens": fname_toks,
                    "cycle": cycle, "party": row[idx["party"]], "state": row[idx["state"]],
                    "seat": row[idx["seat"]], "recipient_cfscore": row[idx["recipient.cfscore"]],
                }
    return out


def stream_contributors(target_surnames: set[str]) -> dict[str, dict[str, dict]]:
    """surname -> {bonica_cid -> contributor row dict}. Individual donors only."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    n_seen = 0
    with gzip.open(CONTRIB_GZ, "rt", newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {h.strip('"'): i for i, h in enumerate(header)}
        for row in r:
            n_seen += 1
            if row[idx["contributor.type"]] != "I":
                continue
            raw_name = row[idx["most.recent.contributor.name"]]
            if "," not in raw_name:
                continue
            lname_raw, _, rest = raw_name.partition(",")
            lname_n = normalize_name(lname_raw)
            if lname_n not in target_surnames:
                continue
            rest = rest.strip()
            fname_toks = name_tokens_norm(rest)
            cid = row[idx["bonica.cid"]]
            out[lname_n][cid] = {
                "bonica_cid": cid, "name": raw_name, "fname_tokens": fname_toks,
                "occupation": row[idx["most.recent.contributor.occupation"]],
                "employer": row[idx["most.recent.contributor.employer"]],
                "contributor_cfscore": row[idx["contributor.cfscore"]],
                "num_distinct": row[idx["num.distinct"]],
            }
            if n_seen % 10_000_000 == 0:
                print(f"  ...scanned {n_seen:,} contributor rows", file=sys.stderr)
    print(f"  contributors file: {n_seen:,} total rows scanned", file=sys.stderr)
    return out


# ------------------------------------------------------------- candidates ----

def merge_candidates(unique: dict[str, dict], recip_by_surname, contrib_by_surname) -> dict[str, list[dict]]:
    """key -> list of candidate dicts, one per distinct bonica.cid, recipient
    data preferred over contributor data when a cid appears in both (a
    candidate's officeseeking cfscore is the more relevant ideology estimate
    than their personal-donor cfscore)."""
    result: dict[str, list[dict]] = {}
    for key, u in unique.items():
        last, first_word = u["last"], u["first_word"]
        by_cid: dict[str, dict] = {}
        for cid, rec in recip_by_surname.get(last, {}).items():
            if first_word not in rec["fname_tokens"]:
                continue
            by_cid[cid] = {
                "bonica_cid": cid, "source": "recipient", "matched_name": rec["name"],
                "cfscore": rec["recipient_cfscore"], "party": rec["party"], "state": rec["state"],
                "seat": rec["seat"], "occupation": "", "employer": "",
            }
        for cid, rec in contrib_by_surname.get(last, {}).items():
            if first_word not in rec["fname_tokens"]:
                continue
            if cid in by_cid:
                continue  # recipient-side data already preferred for this person
            by_cid[cid] = {
                "bonica_cid": cid, "source": "contributor", "matched_name": rec["name"],
                "cfscore": rec["contributor_cfscore"], "party": "", "state": "",
                "seat": "", "occupation": rec["occupation"], "employer": rec["employer"],
            }
        result[key] = list(by_cid.values())
    return result


# --------------------------------------------------------- stage 2 (text) ----

def candidate_signal_tokens(c: dict) -> list[str]:
    toks = []
    if c["state"] and c["state"].strip():
        ab = c["state"].strip().upper()
        toks.append(ab.lower())
        if ab in STATE_NAMES:
            toks.append(STATE_NAMES[ab])
    if c["party"] and c["party"].strip() in PARTY_MAP:
        toks.append(PARTY_MAP[c["party"].strip()])
    if c["seat"]:
        seat_l = c["seat"].lower()
        for k, extra in SEAT_TOKENS.items():
            if k in seat_l:
                toks.extend(extra)
    for field in (c["occupation"], c["employer"]):
        if field:
            for w in re.split(r"[^a-z]+", field.lower()):
                if len(w) >= 4 and w not in STOPWORDS:
                    toks.append(w)
    return list(dict.fromkeys(toks))  # dedupe, preserve order


def disambiguate_stage2(candidates: list[dict], context_texts: list[str]) -> dict | None:
    """Returns the uniquely-resolved candidate dict, or None if still ambiguous."""
    combined_text = " ".join(context_texts)
    scores = []
    for c in candidates:
        toks = candidate_signal_tokens(c)
        hits = sum(1 for t in toks if t and t in combined_text)
        scores.append((hits, c))
    scores.sort(key=lambda x: -x[0])
    if not scores:
        return None
    top_hits, top_c = scores[0]
    if top_hits == 0:
        return None
    runner_hits = scores[1][0] if len(scores) > 1 else 0
    if top_hits > runner_hits:
        return top_c
    return None


# ------------------------------------------------------------------ main ----

DROP_THRESHOLD = 20  # donor-only same-name pools bigger than this aren't search-tractable
                      # (a single search can't disambiguate a person among dozens of
                      # coincidental namesakes with no other identifying info) --
                      # dropped as documented exclusions, not sent to stage 3.


def bucket_and_resolve(unique: dict, appearances: list, candidates_by_key: dict):
    """Recipient (candidate/officeholder) matches are preferred over contributor
    (donor) matches for the same key: a guest interviewed on a political podcast
    is far more likely to BE a named officeholder/candidate than one of several
    coincidental same-name individual donors. This also keeps donor-file noise
    (44M individual records over 45 years inflate same-name pools for ANY common
    name) from blowing up the disambiguation-needed bucket."""
    appearances_by_key = defaultdict(list)
    for a in appearances:
        appearances_by_key[a["key"]].append(a)

    resolution: dict[str, tuple] = {}
    bucket0, bucket1, bucket_ambig = [], [], []

    for key, cands in candidates_by_key.items():
        recip = [c for c in cands if c["source"] == "recipient"]
        donor = [c for c in cands if c["source"] == "contributor"]

        if not recip and not donor:
            bucket0.append(key)
            resolution[key] = ("unmatched", None, None)
        elif len(recip) == 1:
            bucket1.append(key)
            resolution[key] = ("single_recipient", recip[0], None)
        elif not recip and len(donor) == 1:
            bucket1.append(key)
            resolution[key] = ("single_donor", donor[0], None)
        else:
            bucket_ambig.append(key)

    print("\n=== STAGE 1: candidate-bucket sizes (unique guest names) ===")
    print(f"  0 candidates (unmatched, not in DIME):  {len(bucket0)}")
    print(f"  1 candidate  (RESOLVED, no disambig):   {len(bucket1)}")
    print(f"  2+ candidates (ambiguous -> stage 2):   {len(bucket_ambig)}")
    print(f"  total unique guest names:                {len(unique)}")

    stage2_resolved = 0
    residual = []
    dropped_too_common = []
    for key in bucket_ambig:
        cands = candidates_by_key[key]
        recip = [c for c in cands if c["source"] == "recipient"]
        donor = [c for c in cands if c["source"] == "contributor"]
        # recip is 0 or >=2 here (the ==1 case was resolved above)
        pool = recip if len(recip) >= 2 else donor
        context_texts = [a["context_text"] for a in appearances_by_key[key]]
        winner = disambiguate_stage2(pool, context_texts)
        if winner is not None:
            resolution[key] = ("description", winner, None)
            stage2_resolved += 1
        elif len(recip) == 0 and len(donor) > DROP_THRESHOLD:
            reason = f"name too common in DIME donor pool (n={len(donor)} same-name individuals, no recipient/candidate match) -- not search-tractable"
            resolution[key] = ("dropped_too_common", None, reason)
            dropped_too_common.append(key)
        else:
            resolution[key] = ("ambiguous_residual", None, None)
            residual.append(key)

    print("\n=== STAGE 2: description-based disambiguation (no search) ===")
    print(f"  input (post-stage-1 ambiguous):        {len(bucket_ambig)}")
    print(f"  resolved by description:                {stage2_resolved}")
    print(f"  dropped (too-common name, not search-tractable, n>{DROP_THRESHOLD} same-name donors): {len(dropped_too_common)}")
    print(f"  POST-DESCRIPTION RESIDUAL (-> stage 3 candidate set): {len(residual)}")

    return bucket0, bucket1, bucket_ambig, residual, dropped_too_common, resolution, appearances_by_key


def write_stage3_residual(residual, candidates_by_key, unique, appearances_by_key):
    """n_candidates reports the POOL ACTUALLY USED for disambiguation (recipient
    candidates if 2+, else donor candidates) -- not the raw merged total, which
    can include large irrelevant same-name donor noise even when the real
    disambiguation is among a handful of recipient/officeholder entries."""
    with STAGE3_RESIDUAL_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["guest_key", "display_name", "n_candidates", "n_merged_total",
                    "n_appearances", "candidate_names", "candidate_cfscores",
                    "candidate_sources", "example_show", "example_title", "example_descriptor"])
        for key in residual:
            cands = candidates_by_key[key]
            recip = [c for c in cands if c["source"] == "recipient"]
            donor = [c for c in cands if c["source"] == "contributor"]
            pool = recip if len(recip) >= 2 else donor
            ex = appearances_by_key[key][0]
            w.writerow([
                key, unique[key]["display_name"], len(pool), len(cands), unique[key]["n_appearances"],
                "; ".join(c["matched_name"] for c in pool),
                "; ".join(str(c["cfscore"]) for c in pool),
                "; ".join(c["source"] for c in pool),
                ex["show_name"], ex["episode_title"], ex["descriptor"],
            ])
    print(f"\n[stage3 residual written] -> {STAGE3_RESIDUAL_PATH.name} ({len(residual)} rows -- review before searching)")


def write_dropped_too_common(dropped_keys, resolution, candidates_by_key, unique):
    path = OUT / "dime_dropped.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["guest_key", "display_name", "n_candidates", "reason"])
        for key in dropped_keys:
            _, _, reason = resolution[key]
            w.writerow([key, unique[key]["display_name"], len(candidates_by_key[key]), reason])
    print(f"[dropped (too-common) written] -> {path.name} ({len(dropped_keys)} rows)")


def reprocess() -> None:
    """Re-run bucketing/stage-2 from the cached pickle (from a prior `run`) --
    skips re-streaming the 44M-row DIME files, for fast iteration on the
    disambiguation logic itself."""
    import pickle
    state_path = OUT / "_dime_guest_stage2_state.pkl"
    with state_path.open("rb") as f:
        state = pickle.load(f)
    unique, appearances, candidates_by_key = state["unique"], state["appearances"], state["candidates_by_key"]

    bucket0, bucket1, bucket_ambig, residual, dropped_too_common, resolution, appearances_by_key = \
        bucket_and_resolve(unique, appearances, candidates_by_key)

    write_stage3_residual(residual, candidates_by_key, unique, appearances_by_key)
    write_dropped_too_common(dropped_too_common, resolution, candidates_by_key, unique)

    print(f"\n[stage 3] {len(residual)} names ready for search "
          f"(full-corpus search volume approved -- see module docstring).")

    state["resolution"] = resolution
    with state_path.open("wb") as f:
        pickle.dump(state, f)
    print(f"[state updated] -> {state_path.name}")


def run() -> None:
    print("[load] flattening guests_raw.jsonl to unique guest names...", file=sys.stderr)
    unique, appearances = load_unique_guests()
    print(f"[load] {len(appearances)} guest-appearances, {len(unique)} unique guest names "
          f"(single-token names dropped from DIME matching)", file=sys.stderr)

    target_surnames = {u["last"] for u in unique.values()}
    print(f"[stage1] {len(target_surnames)} distinct target surnames", file=sys.stderr)

    print("[stage1] streaming recipients.csv.gz (candidates, small file)...", file=sys.stderr)
    recip_by_surname = stream_recipients(target_surnames)

    print("[stage1] streaming contributors.csv.gz (donors, ~44M rows -- this takes a few minutes)...", file=sys.stderr)
    contrib_by_surname = stream_contributors(target_surnames)

    candidates_by_key = merge_candidates(unique, recip_by_surname, contrib_by_surname)

    bucket0, bucket1, bucket_ambig, residual, dropped_too_common, resolution, appearances_by_key = \
        bucket_and_resolve(unique, appearances, candidates_by_key)

    write_stage3_residual(residual, candidates_by_key, unique, appearances_by_key)
    write_dropped_too_common(dropped_too_common, resolution, candidates_by_key, unique)

    print(f"\n[stage 3] {len(residual)} names ready for search "
          f"(full-corpus search volume approved -- see module docstring).")

    # persist intermediate state for the (separate) stage-3/finalize step
    import pickle
    state_path = OUT / "_dime_guest_stage2_state.pkl"
    with state_path.open("wb") as f:
        pickle.dump({"unique": unique, "appearances": appearances,
                     "candidates_by_key": candidates_by_key, "resolution": resolution}, f)
    print(f"[state saved] -> {state_path.name} (for stage-3/finalize step)")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "reprocess"):
        print(__doc__)
        raise SystemExit(1)
    if sys.argv[1] == "run":
        run()
    else:
        reprocess()


if __name__ == "__main__":
    main()
