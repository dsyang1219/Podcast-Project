"""Tier 1 of the guest-extraction cascade (DeMets & Spiro Sec 3.2): free,
per-show regex patterns for shows that put guest names in a rigid title
format. AUTO-DISCOVERS candidate templates and their per-show coverage, but
never applies them -- writes show_patterns.json with approved: false for
every show. A human reviews and flips approved:true before
nlp/extract_guests.py will use a show's pattern; auto-approving a wrong
pattern would poison the whole downstream network, so this script has no
"approve automatically above threshold X" path at all.

    .venv/bin/python -m nlp.discover_patterns [--min-coverage 0.15] [--min-matches 5]
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

from pipeline import config as pipeline_config
from .guest_common import load_episodes

# A capture is only trusted as a name if it looks like 1-4 Title-Case tokens
# (allows hyphens/apostrophes: "Jean-Pierre", "O'Brien") -- this alone
# rejects ALL-CAPS junk ("TRUMAN SHOW") since .istitle()-style tokens
# require exactly one leading capital per word, and rejects topic phrases
# that happen to be capitalized sentence-starts by requiring EVERY token
# capitalized, not just the first.
_NAME_TOKEN = r"[A-Z][a-zA-Z'’\-]+"
_NAME_1TO4 = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}}"
_TWO_NAMES = rf"{_NAME_1TO4}\s+(?:&|and)\s+{_NAME_1TO4}"

TEMPLATES: dict[str, re.Pattern] = {
    "bare_name": re.compile(rf"^({_NAME_1TO4})$"),
    "bare_two_names": re.compile(rf"^({_TWO_NAMES})$"),
    "numbered_dash_name": re.compile(rf"^#?\d+\s*[-:–]\s*({_NAME_1TO4})\s*$"),
    "name_colon_topic": re.compile(rf"^({_NAME_1TO4}(?:\s+(?:&|and)\s+{_NAME_1TO4})?)\s*:\s*\S.+$"),
    "with_name_suffix": re.compile(rf"\bwith\s+({_NAME_1TO4})\s*$", re.I),
    "feat_name": re.compile(rf"\b(?:feat\.?|featuring)\s+({_NAME_1TO4})", re.I),
    "interview_with_name": re.compile(rf"\binterview\s+with\s+({_NAME_1TO4})", re.I),
}

MIN_SHOW_EPISODES = 10  # below this, coverage estimates are too noisy to be worth reporting


def evaluate_show(titles: list[str], min_matches: int) -> list[dict]:
    n = len(titles)
    results = []
    for name, pattern in TEMPLATES.items():
        matches = [t for t in titles if pattern.search(t)]
        if len(matches) < min_matches:
            continue
        examples = []
        for t in matches[:5]:
            m = pattern.search(t)
            examples.append({"title": t, "captured": m.group(1)})
        results.append({
            "template": name,
            "n_matches": len(matches),
            "coverage": len(matches) / n,
            "examples": examples,
        })
    results.sort(key=lambda r: r["coverage"], reverse=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-coverage", type=float, default=0.15)
    ap.add_argument("--min-matches", type=int, default=5)
    args = ap.parse_args()

    eps = load_episodes()
    print(f"[data] {len(eps)} episodes, {eps['collection_id'].nunique()} shows")

    show_patterns = {}
    n_with_candidate = 0
    for sid, g in eps.groupby("collection_id"):
        titles = g["episode_title"].dropna().tolist()
        if len(titles) < MIN_SHOW_EPISODES:
            continue
        candidates = evaluate_show(titles, args.min_matches)
        candidates = [c for c in candidates if c["coverage"] >= args.min_coverage]
        if not candidates:
            continue
        n_with_candidate += 1
        best = candidates[0]
        show_patterns[sid] = {
            "show_id": sid,
            "show_name": g["show_name"].iloc[0],
            "n_episodes": len(titles),
            "best_template": best["template"],
            "best_coverage": round(best["coverage"], 3),
            "best_n_matches": best["n_matches"],
            "best_examples": best["examples"],
            "all_candidates": candidates,
            "approved": False,
        }

    print(f"[discover] {n_with_candidate}/{eps['collection_id'].nunique()} shows have >=1 candidate "
          f"template (coverage>={args.min_coverage}, matches>={args.min_matches})")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_coverage": args.min_coverage,
        "min_matches": args.min_matches,
        "note": "ALL entries have approved:false. A human must review best_examples and flip "
                "approved:true per show before nlp/extract_guests.py will use Tier 1 for that show. "
                "This script does not and will not auto-approve.",
        "shows": show_patterns,
    }
    out_path = pipeline_config.OUTPUT_DIR / "show_patterns.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[report] -> {out_path.name}")

    print("\n=== top candidates by coverage (for quick eyeball) ===")
    ranked = sorted(show_patterns.values(), key=lambda s: -s["best_coverage"])
    for s in ranked[:15]:
        print(f"  {s['show_name']:<45} {s['best_template']:<20} cov={s['best_coverage']:.2f} "
              f"n={s['best_n_matches']:>4}  e.g. {s['best_examples'][0]['captured']!r}")


if __name__ == "__main__":
    main()
