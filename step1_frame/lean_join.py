"""Step 4 — fuzzy join to Brookings Political Podcast Project lean labels.

The external labels are for AXIS VALIDATION ONLY: they are joined onto the
corpus but never used to filter or stratify it.

Matching: rapidfuzz WRatio on normalized names.
  score >= FUZZY_ACCEPT                    -> matched
  FUZZY_REVIEW <= score < FUZZY_ACCEPT     -> review   (manual check)
  best - runner_up < FUZZY_AMBIGUOUS_GAP   -> ambiguous (manual check)
  score < FUZZY_REVIEW                     -> none
Nothing below `matched` is treated as a join downstream.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

from . import config

_STOPWORDS = {"the", "a", "an", "podcast", "show", "with"}


def normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    tokens = [t for t in name.split() if t not in _STOPWORDS]
    return " ".join(tokens)


def load_brookings(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    name_col = next((c for c in df.columns if c.strip().lower() == "show name"), None)
    lean_col = next((c for c in df.columns if c.strip().lower() == "partisan leaning"), None)
    if name_col is None or lean_col is None:
        raise ValueError(
            f"expected 'Show Name' and 'Partisan Leaning' columns, got {list(df.columns)}")
    series = (df[[name_col, lean_col]]
              .rename(columns={name_col: "brookings_show_name",
                               lean_col: "brookings_partisan_leaning"})
              .dropna(subset=["brookings_show_name"])
              .drop_duplicates(subset=["brookings_show_name"])
              .reset_index(drop=True))
    series["norm"] = series["brookings_show_name"].map(normalize)
    return series


def join_leans(corpus: pd.DataFrame, brookings: pd.DataFrame) -> pd.DataFrame:
    choices = brookings["norm"].tolist()
    rows = []
    for _, show in corpus.iterrows():
        query = normalize(show["show_name"])
        result = {
            "collection_id": show["collection_id"],
            "show_name": show["show_name"],
            "brookings_show_name": "",
            "brookings_partisan_leaning": "",
            "match_score": 0.0,
            "runner_up_score": 0.0,
            "match_status": "none",
        }
        if query and choices:
            top2 = process.extract(query, choices, scorer=fuzz.WRatio, limit=2)
            best_norm, best_score, best_idx = top2[0]
            runner_up = top2[1][1] if len(top2) > 1 else 0.0
            result["match_score"] = round(best_score, 1)
            result["runner_up_score"] = round(runner_up, 1)
            if best_score >= config.FUZZY_REVIEW:
                result["brookings_show_name"] = brookings.loc[best_idx, "brookings_show_name"]
                result["brookings_partisan_leaning"] = brookings.loc[
                    best_idx, "brookings_partisan_leaning"]
                if best_score >= config.FUZZY_ACCEPT:
                    result["match_status"] = "matched"
                else:
                    result["match_status"] = "review"
                # near-tie with a different runner-up name -> ambiguous
                if (best_score - runner_up < config.FUZZY_AMBIGUOUS_GAP
                        and len(top2) > 1 and top2[1][0] != best_norm):
                    result["match_status"] = "ambiguous"
                    result["runner_up_name"] = brookings.loc[top2[1][2],
                                                             "brookings_show_name"]
        rows.append(result)
    out = pd.DataFrame(rows)
    if "runner_up_name" not in out.columns:
        out["runner_up_name"] = ""
    out["runner_up_name"] = out["runner_up_name"].fillna("")
    return out
