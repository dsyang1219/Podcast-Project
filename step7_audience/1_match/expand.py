"""Audited host-name expansion of the respondent match (match_expanded.csv), with the excluded-host audit trail. Reported as a sensitivity that does not reproduce the strict-match results."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, re

# Verbatims, show scores, and the show-host table (to map host names to shows); the strict match is the base.
B = str(ROOT) + "/"
L = pd.read_csv(INPUTS / "verbatims_long.csv")
S = pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "dir_z", "lean"]]
sh = pd.read_csv(DATA / "output/shows_205_host_dime_scores.csv")[["show_id", "show", "host"]]  # host names per show, from the DIME-score table
S = S.merge(sh, on="show_id", how="left")
strict = pd.read_csv(INPUTS / "match_strict.csv")

# ---- AUDITED host -> show_id map.
# CLEAN: host name is the podcast brand; no concurrently-airing major TV program.
CLEAN = {
    "heather cox richardson": "Letters from an American",
    "benny johnson": "The Benny Show",
    "mark levin": "Life, Liberty & Levin Podcast",
    "candace owens": "Candace",
    "brian tyler cohen": "No Lie with Brian Tyler Cohen",
    "steven crowder": "Louder with Crowder",
    "robert reich": "The Coffee Klatch with Robert Reich",
    "russell brand": "Stay Free with Russell Brand",
    "sam seder": "The Majority Report with Sam Seder",
    "tim miller": "The Bulwark Podcast",
    "hugh hewitt": "The Hugh Hewitt Show: Highly Concentrated",
    "james carville": "Politics War Room with James Carville & Al Hunt",
    "keith olbermann": "Countdown with Keith Olbermann",
    "kyle kulinski": "The Kyle Kulinski Show",
    "david pakman": "The David Pakman Show",
    "glenn greenwald": "System Update with Glenn Greenwald",
    "michael berry": "The Michael Berry Show",
    "graham allen": "The Graham Allen Show",
    "lars larson": "The Lars Larson Show Pacific Northwest Podcast",
    "wendy bell": "Wendy Bell Radio Podcast",
    "marc elias": "Defending Democracy with Marc Elias",
    "harry litman": "Talking Feds",
    "nick freitas": "Making the Argument with Nick Freitas",
    "kyle seraphin": "The Kyle Seraphin Show",
    "harry sisson": "The Harry Sisson Show",
    "victor davis hanson": "The Victor Davis Hanson Show",
    "steve bannon": "Bannon`s War Room",
    "mary trump": "The Mary Trump Podcast",
    "ben meiselas": "The MeidasTouch Podcast",
    "rick wilson": "The Lincoln Project",
    "brian lehrer": "The Brian Lehrer Show",
}
# EXCLUDED and why (kept for the audit trail / paper appendix)
EXCLUDED = {
    "chris hayes": "MSNBC All In is a TV program",
    "joy reid": "MSNBC TV program",
    "jim acosta": "CNN/TV identity",
    "ari melber": "MSNBC The Beat is TV",
    "nicolle wallace": "MSNBC Deadline White House is TV",
    "katie phang": "MSNBC TV",
    "chris cuomo": "NewsNation TV host",
    "john dickerson": "CBS News anchor",
    "jimmy failla": "Fox News TV host",
    "ted cruz": "sitting senator, name != show",
    "michael cohen": "famous for non-podcast reasons",
    "scott galloway": "hosts several podcasts",
    "mike baker": "ambiguous",
    "joyce vance": "MSNBC contributor",
}
NEWSLETTER = re.compile(r"substack|newsletter|news letter|letters from")  # a newsletter/Substack is the wrong medium, not the podcast

# Look up each CLEAN show's id, then build one regex per host that accepts an optional 'the' before and an
# optional 'show/podcast/pod/radio' after the name and nothing else (anchored ^...$, so 'X interview' does not match).
id_of = dict(zip(S.show.str.lower(), S.show_id))
rows = []
dropped = []
for h, showname in CLEAN.items():
    sid = id_of.get(showname.lower())  # show title not in the scored corpus: report and skip
    if sid is None:
        print("!! no dir_z for", showname)
        continue
    pat = re.compile(
        r"^(the\s+)?" + re.escape(h) + r"(\s+(show|podcast|pod|podcasts|pod\s*cast|radio))?$"
    )
    # Scan every distinct verbatim string against this host's pattern.
    for v in L.v.unique():
        vv = re.sub(r"[^a-z0-9 ]", " ", str(v))  # strip punctuation (verbatims are already lower-case)
        vv = re.sub(r"\s+", " ", vv).strip()  # collapse repeated whitespace
        if pat.match(vv):
            if NEWSLETTER.search(str(v)):  # host-name match that mentions a newsletter: drop and log
                dropped.append((v, "newsletter/substack"))
                continue
            rows.append((v, h, showname, sid))
alias = pd.DataFrame(rows, columns=["verbatim", "host", "show", "show_id"]).drop_duplicates()  # alias table: verbatim string -> host -> show -> show_id
print(f"clean host aliases matched: {len(alias)} verbatim strings")
print("dropped as wrong-medium:", dropped)

# Respondents whose verbatim is a host alias become new matches; union with the strict match,
# one row per respondent x show.
new = L.merge(alias, left_on="v", right_on="verbatim", how="inner")[
    ["resp", "verbatim", "show", "show_id"]
]
comb = pd.concat([strict[["resp", "verbatim", "show", "show_id"]], new]).drop_duplicates(
    ["resp", "show_id"]
)
# Bookkeeping: how many respondents the host expansion adds beyond the strict match.
print(f"\nstrict respondents      : {strict.resp.nunique()}")
print(f"host-expansion adds     : {new.resp.nunique() - len(set(new.resp)&set(strict.resp))}")
print(f"EXPANDED respondents    : {comb.resp.nunique()}   shows: {comb.show_id.nunique()}")
# Outputs: the expanded match (a sensitivity analysis), the alias table, and the excluded-host audit trail.
comb.to_csv(INPUTS / "match_expanded.csv", index=False)
alias.to_csv(INPUTS / "alias_table_hosts.csv", index=False)
pd.DataFrame([(k, v) for k, v in EXCLUDED.items()], columns=["host", "reason"]).to_csv(
    INPUTS / "alias_excluded.csv", index=False
)
