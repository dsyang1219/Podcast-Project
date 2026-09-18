#!/usr/bin/env python3
"""Resolve corpus shows and hosts to Wikipedia articles via the Wikidata Query Service.

    ./resolve_wikipedia_sparql.py --run
    ./resolve_wikipedia_sparql.py --audit

Why SPARQL instead of the search API
------------------------------------
collect_wikipedia_v2.py resolved through wbsearchentities + Wikipedia search, ~7
requests per target, ~1,500 total. Wikimedia answered with a sustained per-IP
penalty: HTTP 429 carrying Retry-After ~40s on en.wikipedia.org, www.wikidata.org
AND wikimedia.org alike. At 40s per request that design needs seven hours, and
worse, the throttling is silent — an api() that swallows the exception records
"no article" and the run looks merely disappointing rather than broken. That is
most of why v1 reported 5% coverage.

query.wikidata.org is a different service with a different budget. Two queries
replace the entire search phase, and they return strictly more information:

    Tim Miller (politician)            | American politician
    Tim Miller (political strategist)  | American political consultant and writer

Both Tim Millers come back in one row set, each with its description and its
occupations, so the choice is made on evidence rather than on which one a search
ranker happened to put first. The old path took the search ranker's word for it
and picked the politician.

What still uses the throttled API
---------------------------------
Only the intro-extract verification, batched 20 titles per request (~15 calls for
the whole corpus), and the pageview pull itself. Both are paced and both treat
429 as wait-and-retry, never as a result.

Matching rules
--------------
Exact label or altLabel match only — no fuzzy matching. Shows are additionally
tried under name variants ("The X Podcast" -> "X"), because an article is
routinely titled for the brand rather than the feed. Candidates are then filtered
on type evidence (a show must look like a programme or media brand; a host must be
a human) and, where several survive, ranked by how well their description and
occupations agree with the show context and with real_occupation from
host_dime_lookup_v2.csv.

Precision is still the priority: "Bulwark Takes" surfaces HMS Bulwark (L15), a
Royal Navy warship, and a wrong article is worse in an attention panel than a
missing one. Every acceptance records the evidence that justified it.
"""
import argparse, csv, json, re, sys, time
import urllib.error, urllib.parse, urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root (this script lives one folder down)
SHOWS = ROOT / "data/output/corpus_shows.csv"
HOSTS = ROOT / "data/output/host_dime_lookup_v2.csv"
OUT = ROOT / "data/output/wikipedia_resolved_v2.csv"
CAND = ROOT / "data/output/wikipedia_candidates_sparql.csv"

UA = {"User-Agent": "PoliticalPodcastCorpus/2.0 (academic research; podcast attention panel) python-urllib/3.12"}
SPARQL = "https://query.wikidata.org/sparql"

STOP = {"the", "a", "an", "with", "and", "podcast", "podcasts", "show", "on", "of",
        "in", "for", "to", "at", "from", "by", "is", "his", "her", "their"}
MEDIA = ("podcast", "radio", "talk show", "broadcast", "web series", "television series",
         "television programme", "television program", "news program", "news programme",
         "news commentary", "youtube channel", "internet show", "talk radio", "programme",
         "media network", "media company", "media organization", "news organization",
         "news website", "news outlet", "magazine", "newsletter", "political action committee",
         "media franchise", "web show", "series")
PERSON_HINT = ("journalist", "commentator", "host", "broadcaster", "pundit", "writer",
               "podcaster", "consultant", "strategist", "attorney", "lawyer", "activist",
               "economist", "professor", "comedian", "reporter", "anchor", "editor",
               "correspondent", "presenter", "author", "analyst", "spokesman",
               "spokesperson", "political scientist", "historian", "blogger")


class RateLimited(Exception):
    """Throttled (429). The fix is FEWER requests, or waiting."""


class QueryTooHeavy(Exception):
    """WDQS refused the query itself (503 / timeout). The fix is a SMALLER query.

    These two failures pull in opposite directions and must not be conflated:
    batching harder to dodge a 429 is exactly what causes a 503.
    """


def words(s):
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split() if w not in STOP}


def sparql(query, tries=6):
    # Wikimedia applies a per-IP penalty across ALL its services, including this
    # one, and it is not query cost: the same query that answers in 0.4s cold
    # returns 429 once the budget is spent. So back off long and patiently rather
    # than retrying fast, and never let an exhausted retry look like "no match".
    backoff = 30.0
    for k in range(tries):
        u = SPARQL + "?" + urllib.parse.urlencode({"query": query, "format": "json"})
        try:
            req = urllib.request.Request(u, headers={**UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())["results"]["bindings"]
        except urllib.error.HTTPError as e:
            if e.code == 503:
                # Server refused the query, not the client. Retrying the same
                # query at the same size just burns the clock; the caller splits.
                raise QueryTooHeavy(f"503 at {len(query)} chars")
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                w = float(ra) if (ra or "").isdigit() else backoff
                print(f"    [sparql 429] backoff {w:.0f}s", flush=True)
                time.sleep(w); backoff = min(backoff * 2, 300); continue
            print(f"    [sparql {e.code}] {e.reason}", flush=True)
            return []
        except Exception as e:
            if k == tries - 1:
                print(f"    [sparql fail] {type(e).__name__}", flush=True)
                return []
            time.sleep(backoff); backoff = min(backoff * 2, 60)
    # NEVER return empty here. An exhausted retry is throttling, not an absence of
    # matches, and returning [] would record every name in this batch as "no
    # article" — the exact silent failure that made the search-API path report 5%.
    raise RateLimited("sparql retries exhausted; rerun after the penalty expires")


def q_literal(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"@en'


QUERY = """
SELECT ?name ?item ?article ?desc
       (GROUP_CONCAT(DISTINCT ?typeL;  separator=" | ") AS ?types)
       (GROUP_CONCAT(DISTINCT ?occL;   separator=" | ") AS ?occs)
WHERE {
  VALUES ?name { %s }
  { ?item rdfs:label ?name } UNION { ?item skos:altLabel ?name }
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
  OPTIONAL { ?item schema:description ?desc FILTER(lang(?desc) = "en") }
  OPTIONAL { ?item wdt:P31  ?type . ?type rdfs:label ?typeL FILTER(lang(?typeL) = "en") }
  OPTIONAL { ?item wdt:P106 ?occ  . ?occ  rdfs:label ?occL  FILTER(lang(?occL)  = "en") }
}
GROUP BY ?name ?item ?article ?desc
"""


def _fetch_chunk(chunk, out, depth=0):
    """Run one batch, halving it if WDQS says the query is too heavy."""
    try:
        rows = sparql(QUERY % " ".join(q_literal(n) for n in chunk))
    except QueryTooHeavy:
        if len(chunk) <= 4:
            print(f"    [503] giving up on {len(chunk)} names: {chunk[:4]}", flush=True)
            return 0
        mid = len(chunk) // 2
        print(f"    [503] splitting {len(chunk)} -> {mid}+{len(chunk)-mid}", flush=True)
        time.sleep(5.0)
        n = _fetch_chunk(chunk[:mid], out, depth + 1)
        time.sleep(5.0)
        return n + _fetch_chunk(chunk[mid:], out, depth + 1)
    for r in rows:
        title = urllib.parse.unquote(r["article"]["value"].rsplit("/", 1)[1]).replace("_", " ")
        out[r["name"]["value"]].append({
            "title": title,
            "desc": r.get("desc", {}).get("value", ""),
            "types": r.get("types", {}).get("value", ""),
            "occs": r.get("occs", {}).get("value", ""),
        })
    return len(rows)


def fetch_candidates(names, batch=40):
    """{queried_name: [ {title, desc, types, occs} ]}.

    batch=40 is measured, not guessed: 40 names answer in ~0.4s, 400 names 503.
    """
    out = defaultdict(list)
    names = [n for n in dict.fromkeys(names) if n]
    nb = (len(names) + batch - 1) // batch
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        got = _fetch_chunk(chunk, out)
        print(f"  batch {i//batch + 1}/{nb}: {len(chunk)} names -> {got} candidates", flush=True)
        time.sleep(8.0)
    return out


def show_variants(name):
    """An article is often titled for the brand, not the feed."""
    v = [name]
    s = re.sub(r"^The\s+", "", name).strip()
    for cand in (s,
                 re.sub(r"\s+Podcast$", "", name, flags=re.I).strip(),
                 re.sub(r"\s+Podcast$", "", s, flags=re.I).strip(),
                 re.sub(r"\s+(Show|Podcast)$", "", s, flags=re.I).strip()):
        if cand and cand not in v:
            v.append(cand)
    # "X with Y" and "X w/ Y": the article is under X, and Y is the host
    m = re.split(r"\s+(?:with|w/|feat\.?|featuring)\s+", name, flags=re.I)
    if len(m) > 1 and m[0].strip() and m[0].strip() not in v:
        v.append(m[0].strip())
        v.append(re.sub(r"^The\s+", "", m[0]).strip())
    return [x for x in dict.fromkeys(v) if x]


def is_disambig(c):
    return "disambiguation" in (c["desc"] + " " + c["types"]).lower()


def score_show(name, c):
    blob = (c["desc"] + " " + c["types"]).lower()
    if is_disambig(c):
        return None
    nw, tw = words(name), words(c["title"])
    if not nw or not (nw & tw):
        return None
    exact = tw == nw
    media = any(m in blob for m in MEDIA)
    if not (exact or media):
        return None
    # coverage of the show's distinctive words by the article title
    cov = len(nw & tw) / len(nw)
    if cov < 0.6 and not exact:
        return None
    return (2.0 if exact else 0.0) + (1.0 if media else 0.0) + cov


def score_host(name, c, ctx, occupation):
    blob = (c["desc"] + " " + c["types"] + " " + c["occs"]).lower()
    if is_disambig(c):
        return None, ""
    if "human" not in c["types"].lower() and not any(p in blob for p in PERSON_HINT):
        return None, ""
    nw, tw = words(name), words(c["title"])
    if not nw or not (nw & tw):
        return None, ""
    if len(nw & tw) < len(nw):
        return None, ""                      # every name token must appear
    ev, sc = [], 0.0
    if any(p in blob for p in PERSON_HINT):
        ev.append("person"); sc += 0.5
    if any(m in blob for m in ("podcast", "radio", "broadcast", "talk show", "journalis",
                               "commentat", "television", "media")):
        ev.append("media-adjacent"); sc += 1.5
    shared = {w for w in ctx if len(w) > 3} & words(blob)
    if shared:
        ev.append("show:" + ",".join(sorted(shared)[:3])); sc += 2.0
    if occupation:
        ov = words(occupation) & words(blob)
        if ov:
            ev.append("occ:" + ",".join(sorted(ov)[:3])); sc += 1.0 + 0.4 * len(ov)
    # A bare human with a matching name is exactly how "Tim Miller (politician)"
    # got in. Require something beyond personhood.
    if sc <= 0.5:
        return None, ""
    return sc, "+".join(ev)


def load_targets():
    shows = [r for r in csv.DictReader(open(SHOWS))
             if not (r.get("excluded_reason") or "").strip()]
    shows.sort(key=lambda r: int(r["chart_rank"] or 9999))
    hosts = defaultdict(list)
    for r in csv.DictReader(open(HOSTS)):
        n = (r.get("host_name") or "").strip()
        if n and 1 < len(n.split()) <= 4 and not re.search(r"unknown|n/?a|none", n, re.I):
            alts = [x.strip() for x in (r.get("alt_names") or "").split(";") if x.strip()]
            hosts[r["show_id"]].append((n, alts, (r.get("real_occupation") or "").strip()))
    return shows, hosts


def cmd_run():
    shows, hosts = load_targets()
    show_names, host_names = [], []
    for s in shows:
        show_names += show_variants(s["show_name"])
    for lst in hosts.values():
        for n, alts, _ in lst:
            host_names.append(n); host_names += alts

    print(f"resolving {len(shows)} shows ({len(set(show_names))} name variants) "
          f"and {len(set(host_names))} host names\n")
    print("shows:")
    cs = fetch_candidates(show_names)
    print("hosts:")
    ch = fetch_candidates(host_names)

    with open(CAND, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["queried_name", "title", "desc", "types", "occs"])
        for src in (cs, ch):
            for n, lst in src.items():
                for c in lst:
                    w.writerow([n, c["title"], c["desc"], c["types"], c["occs"]])
    print(f"\n  all candidates written to {CAND} (audit trail)\n")

    rows, n_s, n_h = [], 0, 0
    for s in shows:
        ctx = words(s["show_name"]) | words(s.get("publisher", ""))
        best = None
        for v in show_variants(s["show_name"]):
            for c in cs.get(v, []):
                sc = score_show(s["show_name"], c)
                if sc is not None and (best is None or sc > best[0]):
                    best = (sc, c)
        if best:
            n_s += 1
            sc, c = best
            rows.append([s["show_id"], s["show_name"], "show", s["show_name"], c["title"],
                         1, "sparql", round(sc, 2), (c["desc"] or c["types"])[:110]])
        else:
            rows.append([s["show_id"], s["show_name"], "show", s["show_name"], "", 0, "", "", ""])

        for n, alts, occ in hosts.get(s["show_id"], []):
            best = None
            for v in [n] + alts:
                for c in ch.get(v, []):
                    sc, ev = score_host(n, c, ctx, occ)
                    if sc is not None and (best is None or sc > best[0]):
                        best = (sc, c, ev)
            if best:
                n_h += 1
                sc, c, ev = best
                rows.append([s["show_id"], s["show_name"], "host", n, c["title"], 1,
                             "sparql", round(sc, 2), f"{(c['desc'] or '')[:70]} || {ev}"])
            else:
                rows.append([s["show_id"], s["show_name"], "host", n, "", 0, "", "", ""])

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["show_id", "show_name", "target_kind", "target_name",
                    "wiki_title", "found", "tier", "match_score", "evidence"])
        w.writerows(rows)
    print(f"{OUT}\n  {n_s}/{len(shows)} show articles, {n_h} host articles")
    cmd_audit()


def cmd_audit():
    rows = list(csv.DictReader(open(OUT)))
    for kind in ("show", "host"):
        sub = [r for r in rows if r["target_kind"] == kind]
        hit = [r for r in sub if r["found"] == "1"]
        if sub:
            print(f"\n{kind}: {len(hit)}/{len(sub)} resolved ({len(hit)/len(sub):.0%})")
    reached = {r["show_id"] for r in rows if r["found"] == "1"}
    allids = {r["show_id"] for r in rows}
    print(f"\nshows reachable (own article OR a host's): {len(reached)}/{len(allids)} "
          f"({len(reached)/len(allids):.0%})")
    dup = defaultdict(set)
    for r in rows:
        if r["found"] == "1":
            dup[r["wiki_title"]].add(r["show_id"])
    shared = {t: v for t, v in dup.items() if len(v) > 1}
    if shared:
        print(f"\narticles serving >1 show ({len(shared)}) — verify these are genuinely "
              f"shared (networks, co-hosts) and not collapsed matches:")
        for t, v in sorted(shared.items(), key=lambda kv: -len(kv[1]))[:10]:
            print(f"    {t[:44]:<46} {len(v)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    if a.run: cmd_run()
    elif a.audit: cmd_audit()
    else: ap.print_help()


if __name__ == "__main__":
    main()
