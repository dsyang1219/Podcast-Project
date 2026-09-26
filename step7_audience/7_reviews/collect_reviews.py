"""Collect public Apple Podcasts customer reviews for every corpus show (up to 10 pages x 50 = 500 most recent).
Writes data/output/apple_reviews.csv (show_id, show, review_id, date, rating, title, text, author_hash) and a per-show
summary. Public RSS endpoint; polite pacing; resumable (skips shows already collected)."""
import requests, pandas as pd, time, hashlib, os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/output/apple_reviews.csv"
shows = pd.read_csv(ROOT / "data/output/corpus.csv")[["collection_id", "show_name"]]
done = set(pd.read_csv(OUT, usecols=["show_id"]).show_id.astype(str)) if OUT.exists() else set()
rows = []
for cid, name in shows.values:
    if str(cid) in done:
        continue
    n = 0
    for page in range(1, 11):
        u = f"https://itunes.apple.com/us/rss/customerreviews/page={page}/id={cid}/sortby=mostrecent/json"
        try:
            r = requests.get(u, timeout=25, headers={"User-Agent": "Mozilla/5.0 (research corpus; contact via GitHub dsyang1219)"})
            if r.status_code != 200:
                break
            ents = [e for e in r.json().get("feed", {}).get("entry", []) if "content" in e]
        except Exception as e:
            print("  error", name, page, str(e)[:80], flush=True); break
        if not ents:
            break
        def lab(x):
            return x.get("label", "") if isinstance(x, dict) else (str(x) if x is not None else "")
        for e in ents:
            au = e.get("author", {}); au = au.get("name", {}) if isinstance(au, dict) else {}
            rows.append(dict(show_id=str(cid), show=name, review_id=lab(e.get("id")), date=lab(e.get("updated")),
                             rating=lab(e.get("im:rating")), title=lab(e.get("title")), text=lab(e.get("content")),
                             author_hash=hashlib.sha1(lab(au).encode()).hexdigest()[:12]))
        n += len(ents)
        if len(ents) < 50:
            break
        time.sleep(0.6)
    print(f"{name[:45]:<45} {n:>4} reviews", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False); rows = []
    time.sleep(0.4)
d = pd.read_csv(OUT)
d = d.drop_duplicates(["show_id", "review_id"]); d.to_csv(OUT, index=False)
print(f"\ntotal {len(d):,} reviews, {d.show_id.nunique()} shows; median per show {d.groupby('show_id').size().median():.0f}")
d.groupby(["show_id", "show"]).agg(reviews=("review_id", "size"), mean_rating=("rating", lambda s: pd.to_numeric(s, errors="coerce").mean())).reset_index().to_csv(ROOT / "data/output/apple_reviews_summary.csv", index=False)
