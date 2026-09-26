"""CONFIRMATORY Year 2 test, executing handoff/prereg/prereg_y2.md as frozen (2026-09-23T17:09Z).
Address (mfte_z) -> election distrust (primary), 4-item cynicism (secondary), education (tertiary); intensity as the
named rival. Prints hashes, per-show table, specifications (i)-(iii), LOSO, leave-outs and the pooled specifications.
Writes outputs/y2_test_results.csv and handoff/results/y2_RUN_<ts>.log (via redirect)."""
import hashlib, sys, os, datetime, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT, INPUTS, OUTPUTS, HANDOFF
from common import cr_all

H = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
print(f"Y2 RUN {datetime.datetime.now(datetime.UTC).isoformat()} | prereg {H(HANDOFF/'prereg/prereg_y2.md')} | alias {H(HANDOFF/'prereg/prereg_y2_alias_FROZEN.csv')} | mfte {H(INPUTS/'mfte_show_scores.csv')}")
z = lambda s: (s - s.mean()) / s.std(ddof=0)
num = lambda g, c: pd.to_numeric(g[c], errors="coerce").where(lambda s: s > 0)

# ---- survey: outcomes standardised on the FULL Year 2 sample; controls as in common.py minus the platform count
K = pd.read_pickle(ROOT / "data/external/kettering/y2/y2_raw.pkl")
e = 6 - num(K, "Q33E"); f = num(K, "Q32C")
K["elec"] = z(((e - e.mean()) / e.std() + (f - f.mean()) / f.std()) / 2)
Z4 = pd.DataFrame({c: -z(num(K, c)) for c in ["Q31", "Q33E", "Q33H", "Q34B"]})
K["cyn4"] = z(Z4.mean(axis=1).where(Z4.notna().sum(axis=1) >= 3))
K["edu"] = num(K, "EDU"); K["edu_z"] = z(K.edu); K["age"] = num(K, "AGE"); K["attn"] = num(K, "Q19"); K["w"] = pd.to_numeric(K.WEIGHT, errors="coerce")
q41, q43 = num(K, "Q41"), num(K, "Q43")
K["pid"] = np.where(q41 == 1, "R", np.where(q41 == 2, "D", np.where(q43 == 2, "leanR", np.where(q43 == 1, "leanD", "I"))))
K["lean_num"] = K.pid.map({"R": 1, "leanR": 1, "D": -1, "leanD": -1, "I": 0})
print(f"survey n={len(K):,} | elec sd {K.elec.std():.2f} | cyn4 alpha-proxy mean inter-item r {Z4.corr().values[np.triu_indices(4,1)].mean():.2f}")

# ---- exposure
M = pd.read_csv(INPUTS / "mfte_show_scores.csv")[["show_id", "mfte_z"]]; M["show_id"] = M.show_id.astype(str)
S = pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "inten", "lean"]]; S["show_id"] = S.show_id.astype(str)
ID = pd.read_csv(INPUTS / "inten_distilled.csv")[["show_id", "inten_use"]]; ID["show_id"] = ID.show_id.astype(str)
mu, sd = S.inten.mean(), S.inten.std()
LEAN = S.set_index("show_id").lean.to_dict()
LEAN.update({f"HOLDOUT_{s}": "R" for s in ["beck", "bongino", "kelly", "kirk", "rogan", "ryan", "shapiro", "tucker"]}); LEAN.update({"HOLDOUT_thedaily": "L", "HOLDOUT_parnas": "L"})
A = pd.read_csv(HANDOFF / "prereg/prereg_y2_alias_FROZEN.csv", dtype={"show_id": str})
A = A.merge(M, on="show_id", how="left").merge(S[["show_id", "inten"]], on="show_id", how="left").merge(ID, on="show_id", how="left")
A["inten_fill"] = A.inten.fillna(A.inten_use); A["inten_z"] = (A.inten_fill - mu) / sd; A["right"] = A.show_id.map(LEAN).eq("R").astype(float)
A["inten_flag"] = A.inten.isna()

def frame(sub):
    prim = sub.sort_values("show").groupby("resp").first().reset_index()[["resp", "show_id", "show"]]
    R = sub.groupby("resp").agg(x=("mfte_z", "mean"), inten_z=("inten_z", "mean"), share_right=("right", "mean"), n_shows=("show_id", "nunique")).reset_index().merge(prim, on="resp").merge(K, on="resp")
    return R

def fit(R, y, x, extra=(), weighted=False, B=2000):
    X = pd.get_dummies(R.pid, prefix="pid", drop_first=True).astype(float); X["attn"] = R.attn; X["age"] = R.age
    if y != "edu_z": X["edu"] = R.edu
    if R.share_right.nunique() > 1: X["share_right"] = R.share_right
    for c in extra: X[c] = R[c]
    X = pd.concat([pd.Series(1.0, index=R.index, name="const"), X], axis=1); X["x"] = R[x]
    m = pd.concat([X, R[[y, "show_id"] + (["w"] if weighted else [])]], axis=1).dropna()
    if weighted:
        sw = np.sqrt(m.w.values); r = cr_all(m[y].values * sw, m[X.columns].values * sw[:, None], m.show_id.values, list(X.columns).index("x"), B=B)
    else:
        r = cr_all(m[y].values, m[X.columns].values, m.show_id.values, list(X.columns).index("x"), B=B)
    r["p1"] = r["p2"] / 2 if r["b"] > 0 else 1 - r["p2"] / 2; r["pw1"] = r["pw"] / 2 if r["b"] > 0 else 1 - r["pw"] / 2
    return r

ROWS = []
def report(tag, R, y, x, extra=(), weighted=False, B=2000):
    r = fit(R, y, x, extra, weighted, B)
    ROWS.append(dict(spec=tag, outcome=y, exposure=x, **{k: r[k] for k in ("b", "se2", "df", "p2", "pw", "p1", "pw1", "G", "N")}))
    print(f"  {tag:<44} {y:<6} b={r['b']:+.3f} se={r['se2']:.3f} df={r['df']:.1f}  one-sided CR2 p={r['p1']:.3f} wild p={r['pw1']:.3f}  (two-sided {r['p2']:.3f}/{r['pw']:.3f})  G={r['G']} N={r['N']}")
    return r

# ---- primary frame
P = A[A.primary]; R = frame(P)
print(f"\nPRIMARY FRAME: {len(R)} respondents, {R.show_id.nunique()} shows | intensity from distilled classifier for {P[P.inten_flag].show.unique().tolist()}")
per = P.merge(K[["resp", "lean_num"]], on="resp").groupby("show").agg(n=("resp", "nunique"), address=("mfte_z", "first"), intensity=("inten_z", "first"), aud_lean=("lean_num", "mean")).sort_values("n", ascending=False)
print(per.round(2).to_string())

print("\n== (i) address alone ==")
for y in ("elec", "cyn4", "edu_z"): report("(i) address", R, y, "x")
print("== (ii) address + intensity ==")
for y in ("elec", "cyn4", "edu_z"): report("(ii) address | intensity", R, y, "x", ("inten_z",))
print("== (iii) intensity alone ==")
for y in ("elec", "cyn4", "edu_z"): report("(iii) intensity", R, y, "inten_z")
print("== (ii') intensity | address ==")
for y in ("elec", "cyn4", "edu_z"): report("(ii') intensity | address", R, y, "inten_z", ("x",))

print("\n== VERDICT (one-sided .05 on BOTH p-values, primary outcome = election distrust) ==")
def v(tag, y):
    r = [r_ for r_ in ROWS if r_["spec"] == tag and r_["outcome"] == y][0]
    ok = r["b"] > 0 and r["p1"] < .05 and r["pw1"] < .05
    return "PASS" if ok else "FAIL", r
for h, tag, y, sign in [("H1", "(i) address", "elec", 1), ("H2", "(ii) address | intensity", "elec", 1), ("H3", "(ii) address | intensity", "edu_z", -1)]:
    r = [r_ for r_ in ROWS if r_["spec"] == tag and r_["outcome"] == y][0]
    b = r["b"] * sign; p1 = r["p1"] if sign == 1 else (r["p2"] / 2 if r["b"] < 0 else 1 - r["p2"] / 2); pw1 = r["pw1"] if sign == 1 else (r["pw"] / 2 if r["b"] < 0 else 1 - r["pw"] / 2)
    print(f"  {h}: {'PASS' if (b > 0 and p1 < .05 and pw1 < .05) else 'FAIL'}   (b={r['b']:+.3f}, one-sided CR2 p={p1:.3f}, wild p={pw1:.3f})")

print("\n== robustness (reported regardless) ==")
for lab, sub in [("outlet strings (Bulwark/Breitbart/Reich) dropped", A[A.primary & ~A.outlet_string]), ("Clay & Buck included", A[(A.frame == "corpus") & A.show_id.isin(P.show_id.unique()) | A.clay_and_buck]), ("MeidasTouch dropped", A[A.primary & (A.show != "The MeidasTouch Podcast")])]:
    Rr = frame(sub)
    for y in ("elec", "cyn4", "edu_z"):
        report(f"{lab[:38]} (i)", Rr, y, "x", B=800); report(f"{lab[:38]} (ii)", Rr, y, "x", ("inten_z",), B=800)
for y in ("elec", "cyn4", "edu_z"): report("survey-weighted (i)", R, y, "x", weighted=True, B=800)
print("\n== leave-one-show-out, primary frame ==")
for tag, x, ex in [("(i) address", "x", ()), ("(ii) address | intensity", "x", ("inten_z",))]:
    for y in ("elec", "cyn4", "edu_z"):
        bs = []; ps = []
        for s in R.show_id.unique():
            r = fit(R[R.show_id != s], y, x, ex, B=2); bs.append(r["b"]); ps.append(r["p1"])
        print(f"  {tag:<26} {y:<6} b in [{min(bs):+.3f}, {max(bs):+.3f}]  one-sided CR2 p<.05 in {sum(p < .05 for p in ps)}/{len(bs)} drops")

print("\n== secondary: pooled with out-of-frame shows ==")
Rp = frame(A[A.primary | (A.frame == "out_of_frame")])
print(f"  {len(Rp)} respondents, {Rp.show_id.nunique()} shows")
for y in ("elec", "cyn4", "edu_z"): report("pooled (i) address", Rp, y, "x", B=1000); report("pooled (ii) address | intensity", Rp, y, "x", ("inten_z",), B=1000)
for y in ("elec", "cyn4", "edu_z"): report("pooled, Rogan excl. (ii)", Rp[Rp.show_id != "HOLDOUT_rogan"], y, "x", ("inten_z",), B=1000)

pd.DataFrame(ROWS).to_csv(OUTPUTS / "y2_test_results.csv", index=False)
print("\nwritten:", OUTPUTS / "y2_test_results.csv")
