"""Discovery-sample estimates under CR1, CR2 (Bell-McCaffrey + Satterthwaite df) and the wild cluster bootstrap: favourability gap, cynicism composite, exclusivity. Full record section F.3-F.4."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
from common import *

# Each entry below is (label, result dict from cr_all): the dir_z coefficient with CR1, CR2 and wild-bootstrap p-values.
rows = []
g = st.copy()  # strict (exact-title) match: the operationalisation of record
# 1. Party favourability gap (|Q39A - Q39B|) on dir_z with the base controls only.
y, X, gid, j = design_np(g, "ap_gap", ())
rows.append(("Favourability gap, strict, base controls", cr_all(y, X, gid, j)))
# 2. Institutional cynicism composite, adding the two ideology controls (share_right, rw_plat).
y, X, gid, j = design_np(g, "inst_cyn", IDEO)
rows.append(("Cynicism composite, strict, +ideology", cr_all(y, X, gid, j)))
# 3. Exclusivity: respondent named no mainstream outlet. Built from R (any_main flag from common.py).
ge = st.merge(R, on="resp")
ge["excl"] = 1 - ge.any_main  # excl = 1 if none of the respondent's named sources is mainstream
ge = ge[~ge.show.str.contains("NBC", na=False)] if "show" in ge else ge  # no-op unless a 'show' title column exists
# Drop NBC-branded shows (e.g. Meet the Press): naming them already counts as naming a mainstream outlet,
# so they would mechanically pull exclusivity toward zero.
nb = pd.read_csv(INPUTS / "directive_final.csv")
nb = set(nb[nb.show.str.contains("NBC")].show_id)
ge = ge[~ge.show_id.isin(nb)]
y, X, gid, j = design_np(ge, "excl", ("side",))  # control for the show's own ideology (side, -1..+1)
rows.append(("Exclusivity, strict, NBC excl., +show ideology", cr_all(y, X, gid, j)))
# 4-5. Same exclusivity and cynicism models on the EXPANDED (host-name) match, as a sensitivity check.
gx = ex.merge(R, on="resp")
gx["excl"] = 1 - gx.any_main
gx = gx[~gx.show_id.isin(nb)]
y, X, gid, j = design_np(gx, "excl", ("side",))
rows.append(("Exclusivity, EXPANDED, NBC excl., +show ideology", cr_all(y, X, gid, j)))
y, X, gid, j = design_np(ex, "inst_cyn", IDEO)
rows.append(("Cynicism composite, EXPANDED, +ideology", cr_all(y, X, gid, j)))

# Print one line per model: coefficient, the three p-values, Satterthwaite df, number of clusters (shows) and respondents.
print(f"{'estimate':<48}{'b':>8}{'CR1 p':>9}{'CR2 p':>9}{'Satt df':>9}{'WILD p':>9}{'G':>5}{'N':>6}")
for lab, r in rows:
    print(
        f"{lab:<48}{r['b']:+8.3f}{r['p1']:9.4f}{r['p2']:9.4f}{r['df']:9.1f}{r['pw']:9.4f}{r['G']:5d}{r['N']:6d}"
    )
print(
    "\nCR1 = conventional cluster-robust, t on G-1 df.  CR2 = Bell-McCaffrey small-sample correction with Satterthwaite df (Pustejovsky & Tipton 2018)."
)
print(
    "Satterthwaite df is the EFFECTIVE number of clusters the coefficient is identified from; when it is far below G, the CR1 p is unreliable."
)
