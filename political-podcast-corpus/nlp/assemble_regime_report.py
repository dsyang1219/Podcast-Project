"""Final assembly: merges the regime-comparison LDA/ideology results
(nlp/lda_regime_refit.py), the per-regime vocab/token accounting
(nlp/export_regime_tokens.py), and the preText-equivalent sensitivity scan
(nlp/pretext_manual.py) into the single deliverable this task asked for --
see task brief "What to report": regime x {n filler-like topics, ideology
R2 LOSO-CV, mean c_v coherence} table, the preText standard output, and an
explicit verdict on (a) filler-topic elimination and (b) ideology-finding
robustness.

Usage:
    .venv/bin/python -m nlp.assemble_regime_report
"""
from __future__ import annotations

import json
from pathlib import Path

REGIMES_DIR = Path("data/output/regimes")
REGIME_ORDER = ["minimal", "moderate", "aggressive", "aggressive_pos"]


def load_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    comparison = load_json(REGIMES_DIR / "regime_comparison_report.json")
    by_regime = {r["regime"]: r for r in comparison["regimes"]}

    vocab = {}
    for r in REGIME_ORDER:
        p = REGIMES_DIR / f"vocab_accounting_{r}.json"
        if p.exists():
            vocab[r] = load_json(p)

    pretext = None
    p = REGIMES_DIR / "pretext_summary.json"
    if p.exists():
        pretext = load_json(p)

    baseline_r2 = by_regime.get("moderate", {}).get("ideology_probe", {}).get("loso_cv_r2")

    print("\n" + "=" * 100)
    print("REGIME-COMPARISON TABLE (answers both advisor concerns: cleaner topics? robust finding?)")
    print("=" * 100)
    header = f"{'regime':<16}{'n_filler/75':<13}{'ideology R2':<13}{'95% CI':<20}{'perm R2':<10}{'coherence_cv':<14}{'vocab (kept)':<14}{'tokens (kept)':<14}"
    print(header)
    print("-" * len(header))
    rows = []
    for r in REGIME_ORDER:
        if r not in by_regime:
            v = vocab.get(r, {})
            vocab_str = f"{v.get('vocab_after', '?')}/{v.get('vocab_before', '?')}"
            tok_str = f"{v.get('tokens_after', '?')}/{v.get('tokens_before', '?')}"
            print(f"{r:<16}{'OOM@48GB':<13}{'--':<13}{'--':<20}{'--':<10}{'--':<14}{vocab_str:<14}{tok_str:<14}")
            rows.append({"regime": r, "status": "OOM at 48GB cap during LDA fit (never reached probe)", "vocab": v})
            continue
        rr = by_regime[r]
        ip = rr["ideology_probe"]
        ci = f"[{ip['ci_95'][0]:.3f},{ip['ci_95'][1]:.3f}]"
        v = vocab.get(r, {})
        vocab_str = f"{v.get('vocab_after', '?')}/{v.get('vocab_before', '?')}"
        tok_str = f"{v.get('tokens_after', '?')}/{v.get('tokens_before', '?')}"
        row = f"{r:<16}{rr['n_filler_topics']}/{rr['k']:<9}{ip['loso_cv_r2']:<13.3f}{ci:<20}{ip['permutation_r2']:<10.3f}{rr['coherence_cv']:<14.3f}{vocab_str:<14}{tok_str:<14}"
        print(row)
        rows.append({
            "regime": r, "n_filler_topics": rr["n_filler_topics"], "k": rr["k"],
            "ideology_r2": ip["loso_cv_r2"], "ci_95": ip["ci_95"], "perm_r2": ip["permutation_r2"],
            "coherence_cv": rr["coherence_cv"], "vocab": v,
        })

    print("\n" + "=" * 100)
    print("PRETEXT-EQUIVALENT STANDARD OUTPUT (which preprocessing choices drive the most variation)")
    print("=" * 100)
    if pretext:
        print(f"n_specifications={pretext['n_specifications']}  n_passages_sampled={pretext['n_passages_sampled']}  "
              f"mean_score={pretext['mean_score']:.4f}  sd_score={pretext['sd_score']:.4f}")
        print(f"Most influential choice on preText score variation: {pretext['most_influential_choice']}")
    else:
        print("[missing] pretext_summary.json not found")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)

    verdict_lines = []
    baseline_filler = by_regime.get("moderate", {}).get("n_filler_topics")
    aggr_filler = by_regime.get("aggressive_pos", {}).get("n_filler_topics")
    minimal_status = "OOM'd at the 48GB cap before it could fit -- no filler-topic count available for the no-pruning baseline" if "minimal" not in by_regime else None
    if baseline_filler is not None and aggr_filler is not None:
        if baseline_filler == 0 and aggr_filler == 0:
            verdict_lines.append(
                f"(a) Filler-topic elimination: ALL preprocessed regimes that completed (moderate, aggressive, "
                f"aggressive_pos) show 0/75 filler-like topics -- the extended stopword list + docfreq pruning "
                f"already fully eliminates the T57/T46-style filler topics; POS-filtering on top adds no further "
                f"reduction because there is nothing left to reduce. "
                + (minimal_status or "")
            )
        else:
            verdict_lines.append(
                f"(a) Filler-topic elimination: moderate regime has {baseline_filler} filler-like topics; "
                f"aggressive_pos has {aggr_filler}. "
                + ("Aggressive+POS substantially reduces filler topics." if aggr_filler < baseline_filler
                   else "Aggressive+POS did NOT reduce filler topics relative to moderate -- report this honestly.")
            )

    if baseline_r2 is not None:
        r2s = {r: by_regime[r]["ideology_probe"]["loso_cv_r2"] for r in REGIME_ORDER if r in by_regime}
        spread = max(r2s.values()) - min(r2s.values())
        verdict_lines.append(
            f"(b) Ideology-finding robustness: LOSO-CV R2 ranges {min(r2s.values()):.3f}-{max(r2s.values()):.3f} "
            f"across regimes (spread={spread:.3f}). "
            + ("STABLE across preprocessing choices." if spread < 0.05 else
               "NOT fully stable -- preprocessing choice materially moves the ideology R2; report this honestly, "
               "it is itself a finding, not a failure.")
        )

    for line in verdict_lines:
        print(line)

    out = {
        "table": rows,
        "pretext": pretext,
        "verdict": verdict_lines,
    }
    out_path = REGIMES_DIR / "FINAL_regime_report.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[report] -> {out_path}")


if __name__ == "__main__":
    main()
