# 7.3 — the population finding

Uses the diet segments from `1_match/diet.py` on the full survey (not just corpus listeners): people whose news
diet is podcasts are more institutionally cynical than mainstream-diet respondents, in both parties and at every
education level.

```
diet segments  --pop_profile.py-->  outputs/pop_profile.csv   (every usable survey item, BH-corrected)
               --gradient.py------>  the dose gradient: podcast-only / platform-only / mixed vs mainstream-only, equivalence test
               --popcontrast.py--->  the same contrast under narrower and broader podcast definitions
               --survive.py------->  inputs/d_pop.pkl  (composites that survive the full control ladder)
                     d_pop.pkl  --perm.py-->  show-level permutation p-values
                     d_pop.pkl  --scan.py-->  outputs/outcome_scan.csv  (the 113-item scan, disclosure G.1)
```

| script | role |
|---|---|
| `pop_profile.py` | Profiles all usable items across the segments; the source of the population table. |
| `gradient.py` | The gradient and the TOST equivalence test between the two parties. |
| `popcontrast.py` | Sensitivity to how "podcast listener" is defined (corpus shows only, any political podcast, the word "podcast"). |
| `survive.py` | Runs the control ladder and saves what survives for the two scripts below. |
| `perm.py` | Show-level permutation test for those items. |
| `scan.py` | The full outcome scan on the discovery sample, reported in the disclosure section. |
