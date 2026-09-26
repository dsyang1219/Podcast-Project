# step7_audience/9_education_probes

Scripts behind the education-composition results (24–25 September 2026), all run from the repository root with
`.venv/bin/python`. They read the restricted Kettering-Gallup microdata (`data/external/kettering/`, not committed)
and the show-level tables in `step7_audience/inputs/`. Each prints to stdout; the logs are in `results/`
and the tables in `scans/`.

| script | what it does | record |
|---|---|---|
| `edu_consequences.py` | builds the pooled two-wave listener frame (1,247 / 64 matched; 1,200 / 63 with complete data) and the CR2 + wild-bootstrap `fit()`; indirect paths, moderation by education, reach, show-level profile | `scans/address_edu_indirect_paths.csv`, `address_by_education_moderation.csv` |
| `edu_followup.py` | life-ladder × education robustness; composition by address tercile and by side | y2_RESULTS.md |
| `imp_classes_audience.py` | Biber command classes (from `step6_register/mfte_imperative_classes.py`) against education, by year and head to head | `results/imperative_classes_audience.log` |
| `diet_radio.py` | other named sources (radio, cable, newspaper, YouTube ...) by address; talk-radio-succession check | `results/address_news_diet.log` |
| `survey_new_angles.py` | never-scanned survey variables, address × own party, multi-show namers, all address factors jointly | `results/survey_new_angles.log` |
| `ceiling.py` | between-show variance ceiling per outcome; time-matched address | `results/between_show_ceiling.log`, `scans/between_show_ceiling.csv` |
| `edu_adjacent.py` | education vs income, credential levels, employment, civic socialisation | `results/address_education_adjacent.log` |
| `stance_index.py` | guide-vs-peer stance index fitted on Year 1, tested on Year 2 | `results/stance_index_y1_to_y2.log` |

The other scripts `exec()` the header of `edu_consequences.py` to share the frame and the estimator, so run them
from the repository root. `figures/build_figures.py` does the same to draw Figures 4 to 7.
