# handoff — the paper record

Everything a reader of the paper would need to check a number, in six folders.

| folder | contents |
|---|---|
| `prereg/` | The three pre-registrations as frozen (`prereg_Q28.md`, `prereg_holdout.md`, `prereg_withinframe.md`) with their SHA-256 records, the frozen respondent alias tables and patterns, and `dirz_reference_scale.csv`, the 194-show reference scale every address score is standardised against. **Do not edit these files**; their hashes are recorded and printed by the test scripts. |
| `results/` | The run logs and write-ups of the pre-registered tests (`holdout_RUN_*.log`, `holdout_RESULTS.md`, `withinframe_RUN_*.log`, `withinframe_RESULTS.md`, the final holdout show scores), the Kettering Year-1 results record (`kettering_y1_test_results.md`, rounds 19-25) and the populism / intensity record (`exploratory_populism.md`). |
| `scans/` | The result tables the paper cites: the address-outcome and political-frame scans, the show feature table (`show_features_ALL.csv`), the holdout audience profile and political-density tables, the distilled intensity scores, the populism word table, and the 13 missing episodes. |
| `paper/` | The paper text and its supporting documents: `abstract_v3.md` (four abstract variants), `paper_story.md`, `progress_report_revised.md`, `methods_gaps_answered.md`, and `artifacts_html/` (local copies of the three published pages: précis, full record, slides download). |
| `slides/` | The talk: `Talking_at_the_Audience_slides.pptx` and `.pdf`, `build_slides.py` (rebuilds the deck), `speaker_notes.py` (writes the plain-language notes into it), `talk_script.md`. |
| `notes/` | Operational notes: `RMA2_SETUP.md` (GPU machine set-up), `PIPELINE_LOAD_PROFILE.md` (written for the cluster administrators), `kettering_data_request_email.md`. |

## Where a file moved

Until 12 September 2026 every file sat directly in `handoff/`. The frozen pre-registrations therefore cite paths
such as `handoff/holdout_alias_table_FROZEN.csv`; those files are now in `handoff/prereg/`. Other documents were
updated; the pre-registration texts were not, because their content is hashed. The mapping is the table above:
`prereg_*`, `*_FROZEN.*` and `dirz_reference_scale.csv` are in `prereg/`; `*_RESULTS.md`, `*_RUN_*.log` and the
two results records in `results/`; every other `.csv` in `scans/`.

## Published pages

- Précis: https://claude.ai/code/artifact/1dd88b53-e0c1-4417-9965-37fe88d8bdae
- Full record (with the disclosure section and 22 retractions): https://claude.ai/code/artifact/ef053c6d-84f8-44f2-b845-fae12f0671ec
- Slides download page: https://claude.ai/code/artifact/67b804b2-3b26-4ebd-a09a-117bd79ec212
