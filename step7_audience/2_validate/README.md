# 7.2 — validating the address measure

Shows that `dir_z` (listener-directed address) is a real, stable property of a show and that the ideology
labels it is compared against agree with two independent sources.

| script | what it establishes |
|---|---|
| `tri_strong.py` | The validation triangle at the listener level: the LLM side label, the host's DIME score and the party of the show's own listeners agree, across all 59 matched shows; plus a Pew cross-check. |
| `audience.py` | The label-versus-listeners row of that triangle on the expanded match (validation only). |
| `event_study.py` | Within-show event study around eight political shocks 2020-2025 with placebo dates: address does not move with the news, so it is a trait, not a reaction. |
| `topics_lda.py` | Topic invariance: within-show address is flat across the 75 topics of the topic-arm model. |
| `imp_decomp.py` | Breaks imperatives and second person into subtypes (epistemic / attention / action; deontic / addressive / filler) on a stratified corpus sample. Writes `inputs/show_imp_decomp.csv`. |
| `imp_decomp_holdout.py` | The same decomposition for the out-of-frame shows and The Majority Report. Writes `inputs/holdout_imp_decomp.csv`. |
| `you_split.py` | Separates guest-directed from listener-directed second person (questions, named persons, "you know", deontic) on holdout passages and a corpus sample. Writes `inputs/corpus_you_split.csv`, `inputs/holdout_you_split.csv`. |

The decomposition scripts load `step4_ideology/build_scoring_chunks.py` so passages are cut exactly as in the
corpus, and read episode-level features from `inputs/remeasured.csv`.
