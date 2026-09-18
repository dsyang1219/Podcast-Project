import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nlp.pilot.common import (
    build_dictionary_and_dtm, correspondence_analysis_d_eff, fit_lda, fit_tlda,
    lda_doc_topic, load_tokenized_docs, tlda_doc_topic,
)

K = 20
SEED = 0
MIN_CHUNKS = 8

df, texts = load_tokenized_docs()
dictionary, bow_corpus, dtm, vocab = build_dictionary_and_dtm(texts)

lda = fit_lda(dtm, K, SEED)
lda_theta = lda_doc_topic(lda, dtm)

tlda = fit_tlda(dtm, K, SEED)
tlda_theta = tlda_doc_topic(tlda, dtm)

show_counts = df["show"].value_counts()
shows = show_counts[show_counts >= MIN_CHUNKS].index.tolist()
print(f"{len(shows)} shows with >= {MIN_CHUNKS} chunks: {shows}")

rows = []
for show in shows:
    idx = df.index[df["show"] == show].to_numpy()
    d_eff_lda = correspondence_analysis_d_eff(lda_theta[idx])
    d_eff_tlda = correspondence_analysis_d_eff(tlda_theta[idx])
    rows.append({"show": show, "n_chunks": len(idx), "d_eff_lda": d_eff_lda, "d_eff_tlda": d_eff_tlda})

out = pd.DataFrame(rows).sort_values("n_chunks", ascending=False)
print(out.to_string(index=False))

nan_lda = out["d_eff_lda"].isna().sum()
nan_tlda = out["d_eff_tlda"].isna().sum()
print(f"\nNaN d_eff: LDA={nan_lda}, TLDA={nan_tlda}")
print(f"d_eff varies across shows (LDA): std={out['d_eff_lda'].std():.3f}, range=[{out['d_eff_lda'].min():.2f}, {out['d_eff_lda'].max():.2f}]")
print(f"d_eff varies across shows (TLDA): std={out['d_eff_tlda'].std():.3f}, range=[{out['d_eff_tlda'].min():.2f}, {out['d_eff_tlda'].max():.2f}]")

rho, p = spearmanr(out["d_eff_lda"], out["d_eff_tlda"])
print(f"\nSpearman rank correlation of show d_eff rankings (LDA vs TLDA): rho={rho:.3f} (p={p:.3f})")

out.to_csv("data/output/pilot_d_eff_sanity.csv", index=False)
print("\nSaved data/output/pilot_d_eff_sanity.csv")
