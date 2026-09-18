import itertools
import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from nlp.pilot.common import (
    best_match_alignment, build_dictionary_and_dtm, fit_lda, fit_tlda,
    jaccard_best_match, lda_topic_word, load_tokenized_docs, tlda_topic_word,
    top_words,
)

K = 20
SEEDS = [0, 1, 2, 3, 4]

df, texts = load_tokenized_docs()
dictionary, bow_corpus, dtm, vocab = build_dictionary_and_dtm(texts)
print(f"{len(texts)} chunks, vocab={len(vocab)}, K={K}, seeds={SEEDS}")

results = {"LDA": [], "TLDA": []}
for model_name in ["LDA", "TLDA"]:
    for seed in SEEDS:
        if model_name == "LDA":
            model = fit_lda(dtm, K, seed)
            tw = lda_topic_word(model)
        else:
            model = fit_tlda(dtm, K, seed)
            tw = tlda_topic_word(model)
        results[model_name].append(tw)

rows = []
for model_name, topic_word_runs in results.items():
    cosine_scores = []
    jaccard_scores = []
    for i, j in itertools.combinations(range(len(SEEDS)), 2):
        row_ind, col_ind, sim = best_match_alignment(topic_word_runs[i], topic_word_runs[j])
        cosine_scores.append(sim[row_ind, col_ind].mean())

        words_i = top_words(topic_word_runs[i], vocab, n_words=20)
        words_j = top_words(topic_word_runs[j], vocab, n_words=20)
        jaccard_scores.append(jaccard_best_match(words_i, words_j))

    print(f"\n=== {model_name} stability across {len(SEEDS)} seeds (K={K}) ===")
    print(f"  best-match cosine: mean={np.mean(cosine_scores):.3f} std={np.std(cosine_scores):.3f}")
    print(f"  best-match Jaccard(top20): mean={np.mean(jaccard_scores):.3f} std={np.std(jaccard_scores):.3f}")
    rows.append({
        "model": model_name, "k": K, "n_seeds": len(SEEDS),
        "cosine_mean": np.mean(cosine_scores), "cosine_std": np.std(cosine_scores),
        "jaccard_mean": np.mean(jaccard_scores), "jaccard_std": np.std(jaccard_scores),
    })

out = pd.DataFrame(rows)
out.to_csv("data/output/pilot_stability_results.csv", index=False)
print("\nSaved data/output/pilot_stability_results.csv")
print(out.to_string(index=False))
