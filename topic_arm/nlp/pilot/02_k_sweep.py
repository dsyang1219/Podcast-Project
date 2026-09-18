import sys
import time

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from gensim.models import CoherenceModel

from nlp.pilot.common import (
    build_dictionary_and_dtm, fit_lda, fit_tlda, flag_degenerate_topics,
    k_eff_and_entropy, lda_doc_topic, lda_topic_word, load_tokenized_docs,
    tlda_doc_topic, tlda_topic_word, top_words,
)

SEED = 0
KS = [5, 10, 15, 20, 30, 40]

df, texts = load_tokenized_docs()
dictionary, bow_corpus, dtm, vocab = build_dictionary_and_dtm(texts)
print(f"{len(texts)} chunks, vocab={len(vocab)}")

rows = []
for k in KS:
    for model_name in ["LDA", "TLDA"]:
        t0 = time.perf_counter()
        if model_name == "LDA":
            model = fit_lda(dtm, k, SEED)
            topic_word = lda_topic_word(model)
            doc_topic = lda_doc_topic(model, dtm)
        else:
            model = fit_tlda(dtm, k, SEED)
            topic_word = tlda_topic_word(model)
            doc_topic = tlda_doc_topic(model, dtm)
        fit_time = time.perf_counter() - t0

        topics15 = top_words(topic_word, vocab, n_words=15)
        topics10 = [t[:10] for t in topics15]

        cv = CoherenceModel(topics=topics10, texts=texts, dictionary=dictionary, coherence="c_v").get_coherence()
        npmi = CoherenceModel(topics=topics10, texts=texts, dictionary=dictionary, coherence="c_npmi").get_coherence()

        k_eff, mean_entropy = k_eff_and_entropy(doc_topic)
        degeneracy = flag_degenerate_topics(doc_topic, topic_word, vocab)

        print(f"\n=== {model_name} K={k} (fit {fit_time:.1f}s) ===")
        print(f"  C_v={cv:.3f}  NPMI={npmi:.3f}  K_eff={k_eff:.2f}  mean_chunk_entropy={mean_entropy:.3f}")
        print(f"  near-empty topics: {degeneracy['near_empty_topics']}")
        print(f"  near-duplicate pairs (jaccard>=0.5): {degeneracy['near_duplicate_pairs']}")
        for i, words in enumerate(topics15):
            print(f"    topic {i}: {', '.join(words)}")

        rows.append({
            "model": model_name, "k": k, "fit_time_s": fit_time,
            "c_v": cv, "npmi": npmi, "k_eff": k_eff, "mean_chunk_entropy": mean_entropy,
            "n_near_empty": len(degeneracy["near_empty_topics"]),
            "n_near_duplicate_pairs": len(degeneracy["near_duplicate_pairs"]),
        })

out = pd.DataFrame(rows)
out.to_csv("data/output/pilot_k_sweep_results.csv", index=False)
print("\nSaved data/output/pilot_k_sweep_results.csv")
print(out.to_string(index=False))
