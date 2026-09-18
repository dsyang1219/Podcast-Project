import sys

sys.path.insert(0, ".")
from nlp.pilot.common import build_dictionary_and_dtm, load_tokenized_docs

df, texts = load_tokenized_docs()
print(f"{len(texts)} chunks loaded")

dictionary, bow_corpus, dtm, vocab = build_dictionary_and_dtm(texts)
print(f"vocab size after no_below=3, no_above=0.99: {len(vocab)}")
print(f"DTM shape: {dtm.shape}, nnz={dtm.nnz}")

survives = "trump" in vocab
print(f"'trump' survives DF pruning: {survives}")
if survives:
    df_count = (dtm[:, list(vocab).index("trump")] > 0).sum()
    print(f"  'trump' document frequency: {df_count}/{dtm.shape[0]} = {df_count/dtm.shape[0]:.2%}")

for w in ["biden", "harris", "epstein", "gaza", "china", "court"]:
    print(f"'{w}' in vocab: {w in vocab}")
