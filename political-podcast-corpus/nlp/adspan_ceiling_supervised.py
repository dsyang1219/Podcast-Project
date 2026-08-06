"""Arm 3 -- the recoverability CEILING under direct supervision.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
---------------------------------------
This arm learns features TOWARD the ideology target. It therefore answers
"can ideology be extracted from this discourse if we try directly", which is a
DIFFERENT QUESTION from what the unsupervised arms answer ("is ideology a
natural axis of this discourse"). A higher R2 here is NOT evidence of a better
model of discourse, and it is NOT evidence about dominance. Do not place this
number in a ranking with the Arm 1/2 results as though bigger were better --
they are answers to different questions and the comparison is a category error.
The PCA finding that ideology is a low-variance/secondary axis is untouched.

TWO SUPERVISED MODELS
---------------------
1. Wordscores (Laver, Benoit & Garry 2003) -- the canonical supervised
   text-scaling estimator in political science. Word scores are estimated from
   TRAINING shows only; the held-out show's text never contributes to any word
   score, and the score->target mapping is an OLS fit on training shows only.
   Runs under the same outer LOSO folds as Arms 1-2, so its per-show errors are
   directly pairable with them.

2. A fine-tuned transformer regressor on the same mpnet backbone the
   unsupervised embedding arm uses, so the supervised/unsupervised contrast
   holds the encoder constant and varies only whether the target was seen.
   Chunk-level regression against the show's ideology, averaged to a show
   prediction. Outer CV is GROUPED 5-FOLD by show rather than LOSO -- 204
   fine-tunes is not affordable, and the task brief permits grouped k-fold.

   Its hyperparameters are FIXED A PRIORI (lr 2e-5, 2 epochs, batch 16), not
   tuned. That is deliberate: with no inner tuning loop there is nothing to
   overfit to the outer folds, so the number stays honest. It also means this
   is one configuration's performance, not a tuned ceiling, and it may
   understate what a fully tuned fine-tune could reach.

    .venv/bin/python -m nlp.adspan_ceiling_supervised --model wordscores
    .venv/bin/python -m nlp.adspan_ceiling_supervised --model transformer
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .adspan_ceiling_nested import (
    ADSPAN_DIR,
    CHUNKS_CLEAN,
    build_counts,
    evaluate,
    save,
)
from .adspan_target_robustness import build_targets
from .embed_ideology_nonlinear import bootstrap_r2_ci, loso_cv, r2_from_predictions

SEED = 0
MAX_LEN = 384
TRAIN_CHUNKS_PER_SHOW = 40
EVAL_CHUNKS_PER_SHOW = 100
N_FOLDS = 5


# ------------------------------------------------------------- wordscores ----
def wordscores_fit_predict(Xtr, ytr, Xte):
    """Laver-Benoit-Garry wordscores, estimated on training rows only.

    F_wr  = relative frequency of word w in reference doc r
    P(r|w)= F_wr / sum_r' F_w r'          (word's distribution over references)
    S_w   = sum_r P(r|w) * y_r            (word score)
    score = sum_w f_w_virgin * S_w        (virgin doc score)

    The raw virgin score is compressed toward the mean by construction, so a
    training-only OLS maps it back onto the target scale. Both the word scores
    and the mapping use training rows exclusively.
    """
    Ftr = np.asarray(Xtr.todense(), dtype=np.float64) if hasattr(Xtr, "todense") else np.asarray(Xtr, float)
    Fte = np.asarray(Xte.todense(), dtype=np.float64) if hasattr(Xte, "todense") else np.asarray(Xte, float)
    Ftr = Ftr / np.maximum(Ftr.sum(axis=1, keepdims=True), 1.0)
    Fte = Fte / np.maximum(Fte.sum(axis=1, keepdims=True), 1.0)

    col = Ftr.sum(axis=0)
    keep = col > 0
    P = Ftr[:, keep] / col[keep]
    S = P.T @ ytr                                  # word scores

    raw_tr = Ftr[:, keep] @ S
    raw_te = Fte[:, keep] @ S
    # OLS rescale on training rows only
    A = np.vstack([raw_tr, np.ones_like(raw_tr)]).T
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return raw_te * coef[0] + coef[1]


def run_wordscores(targets: dict[str, pd.Series]) -> dict:
    cnt_idx, counts, _ = build_counts()
    out = {"arm": 3, "model": "wordscores",
           "question": "RECOVERABILITY under direct supervision -- NOT dominance",
           "results": {}}
    for tname, y_series in targets.items():
        idx = pd.Index(sorted(y_series.index.intersection(cnt_idx)))
        pos = [cnt_idx.get_loc(s) for s in idx]
        X = counts[pos]
        y = y_series.loc[idx].to_numpy(dtype=float)
        print(f"\n[wordscores] target={tname} n={len(y)} vocab={X.shape[1]}")
        r = evaluate(X, y, wordscores_fit_predict, f"{tname}/wordscores")
        print(f"  outer R2 = {r['outer_r2']:+.3f} "
              f"[{r['ci_95'][0]:+.3f}, {r['ci_95'][1]:+.3f}]")
        out["results"][tname] = {"show_ids": list(idx), "model": r}
    return out


# ------------------------------------------------------------ transformer ----
def run_transformer(targets: dict[str, pd.Series], model_name: str,
                    epochs: int, batch_size: int, lr: float) -> dict:
    import torch
    from sklearn.model_selection import GroupKFold
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ch = pd.read_csv(CHUNKS_CLEAN, usecols=["collection_id", "text"],
                     dtype={"collection_id": str})
    ch["text"] = ch["text"].fillna("")
    tok = AutoTokenizer.from_pretrained(model_name)

    class DS(Dataset):
        def __init__(self, texts, ys):
            self.t, self.y = texts, ys

        def __len__(self):
            return len(self.t)

        def __getitem__(self, i):
            return self.t[i], self.y[i]

    def collate(b):
        txt, ys = zip(*b)
        enc = tok(list(txt), truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(ys, dtype=torch.float32)
        return enc

    out = {"arm": 3, "model": "transformer_finetune", "backbone": model_name,
           "cv": f"grouped {N_FOLDS}-fold by show (LOSO infeasible for fine-tuning)",
           "hyperparameters_fixed_a_priori": {"lr": lr, "epochs": epochs,
                                              "batch_size": batch_size,
                                              "max_len": MAX_LEN},
           "question": "RECOVERABILITY under direct supervision -- NOT dominance",
           "results": {}}

    for tname, y_series in targets.items():
        idx = pd.Index(sorted(y_series.index.intersection(set(ch.collection_id))))
        y_map = y_series.loc[idx].to_dict()
        sub = ch[ch.collection_id.isin(set(idx))].copy()
        sub["y"] = sub.collection_id.map(y_map)
        shows = np.array(sorted(idx))
        y_show = np.array([y_map[s] for s in shows], dtype=float)
        print(f"\n[transformer] target={tname} n_shows={len(shows)} "
              f"n_chunks={len(sub)} device={device}")

        rng = np.random.default_rng(SEED)
        preds_show = np.full(len(shows), np.nan)
        gkf = GroupKFold(n_splits=N_FOLDS)
        for fold, (tr, te) in enumerate(gkf.split(shows, groups=shows)):
            tr_shows, te_shows = set(shows[tr]), set(shows[te])
            tr_df = sub[sub.collection_id.isin(tr_shows)]
            tr_df = (tr_df.groupby("collection_id", group_keys=False)
                          .apply(lambda g: g.sample(min(len(g), TRAIN_CHUNKS_PER_SHOW),
                                                    random_state=SEED)))
            te_df = sub[sub.collection_id.isin(te_shows)]
            te_df = (te_df.groupby("collection_id", group_keys=False)
                          .apply(lambda g: g.sample(min(len(g), EVAL_CHUNKS_PER_SHOW),
                                                    random_state=SEED)))

            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            dl = DataLoader(DS(tr_df.text.tolist(), tr_df.y.to_numpy(float)),
                            batch_size=batch_size, shuffle=True, collate_fn=collate)
            model.train()
            for ep in range(epochs):
                for step, batch in enumerate(dl):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss = model(**batch).loss
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                print(f"  fold{fold} ep{ep} last_loss={loss.item():.4f}", flush=True)

            model.eval()
            edl = DataLoader(DS(te_df.text.tolist(), te_df.y.to_numpy(float)),
                             batch_size=batch_size * 2, shuffle=False, collate_fn=collate)
            chunk_pred = []
            with torch.no_grad():
                for batch in edl:
                    batch.pop("labels")
                    batch = {k: v.to(device) for k, v in batch.items()}
                    chunk_pred.append(model(**batch).logits.squeeze(-1).float().cpu().numpy())
            te_df = te_df.assign(pred=np.concatenate(chunk_pred))
            per_show = te_df.groupby("collection_id")["pred"].mean()
            for s, p in per_show.items():
                preds_show[np.searchsorted(shows, s)] = p
            del model
            torch.cuda.empty_cache()

        ok = ~np.isnan(preds_show)
        r2 = r2_from_predictions(y_show[ok], preds_show[ok])
        lo, hi = bootstrap_r2_ci(y_show[ok], preds_show[ok])
        print(f"  outer R2 = {r2:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        out["results"][tname] = {
            "show_ids": list(shows[ok]), "n": int(ok.sum()),
            "outer_r2": float(r2), "ci_95": [float(lo), float(hi)],
            "sq_err": ((y_show[ok] - preds_show[ok]) ** 2).tolist(),
            "preds": preds_show[ok].tolist()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["wordscores", "transformer"])
    ap.add_argument("--backbone", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    t = build_targets()
    targets = {"extended_all": t["extended_all"], "host_only": t["host_only"]}
    if args.model == "wordscores":
        save("ceiling_nested_arm3_wordscores.json", run_wordscores(targets))
    else:
        save("ceiling_nested_arm3_transformer.json",
             run_transformer(targets, args.backbone, args.epochs,
                             args.batch_size, args.lr))


if __name__ == "__main__":
    main()
