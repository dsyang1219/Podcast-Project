import sys

sys.path.insert(0, ".")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

k_sweep = pd.read_csv("data/output/pilot_k_sweep_results.csv")
stability = pd.read_csv("data/output/pilot_stability_results.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for model, sub in k_sweep.groupby("model"):
    axes[0].plot(sub["k"], sub["c_v"], marker="o", label=model)
    axes[1].plot(sub["k"], sub["npmi"], marker="o", label=model)
axes[0].set_xlabel("K"); axes[0].set_ylabel("C_v coherence"); axes[0].set_title("C_v vs K"); axes[0].legend()
axes[1].set_xlabel("K"); axes[1].set_ylabel("NPMI coherence"); axes[1].set_title("NPMI vs K"); axes[1].legend()
fig.tight_layout()
fig.savefig("data/output/coherence_vs_k.png", dpi=150)
print("Saved data/output/coherence_vs_k.png")

fig2, axes2 = plt.subplots(1, 2, figsize=(9, 5))
axes2[0].bar(stability["model"], stability["cosine_mean"], yerr=stability["cosine_std"])
axes2[0].set_title("Topic stability across 5 seeds (K=20)\nbest-match cosine")
axes2[0].set_ylim(0, 1)
axes2[1].bar(stability["model"], stability["jaccard_mean"], yerr=stability["jaccard_std"])
axes2[1].set_title("Topic stability across 5 seeds (K=20)\nbest-match Jaccard (top-20 words)")
axes2[1].set_ylim(0, 1)
fig2.tight_layout()
fig2.savefig("data/output/stability_comparison.png", dpi=150)
print("Saved data/output/stability_comparison.png")

print("\n" + k_sweep.to_string(index=False))
print("\n" + stability.to_string(index=False))
