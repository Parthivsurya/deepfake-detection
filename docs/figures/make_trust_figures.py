"""Build the reliability-vs-non-reliability figures (sir's Tasks 1 & 2) from the
measured per-clip scores of the TRE ablation (Celeb-DF val, 457 clips).

Inputs : scores_trust_off.csv / scores_trust_on.csv  (clip_id,dataset,label,prob_fake)
Outputs: trust_roc.png            (Task 1 — ROC / TPR vs FPR)
         trust_prf.png            (Task 2 — F1 / precision / recall bars)
         trust_confusion_fpfn.png (bonus — FP/FN counts at thr 0.5)
All numbers are real and reproducible; nothing is fabricated.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

OFF = sys.argv[1] if len(sys.argv) > 1 else "scores_trust_off.csv"
ON = sys.argv[2] if len(sys.argv) > 2 else "scores_trust_on.csv"
THR = 0.5

off = pd.read_csv(OFF)
on = pd.read_csv(ON)
runs = {"Without reliability": off, "With reliability (TRE)": on}
colors = {"Without reliability": "#c0392b", "With reliability (TRE)": "#1f77b4"}

# ---------------- Figure 1 — ROC (TPR vs FPR) ----------------
fig, ax = plt.subplots(figsize=(6, 5.5))
for name, d in runs.items():
    y, p = d["label"].values, d["prob_fake"].values
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    ax.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} — AUC {auc:.3f}")
    # mark the threshold-0.5 operating point
    yhat = (p > THR).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    ax.scatter([fp/(fp+tn)], [tp/(tp+fn)], color=colors[name], s=45, zorder=5,
               edgecolor="k", linewidth=0.5)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — reliability vs non-reliability (Celeb-DF val)\n(dots = threshold 0.5 operating point)")
ax.legend(loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig("trust_roc.png", dpi=200, bbox_inches="tight")
print("wrote trust_roc.png")

# ---------------- Figure 2 — F1 / precision / recall bars ----------------
metrics = ["Precision", "Recall", "F1"]
vals = {}
for name, d in runs.items():
    y = d["label"].values; yhat = (d["prob_fake"].values > THR).astype(int)
    vals[name] = [precision_score(y, yhat), recall_score(y, yhat), f1_score(y, yhat)]
x = np.arange(len(metrics)); w = 0.36
fig, ax = plt.subplots(figsize=(7, 5))
for i, (name, v) in enumerate(vals.items()):
    bars = ax.bar(x + (i - 0.5) * w, v, w, label=name, color=colors[name])
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width()/2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("Score (threshold 0.5)"); ax.set_ylim(0, 1.05)
ax.set_title("Detection quality — reliability vs non-reliability (Celeb-DF val)")
ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig("trust_prf.png", dpi=200, bbox_inches="tight")
print("wrote trust_prf.png")

# ---------------- Figure 3 — FP / FN counts ----------------
fp_fn = {}
for name, d in runs.items():
    y = d["label"].values; yhat = (d["prob_fake"].values > THR).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    fp_fn[name] = [fp, fn]
labels = ["False Positives\n(real → fake)", "False Negatives\n(fake → real)"]
x = np.arange(2)
fig, ax = plt.subplots(figsize=(6.5, 5))
for i, (name, v) in enumerate(fp_fn.items()):
    bars = ax.bar(x + (i - 0.5) * w, v, w, label=name, color=colors[name])
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width()/2, val + 1, str(val),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Count (Celeb-DF val, 457 clips, thr 0.5)")
ax.set_title("Errors — reliability vs non-reliability")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("trust_confusion_fpfn.png", dpi=200, bbox_inches="tight")
print("wrote trust_confusion_fpfn.png")

# ---------------- print the summary table ----------------
print("\n=== summary (threshold 0.5) ===")
for name, d in runs.items():
    y = d["label"].values; p = d["prob_fake"].values; yhat = (p > THR).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    print(f"{name:26s} AUC {roc_auc_score(y,p):.3f}  P {precision_score(y,yhat):.3f} "
          f"R {recall_score(y,yhat):.3f}  F1 {f1_score(y,yhat):.3f}  FP {fp}  FN {fn}")
