# A file to help plot different relationships in the results of the models' testing.
# Was generated with AI.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


if len(sys.argv) < 2:
    raise ValueError("File path needed!")

CSV_PATH = sys.argv[1]

# ---------- load ----------
df = pd.read_csv(CSV_PATH)

cols = [
    "episode", "return", "length", "success", "terminated", "truncated",
    "init_dist_goal", "best_dist_goal", "final_dist_goal"
]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df = df.sort_values("episode").reset_index(drop=True)

for c in ["success", "terminated", "truncated"]:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])

df["progress_ratio"] = np.where(
    df["init_dist_goal"] > 0,
    (df["init_dist_goal"] - df["best_dist_goal"]) / df["init_dist_goal"],
    0.0
).clip(0.0, 1.0)


# ---------- 1) Return histogram ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["return"], bins=15, color="#378ADD", edgecolor="#185FA5", alpha=0.85)
ax.set_xlabel("Return")
ax.set_ylabel("Episodes")
ax.set_title("Return distribution (test)")
mu = df["return"].mean()
med = df["return"].median()
ax.axvline(mu,  color="#D85A30", linewidth=1.5, linestyle="--", label=f"mean {mu:.1f}")
ax.axvline(med, color="#1D9E75", linewidth=1.5, linestyle="--", label=f"median {med:.1f}")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


# ---------- 2) Success rate bar by distance bin ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: overall metric as large text
axes[0].axis("off")
rate = df["success"].astype(float).mean()
axes[0].text(0.5, 0.55, f"{rate:.0%}", fontsize=52, fontweight="bold",
             ha="center", va="center", transform=axes[0].transAxes, color="#1D9E75")
axes[0].text(0.5, 0.28, "success rate", fontsize=13,
             ha="center", va="center", transform=axes[0].transAxes, color="#5F5E5A")
axes[0].text(0.5, 0.18, f"n = {len(df)}", fontsize=10,
             ha="center", va="center", transform=axes[0].transAxes, color="#888780")

# Right: success rate by distance bin
bins = pd.qcut(df["init_dist_goal"], q=5, duplicates="drop")
grp = df.groupby(bins, observed=True)["success"].mean()
colors = ["#1D9E75", "#5DCAA5", "#BA7517", "#D85A30", "#993C1D"][:len(grp)]
bars = axes[1].bar(range(len(grp)), grp.values * 100, color=colors, edgecolor="none")
axes[1].set_xticks(range(len(grp)))
axes[1].set_xticklabels(["very close", "close", "mid", "far", "very far"][:len(grp)],
                         fontsize=9)
axes[1].set_ylabel("Success rate (%)")
axes[1].set_ylim(0, 110)
axes[1].set_title("Success rate by initial distance")
for bar, val in zip(bars, grp.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val * 100 + 2,
                 f"{val:.0%}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()


# ---------- 3) Scatter: init distance vs best reached, colored by success ----------
fig, ax = plt.subplots(figsize=(7, 5))
success_mask = df["success"].astype(bool)
ax.scatter(df.loc[success_mask, "init_dist_goal"],
           df.loc[success_mask, "best_dist_goal"],
           color="#1D9E75", alpha=0.7, s=40, label=f"success (n={success_mask.sum()})", zorder=3)
ax.scatter(df.loc[~success_mask, "init_dist_goal"],
           df.loc[~success_mask, "best_dist_goal"],
           color="#D85A30", alpha=0.5, s=40, label=f"failure (n={(~success_mask).sum()})", zorder=2)
lim = max(df["init_dist_goal"].max(), df["best_dist_goal"].max()) * 1.05
ax.plot([0, lim], [0, lim], color="#888780", linewidth=1, linestyle="--",
        label="no progress (diagonal)")
ax.set_xlabel("Initial distance to goal")
ax.set_ylabel("Best distance reached")
ax.set_title("Initial distance vs best reached (per episode)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


# ---------- 4) Termination breakdown donut ----------
succ_n   = int(df["success"].sum())
timeout_n = int(df["truncated"].sum())
oob_n    = int((df["terminated"] & ~df["success"]).sum())
total    = len(df)

labels = []
sizes  = []
colors = []
for label, n, color in [("success", succ_n, "#1D9E75"),
                         ("timeout", timeout_n, "#BA7517"),
                         ("out of bounds", oob_n, "#D85A30")]:
    if n > 0:
        labels.append(f"{label}\n{n/total:.0%}")
        sizes.append(n)
        colors.append(color)

fig, ax = plt.subplots(figsize=(6, 5))
wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                   wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2})
ax.legend(wedges, labels, loc="lower center", ncol=len(labels),
          fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.08))
ax.set_title("Episode termination breakdown")
plt.tight_layout()
plt.show()