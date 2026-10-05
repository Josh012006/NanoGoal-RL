# The equivalent of the plots.py program but here, instead of showing an interactive
# view of the plots, they are automatically saved in the appropriate way.
# Was reviewed with CHATGPT and CLAUDE.

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DIFFICULTIES = ["easy", "medium", "hard"]
SEED_MODES   = ["easy", "medium", "hard", "mix"]

parser = argparse.ArgumentParser(
    description="Generate and save evaluation plots for a trained PPO model's "
                "results CSV (see eval.py), without popping up an interactive window."
)
parser.add_argument(
    "--model", required=True, choices=DIFFICULTIES,
    help="Difficulty the model was trained for."
)
parser.add_argument(
    "--seed", required=True, choices=SEED_MODES,
    help="Difficulty of the world seeds the model was evaluated on ('mix' combines all three)."
)
parser.add_argument(
    "--csv", required=False, default=None,
    help="Path to the evaluation CSV. Defaults to results/<model>/ppo_eval_<seed>.csv, "
         "matching eval.py's own output path convention."
)
parser.add_argument(
    "--output-folder", required=False, default=None,
    help="Folder to save the plots to. Defaults to plots/<model>."
)
args = parser.parse_args()

CSV_PATH      = args.csv or f"results/{args.model}/ppo_eval_{args.seed}.csv"
OUTPUT_FOLDER = args.output_folder or f"plots/{args.model}"
TEXT_MODE     = args.seed

# True exactly when the CSV is the model's "native" evaluation (world seeds of
# the same difficulty it was trained for) rather than a cross-difficulty test.
# Only native evaluations get the extra per-model diagnostic plots below
# (termination breakdown, episode length, regret) -- cross-evaluations (e.g.
# the easy model tested on hard seeds) only get the return/success/distance
# plots. Derived from --model/--seed instead of a hand-passed 0/1 flag, so
# there's no longer a raw boolean to mis-parse.
SAME_MODEL = args.model == args.seed

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

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


# ── 1) Return histogram ───────────────────────────────────────────────────────
# Replaces the return-vs-episode line chart — episode order is arbitrary in test.
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["return"], bins=15, color="#378ADD", edgecolor="#185FA5", alpha=0.85)
ax.set_xlabel("Return")
ax.set_ylabel("Episodes")
ax.set_title(f"Return distribution (test — {TEXT_MODE} seeds)")
mu  = df["return"].mean()
med = df["return"].median()
ax.axvline(mu,  color="#D85A30", linewidth=1.5, linestyle="--", label=f"mean {mu:.1f}")
ax.axvline(med, color="#1D9E75", linewidth=1.5, linestyle="--", label=f"median {med:.1f}")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUTPUT_FOLDER + "/return-episode-" + TEXT_MODE + ".png", dpi=150)
plt.close(fig)


# ── 2) Success rate metric + bar by distance bin ──────────────────────────────
# Overall rate as a large number + breakdown showing where the model struggles.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].axis("off")
rate = df["success"].astype(float).mean()
axes[0].text(0.5, 0.55, f"{rate:.0%}", fontsize=52, fontweight="bold",
             ha="center", va="center", transform=axes[0].transAxes, color="#1D9E75")
axes[0].text(0.5, 0.28, "success rate", fontsize=13,
             ha="center", va="center", transform=axes[0].transAxes, color="#5F5E5A")
axes[0].text(0.5, 0.18, f"n = {len(df)}", fontsize=10,
             ha="center", va="center", transform=axes[0].transAxes, color="#888780")

bins = pd.qcut(df["init_dist_goal"], q=5, duplicates="drop")
grp  = df.groupby(bins, observed=True)["success"].mean()
bar_colors = ["#1D9E75", "#5DCAA5", "#BA7517", "#D85A30", "#993C1D"][:len(grp)]
bars = axes[1].bar(range(len(grp)), grp.values * 100, color=bar_colors, edgecolor="none")
axes[1].set_xticks(range(len(grp)))
axes[1].set_xticklabels(["very close", "close", "mid", "far", "very far"][:len(grp)], fontsize=9)
axes[1].set_ylabel("Success rate (%)")
axes[1].set_ylim(0, 110)
axes[1].set_title(f"Success rate by initial distance ({TEXT_MODE} seeds)")
for bar, val in zip(bars, grp.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val * 100 + 2,
                 f"{val:.0%}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
fig.savefig(OUTPUT_FOLDER + "/success-episode-" + TEXT_MODE + ".png", dpi=150)
plt.close(fig)


# ── 3) Scatter: init distance vs best reached, colored by success ─────────────
# Replaces the 3-line distance chart — shows per-episode story more clearly.
fig, ax = plt.subplots(figsize=(7, 5))
success_mask = df["success"].astype(bool)
ax.scatter(df.loc[success_mask,  "init_dist_goal"],
           df.loc[success_mask,  "best_dist_goal"],
           color="#1D9E75", alpha=0.7, s=40,
           label=f"success (n={success_mask.sum()})", zorder=3)
ax.scatter(df.loc[~success_mask, "init_dist_goal"],
           df.loc[~success_mask, "best_dist_goal"],
           color="#D85A30", alpha=0.5, s=40,
           label=f"failure (n={(~success_mask).sum()})", zorder=2)
lim = max(df["init_dist_goal"].max(), df["best_dist_goal"].max()) * 1.05
ax.plot([0, lim], [0, lim], color="#888780", linewidth=1, linestyle="--",
        label="no progress (diagonal)")
ax.set_xlabel("Initial distance to goal")
ax.set_ylabel("Best distance reached")
ax.set_title(f"Initial distance vs best reached ({TEXT_MODE} seeds)")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUTPUT_FOLDER + "/distances-" + TEXT_MODE + ".png", dpi=150)
plt.close(fig)


# ── 4) Termination breakdown donut ────────────────────────────────────────────
# Replaces terminated/truncated line chart — shows episode outcomes clearly.
if SAME_MODEL:
    succ_n    = int(df["success"].sum())
    timeout_n = int(df["truncated"].sum())
    oob_n     = int((df["terminated"] & ~df["success"]).sum())
    total     = len(df)

    labels  = []
    sizes   = []
    colors  = []
    for label, n, color in [("success",       succ_n,    "#1D9E75"),
                              ("timeout",       timeout_n, "#BA7517"),
                              ("out of bounds", oob_n,     "#D85A30")]:
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
    fig.savefig(OUTPUT_FOLDER + "/terminated-truncated.png", dpi=150)
    plt.close(fig)


# ── 5) Episode length histogram ───────────────────────────────────────────────
if SAME_MODEL:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["length"], bins=15, color="#888780", edgecolor="#5F5E5A", alpha=0.85)
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel("Episodes")
    ax.set_title("Episode length distribution (test)")
    mu_l  = df["length"].mean()
    med_l = df["length"].median()
    ax.axvline(mu_l,  color="#D85A30", linewidth=1.5, linestyle="--", label=f"mean {mu_l:.0f}")
    ax.axvline(med_l, color="#1D9E75", linewidth=1.5, linestyle="--", label=f"median {med_l:.0f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_FOLDER + "/episode-length.png", dpi=150)
    plt.close(fig)


# ── 6) Regret histogram ───────────────────────────────────────────────────────
if SAME_MODEL:
    df["regret"] = df["final_dist_goal"] - df["best_dist_goal"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["regret"], bins=15, color="#BA7517", edgecolor="#854F0B", alpha=0.85)
    ax.axvline(0, color="#888780", linewidth=1, linestyle="--")
    ax.set_xlabel("Regret = final − best distance")
    ax.set_ylabel("Episodes")
    ax.set_title("Regret distribution (test) — how much progress is lost at end")
    mu_r = df["regret"].mean()
    ax.axvline(mu_r, color="#D85A30", linewidth=1.5, linestyle="--", label=f"mean {mu_r:.1f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_FOLDER + "/regret-episode.png", dpi=150)
    plt.close(fig)