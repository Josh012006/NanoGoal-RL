# A file to help plot different relationships in the results of the models' testing.
# Was generated with CHATGPT.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

CSV_PATH = sys.argv[1]
if CSV_PATH is None:
    raise ValueError("File path needed !")

EPS = 1e-8

# ---------- load ----------
df = pd.read_csv(CSV_PATH)

cols = [
    "episode","return","length","success","terminated","truncated",
    "init_dist_goal","best_dist_goal","final_dist_goal"
]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df = df.sort_values("episode").reset_index(drop=True)

for c in ["success", "terminated", "truncated"]:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])

# Derived metrics
df["regret"] = df["final_dist_goal"] - df["best_dist_goal"]

def running_mean(s: pd.Series) -> pd.Series:
    """Cumulative mean: meaningful for i.i.d. test episodes (convergence of the mean)."""
    s = s.astype(float)
    return s.expanding(min_periods=1).mean()

# ---------- robust plot helpers (for outliers) ----------
def robust_ylim(y, q_low=0.05, q_high=0.95, pad=0.10):
    y = pd.Series(y).dropna()
    if len(y) == 0:
        return None
    lo, hi = y.quantile([q_low, q_high]).to_numpy()
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    r = hi - lo
    return lo - pad * r, hi + pad * r

def annotate_outliers(ax, x, y, q_low=0.05, q_high=0.95, max_marks=6):
    y_s = pd.Series(y).dropna()
    if len(y_s) == 0:
        return

    lo, hi = y_s.quantile([q_low, q_high]).to_numpy()

    can_mark = (x is not None) and (len(x) == len(y))

    y_full = pd.Series(y)
    mask = (y_full < lo) | (y_full > hi)
    n_out = int(mask.sum())
    if n_out == 0:
        return

    ax.text(
        0.01, 0.01,
        f"{n_out} outliers outside [{q_low:.0%}, {q_high:.0%}] quantiles",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom"
    )

    if not can_mark:
        return

    yv = y_full.to_numpy()
    d = np.where(yv < lo, lo - yv, yv - hi)
    d = np.where(mask.to_numpy(), d, -np.inf)
    idx = np.argsort(d)[::-1][:min(max_marks, n_out)]

    ax.scatter(np.asarray(x)[idx], np.asarray(y)[idx], s=20, zorder=3)

def apply_robust_view(ax, y, q_low=0.05, q_high=0.95, pad=0.10, show_outlier_note=True, x=None):
    lim = robust_ylim(y, q_low=q_low, q_high=q_high, pad=pad)
    if lim is not None:
        ax.set_ylim(*lim)
    if show_outlier_note:
        annotate_outliers(ax, x, y, q_low=q_low, q_high=q_high)

QLOW, QHIGH = 0.05, 0.95
PAD = 0.10

# ---------- helpers for test summaries ----------
def summary_text(s: pd.Series, name: str) -> str:
    s = s.dropna().astype(float)
    if len(s) == 0:
        return f"{name}: empty"
    q05, q50, q95 = s.quantile([0.05, 0.50, 0.95]).to_numpy()
    mu = s.mean()
    return f"{name}: mean={mu:.3f} | median={q50:.3f} | q05={q05:.3f} | q95={q95:.3f} | n={len(s)}"

# ---------- 1) Return (raw + running mean) ----------
fig, ax = plt.subplots()
ax.plot(df["episode"], df["return"], alpha=0.35, label="raw")
ax.plot(df["episode"], running_mean(df["return"]), linewidth=2, label="running mean")
ax.set_xlabel("Episode (arbitrary order in test)")
ax.set_ylabel("Return")
ax.set_title("Return (test): raw + running mean (convergence)")
ax.legend()
apply_robust_view(ax, df["return"], q_low=QLOW, q_high=QHIGH, pad=PAD, x=df["episode"])
ax.text(0.01, 0.98, summary_text(df["return"], "return"), transform=ax.transAxes,
        fontsize=9, verticalalignment="top")
plt.show()

# ---------- 2) Success rate (cumulative) ----------
fig, ax = plt.subplots()
succ = df["success"].astype(float)
ax.plot(df["episode"], running_mean(succ), linewidth=2)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Episode (arbitrary order in test)")
ax.set_ylabel("Success rate (cumulative)")
ax.set_title("Success rate (test): cumulative mean")
ax.text(0.01, 0.98, f"success rate = {succ.mean():.3f} (n={len(succ)})",
        transform=ax.transAxes, fontsize=9, verticalalignment="top")
plt.show()

# ---------- 3) Episode length (raw + running mean) ----------
fig, ax = plt.subplots()
ax.plot(df["episode"], df["length"], alpha=0.35, label="raw")
ax.plot(df["episode"], running_mean(df["length"]), linewidth=2, label="running mean")
ax.set_xlabel("Episode (arbitrary order in test)")
ax.set_ylabel("Length")
ax.set_title("Episode length (test): raw + running mean")
ax.legend()
apply_robust_view(ax, df["length"], q_low=QLOW, q_high=QHIGH, pad=PAD, x=df["episode"])
ax.text(0.01, 0.98, summary_text(df["length"], "length"), transform=ax.transAxes,
        fontsize=9, verticalalignment="top")
plt.show()

# ---------- 4) Terminated vs Truncated rates (global + cumulative) ----------
fig, ax = plt.subplots()
term = df["terminated"].astype(float)
trun = df["truncated"].astype(float)
ax.plot(df["episode"], running_mean(term), linewidth=2, label="terminated (cumulative)")
ax.plot(df["episode"], running_mean(trun), linewidth=2, label="truncated (cumulative)")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Episode (arbitrary order in test)")
ax.set_ylabel("Rate (cumulative)")
ax.set_title("Terminated vs Truncated (test): cumulative means")
ax.legend()
ax.text(
    0.01, 0.98,
    f"terminated={term.mean():.3f} | truncated={trun.mean():.3f} | n={len(df)}",
    transform=ax.transAxes, fontsize=9, verticalalignment="top"
)
plt.show()

# ---------- 5) Distances (distributions + cumulative mean) ----------
# In test, plotting distances vs episode is not a temporal story.
# We still can show cumulative mean convergence + robust view.
fig, ax = plt.subplots()
ax.plot(df["episode"], running_mean(df["init_dist_goal"]), linewidth=2, label="init (cumul mean)")
ax.plot(df["episode"], running_mean(df["best_dist_goal"]), linewidth=2, label="best (cumul mean)")
ax.plot(df["episode"], running_mean(df["final_dist_goal"]), linewidth=2, label="final (cumul mean)")
ax.set_xlabel("Episode (arbitrary order in test)")
ax.set_ylabel("Distance to goal")
ax.set_title("Distances (test): cumulative mean convergence")
ax.legend()

# robust y-limits based on all three distance series
all_d = pd.concat(
    [df["init_dist_goal"], df["best_dist_goal"], df["final_dist_goal"]],
    ignore_index=True
)
apply_robust_view(ax, all_d, q_low=QLOW, q_high=QHIGH, pad=PAD, x=None)
plt.show()

# ---------- 6) Regret (raw + running mean) ----------
fig, ax = plt.subplots()
ax.plot(df["episode"], df["regret"], alpha=0.35, label="raw")
ax.plot(df["episode"], running_mean(df["regret"]), linewidth=2, label="running mean")
ax.axhline(0.0, linewidth=1)
ax.set_xlabel("Episode (arbitrary order in test)")
ax.set_ylabel("Regret = final - best")
ax.set_title("Regret (test): raw + running mean")
ax.legend()
apply_robust_view(ax, df["regret"], q_low=QLOW, q_high=QHIGH, pad=PAD, x=df["episode"])
ax.text(0.01, 0.98, summary_text(df["regret"], "regret"), transform=ax.transAxes,
        fontsize=9, verticalalignment="top")
plt.show()


# ---------- 7) Success vs init distance (binned) ----------
bins = pd.qcut(df["init_dist_goal"], q=6, duplicates="drop")
grp = df.groupby(bins)["success"].mean()

plt.figure()
plt.plot(range(len(grp)), grp.values, marker="o")
plt.ylim(-0.05, 1.05)
plt.xlabel("Init distance bin (quantiles)")
plt.ylabel("Success rate")
plt.title("Success rate vs Init Distance (binned)")
plt.show()
