# A file to help plot different relationships in the results of the models' testing.
# Was generated with CHATGPT.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

CSV_PATH = sys.argv[1]
if CSV_PATH is None:
    raise ValueError("File path needed !")

ROLL = 50
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

df["progress"] = (df["init_dist_goal"] - df["final_dist_goal"]) / np.maximum(df["init_dist_goal"], EPS)
df["regret"] = df["final_dist_goal"] - df["best_dist_goal"]

def rolling_mean(s, w=ROLL):
    return s.rolling(w, min_periods=max(5, w//5)).mean()

# ---------- robust plot helpers (for outliers) ----------
def robust_ylim(y, q_low=0.05, q_high=0.95, pad=0.10):
    """
    y-limits based on quantiles so extreme outliers don't dictate the scale.
    pad is relative padding added on both sides.
    """
    y = pd.Series(y).dropna()
    if len(y) == 0:
        return None
    lo, hi = y.quantile([q_low, q_high]).to_numpy()
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        # fallback if constant series
        lo -= 1.0
        hi += 1.0
    r = hi - lo
    return lo - pad * r, hi + pad * r

def annotate_outliers(ax, x, y, q_low=0.05, q_high=0.95, max_marks=6):
    """
    Adds a note with how many points are outside quantile range.
    Marks a few extreme outliers ONLY if x and y have same length.
    """
    y_s = pd.Series(y).dropna()
    if len(y_s) == 0:
        return

    lo, hi = y_s.quantile([q_low, q_high]).to_numpy()

    # If x is provided, ensure same length for marking
    can_mark = (x is not None) and (len(x) == len(y))

    # Build mask on the original (non-dropna) shape for consistency
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

    # mark a few most extreme points (by distance to nearest bound)
    yv = y_full.to_numpy()
    d = np.where(yv < lo, lo - yv, yv - hi)
    d = np.where(mask.to_numpy(), d, -np.inf)  # ignore non-outliers
    idx = np.argsort(d)[::-1][:min(max_marks, n_out)]

    ax.scatter(np.asarray(x)[idx], np.asarray(y)[idx], s=20, zorder=3)

def apply_robust_view(ax, y, q_low=0.05, q_high=0.95, pad=0.10, show_outlier_note=True, x=None):
    lim = robust_ylim(y, q_low=q_low, q_high=q_high, pad=pad)
    if lim is not None:
        ax.set_ylim(*lim)
    if show_outlier_note:
        annotate_outliers(ax, x, y, q_low=q_low, q_high=q_high)


# You can tweak these once and all "raw+rolling" plots benefit.
QLOW, QHIGH = 0.05, 0.95
PAD = 0.10

# ---------- 1) Return ----------
fig, ax = plt.subplots()
ax.plot(df["episode"], df["return"], alpha=0.35)
ax.plot(df["episode"], rolling_mean(df["return"]), linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Return")
ax.set_title("Return vs Episode (raw + rolling mean)")
apply_robust_view(ax, df["return"], q_low=QLOW, q_high=QHIGH, pad=PAD, x=df["episode"])
plt.show()

# ---------- 2) Success rate ----------
fig, ax = plt.subplots()
ax.plot(df["episode"], rolling_mean(df["success"].astype(float)), linewidth=2)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Episode")
ax.set_ylabel("Success rate (rolling)")
ax.set_title("Success Rate vs Episode")
plt.show()

# ---------- 3) Episode length ----------
fig, ax = plt.subplots()
ax.plot(df["episode"], df["length"], alpha=0.35)
ax.plot(df["episode"], rolling_mean(df["length"]), linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Length")
ax.set_title("Episode Length vs Episode")
apply_robust_view(ax, df["length"], q_low=QLOW, q_high=QHIGH, pad=PAD, x=df["episode"])
plt.show()

# ---------- 4) Terminated vs Truncated rates ----------
fig, ax = plt.subplots()
ax.plot(df["episode"], rolling_mean(df["terminated"].astype(float)), linewidth=2, label="terminated (rolling)")
ax.plot(df["episode"], rolling_mean(df["truncated"].astype(float)), linewidth=2, label="truncated (rolling)")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Episode")
ax.set_ylabel("Rate")
ax.set_title("Terminated vs Truncated Rate")
ax.legend()
plt.show()

# ---------- 5) Distances ----------
fig, ax = plt.subplots()
ax.plot(df["episode"], rolling_mean(df["init_dist_goal"]), linewidth=2, label="init (rolling)")
ax.plot(df["episode"], rolling_mean(df["best_dist_goal"]), linewidth=2, label="best (rolling)")
ax.plot(df["episode"], rolling_mean(df["final_dist_goal"]), linewidth=2, label="final (rolling)")
ax.set_xlabel("Episode")
ax.set_ylabel("Distance to goal")
ax.set_title("Distances vs Episode (rolling)")
ax.legend()
# robust y-limits from all distance series together, but no outlier markers
all_d = pd.concat(
    [
        rolling_mean(df["init_dist_goal"]),
        rolling_mean(df["best_dist_goal"]),
        rolling_mean(df["final_dist_goal"]),
    ],
    ignore_index=True
)
apply_robust_view(ax, all_d, q_low=QLOW, q_high=QHIGH, pad=PAD, x=None)
plt.show()

# ---------- 6) Progress ----------
fig, ax = plt.subplots()
ax.plot(df["episode"], df["progress"], alpha=0.35)
ax.plot(df["episode"], rolling_mean(df["progress"]), linewidth=2)
ax.axhline(0.0, linewidth=1)
ax.set_xlabel("Episode")
ax.set_ylabel("Progress (normalized)")
ax.set_title("Progress vs Episode")
apply_robust_view(ax, df["progress"], q_low=QLOW, q_high=QHIGH, pad=PAD, x=df["episode"])
plt.show()

# ---------- 7) Regret ----------
fig, ax = plt.subplots()
ax.plot(df["episode"], df["regret"], alpha=0.35)
ax.plot(df["episode"], rolling_mean(df["regret"]), linewidth=2)
ax.axhline(0.0, linewidth=1)
ax.set_xlabel("Episode")
ax.set_ylabel("Regret = final - best")
ax.set_title("Regret vs Episode (how often it loses progress)")
apply_robust_view(ax, df["regret"], q_low=QLOW, q_high=QHIGH, pad=PAD, x=df["episode"])
plt.show()

# ---------- 8) Return vs Progress scatter ----------
plt.figure()
plt.scatter(df["progress"], df["return"], alpha=0.35, s=12)
plt.xlabel("Progress")
plt.ylabel("Return")
plt.title("Return vs Progress (reward alignment check)")
plt.show()

# ---------- 9) Success vs init distance (binned) ----------
bins = pd.qcut(df["init_dist_goal"], q=6, duplicates="drop")
grp = df.groupby(bins)["success"].mean()

plt.figure()
plt.plot(range(len(grp)), grp.values, marker="o")
plt.ylim(-0.05, 1.05)
plt.xlabel("Init distance bin (quantiles)")
plt.ylabel("Success rate")
plt.title("Success rate vs Init Distance (binned)")
plt.show()
