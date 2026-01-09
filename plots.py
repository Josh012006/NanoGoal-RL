import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/ppo_eval.csv"
ROLL = 50                    
EPS = 1e-8

# ---------- load ----------
df = pd.read_csv(CSV_PATH)

# Ensure expected columns exist
cols = [
    "episode","return","length","success","terminated","truncated",
    "init_dist_goal","best_dist_goal","final_dist_goal"
]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Sort by episode 
df = df.sort_values("episode").reset_index(drop=True)

# Convert booleans if they are strings
for c in ["success", "terminated", "truncated"]:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])

# Derived metrics
df["progress"] = (df["init_dist_goal"] - df["final_dist_goal"]) / np.maximum(df["init_dist_goal"], EPS)
df["regret"] = df["final_dist_goal"] - df["best_dist_goal"]

def rolling_mean(s, w=ROLL):
    return s.rolling(w, min_periods=max(5, w//5)).mean()

# ---------- 1) Return ----------
plt.figure()
plt.plot(df["episode"], df["return"], alpha=0.35)
plt.plot(df["episode"], rolling_mean(df["return"]), linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Return vs Episode (raw + rolling mean)")
plt.show()

# ---------- 2) Success rate ----------
plt.figure()
plt.plot(df["episode"], rolling_mean(df["success"].astype(float)), linewidth=2)
plt.ylim(-0.05, 1.05)
plt.xlabel("Episode")
plt.ylabel("Success rate (rolling)")
plt.title("Success Rate vs Episode")
plt.show()

# ---------- 3) Episode length ----------
plt.figure()
plt.plot(df["episode"], df["length"], alpha=0.35)
plt.plot(df["episode"], rolling_mean(df["length"]), linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Length")
plt.title("Episode Length vs Episode")
plt.show()

# ---------- 4) Terminated vs Truncated rates ----------
plt.figure()
plt.plot(df["episode"], rolling_mean(df["terminated"].astype(float)), linewidth=2, label="terminated (rolling)")
plt.plot(df["episode"], rolling_mean(df["truncated"].astype(float)), linewidth=2, label="truncated (rolling)")
plt.ylim(-0.05, 1.05)
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.title("Terminated vs Truncated Rate")
plt.legend()
plt.show()

# ---------- 5) Distances ----------
plt.figure()
plt.plot(df["episode"], rolling_mean(df["init_dist_goal"]), linewidth=2, label="init (rolling)")
plt.plot(df["episode"], rolling_mean(df["best_dist_goal"]), linewidth=2, label="best (rolling)")
plt.plot(df["episode"], rolling_mean(df["final_dist_goal"]), linewidth=2, label="final (rolling)")
plt.xlabel("Episode")
plt.ylabel("Distance to goal")
plt.title("Distances vs Episode (rolling)")
plt.legend()
plt.show()

# ---------- 6) Progress ----------
plt.figure()
plt.plot(df["episode"], df["progress"], alpha=0.35)
plt.plot(df["episode"], rolling_mean(df["progress"]), linewidth=2)
plt.axhline(0.0, linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Progress (normalized)")
plt.title("Progress vs Episode")
plt.show()

# ---------- 7) Regret ----------
plt.figure()
plt.plot(df["episode"], df["regret"], alpha=0.35)
plt.plot(df["episode"], rolling_mean(df["regret"]), linewidth=2)
plt.axhline(0.0, linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Regret = final - best")
plt.title("Regret vs Episode (how often it loses progress)")
plt.show()

# ---------- 8) Return vs Progress scatter ----------
plt.figure()
plt.scatter(df["progress"], df["return"], alpha=0.35, s=12)
plt.xlabel("Progress")
plt.ylabel("Return")
plt.title("Return vs Progress (reward alignment check)")
plt.show()

# ---------- 9) Success vs init distance (binned) ----------
# Bin init distances into quantiles (robust)
bins = pd.qcut(df["init_dist_goal"], q=6, duplicates="drop")
grp = df.groupby(bins)["success"].mean()

plt.figure()
plt.plot(range(len(grp)), grp.values, marker="o")
plt.ylim(-0.05, 1.05)
plt.xlabel("Init distance bin (quantiles)")
plt.ylabel("Success rate")
plt.title("Success rate vs Init Distance (binned)")
plt.show()
