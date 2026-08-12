"""
sanity_check_seeds.py
Run this IMMEDIATELY after classify_seeds.py, in the SAME terminal session,
BEFORE committing seeds.json. It spot-checks 30 random "easy" seeds by
recomputing their deviation from scratch and comparing to the threshold.

If this reports 0 failures but the droplet/CI later shows contamination,
that proves the discrepancy happens between your local run and what gets
committed/pulled (e.g. uncommitted changes, stale seeds.json from a prior
run, or a git issue) rather than in the classification algorithm itself.

Usage: python sanity_check_seeds.py
"""
import json
import numpy as np
import env
from classify_seeds import astar_path_and_deviation, EASY_MAX_DEG

with open("seeds.json") as f:
    seeds = json.load(f)

easy_seeds = seeds["easy"]
rng = np.random.default_rng()  # fresh, unseeded — different sample every run
sample = rng.choice(easy_seeds, size=min(30, len(easy_seeds)), replace=False)

environment = env.NanoEnv()
failures = []
for s in sample:
    s = int(s)
    environment.reset(seed=s)
    start = (int(environment._agent_location[0]), int(environment._agent_location[1]))
    goal  = (int(environment._target_location[0]), int(environment._target_location[1]))
    pl, dev = astar_path_and_deviation(
        environment._vessel_topology, start, goal, environment._agent_radius
    )
    if pl == np.inf or dev >= EASY_MAX_DEG:
        failures.append((s, dev if pl != np.inf else "unreachable"))
environment.close()

print(f"Checked {len(sample)} random 'easy' seeds.")
if failures:
    print(f"❌ {len(failures)} FAILED the easy criterion: {failures}")
    print("DO NOT COMMIT seeds.json — something is inconsistent in this run.")
else:
    print("✅ All sampled seeds pass. Safe to commit seeds.json.")