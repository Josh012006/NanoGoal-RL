"""
diagnose_cache_vs_fresh.py
Runs ON THE SAME MACHINE, comparing:
  (a) what the topology_cache returns for a given seed (cache-hit path)
  (b) a completely fresh computation of the same seed, bypassing the cache
      entirely (main_related_component + _filter_by_clearance called directly)

If topology matches but available_space differs, the bug is in HOW the cache
stores/reconstructs available_space, not in topology generation itself.

Usage: python diagnose_cache_vs_fresh.py <seed>
"""
import sys
import numpy as np
import env
from utils import main_related_component

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3271

# ── (a) Cache-hit path ──────────────────────────────────────────────────────
e = env.NanoEnv()
cache_hit = str(seed) in e._topology_cache
if not cache_hit:
    print(f"No cache entry for seed {seed} -- nothing to compare.")
    sys.exit(1)

entry = e._topology_cache[str(seed)]
cached_topology = entry["topology"].copy()
cached_available = np.array([entry["available"][k] for k in range(len(entry["available"]))])

print(f"=== CACHE-HIT for seed {seed} ===")
print(f"topology checksum: {cached_topology.sum()}")
print(f"available_space count: {len(cached_available)}")
print(f"available_space first 5: {cached_available[:5].tolist()}")

# ── (b) Fresh computation, bypassing cache entirely ─────────────────────────
e2 = env.NanoEnv()
e2._topology_cache = {}  # force bypass

new_seed = 1 + seed
found = False
attempts = 0
while not found:
    fresh_topology = e2._generate_logical_topology(new_seed)
    fresh_available = main_related_component(fresh_topology)
    fresh_available = e2._filter_by_clearance(
        fresh_available, max(e2._agent_radius, e2._target_radius)
    )
    attempts += 1
    new_seed += 1
    if len(fresh_available) > 100:
        found = True

fresh_available_arr = np.array(fresh_available)

print(f"\n=== FRESH computation for seed {seed} (took {attempts} attempt(s)) ===")
print(f"topology checksum: {fresh_topology.sum()}")
print(f"available_space count: {len(fresh_available_arr)}")
print(f"available_space first 5: {fresh_available_arr[:5].tolist()}")

# ── Comparison ───────────────────────────────────────────────────────────────
print(f"\n=== COMPARISON ===")
print(f"Topology identical?        {np.array_equal(cached_topology, fresh_topology)}")
print(f"available_space count eq?  {len(cached_available) == len(fresh_available_arr)}")
if len(cached_available) == len(fresh_available_arr):
    print(f"available_space content eq? {np.array_equal(cached_available, fresh_available_arr)}")

e.close()
e2.close()
