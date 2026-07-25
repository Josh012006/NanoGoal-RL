"""
diagnose_reset.py
Prints every value produced by reset(seed=X), step by step, so we can compare
Windows vs Linux output line-by-line and find EXACTLY where the first
divergence appears (topology / agent / target / nb_red / nb_white / cell
positions).

Usage: python diagnose_reset.py <seed>
"""
import sys
import env
import numpy as np

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3271

e = env.NanoEnv()
obs, info = e.reset(seed=seed)

print(f"=== reset(seed={seed}) diagnostic ===")
print(f"topology checksum (sum):      {e._vessel_topology.sum()}")
print(f"topology shape:               {e._vessel_topology.shape}")
print(f"agent_location:               {e._agent_location.tolist()}")
print(f"target_location:              {e._target_location.tolist()}")
print(f"nb_red:                       {e._nb_red}")
print(f"nb_white:                     {e._nb_white}")
print(f"red_cells (first 3):          {e._red_cells[:3].tolist()}")
print(f"white_cells (first 3):        {e._white_cells[:3].tolist()}")
print(f"orientation:                  {e._orientation}")
print(f"init distance (info):         {info['distance']}")

# Also print whether the cache was used
cache_used = str(seed) in e._topology_cache
print(f"cache hit for this seed:      {cache_used}")
print(f"topology_cache type:          {type(e._topology_cache)}")

e.close()
