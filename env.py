import os
import shelve
from typing import Optional, Literal
import gymnasium as gym
import numpy as np
import pygame
import json
from numba import njit
from perlin_noise import fbm2d
from utils import main_related_component, clearance_mask_jit, is_navigable_jit

from gymnasium.envs.registration import register

Difficulty = Literal["easy", "medium", "hard"]


@njit(cache=True, fastmath=True)
def _lidar_walls_jit(x0, y0, orientation, n, max_range, lidar_step, start, size, topology):
    """Ray-marches `n` lidar rays outward from (x0, y0) until each hits a wall
    (topology[i, j] == 1), leaves the grid, or reaches max_range. Kept as a
    free function (not a method) and decorated with @njit because numba's
    nopython mode can't compile arbitrary Python objects like `self` -- only
    plain scalars and numpy arrays, which is exactly what's passed in here.
    cache=True writes the compiled machine code to disk next to this file
    after the first-ever compilation, so later launches (including each new
    SubprocVecEnv worker process) load it directly instead of re-JIT-compiling
    from scratch every time.

    Same logic and same output as the original pure-Python version (verified
    bit-identical across ~100 random steps): for each ray, step outward by
    `lidar_step` from `start` until a wall, the grid boundary, or max_range is
    hit, then normalize the resulting distance to [0, 1] (1.0 = clear all the
    way to max_range, 0.0 = touching a wall).
    """
    out = np.empty(n, dtype=np.float32)
    for k in range(n):
        a = orientation + (2.0 * np.pi * k) / n
        # Aligned with the movement direction formula used in step()
        # (v_agent = [-v*cos(orientation), v*sin(orientation)]), so ray k=0
        # (a == orientation) points exactly straight ahead instead of 90
        # degrees off from the true heading.
        dx = -np.cos(a)
        dy = np.sin(a)

        dist = start
        while dist <= max_range:
            x = x0 + dx * dist
            y = y0 + dy * dist

            # We consider outside the grid's bounds as a wall
            if x < 0.0 or y < 0.0 or x >= size or y >= size:
                break

            i = int(np.floor(x))
            j = int(np.floor(y))

            if topology[i, j] == 1:
                break

            dist += lidar_step

        d = min(dist, max_range)
        out[k] = np.float32(d / max_range)

    return out


class NanoEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, 
        render_mode: str = None, 
        difficulty: Optional[Difficulty] = None, 
        max_v: float = 6.0, 
        max_red: int = 8, 
        max_white: int = 4,
        worker_seed_offset: int = 0
    ):
        """Initialize the NanoEnv environment.
        Args:
            render_mode: The mode for rendering the environment. Can be "human", "rgb_array" or None.
            difficulty: The difficulty level for the learning curriculum. If None, the seeds will be drawn randomly from the whole set of seeds. If not None, it will be one of "easy", "medium" or "hard" and it will determine the seeds that will be drawn at the start of each episode. The seeds are divided into three sets of increasing difficulty and they are drawn from these sets with increasing pool sizes as the episodes go by. This allows a smooth learning curriculum for the agent.
            max_v: The maximum velocity of the agent in cell units per second.
            max_red: The maximum number of red cells in the environment. Red cells are obstacles that reduce the agent's velocity when it collides with them.
            max_white: The maximum number of white cells in the environment. White cells are obstacles that reduce the agent's velocity more than red cells when it collides with them.
            worker_seed_offset: When training with multiple parallel environments
                (SubprocVecEnv), each worker process constructs its own NanoEnv
                instance. Passing a distinct offset per worker (e.g. its rank)
                gives each one an independent per-episode sampling stream (see
                `_sampling_rng` below), so parallel workers explore genuinely
                different seeds instead of following the same sequence in
                lockstep. Defaults to 0 for single-environment usage
                (eval.py, visual_eval.py), where this has no effect anyway.
        """

        # Introducing difficulty levels for the learning curriculum
        self.difficulty = difficulty
        
        # Governs POOL CONSTRUCTION only (which seeds make it into the 40%/60%
        # training subset, and their permutation order) -- kept at a FIXED
        # seed regardless of worker_seed_offset so every parallel worker
        # agrees on the exact same pool definition. Do not use this for
        # anything drawn repeatedly during training (see _sampling_rng).
        self._episode_rng = np.random.default_rng(99999)


        if os.path.exists("seeds.json"):

            # Load the classified seeds from the JSON file
            with open("seeds.json") as f:
                _all_seeds = json.load(f)

            # Training split: 40% for easy (large pool), 60% for medium and hard (smaller pools).
            # The rest is reserved for evaluation in eval.py.
            def _sample_category(seeds_list, pct=0.40):
                arr = np.array(seeds_list)
                k = max(1, int(len(arr) * pct))
                idx = self._episode_rng.choice(len(arr), size=k, replace=False)
                return arr[idx].tolist()

            self.__easy_seeds   = _sample_category(_all_seeds["easy"],   pct=0.40)
            self.__medium_seeds = _sample_category(_all_seeds["medium"], pct=0.60)
            self.__hard_seeds   = _sample_category(_all_seeds["hard"],   pct=0.60)
        else : 
            # Fallback pendant la génération de seeds.json
            self.__easy_seeds   = []
            self.__medium_seeds = []
            self.__hard_seeds   = []

        self._easy_perm = [self.__easy_seeds[i] for i in self._episode_rng.permutation(len(self.__easy_seeds))]
        self._medium_perm = [self.__medium_seeds[i] for i in self._episode_rng.permutation(len(self.__medium_seeds))]
        self._hard_perm = [self.__hard_seeds[i] for i in self._episode_rng.permutation(len(self.__hard_seeds))]

        # Governs WHICH seed gets drawn each episode (and the 20/80, 10/20/70
        # curriculum-mixing coin flips) -- unlike _episode_rng above, this IS
        # meant to differ per worker (via worker_seed_offset), so that
        # parallel SubprocVecEnv workers explore genuinely different episodes
        # instead of following the identical draw sequence in lockstep (which
        # happened before, since every worker used to share the same fixed
        # _episode_rng seed for this too). Still fully deterministic and
        # reproducible for a given worker_seed_offset.
        self._sampling_rng = np.random.default_rng(13579 + worker_seed_offset)

        # Tracks how many times each individual seed has actually been drawn
        # during training, per pool ({"easy": {seed: count}, ...}). This gives
        # real visibility into training coverage (e.g. "92% of the easy pool
        # has been seen at least once by step 8M"), independent of raw
        # timestep count. A dict is used (not a boolean set) so we also know
        # HOW MANY times each seed was revisited, not just whether it was seen.
        # Memory cost is negligible: a few thousand int keys per pool at most.
        self._seed_counts = {"easy": {}, "medium": {}, "hard": {}}

        # Tracks, per pool category, how many episodes have actually been run
        # and how many of those were successes ({"easy": {"episodes": n,
        # "successes": m}, ...}). This is what lets SeedCoverageCallback report
        # a live per-category success rate over the seeds already seen during
        # training -- e.g. "the agent succeeds on 62% of the hard episodes it
        # has actually been run on so far", as opposed to a global success
        # rate that would blur together seeds of very different difficulty.
        self._category_episode_stats = {
            "easy":   {"episodes": 0, "successes": 0},
            "medium": {"episodes": 0, "successes": 0},
            "hard":   {"episodes": 0, "successes": 0},
        }

        # The category ("easy", "medium" or "hard") the CURRENT episode's seed
        # was drawn from -- set by _sample_from() at reset() time, consumed in
        # step() once the episode actually ends, to attribute its outcome to
        # the right category above. Stays None outside curriculum training
        # (difficulty=None with an explicit seed, e.g. eval.py/visual_eval.py),
        # where per-category success tracking doesn't apply.
        self._current_category = None


        # Load topology cache if available
        if os.path.exists("topology_cache") or os.path.exists("topology_cache.db") or os.path.exists("topology_cache.dir"):
            self._topology_cache = shelve.open("topology_cache", flag="r")  # read-only
        else:
            self._topology_cache = {}
        
        # Learn by using increasing pools of seeds
        self._ep = 0               # episodes count
        self._pool_init = 4        # initial pool's size
        self._expand_every = 1500 if difficulty == "easy" else \
                            4000  if difficulty == "medium" else \
                            10000  # expansion frequency

        # Discrete representation as a grid
        self._size = 125  # grid's size
        self._vessel_topology = np.zeros(shape=(self._size, self._size), dtype=int) # the vessels layout as a grid with 0 being the empty spaces and 1 being occupied ones by walls

        # Lidar (raycasts) parameters
        self._lidar_n = 16
        self._lidar_max_range = 60.0  
        self._lidar_step = 0.25        

        # Entities characteristics
        self._agent_radius = 1.2
        self._cell_radius = 0.8
        self._target_radius = 2.0

        # The number of red and white cells in the simulation. An exact number will be chosen randomly
        self._nb_red = 0
        self._nb_white = 0
        self._max_red = np.clip(int(max_red), 0, 20)
        self._max_white = np.clip(int(max_white), 0, 20)

        # The maximum velocity of the agent. They are useful to define real-life constraint on the agent
        self._max_v = max_v

        # The blood's velocity
        self.__v_blood = np.array([1.4, 1.1], dtype=np.float32)

        # Agent and target initial locations
        self._agent_location = np.array([-1, -1], dtype=np.float32)
        self._target_location = np.array([-1, -1], dtype=np.float32)
        self.__initial_distance = 0
        self._agent_trail = []  # visited positions, drawn as a dotted trail when rendering

        # Blood and white cells initial locations
        self._red_cells = np.full(shape=(max_red, 2), fill_value=-1, dtype=np.float32)
        self._white_cells = np.full(shape=(max_white, 2), fill_value=-1, dtype=np.float32)

        # Initial agent's velocity and orientation
        self._velocity = 0.0
        self._orientation = 0.0

        # Prepare the reward for improvement in distance
        self._best_dist = np.inf
        self._is_success = False

        # The limits on the variation of velocity and orientation in the action
        self.__action_v_limit = 2.0
        self.__action_theta_limit = np.pi/6

        # Penalty collision
        self.__penalty_red_cell = -3.0
        self.__penalty_white_cell = -7.0

        # Time management
        self._time = 0
        self.__timestep = 0.05
        self.__timelimit = 0

        # The rendering parameters
        assert render_mode is None or render_mode in self.metadata["render_modes"] # set the rendering mode
        self.render_mode = render_mode

        self._window_size = 625  # The size of the PyGame window
        self.__pix_square_size = (
            self._window_size / self._size
        )  # The size of a single grid square in pixels

        # The window and the clock we will use for rendering
        self._window = None 
        self._clock = None
        self._last_frame = None


        # What the agent can observe
        self.observation_space = gym.spaces.Dict(
            {
                # -1.0 and size will be used to represent element outside of the visible box
                "agent" : gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "mvt" : gym.spaces.Box(
                    low=np.array([0.0, -1.0, -1.0], dtype=np.float32), 
                    high=np.array([1.0, 1.0, 1.0], dtype=np.float32), 
                    shape=(3,), 
                    dtype=np.float32
                ),
                "delta_goal":gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "lidar": gym.spaces.Box(0.0, 1.0, shape=(self._lidar_n,), dtype=np.float32)
            }
        )

        # The actions available to the agent
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(2,), 
            dtype=np.float32
        )



    def _lidar_walls(self):
        """Returns an array of shape (n,) with normalized distances [0, 1] toward the first wall
        encountered.
        1.0 = empty until max_range, 0.0 = very close to a wall.

        The actual ray-marching loop lives in the module-level @njit function
        _lidar_walls_jit below. Profiling showed this single call was ~100% of
        a step()'s wall-clock cost (up to 16 rays * up to 240 marching steps
        each = up to 3,840 pure-Python scalar loop iterations per step, on
        every environment, every timestep, for hundreds of millions of
        timesteps) -- by far the dominant cost in the whole simulation, well
        above anything the LSTM itself adds. A pure-numpy vectorization across
        the 16 rays was tried first and was actually SLOWER (numpy's per-call
        overhead on 16-element arrays outweighs the savings when repeated up
        to 240 times per call); compiling the original scalar loop with numba
        instead measured ~130x faster with bit-identical output, so that's
        what's used here.
        """
        x0, y0 = float(self._agent_location[0]), float(self._agent_location[1])
        start = float(self._agent_radius) * 1.05
        return _lidar_walls_jit(
            x0, y0, self._orientation, self._lidar_n, self._lidar_max_range,
            self._lidar_step, start, self._size, self._vessel_topology
        )


    
    def _get_obs(self):
        """Convert the internal state to observation format

        Returns:
            dict: Observation with agent and target positions, obstacles positions and  current motion parameters
        """

        def normalize(x):
            x = np.asarray(x, dtype=np.float32)
            x = np.clip(x, 0.0, self._size - 1)
            return 2.0 * (x / (self._size - 1)) - 1.0

        return {
            "agent" : normalize(self._agent_location),
            "mvt" : np.array([
                    self._velocity / self._max_v, 
                    np.sin(self._orientation), 
                    -np.cos(self._orientation)
                ], 
                dtype=np.float32
            ),
            "delta_goal": np.clip((self._target_location - self._agent_location) / self._size, -1.0, 1.0),
            "lidar": self._lidar_walls()
        }
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target and if the experience is a success or not
        """

        distance = np.linalg.norm(self._agent_location - self._target_location)

        return {
            "distance": distance,
            "is_success": self._is_success,
            "best_dist": self._best_dist
        }


    def _generate_logical_topology(self, seed: Optional[int] = None):
        """Generates a size x size matrix describing the vessel's topology. It must be a valid topology
            but it must also have a bit of diversity (randomness). It uses fractal Perlin noise
            (see perlin_noise.py — a fully deterministic, vectorized, pure-numpy implementation;
            the third-party `noise` package was found to be non-deterministic across separate
            process launches for some (base, octaves) combinations, which silently corrupted
            seed-based difficulty classification).

        Returns:
            matrix: a numpy array describing the generated topology
        """
        base = int(seed) if seed is not None else self.np_random.integers(25, 10000)

        # IMPORTANT: use a dedicated RNG derived purely from `base`, NOT self.np_random.
        # self.np_random is shared with agent/target/cell placement later in reset().
        # If topology params were drawn from self.np_random, the number of draws
        # consumed here (which varies with cache hit/miss and with how many fallback
        # attempts were needed) would shift the RNG state before agent/target
        # placement, making positions depend on cache state instead of purely on
        # the seed. Using an independent RNG makes topology generation a pure,
        # fully reproducible function of `base` alone, regardless of process,
        # caching, or prior calls.
        topo_rng = np.random.default_rng(base)
        gamma = topo_rng.uniform(1.1, 1.5)
        ox, oy = topo_rng.uniform(0, 10000, size=2)

        # Vectorized coordinate grids (i = row, j = col) — computes the whole
        # grid in one shot instead of looping cell by cell in Python.
        jj, ii = np.meshgrid(np.arange(self._size), np.arange(self._size))
        x1, y1 = (jj + ox) / 153.0, (ii + oy) / 153.0
        x2, y2 = (jj + ox) / 67.0,  (ii + oy) / 67.0
        x3, y3 = (jj + ox) / 23.0,  (ii + oy) / 23.0

        n1 = fbm2d(x1, y1, base=base,        octaves=4, persistence=0.5, lacunarity=2.0)
        n2 = fbm2d(x2, y2, base=base + 1337, octaves=6, persistence=0.5, lacunarity=2.0)
        micro = 0.05 * fbm2d(x3, y3, base=base + 999, octaves=1)

        n = (((0.6 * n1 + 0.4 * n2) + 1) / 2 + micro) ** gamma
        computed = np.clip(n, 0.0, 1.0)

        q = topo_rng.uniform(0.36, 0.44)
        t = np.quantile(computed, q)
        grid = (computed <= t).astype(int)

        return grid
    

    def _has_clearance(self, pos: np.ndarray, r: float):
        """
        Returns True if a disk of radius r centered at pos (x=i, y=j in continuous coordinates)
        does not intersect any wall (topology == 1) and stays within the grid.
        Args:
            pos: np.array([x, y]) in float32
            r: radius in "cell units" (same unit as _agent_radius)
        """

        x = float(pos[0])
        y = float(pos[1])

        # Si le disque sort de la grille -> pas acceptable
        if x - r < 0.0 or y - r < 0.0 or x + r >= self._size or y + r >= self._size:
            return False

        # Bounding box de cellules candidates
        i0 = int(np.floor(x - r))
        i1 = int(np.floor(x + r))
        j0 = int(np.floor(y - r))
        j1 = int(np.floor(y + r))

        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                if self._vessel_topology[i, j] != 1:
                    continue

                # collision disque (x,y,r) vs carré cellule [i,i+1]x[j,j+1]
                cx = np.clip(x, i, i + 1.0)
                cy = np.clip(y, j, j + 1.0)
                dx = x - cx
                dy = y - cy

                if dx * dx + dy * dy < r * r:
                    return False

        return True

    def _filter_by_clearance(self, cells, r: float):
        """Filters a list of cells by clearance. It returns the cells that satisfy the clearance constraint for a disk of radius r. 
        This is useful to filter the available space in the environment and keep only the positions where the agent can be placed without colliding with walls.
            cells: List/set of tuples (i, j) coming from main_related_component.

        Returns:
            list: a list of np.array([i, j]) that satisfy the clearance constraint.
        """
        out = []
        for (i, j) in cells:
            p = np.array([i, j], dtype=np.float32)
            if self._has_clearance(p, r):
                out.append(p)
        return out


    def _pool_size(self, max_len: int):
        steps = self._ep // self._expand_every
        k = self._pool_init * (2 ** steps)
        return int(min(max_len, max(1, k)))
    
    def _sample_from(self, seeds, k: int, category: str):
        # pool = k first seeds
        pool = seeds[:k]
        seed = pool[self._sampling_rng.integers(0, len(pool))]
        self._seed_counts[category][seed] = self._seed_counts[category].get(seed, 0) + 1
        self._current_category = category
        return seed

    def _get_seed(self):
        """Generates a seed for the episode depending on the difficulty level chosen"""

        # Actual sizes of the pools
        ke = self._pool_size(len(self._easy_perm))
        km = self._pool_size(len(self._medium_perm))
        kh = self._pool_size(len(self._hard_perm))

        if self.difficulty == "easy":
            print("pool_size_easy: ", ke)
            return self._sample_from(self._easy_perm, ke, "easy")

        if self.difficulty == "medium":
            print("pool_size_easy: ", ke, ", pool_size_medium: ", km)
            # 20% easy, 80% medium 
            if self._sampling_rng.uniform(0.0, 1.0) < 0.2:
                return self._sample_from(self._easy_perm, ke, "easy")
            return self._sample_from(self._medium_perm, km, "medium")

        if self.difficulty == "hard":
            print("pool_size_easy: ", ke, ", pool_size_medium: ", km, ", pool_size_hard: ", kh)
            # 10% easy, 20% medium, 70% hard
            u = self._sampling_rng.uniform(0.0, 1.0)
            if u < 0.1:
                return self._sample_from(self._easy_perm, ke, "easy")
            if u < 0.3:
                return self._sample_from(self._medium_perm, km, "medium")
            return self._sample_from(self._hard_perm, kh, "hard")

        return int(self._sampling_rng.integers(0, 10000))

    def get_seed_visit_counts(self):
        """Returns {"easy": {seed: times_drawn}, "medium": {...}, "hard": {...}}
        for THIS environment instance since it was created. Meant to be called
        via VecEnv.env_method() from a training callback, then aggregated
        across all parallel sub-environments (see seed_coverage_callback.py)."""
        return self._seed_counts

    def get_category_success_stats(self):
        """Returns {"easy": {"episodes": n, "successes": m}, "medium": {...},
        "hard": {...}} for THIS environment instance since it was created.
        Meant to be called via VecEnv.env_method() from a training callback,
        then summed across all parallel sub-environments (see
        seed_coverage_callback.py) to compute a live per-category success
        rate over the seeds already seen so far during training."""
        return self._category_episode_stats

    def get_training_pool_sizes(self):
        """Returns the number of seeds in each training pool (identical across
        every sub-environment, since all are built from the same seeds.json
        with the same fixed _episode_rng seed) -- used as the denominator for
        coverage percentages."""
        return {
            "easy":   len(self.__easy_seeds),
            "medium": len(self.__medium_seeds),
            "hard":   len(self.__hard_seeds),
        }

    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration namely 
                - nb_red which is the wanted number of red cells with 0 <= nb_red <= max_red
                - nb_white which is the wanted number of white cells with 0 <= nb_white <= max_white

        Returns:
            tuple: (observation, info) for the initial state
        """

        # Seed the random number generator
        used_seed = seed if self.difficulty == None and seed != None else self._get_seed()

        super().reset(seed=int(used_seed))
        self._ep += 1

        # Reset the success variable
        self._is_success = False

        # TODO: Remove this print
        print("episode: ", self._ep, " ,seed: ", used_seed)


        # ── World generation (with cache if available) ───────────────────────
        if str(used_seed) in self._topology_cache:
            # Cache hit: precomputed topology and free space
            entry                 = self._topology_cache[str(used_seed)]
            self._vessel_topology = entry["topology"].copy()
            # Reconstruct list of np.array([i, j]) from the (N, 2) array
            available_space       = [entry["available"][k] for k in range(len(entry["available"]))]
        else:
            # Cache miss: normal generation
            new_seed = 1 + int(used_seed)
            found    = False
            while not found:
                self._vessel_topology = self._generate_logical_topology(new_seed)
                available_space       = main_related_component(self._vessel_topology)
                available_space       = self._filter_by_clearance(
                    available_space,
                    max(self._agent_radius, self._target_radius)
                )
                new_seed += 1
                if len(available_space) > 100:
                    found = True

        # Precomputed ONCE per reset (the topology doesn't change within an
        # episode) so the target-placement retry loop below -- which can call
        # this navigability check several times if early candidates get
        # rejected -- does O(1) lookups instead of re-scanning an O(radius^2)
        # box from scratch at every one of the potentially thousands of cells
        # a single BFS call visits. See utils.py's clearance_mask_jit/
        # is_navigable_jit docstrings for the full rationale; verified
        # bit-identical to the original is_navigable/surroundings_ok.
        _clearance_mask = clearance_mask_jit(self._vessel_topology, int(np.ceil(self._agent_radius)))

        # ── Agent and target placement (always random) ───────────────────────
        repeat1 = True
        while repeat1:
            agent_int           = self.np_random.integers(0, len(available_space))
            init_agent_location = available_space[agent_int].copy()
            if 10 <= init_agent_location[0] <= self._size - 10 and 10 <= init_agent_location[1] <= self._size - 10:
                repeat1 = False

        self._agent_location = init_agent_location
        available_space.pop(agent_int)

        # Trail of visited positions, drawn as a permanent dotted line during
        # rendering so the agent's full path through the episode stays visible.
        self._agent_trail = [self._agent_location.copy()]

        repeat2    = True
        to_explore = available_space.copy()
        while repeat2 and len(to_explore) != 0:
            target_int           = self.np_random.integers(0, len(to_explore))
            init_target_location = to_explore[target_int].copy()
            d0                   = np.linalg.norm(self._agent_location - init_target_location)
            if d0 >= 35 and is_navigable_jit(
                self._vessel_topology, _clearance_mask,
                int(self._agent_location[0]), int(self._agent_location[1]),
                int(init_target_location[0]), int(init_target_location[1]),
            ):
                repeat2                  = False
                self._target_location    = init_target_location
                self.__initial_distance  = d0
                self._best_dist          = d0
                available_space          = list(filter(
                    lambda x: x[0] != init_target_location[0] or x[1] != init_target_location[1],
                    available_space
                ))
            else:
                to_explore.pop(target_int)
                
                


        # Velocity and orientation at the start of an episode.
        # Orientation starts facing directly toward the target (rather than a
        # fixed, arbitrary heading) so that any wall-avoidance turning the
        # agent must learn reflects the topology's true complexity, not an
        # unrelated "correct my initial heading" maneuver baked into every
        # episode regardless of difficulty.
        self._velocity = 0.0
        delta = self._target_location - self._agent_location
        self._orientation = float(np.arctan2(delta[1], -delta[0]))

        # Set the time limit 
        self.__timelimit = min(3 + 2 * self.__initial_distance, 40)
        self._time = 0.0

        # Random red and white cells locations in regard to the topology and the options nb_red and nb_white
        nb_red = 0 if self._max_red == 0 else self.np_random.integers(0, self._max_red)
        if options != None and "nb_red" in options:
            nb_red = max(0, options["nb_red"])
        nb_red = min(len(available_space) // 2, min(self._max_red, nb_red))
        self._nb_red = nb_red
        self._red_cells = np.full(shape=(self._max_red, 2), fill_value=-1, dtype=np.float32)
        for i in range(nb_red):
            drawn_int = self.np_random.integers(0, len(available_space))
            self._red_cells[i] = list(available_space[drawn_int])
            available_space.pop(drawn_int)
        
        nb_white = 0 if self._max_white == 0 else self.np_random.integers(0, self._max_white)
        if options != None and "nb_white" in options:
            nb_white = max(0, options["nb_white"])
        nb_white = min(len(available_space), min(self._max_white, nb_white))
        self._nb_white = nb_white
        self._white_cells = np.full(shape=(self._max_white, 2), fill_value=-1, dtype=np.float32)
        for i in range(nb_white):
            drawn_int = self.np_random.integers(0, len(available_space))
            self._white_cells[i] = list(available_space[drawn_int])
            available_space.pop(drawn_int)

        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode is not None:
            # Store every frame so callers (e.g. visual_eval.py) can build a
            # full-episode GIF just by reading self._last_frame after each
            # reset()/step() call, without needing a second, duplicate render
            # pass. Works the same whether render_mode is "human" (live
            # window) or "rgb_array" (headless, no window at all).
            self._last_frame = self._render_frame()

        return observation, info
    

    def _manage_wall_collision(self, old_location, new_location, radius):
        """Logic to verify wall collision with a cell or the agent.
        Args:
            old_location: the previous location of the entity
            new_location: the location it wants to attain after the step
            radius: the entity's radius 
        """
        x0, y0 = old_location[0], old_location[1]
        x1, y1 = new_location[0], new_location[1]
        r = radius

        def touch_wall(x, y):
            # Grid's cells potentially touched by the entity
            i0 = int(np.floor(x - r))
            i1 = int(np.floor(x + r))
            j0 = int(np.floor(y - r))
            j1 = int(np.floor(y + r))

            # We let the element disappear if it goes out of the grid
            if i0 < 0 or j0 < 0 or i1 >= self._size or j1 >= self._size:
                return False

            # Check collision for each cell in the bounding box
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    if self._vessel_topology[i, j] != 1:
                        continue

                    cx = np.clip(x, i, i + 1.0) # clamp sur le carré
                    cy = np.clip(y, j, j + 1.0)
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < r * r:
                        return True
            return False

        # If no collision, just move
        if not touch_wall(x1, y1):
            return np.array([x1, y1], dtype=np.float32)

        # If there is a collision see if the entity can slide along the wall
        if not touch_wall(x1, y0):
            return np.array([x1, y0], dtype=np.float32)

        if not touch_wall(x0, y1):
            return np.array([x0, y1], dtype=np.float32)

        # If the entity is completely blocked, don't move
        return np.array([x0, y0], dtype=np.float32)
    
    

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: array [dv, dtheta] where each component is clipped to [-1, 1]
                    before being scaled to real physical limits.
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        # ── Helpers ───────────────────────────────────────────────────────────────
        wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi  # wrap angle to [-π, π]

        reward     = 0.0
        terminated = False
        truncated  = False


        # ── 1. ACTION → PHYSICS ───────────────────────────────────────────────────
        # Scale the normalised action [-1, 1] to real physical deltas
        delta_v     = np.clip(action[0], -1.0, 1.0) * self.__action_v_limit
        delta_theta = np.clip(action[1], -1.0, 1.0) * self.__action_theta_limit

        # Apply deltas to agent state
        theta_old        = self._orientation
        self._velocity   = np.clip(self._velocity + delta_v, 0.0, self._max_v)
        self._orientation = wrap(self._orientation + delta_theta)


        # ── 2. MOVEMENT ───────────────────────────────────────────────────────────
        # Compute velocity vector (agent propulsion + blood flow drift)
        v_agent = np.array(
            [
                -self._velocity * np.cos(self._orientation),
                self._velocity * np.sin(self._orientation)
            ],
            dtype=np.float32
        )
        old_agent_location = self._agent_location.copy()
        new_agent_location = self._agent_location + (v_agent + 0.5 * self.__v_blood) * self.__timestep
        self._agent_location = self._manage_wall_collision(
            self._agent_location, new_agent_location, self._agent_radius
        )
        self._agent_trail.append(self._agent_location.copy())

        # Move blood cells (purely advected by the blood flow)
        for i in range(self._nb_red):
            new_pos          = self._red_cells[i] + self.__v_blood * self.__timestep
            self._red_cells[i] = self._manage_wall_collision(self._red_cells[i], new_pos, self._cell_radius)

        for i in range(self._nb_white):
            new_pos            = self._white_cells[i] + self.__v_blood * self.__timestep
            self._white_cells[i] = self._manage_wall_collision(self._white_cells[i], new_pos, self._cell_radius)


        # ── 3. REWARD COMPUTATION ─────────────────────────────────────────────────

        # 3a. Control-effort penalty — discourage erratic velocity and orientation changes
        alpha_v     = 0.006
        beta_theta  = 0.001
        dtheta      = wrap(self._orientation - theta_old)
        reward += -alpha_v    * (action[0] ** 2)
        reward += -beta_theta * (dtheta    ** 2)

        # 3b. Cell-collision penalty — hitting blood cells slows the agent and costs reward
        beta = 0.6
        for i in range(self._nb_red):
            if np.linalg.norm(self._red_cells[i] - self._agent_location) < self._agent_radius + self._cell_radius:
                self._velocity  = np.clip(self._velocity - beta, 0.0, self._max_v)
                reward += self.__penalty_red_cell
                break

        for i in range(self._nb_white):
            if np.linalg.norm(self._white_cells[i] - self._agent_location) < self._agent_radius + self._cell_radius:
                self._velocity  = np.clip(self._velocity - beta - 0.1, 0.0, self._max_v)
                reward += self.__penalty_white_cell
                break

        # 3c. Progress reward — reward proportional to reduction in distance to goal
        dbefore = np.linalg.norm(old_agent_location  - self._target_location)
        dafter  = np.linalg.norm(self._agent_location - self._target_location)
        p       = np.clip((dbefore - dafter) / self.__initial_distance, -1.0, 1.0)
        reward += 10.0 * p

        # 3d. Best-distance bonus — extra reward for reaching a new closest point ever
        if dafter < self._best_dist:
            gain           = (self._best_dist - dafter) / (self.__initial_distance + 1e-8)
            reward        += 3.0 * float(np.clip(gain, 0.0, 1.0))
            self._best_dist = dafter

        # 3e. Idleness penalty — small constant cost to push the agent to keep moving
        reward += -0.03 if p <= 0 else -0.01


        # ── 4. TERMINATION CONDITIONS ─────────────────────────────────────────────

        # 4a. Success — agent reached the target
        if np.linalg.norm(self._agent_location - self._target_location) <= self._agent_radius + self._target_radius:
            terminated     = True
            self._is_success = True
            reward        += 100.0

        # 4b. Out-of-bounds — agent drifted outside the vessel (carried by blood flow)
        if self._agent_location[0] < 0 or self._agent_location[1] < 0 \
                or self._agent_location[0] >= self._size or self._agent_location[1] >= self._size:
            terminated = True
            reward    += -50.0

        # 4c. Timeout — episode exceeded the time limit
        truncated = self._time > self.__timelimit
        if truncated:
            reward += -10.0

        # 4d. Category success-rate bookkeeping — once the episode actually
        # ends (success, out-of-bounds, or timeout), attribute its outcome to
        # the pool category its seed was drawn from (see _sample_from()), so
        # SeedCoverageCallback can report a live per-category success rate
        # over seeds already seen during training.
        if (terminated or truncated) and self._current_category is not None:
            stats = self._category_episode_stats[self._current_category]
            stats["episodes"]  += 1
            stats["successes"] += int(self._is_success)


        # ── 5. STEP FINALISATION ──────────────────────────────────────────────────
        self._time  += self.__timestep
        observation  = self._get_obs()
        info         = self._get_info()

        if self.render_mode is not None:
            self._last_frame = self._render_frame()

        return observation, float(reward), terminated, truncated, info
    

    def render(self):
        if self.render_mode is not None:
            return self._render_frame()


    def _render_frame(self):
        """This is a function to render the environment. It represents the elements in the pygame coordinate system. So the
        center is at the top left corner, the x-axis increases as we go further to the right and the y-axis increases as we go further down.
        Concerning the orientation, 0 radians corresponds to pointing upward.
        Positive orientation values rotate the agent clockwise.
        """
        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (self._window_size, self._window_size)
            )

        if not hasattr(self, "_agent_img"):
            # Needed for BOTH "human" and "rgb_array" render modes -- must not
            # live inside the "human"-only block above, or rendering in
            # "rgb_array" mode (e.g. for headless frame capture on a server
            # with no display) crashes with AttributeError since self._agent_img
            # would never get set. convert_alpha() requires pygame's video
            # subsystem to be initialized even if no window is ever shown, so
            # we set up a minimal display context here if one doesn't exist yet
            # (works headlessly with SDL_VIDEODRIVER=dummy).
            if not pygame.display.get_init():
                pygame.init()
                pygame.display.init()
                pygame.display.set_mode((1, 1))
            self._agent_img = pygame.image.load("assets/agent.png").convert_alpha()
            rect = self._agent_img.get_bounding_rect(min_alpha=10)
            self._agent_img = self._agent_img.subsurface(rect).copy()
            d_px = int(round(2 * self._agent_radius * self.__pix_square_size))
            d_px = max(2, d_px)
            self._agent_img = pygame.transform.smoothscale(self._agent_img, (d_px, d_px))
        
        if self._clock is None and self.render_mode == "human":
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface((self._window_size, self._window_size))
        canvas.fill((82, 41, 47))

        # Draw the environment first (the blood vessels with the walls)
        size_px = int(np.ceil(self.__pix_square_size))
        for i in range(self._size):
            for j in range(self._size):
                pygame.draw.rect(
                    canvas,
                    (213, 132, 144) if self._vessel_topology[i][j] == 0 else (75, 41, 47),
                    pygame.Rect(
                        (self.__pix_square_size * np.array([j, i])).astype(int),
                        (size_px, size_px),
                    ),
                )

        # Draw the agent's trail as a dashed blue line: connect consecutive
        # visited positions in short segments, skipping every other segment
        # to create the dashed look (instead of isolated dots, this follows
        # the actual path curvature). Drawn after the walls but before
        # target/cells/agent so it stays visually "on the ground".
        dash_length = 5
        trail_width = max(2, int(self.__pix_square_size * 0.25))
        for start in range(0, len(self._agent_trail) - 1, dash_length * 2):
            end = min(start + dash_length, len(self._agent_trail))
            if end - start < 2:
                continue
            points = [
                tuple(((p[::-1] + 0.5) * self.__pix_square_size).astype(int))
                for p in self._agent_trail[start:end]
            ]
            pygame.draw.lines(canvas, (20, 100, 40), False, points, width=trail_width)

        # First we draw the target
        # Convert [row, col] to pygame (x, y) by reversing the coordinates
        pygame.draw.circle(
            canvas,
            (255, 255, 153),
            tuple(((self._target_location[::-1] + 0.5) * self.__pix_square_size).astype(int)),
            int(np.ceil(self._target_radius * self.__pix_square_size))
        )

        # Red and white blood cells rendering
        for red_cell in self._red_cells:
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                tuple(((red_cell[::-1] + 0.5) * self.__pix_square_size).astype(int)),
                int(np.ceil(self._cell_radius * self.__pix_square_size))
            )

        for white_cell in self._white_cells:
            pygame.draw.circle(
                canvas,
                (255, 255, 255),
                tuple(((white_cell[::-1] + 0.5) * self.__pix_square_size).astype(int)),
                int(np.ceil(self._cell_radius * self.__pix_square_size))
            )

        # Draw the agent

        angle_deg = -np.degrees(self._orientation)
        rotated_img = pygame.transform.rotate(self._agent_img, angle_deg)
        center = ((self._agent_location[::-1] + 0.5) * self.__pix_square_size).astype(int)
        rect = rotated_img.get_rect(center=tuple(center))

        canvas.blit(rotated_img, rect)


        # Always extract the pixel array from the off-screen canvas, regardless
        # of render_mode. This is what lets us capture every frame for saving
        # an episode as a GIF even in "human" mode (live window), and even
        # headlessly with no real display attached (SDL_VIDEODRIVER=dummy
        # still provides a valid off-screen canvas to read pixels from).
        frame = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self._window.blit(canvas, canvas.get_rect())
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise KeyboardInterrupt
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self._clock.tick(self.metadata["render_fps"])

        return frame
        
    
    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
        
        self._window = None
        self._clock = None



register(
    id="Nano-v0",
    entry_point="env:NanoEnv",
)