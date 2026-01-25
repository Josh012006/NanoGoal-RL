from typing import Optional, Literal
import gymnasium as gym
import numpy as np
import pygame
from noise import pnoise2
from utils import main_related_component, is_navigable

from gymnasium.envs.registration import register

class NanoEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    Difficulty = Literal["easy", "medium", "hard"]

    def __init__(self, render_mode: str = None, difficulty: Optional[Difficulty] = None, max_v: float = 6.0, max_red: int = 8, max_white: int = 4):

        # Introducing difficulty levels for the learning curriculum
        self.difficulty = difficulty
        self.__easy_seeds = [0, 2, 10, 11, 30, 42, 49, 101, 121, 205, 775, 951, 1500, 1974, 1976, 2003, 2008, 2013, 2017, 2033]
        self.__medium_seeds = [3, 32, 89, 100, 195, 665, 714, 726, 1001, 1004, 1005, 2006, 2020, 2023, 2029, 2037, 7011, 9151]
        self.__hard_seeds = [1, 265, 721, 728, 729, 989, 999, 2002, 2011, 2012, 2022, 2024, 2027, 2222, 2565, 8188] # Use 1011 too if you want to teach precision

        self._episode_rng = np.random.default_rng(12345)
        self._easy_perm = [self.__easy_seeds[i] for i in self._episode_rng.permutation(len(self.__easy_seeds))]
        self._medium_perm = [self.__medium_seeds[i] for i in self._episode_rng.permutation(len(self.__medium_seeds))]
        self._hard_perm = [self.__hard_seeds[i] for i in self._episode_rng.permutation(len(self.__hard_seeds))]
        
        # Learn by using increasing pools of seeds
        self._ep = 0               # episodes count
        self._pool0 = 2            # initial pool's size
        self._expand_every = 500 if difficulty == "easy" else 1000 if difficulty == "medium" else 2000  # expand the pool's size by 2 every 2000 resets

        # Discrete representation as a grid
        self._size = 125  # grid's size
        self._vessel_topology = np.zeros(shape=(self._size, self._size), dtype=int) # the vessels layout as a grid with 0 being the empty spaces and 1 being occupied ones by walls

        # Lidar (raycasts) parameters
        self._lidar_n = 8
        self._lidar_max_range = 20.0  
        self._lidar_step = 0.25        

        # Entities characteristics
        self.__agent_radius = 1.2
        self.__cell_radius = 0.8
        self.__target_radius = 2.0

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


        # What the agent can observe
        self.observation_space = gym.spaces.Dict(
            {
                # -1.0 and size will be used to reprensent element outside of the visible box
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
        encountered. Generated with CHATGPT.
        1.0 = empty until max_range, 0.0 = very close to a wall.
        """
        x0, y0 = float(self._agent_location[0]), float(self._agent_location[1])

        # Angles of the rays (8 directions)
        angles = self._orientation + np.linspace(0.0, 2.0 * np.pi, num=self._lidar_n, endpoint=False)

        out = np.empty((self._lidar_n,), dtype=np.float32)

        # We start outside the agent's radius
        start = float(self.__agent_radius) * 1.05

        for k, a in enumerate(angles):
            dx = float(np.cos(a))
            dy = float(np.sin(a))

            dist = start

            # Check along the ray's direction
            while dist <= self._lidar_max_range:
                x = x0 + dx * dist
                y = y0 + dy * dist

                # We consider outside the grid's bounds as a wall
                if x < 0.0 or y < 0.0 or x >= self._size or y >= self._size:
                    break

                i = int(np.floor(x))
                j = int(np.floor(y))

                if self._vessel_topology[i, j] == 1:
                    break

                dist += self._lidar_step

            d = min(dist, self._lidar_max_range)
            out[k] = np.float32(d / self._lidar_max_range)

        return out


    
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
                    np.cos(self._orientation)
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
            but it must also have a bit of diversity (randomness). It uses Perlin's noise.
        
        Returns:
            matrix: a numpy array describing the generated topology
        """
        computed = np.zeros(shape=(self._size, self._size))
        base = int(seed) if seed is not None else self.np_random.integers(25, 10000)

        # Add more variety
        gamma = self.np_random.uniform(1.1, 1.5)
        ox, oy = self.np_random.uniform(0, 10000, size=2)

        for i in range(self._size):
            for j in range(self._size):
                n1 = pnoise2((j + ox)/153, (i + oy)/153, base=base, octaves=4, persistence=0.5, lacunarity=2.0)
                n2 = pnoise2((j + ox)/67, (i + oy)/67, base=(base + 1337), octaves=6, persistence=0.5, lacunarity=2.0)
                micro = 0.05 * pnoise2((j + ox) / 23, (i + oy) / 23, base=base + 999)
                n = (((0.6 * n1 + 0.4 * n2) + 1) / 2 + micro) ** gamma
                computed[i][j] = np.clip(n, 0.0, 1.0)
        q = self.np_random.uniform(0.36, 0.44)
        t = np.quantile(computed, q)
        grid = (computed <= t).astype(int)

        return grid
    

    def _has_clearance(self, pos: np.ndarray, r: float):
        """
        Returns True if a disk of radius r centered at pos (x=i, y=j in continuous coordinates)
        does not intersect any wall (topology == 1) and stays within the grid.
        Args:
            pos: np.array([x, y]) in float32
            r: radius in "cell units" (same unit as __agent_radius)
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
        """Generated by CHATGPT
        cells: List/set of tuples (i, j)coming from main_related_component.

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
        k = self._pool0 * (2 ** steps)
        return int(min(max_len, max(1, k)))
    
    def _sample_from(self, seeds, k: int):
        # pool = k first seeds
        pool = seeds[:k]
        return pool[self._episode_rng.integers(0, len(pool))]

    def _get_seed(self):
        """Generates a seed for the episode depending on the difficulty level chosen"""

        # Actual sizes of the pools
        ke = self._pool_size(len(self._easy_perm))
        km = self._pool_size(len(self._medium_perm))
        kh = self._pool_size(len(self._hard_perm))
        print("ke: ", ke, ", km: ", km, ", kh: ", kh)

        if self.difficulty == "easy":
            return self._sample_from(self._easy_perm, ke)

        if self.difficulty == "medium":
            # 20% easy, 80% medium 
            if self._episode_rng.uniform(0.0, 1.0) < 0.2:
                return self._sample_from(self._easy_perm, ke)
            return self._sample_from(self._medium_perm, km)

        if self.difficulty == "hard":
            # 10% easy, 20% medium, 70% hard
            u = self._episode_rng.uniform(0.0, 1.0)
            if u < 0.1:
                return self._sample_from(self._easy_perm, ke)
            if u < 0.3:
                return self._sample_from(self._medium_perm, km)
            return self._sample_from(self._hard_perm, kh)

        return int(self._episode_rng.integers(0, 10000))

    
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


        # Generate a pseudo-random but also valid vessel topology for the episode
        new_seed = 1 + int(used_seed)
        repeter = True
        while repeter:
            self._vessel_topology = self._generate_logical_topology(new_seed)

            available_space = main_related_component(self._vessel_topology)
            available_space = self._filter_by_clearance(available_space, max(self.__agent_radius, self.__target_radius))

            new_seed += 1
            if len(available_space) > 100:
                # Randomly generated target and agent locations in the available space
                repeter1 = True
                while repeter1:
                    agent_int = self.np_random.integers(0, len(available_space))
                    init_agent_location = available_space[agent_int].copy()
                    if 10 <= init_agent_location[0] <= self._size - 10 and 10 <= init_agent_location[1] <= self._size - 10 :
                        repeter1 = False

                self._agent_location = init_agent_location
                available_space.pop(agent_int)

                repeter2 = True
                to_explore = available_space.copy()
                while repeter2 and len(to_explore) != 0:
                    target_int = self.np_random.integers(0, len(to_explore))
                    init_target_location = to_explore[target_int].copy()
                    d0 = np.linalg.norm(self._agent_location - init_target_location)
                    if d0 >= 35 and is_navigable(self._vessel_topology, self._agent_location, init_target_location, self.__agent_radius):
                        repeter2 = False
                        self._target_location = init_target_location
                        self.__initial_distance = d0
                        self._best_dist = d0
                        available_space = list(filter(lambda x: x[0] != init_target_location[0] or x[1] != init_target_location[1], available_space))

                        repeter = False
                    else: 
                        to_explore.pop(target_int)
                
                


        # Velocity and orientation at the start of an episode
        self._velocity = 0.0
        self._orientation = 0.0

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

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    def _manage_wall_collision(self, old_location, new_location, radius):
        """Logic to verify wall collision with a cell or the agent. Was refactored with the help of AI (CHATGPT).
        Args:
            old_location: the previous location of the entity
            new_location: the location it wants to attain after the step
            radius: the entity's radius 
        """
        x0, y0 = old_location[0], old_location[1]
        x1, y1 = new_location[0], new_location[1]
        r = radius

        def touche_mur(x, y):
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
        if not touche_mur(x1, y1):
            return np.array([x1, y1], dtype=np.float32)

        # If there is a collision see if the entity can slide along the wall
        if not touche_mur(x1, y0):
            return np.array([x1, y0], dtype=np.float32)

        if not touche_mur(x0, y1):
            return np.array([x0, y1], dtype=np.float32)

        # If the entity is completely blocked, don't move
        return np.array([x0, y0], dtype=np.float32)
    
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take, namely an array in the format [float, float] where the first component is 
                the component to add to the velocity and the second one is the component to add to
                the orientation. Both can be negative.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        reward = 0 
        terminated = False
        truncated = False

        wrap = lambda x : (x + np.pi) % (2 * np.pi) - np.pi # helper function
        
        # Updating the agent's new velocity and orientation
        delta_v = np.clip(action[0], -1.0, 1.0) * self.__action_v_limit
        delta_theta = np.clip(action[1], -1.0, 1.0) * self.__action_theta_limit

        self._velocity = np.clip(self._velocity + delta_v, 0.0, self._max_v)

        theta_old = self._orientation
        self._orientation = wrap(self._orientation + delta_theta)


        # Discourage spins and changes in orientation that are too great
        alpha_v = 0.006
        reward += - alpha_v * (action[0] ** 2)

        beta_theta = 0.001
        dtheta = wrap(self._orientation - theta_old)
        reward += - beta_theta * (dtheta ** 2)


        # Compute new agent and cells continuous positions with collisions management
        old_agent_location = self._agent_location.copy()
        v_agent = np.array([self._velocity * np.sin(self._orientation), self._velocity * np.cos(self._orientation)], dtype=np.float32)
        new_agent_location = self._agent_location + (v_agent + 0.5 * self.__v_blood) * self.__timestep
        self._agent_location = self._manage_wall_collision(self._agent_location, new_agent_location, self.__agent_radius)

        for i in range(self._nb_red):
            new_red_cell_position = self._red_cells[i] + self.__v_blood * self.__timestep
            self._red_cells[i] = self._manage_wall_collision(self._red_cells[i], new_red_cell_position, self.__cell_radius)
        
        for i in range(self._nb_white):
            new_white_cell_position = self._white_cells[i] + self.__v_blood * self.__timestep
            self._white_cells[i] =  self._manage_wall_collision(self._white_cells[i], new_white_cell_position, self.__cell_radius)


        # Agent-cell collision
        beta = 0.6

        for i in range(self._nb_red):
            if np.linalg.norm(self._red_cells[i] - self._agent_location) < self.__agent_radius + self.__cell_radius:
                self._velocity = np.clip(self._velocity - beta, 0.0, self._max_v)
                reward += self.__penalty_red_cell
                break
        
        for i in range(self._nb_white):
            if np.linalg.norm(self._white_cells[i] - self._agent_location) < self.__agent_radius + self.__cell_radius:
                self._velocity = np.clip(self._velocity - beta - 0.1, 0.0, self._max_v)
                reward += self.__penalty_white_cell
                break
        
        # Reward for decreasing the distance between the agent and the target
        dbefore = np.linalg.norm(old_agent_location - self._target_location)
        dafter = np.linalg.norm(self._agent_location - self._target_location)
        p = np.clip((dbefore - dafter) / self.__initial_distance, -1.0, 1.0)
        reward += 10.0 * p
        if dafter < self._best_dist:
            gain = (self._best_dist - dafter) / (self.__initial_distance + 1e-8)
            reward += 3.0 * float(np.clip(gain, 0.0, 1.0)) 
            self._best_dist = dafter

        # Make sure the agent doesn't stay motionless
        if p <= 0:
            reward += -0.03
        else:
            reward += -0.01

        if np.linalg.norm(self._agent_location - self._target_location) <= self.__agent_radius + self.__target_radius:
            terminated = True
            self._is_success = True
            reward += 100.0
        truncated = self._time > self.__timelimit


        # Prevent the agent from letting the fluid transport it outside the blood vessel
        if self._agent_location[0] < 0 or self._agent_location[1] < 0 or self._agent_location[0] >= self._size or self._agent_location[1] >= self._size:
            reward += -50.0
            terminated = True

        
        if truncated:
            reward += -10.0

        
        # Update the timer  
        self._time += self.__timestep
        

        observation = self._get_obs()
        info = self._get_info()
        reward = float(reward)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        """This is a function to render the environment. It represents the elements in the pygame coordinate system. So the
        center is at the top left corner, the x-axis increases as we go further to the right and the y-axis increases as we go further down.
        Concerning the orientation, now a positive orientation rotates the element clockwise instead of counterclockwise to keep the logic
        consistent with the orientation of the axes.
        """
        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (self._window_size, self._window_size)
            )
            
            self._agent_img = pygame.image.load("assets/agent.png").convert_alpha()
            rect = self._agent_img.get_bounding_rect(min_alpha=10)
            self._agent_img = self._agent_img.subsurface(rect).copy()
            d_px = int(round(2 * self.__agent_radius * self.__pix_square_size))
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

        # First we draw the target
        # Convert [row, col] to pygame (x, y) by reversing the coordinates
        pygame.draw.circle(
            canvas,
            (255, 255, 153),
            tuple(((self._target_location[::-1] + 0.5) * self.__pix_square_size).astype(int)),
            int(np.ceil(self.__target_radius * self.__pix_square_size))
        )
        # Now we draw the agent with the appropriate orientation
        angle_deg = -np.degrees(self._orientation)
        rotated_img = pygame.transform.rotate(self._agent_img, angle_deg)
        center = ((self._agent_location[::-1] + 0.5) * self.__pix_square_size).astype(int)
        rect = rotated_img.get_rect(center=tuple(center))
        canvas.blit(rotated_img, rect)

        # Red and white blood cells rendering
        for red_cell in self._red_cells:
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                tuple(((red_cell[::-1] + 0.5) * self.__pix_square_size).astype(int)),
                int(np.ceil(self.__cell_radius * self.__pix_square_size))
            )

        for white_cell in self._white_cells:
            pygame.draw.circle(
                canvas,
                (255, 255, 255),
                tuple(((white_cell[::-1] + 0.5) * self.__pix_square_size).astype(int)),
                int(np.ceil(self.__cell_radius * self.__pix_square_size))
            )


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self._clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    
    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()



register(
    id="Nano-v0",
    entry_point="env:NanoEnv",
)