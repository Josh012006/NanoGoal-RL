import gymnasium as gym
import numpy as np
from typing import Optional
from noise import pnoise2
from utils import main_related_component


class NanoEnv(gym.Env):

    def __init__(self, size: int = 1000, minV: float = 0.2, maxV: float = 3.0, maxRed: int = 8, maxWhite: int = 4):
        self._size = size # grid's size : preferably really large to allow a lot of modelisation to the environment
        self._vessel_topology = np.zeros(shape=(size, size), dtype=int) # the vessels layout as a grid with 0 being the empty spaces and 1 being occupied ones by walls

        # The minimum and maximum velocity. They are useful to define real-life constraint on the agent
        self._minV = minV
        self._maxV = maxV

        # The maximum number of red and white cells in the simulation. AN exact number will be chosen randomly
        self._maxRed = maxRed
        self._maxWhite = maxWhite

        # Agent and target initial locations
        self._agent_location = np.array([-1, -1], dtype=np.float32)
        self._target_location = np.array([-1, -1], dtype=np.float32)

        # Initial velocity and orientation
        self._velocity = 0.0
        self._orientation = 0.0

        # Blood and white cells initial locations
        self._red_cells = np.full(shape=(maxRed, 2), fill_value=-1, dtype=np.float32)
        self._white_cells = np.full(shape=(maxWhite, 2), fill_value=-1, dtype=np.float32)


        # What the agent can observe
        self.observation_space = gym.spaces.Dict(
            {
                "agent" : gym.spaces.Box(-1.0, float(size), shape=(2,), dtype=np.float32), # -1.0 and size will be used to reprensent element outside of the visible box
                "target" : gym.spaces.Box(-1.0, float(size), shape=(2,), dtype=np.float32),
                "mvt" : gym.spaces.Box([minV, -np.pi], [maxV, np.pi], shape=(2,), dtype=np.float32),
                "obstacles": gym.spaces.Dict(
                    {
                        "red" : gym.spaces.Box(-1.0, float(size), shape=(maxRed, 2), dtype=np.float32),
                        "white" : gym.spaces.Box(-1.0, float(size), shape=(maxWhite, 2), dtype=np.float32)
                    }
                )
            }
        )


        # The actions available to the agent
        self.action_space = gym.spaces.Box([minV, -np.pi], [maxV, np.pi], shape=(2,), dtype=np.float32)


    
    def _get_obs(self):
        """Convert the internal state to observation format

        Returns:
            dict: Observation with agent and target positions, obstacles positions and  current motion parameters
        """

        return {
            "agent" : self._agent_location,
            "target" : self._target_location,
            "mvt" : np.array([self._velocity, self._orientation]),
            "obstacles": {
                "red" : self._red_cells,
                "white" : self._white_cells
            }
        }
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }


    def _generate_logical_topology(self, seed: Optional[int] = None):
        """Generates a size x size matrix describing the vessel's topology. It must be a valid topology
            but it must also have a bit of diversity (randomness). It uses Perlin's noise.
        
        Returns:
            matrix: a numpy array describing the generated topology
        """
        grid = np.zeros(shape=(self._size, self._size), dtype=int) # completely empty initially

        treshold = self.np_random.uniform(0.1, 0.5) # treshold to decide if it is a wall or an empty space
        alpha = self.np_random.uniform(0.2, 0.6) # used so that the cells at the end of the grid are more empty

        for i in range(self._size):
            for j in range(self._size):
                noise_num = pnoise2(j/100, i/100, base=seed, octaves=4, persistence=0.5, lacunarity=2.0) # generate perlin noise 
                if((noise_num + alpha * i/(self._size - 1)) < treshold): grid[i][j] = 1

        return grid
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration namely 
                - nbRed which is the wanted number of red cells with 0 <= nbRed <= maxRed
                - nbWhite which is the wanted number of white cells with 0 <= nbWhite <= maxWhite

        Returns:
            tuple: (observation, info) for the initial state
        """

        # Seed the random number generator
        super().reset(seed=seed)

        # Generate a pseudo-random but also valid vessel topology for the episode
        self._vessel_topology = self._generate_logical_topology(seed)

        available_space = main_related_component(self._vessel_topology)
        new_seed = 1 + 0 if seed == None else seed
        while available_space == []: # making sure there is at least a related component in the generated environment
            self._vessel_topology = self._generate_logical_topology(new_seed)
            available_space = main_related_component(self._vessel_topology)
            new_seed += 1

        # Randomly generated target and agent locations in regard to the topology 
        agent_int = self.np_random.integers(0, len(available_space))
        self._agent_location = np.array(list(available_space[agent_int]), dtype=np.float32)
        available_space.pop(agent_int)

        target_int = self.np_random.integers(0, len(available_space))
        self._target_location = np.array(list(available_space[target_int]), dtype=np.float32)
        available_space.pop(target_int)

        # Velocity and orientation at the start of an episode
        self._velocity = 0.0
        self._orientation = self.np_random.uniform(-np.pi, np.pi)

        # Random red and white cells locations in regard to the topology and the options nbRed and nbWhite
        nbRed = self.np_random.integers(0, self._maxRed)
        if options != None and "nbRed" in options:
            nbRed = max(0, options["nbRed"])
        nbRed = min(len(available_space) // 2, min(self._maxRed, nbRed))
        self._red_cells = np.full(shape=(self._maxRed, 2), fill_value=-1, dtype=np.float32)
        for i in range(nbRed):
            drawn_int = self.np_random.integers(0, len(available_space))
            self._red_cells[i] = list(available_space[drawn_int])
            available_space.pop(drawn_int)
        
        nbWhite = self.np_random.integers(0, self._maxWhite)
        if options != None and "nbWhite" in options:
            nbWhite = max(0, options["nbWhite"])
        nbWhite = min(len(available_space), min(self._maxWhite, nbWhite))
        self._white_cells = np.full(shape=(self._maxWhite, 2), fill_value=-1, dtype=np.float32)
        for i in range(nbWhite):
            drawn_int = self.np_random.integers(0, len(available_space))
            self._red_cells[i] = list(available_space[drawn_int])
            available_space.pop(drawn_int)

        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    
    

