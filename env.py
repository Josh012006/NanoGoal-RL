import gymnasium as gym
import numpy as np
from typing import Optional


class NanoEnv(gym.Env):

    def __init__(self, size: int = 1000, minV: float = 0.2, maxV: float = 3.0, maxRed: int = 8, maxWhite: int = 4):
        self.size = size # grid's size : preferably really large to allow a lot of modelisation to the environment
        self._vessel_topology = np.zeros(shape=(size, size), dtype=int) # the vessels layout as a grid with 0 being the empty spaces and 1 being occupied ones by walls

        # The minimum and maximum velocity. They are useful to define real-life constraint on the agent
        self.minV = minV
        self.maxV = maxV
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

    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration namely 
                - nbRed which is the wanted number of red cells with 0 <= nbRed <= maxRed
                - nbWhite which is the wanted number of white cells with 0 <= nbWhite <= maxWhite
                - minV which is the minimum velocity wanted with 0.1 <= minV <= 0.6
                - maxV which is the maximum velocity wanted with 0.1 <= maxV <= 5

        Returns:
            tuple: (observation, info) for the initial state
        """

        # Seed the random number generator
        super().reset(seed=seed)


        # Set the different key variables in a random yet valid way
        

