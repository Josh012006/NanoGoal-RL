import gymnasium as gym
import numpy as np


class NanoEnv(gym.Env):

    def __init__(self, size: int = 1000, minV: float = 0.2, maxV: float = 3.0, maxRed: int = 8, maxWhite: int = 4):
        self.size = size # grid size : preferably really large to allow a lot of modelisation to the environment
        # The minimum and maximum velocity. They are useful to define real-life constraint on the agent
        self.minV = minV
        self.maxV = maxV
        # The maximum number of red and white cells in the simulation. AN exact number will be chosen randomly
        self._maxRed = maxRed
        self._maxWhite = maxWhite

        # Agent and target initial locations
        self._agent_location = np.array([-1, -1])
        self._target_location = np.array([-1, -1])

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
