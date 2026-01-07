from typing import Optional
import gymnasium as gym
import numpy as np
import pygame
from noise import pnoise2
from utils import main_related_component

from gymnasium.envs.registration import register

class NanoEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str = None, min_v: float = 1.0, max_v: float = 6.0, max_red: int = 8, max_white: int = 4):

        # Discrete representation as a grid
        self._size = 125  # grid's size
        self._vessel_topology = np.zeros(shape=(self._size, self._size), dtype=int) # the vessels layout as a grid with 0 being the empty spaces and 1 being occupied ones by walls

        # Entities characteristics
        self.__agent_radius = 8.0
        self.__cell_radius = 1.0
        self.__target_radius = 2.0

        # The number of red and white cells in the simulation. An exact number will be chosen randomly
        self._nb_red = 0
        self._nb_white = 0
        self._max_red = max(0, min(max_red, 20))
        self._max_white = max(0, min(max_white, 20))

        # The minimum and maximum velocity. They are useful to define real-life constraint on the agent
        self._min_v = min_v
        self._max_v = max_v

        # Agent and target initial locations
        self._agent_location = np.array([-1, -1], dtype=np.float32)
        self._target_location = np.array([-1, -1], dtype=np.float32)

        # Blood and white cells initial locations
        self._red_cells = np.full(shape=(max_red, 2), fill_value=-1, dtype=np.float32)
        self._white_cells = np.full(shape=(max_white, 2), fill_value=-1, dtype=np.float32)

        # Initial agent's velocity and orientation
        self._velocity = 0.0
        self._orientation = 0.0

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
                "agent" : gym.spaces.Box(-1.0, float(self._size), shape=(2,), dtype=np.float32),
                "target" : gym.spaces.Box(-1.0, float(self._size), shape=(2,), dtype=np.float32),
                "mvt" : gym.spaces.Box(
                    low=np.array([min_v, -np.pi], dtype=np.float32), 
                    high=np.array([max_v, np.pi], dtype=np.float32), 
                    shape=(2,), 
                    dtype=np.float32
                ),
                "obstacles": gym.spaces.Dict(
                    {
                        "red" : gym.spaces.Box(-1.0, float(self._size), shape=(self._max_red, 2), dtype=np.float32),
                        "white" : gym.spaces.Box(-1.0, float(self._size), shape=(self._max_white, 2), dtype=np.float32)
                    }
                )
            }
        )

        # The actions available to the agent
        self.action_space = gym.spaces.Box(
            low=np.array([min_v, -np.pi], dtype=np.float32), 
            high=np.array([max_v, np.pi], dtype=np.float32), 
            shape=(2,), 
            dtype=np.float32
        )


    
    def _get_obs(self):
        """Convert the internal state to observation format

        Returns:
            dict: Observation with agent and target positions, obstacles positions and  current motion parameters
        """

        return {
            "agent" : self._agent_location,
            "target" : self._target_location,
            "mvt" : np.array([self._velocity, self._orientation], dtype=np.float32),
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
                self._agent_location - self._target_location
            )
        }


    def _generate_logical_topology(self, seed: Optional[int] = None):
        """Generates a size x size matrix describing the vessel's topology. It must be a valid topology
            but it must also have a bit of diversity (randomness). It uses Perlin's noise.
        
        Returns:
            matrix: a numpy array describing the generated topology
        """
        grid = np.zeros(shape=(self._size, self._size), dtype=int) # completely empty initially
        base = int(seed) if seed is not None else 0

        treshold = self.np_random.uniform(0.1, 0.5) # treshold to decide if it is a wall or an empty space

        for i in range(self._size):
            for j in range(self._size):
                n = pnoise2(j/100, i/100, base=base, octaves=6, persistence=0.5, lacunarity=2.0)
                if(n < treshold): grid[i][j] = 1

        return grid

    
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
        super().reset(seed=seed)

        # Generate a pseudo-random but also valid vessel topology for the episode
        self._vessel_topology = self._generate_logical_topology(seed)

        available_space = main_related_component(self._vessel_topology, self._size, self._size)
        new_seed = 1 + 0 if seed == None else seed
        while len(available_space) < 500: # making sure there is at least a related component in the generated environment
            self._vessel_topology = self._generate_logical_topology(new_seed)
            available_space = main_related_component(self._vessel_topology, self._size, self._size)
            new_seed += 1

        print(len(available_space))

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

        # Set the time limit 
        d0 = np.linalg.norm(self._agent_location - self._target_location)
        self.__timelimit = min(3 + 2 * d0, 30)

        # Random red and white cells locations in regard to the topology and the options nb_red and nb_white
        nb_red = self.np_random.integers(0, self._max_red)
        if options != None and "nb_red" in options:
            nb_red = max(0, options["nb_red"])
        nb_red = min(len(available_space) // 2, min(self._max_red, nb_red))
        self._nb_red = nb_red
        self._red_cells = np.full(shape=(self._max_red, 2), fill_value=-1, dtype=np.float32)
        for i in range(nb_red):
            drawn_int = self.np_random.integers(0, len(available_space))
            self._red_cells[i] = list(available_space[drawn_int])
            available_space.pop(drawn_int)
        
        nb_white = self.np_random.integers(0, self._max_white)
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
    
    def _manage_wall_collision(self, old_location, new_location):
        i = int(np.floor(old_location[0]))
        j = int(np.floor(old_location[1]))
        i_new = int(np.floor(new_location[0]))
        j_new = int(np.floor(new_location[1]))

        old_valid = (
            0 <= i < self._size and
            0 <= j < self._size
        )

        new_valid = (
            0 <= i_new < self._size and
            0 <= j_new < self._size
        )

        if not old_valid:
            return old_location
        if not new_valid:
            return old_location
        if self._vessel_topology[i_new, j_new] == 1:
            if self._vessel_topology[i_new, j] != 1:
                return np.array([new_location[0], old_location[1]], dtype=np.float32)
            if self._vessel_topology[i, j_new] != 1:
                return np.array([old_location[0], new_location[1]], dtype=np.float32)
            return old_location

        return new_location
    
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take, namely an array in the format [float, float] where the first component is 
                the component to add to the velocity and the second one is the component to add to
                the orientation. Both can be negative.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        
        delta_v = max(-0.1, min(action[0], 2.0)) # -0.02 <= delta_v <= 0.02
        delta_theta = max(-0.2, min(action[1], 0.2)) # -0.2 <= delta_theta <= 0.2
        wrap = lambda x : (x + np.pi) % (2 * np.pi) - np.pi

        v_blood = np.array([0.4, 0.2], dtype=np.float32)

        self._velocity = max(self._min_v, min(self._velocity + delta_v, self._max_v))
        self._orientation = wrap(self._orientation + delta_theta)
        v_agent = np.array([self._velocity * np.cos(self._orientation), self._velocity * np.sin(self._orientation)], dtype=np.float32)

        # Compute new agent and cells positions with collisions management
        new_agent_location = self._agent_location + (v_agent + v_blood) * self.__timestep
        self._agent_location = self._manage_wall_collision(self._agent_location, new_agent_location)

        for i in range(self._nb_red):
            new_red_cell_position = self._red_cells[i] + 2 * v_blood * self.__timestep
            self._red_cells[i] = self._manage_wall_collision(self._red_cells[i], new_red_cell_position)
        
        for i in range(self._nb_white):
            new_white_cell_position = self._white_cells[i] + 2 * v_blood * self.__timestep
            self._white_cells[i] =  self._manage_wall_collision(self._white_cells[i], new_white_cell_position)

        
        # Update the timer  
        self._time += self.__timestep

        reward = 0

        # Agent-cell collision
        beta = self.np_random.uniform(0.5, 0.8)
        epsilon = self.np_random.uniform(-0.1, 0.1)

        for i in range(self._nb_red):
            if np.linalg.norm(self._red_cells[i] - self._agent_location) < self.__agent_radius + self.__cell_radius:
                self._velocity *= beta
                self._orientation += epsilon
                reward += self.__penalty_red_cell
                break
        
        for i in range(self._nb_white):
            if np.linalg.norm(self._white_cells[i] - self._agent_location) < self.__agent_radius + self.__cell_radius:
                self._velocity *= beta + 0.1
                self._orientation = wrap(self._orientation + epsilon)
                reward += self.__penalty_white_cell
                break

        reward += -np.linalg.norm(self._agent_location - self._target_location)

        terminated = np.linalg.norm(self._agent_location - self._target_location) <= self.__agent_radius + self.__target_radius
        truncated = self._time > self.__timelimit


        # Prevent the agent from letting the fluid transport it outside the blood vessel
        if self._agent_location[0] < 0 or self._agent_location[1] < 0 or self._agent_location[0] >= self._size or self._agent_location[1] >= self._size:
            reward += -50
            truncated = True
            terminated = True
        

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                (self._window_size, self._window_size)
            )
            self._agent_img = pygame.image.load("assets/agent.png").convert_alpha()
            self._agent_img = pygame.transform.smoothscale(
                self._agent_img,
                (int(2 * self.__agent_radius * self.__pix_square_size), int(2 * self.__agent_radius * self.__pix_square_size))
            )
        
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
            int(self.__target_radius * self.__pix_square_size)
        )
        # Now we draw the agent with the appropriate orientation
        angle_deg = -np.degrees(self._orientation)
        rotated_img = pygame.transform.rotate(self._agent_img, angle_deg)
        center = (self._agent_location[::-1] + 0.5) * self.__pix_square_size
        rect = rotated_img.get_rect(center=center)
        canvas.blit(rotated_img, rect)

        # Red and white blood cells rendering
        for red_cell in self._red_cells:
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                tuple(((red_cell[::-1] + 0.5) * self.__pix_square_size).astype(int)),
                int(self.__cell_radius * self.__pix_square_size)
            )

        for white_cell in self._white_cells:
            pygame.draw.circle(
                canvas,
                (255, 255, 255),
                tuple(((white_cell[::-1] + 0.5) * self.__pix_square_size).astype(int)),
                int(self.__cell_radius * self.__pix_square_size)
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
