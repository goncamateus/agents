import sys
from contextlib import closing
from io import StringIO

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces, utils


class RacetrackEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render.modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
        "render.fps": 4,
    }
    MAP = [
        "022222222222222222222222220",
        "011111111111191111111111110",
        "011111111111191111111111110",
        "011111111111191111111111110",
        "011111111111191111111111110",
        "011111111111191111111111110",
        "011111022222222222220111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111011111111111110111110",
        "011111022222222222220111110",
        "011111111111111111111111110",
        "011111111111111111111111110",
        "011111111111111111111111110",
        "011111111111111111111111110",
        "011111111111111111111111110",
        "022222222222222222222222220",
    ]

    def __init__(self):
        # Define the parameters from your description
        self.desc = np.asarray(self.MAP, dtype="c")
        self.track_width = 30
        self.track_height = 25
        self.infield_width = 20
        self.infield_height = 15
        self.infield_y_start = (self.track_width - self.infield_width) // 2
        self.infield_x_start = (self.track_height - self.infield_height) // 2
        self.grid = np.zeros((self.track_height, self.track_width), dtype=int)
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Discrete(25 * 30)
        self.agent_pos = (2, self.track_width // 2)
        self.agent_velocity = (0, 0)
        self.agent_max_velocity = 5  # Set your desired maximum velocity here
        self.wall_penalty = 1  # Define the penalty for colliding with a wall
        self.window_surface = None
        self.clock = None
        self.cab_image = None
        self.track_image = None
        self.wall_horizontal_image = None
        self.wall_vertical_image = None
        self.window_size = (27 * 20, 32 * 20)
        self.cell_size = (
            self.window_size[0] * (32 / 27) / self.desc.shape[0],
            self.window_size[1] * (27 / 32) / self.desc.shape[1],
        )

    def _get_state(self):
        return self.agent_pos[0] * self.track_width + self.agent_pos[1]

    def _handle_wall_collision(self, new_pos, new_velocity):
        # Handle collisions with walls and adjust position and velocity
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.track_width
            or new_pos[1] < 0
            or new_pos[1] >= self.track_height
        ):
            new_pos = (
                max(0, min(new_pos[0], self.track_width - 1)),
                max(0, min(new_pos[1], self.track_height - 1)),
            )
            new_velocity = (0, 0)
        return new_pos, new_velocity

    def _handle_infield_collision(self, new_pos, new_velocity):
        # Handle collisions with infield and adjust position and velocity
        # if the agent is in the infield put it back on the track parralel to the wall
        if (
            new_pos[0] >= self.infield_x_start
            and new_pos[0] < self.infield_x_start + self.infield_width
            and new_pos[1] >= self.infield_y_start
            and new_pos[1] < self.infield_y_start + self.infield_height
        ):
            if new_pos[0] < self.infield_x_start + self.infield_width // 2:
                new_pos = (
                    self.infield_x_start - 1,
                    max(
                        self.infield_y_start,
                        min(new_pos[1], self.infield_y_start + self.infield_height - 1),
                    ),
                )
            else:
                new_pos = (
                    self.infield_x_start + self.infield_width,
                    max(
                        self.infield_y_start,
                        min(new_pos[1], self.infield_y_start + self.infield_height - 1),
                    ),
                )
            if new_pos[1] < self.infield_y_start + self.infield_height // 2:
                new_pos = (
                    max(
                        self.infield_x_start,
                        min(new_pos[0], self.infield_x_start + self.infield_width - 1),
                    ),
                    self.infield_y_start - 1,
                )
            else:
                new_pos = (
                    max(
                        self.infield_x_start,
                        min(new_pos[0], self.infield_x_start + self.infield_width - 1),
                    ),
                    self.infield_y_start + self.infield_height,
                )
            new_velocity = (0, 0)
        return new_pos, new_velocity

    def _do_action(self, action):
        # Calculate the new velocity based on the action
        action_idx = (
            action // 3
        )  # Convert action to an index for horizontal velocity change (-1, 0, 1)
        action_idy = (
            action % 3
        )  # Convert action to an index for vertical velocity change (-1, 0, 1)
        new_velocity = (
            np.clip(
                self.agent_velocity[0] + action_idx - 1,
                -self.agent_max_velocity,
                self.agent_max_velocity,
            ),
            np.clip(
                self.agent_velocity[1] + action_idy - 1,
                -self.agent_max_velocity,
                self.agent_max_velocity,
            ),
        )

        # Update agent's position based on velocity
        new_pos = (
            self.agent_pos[0] + new_velocity[0],
            self.agent_pos[1] + new_velocity[1],
        )

        new_pos, new_velocity = self._handle_wall_collision(new_pos, new_velocity)
        new_pos, new_velocity = self._handle_infield_collision(new_pos, new_velocity)

        # Update agent's position and velocity
        self.agent_pos = new_pos
        self.agent_velocity = new_velocity

    def step(self, action):
        # Actions
        # 0: Up-Left
        # 1: Up
        # 2: Up-Right
        # 3: Left
        # 4: Stay
        # 5: Right
        # 6: Down-Left
        # 7: Down
        # 8: Down-Right
        action_randomness = np.random.random()
        if action_randomness > 0.9:
            action_randomness -= 0.9
            action_randomness *= 10
            if action_randomness < 0.8:
                if action_randomness < 0.6:
                    if action_randomness < 0.15:
                        action = 0
                    elif action_randomness < 0.3:
                        action = 2
                    elif action_randomness < 0.45:
                        action = 6
                    elif action_randomness < 0.6:
                        action = 8
                else:
                    if action_randomness < 0.65:
                        action = 1
                    elif action_randomness < 0.7:
                        action = 3
                    elif action_randomness < 0.75:
                        action = 5
                    elif action_randomness < 0.8:
                        action = 7
            else:
                action = 4

        self._do_action(action)
        state = self._get_state()
        # Calculate rewards (you'll need to define your reward logic)
        reward = 0  # Implement your reward function here

        # Check for terminal conditions (end of episode)
        done = False  # Implement your termination condition

        return state, reward, done, False, {}

    def reset(self):
        self.agent_pos = (2, self.track_width // 2)
        self.agent_velocity = (0, 0)

        return self._get_state()

    def render(self, mode="human"):
        row, col = self.agent_pos
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row + 1][col + 1] = "5"
        if mode == "ansi":
            self._render_text(desc)
        if mode == "human":
            self._render_gui(mode, desc)

    def _render_gui(self, mode, desc):
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Racetrack")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.cab_image is None:
            self.cab_image = pygame.transform.scale(
                pygame.image.load("img/cab_right.png"), self.cell_size
            )
        if self.track_image is None:
            self.track_image = pygame.transform.scale(
                pygame.image.load("img/track.png"), self.cell_size
            )
        if self.wall_horizontal_image is None:
            self.wall_horizontal_image = pygame.transform.scale(
                pygame.image.load("img/wall_horizontal.png"), self.cell_size
            )
        if self.wall_vertical_image is None:
            self.wall_vertical_image = pygame.transform.scale(
                pygame.image.load("img/wall_vertical.png"), self.cell_size
            )

        for y in range(self.desc.shape[0]):
            for x in range(self.desc.shape[1]):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.track_image, pos)
                if desc[y][x] == "5":
                    self.window_surface.blit(self.cab_image, pos)
                elif desc[y][x] == "9":
                    color_cell = pygame.Surface(self.cell_size)
                    color_cell.set_alpha(128)
                    color_cell.fill((255, 255, 0))
                    self.window_surface.blit(color_cell, pos)
                elif desc[y][x] == "0":
                    self.window_surface.blit(self.wall_vertical_image, pos)
                elif desc[y][x] == "2":
                    self.window_surface.blit(self.wall_horizontal_image, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _render_text(self, desc):
        outfile = sys.stdout
        for row in range(len(desc)):
            for col in range(len(desc[row])):
                color = "white"
                if desc[row][col] == "5":
                    color = "red"
                elif desc[row][col] in "9":
                    color = "green"
                elif desc[row][col] in ["0", "2"]:
                    color = "blue"
                desc[row][col] = utils.colorize(" ", color, highlight=True)
        outfile.write("\n".join(["".join(line) for line in desc]) + "\n")


if __name__ == "__main__":
    env = RacetrackEnv()
    env.reset()
    env.render("human")
    while True:
        # Play on number keyboard
        action = ""
        while action not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            action = input("Enter action: ")
        action_map = {
            "7": 0,
            "8": 1,
            "9": 2,
            "4": 3,
            "5": 4,
            "6": 5,
            "1": 6,
            "2": 7,
            "3": 8,
        }
        s, r, d, t, info = env.step(action_map[action])
        print("state: ", s)
        print("reward: ", r)
        print("done: ", d)
        print("info: ", info)
        env.render("human")
