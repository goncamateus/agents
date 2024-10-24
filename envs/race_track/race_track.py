import sys
from contextlib import closing
from io import StringIO
from os import path

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces, utils


class RacetrackEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render.modes": ["human", "ansi", "rgb_array"],
        "render_fps": 30,
        "render.fps": 30,
    }
    MAP = [
        "022222222222222222222222220",
        "011111111111190111111111110",
        "011111111111190111111111110",
        "011111111111190111111111110",
        "011111111111190111111111110",
        "011111111111190111111111110",
        "011111022222222222220111110",
        "011111002222222222200111110",
        "011111000222222222000111110",
        "011111000222222222000111110",
        "011111000022222220000111110",
        "011111000002222200000111110",
        "011111000000222000000111110",
        "011111000000020000000111110",
        "011111000000000000000111110",
        "011111000000000000000111110",
        "011111000000000000000111110",
        "011111000000000000000111110",
        "011111000000000000000111110",
        "011111000000020000000111110",
        "011111000000222000000111110",
        "011111000002222200000111110",
        "011111000022222220000111110",
        "011111000222222222000111110",
        "011111002222222222200111110",
        "011111022222222222220111110",
        "011111111111111111111111110",
        "011111111111111111111111110",
        "011111111111111111111111110",
        "011111111111111111111111110",
        "011111111111111111111111110",
        "022222222222222222222222220",
    ]

    def __init__(self, render_mode=None):
        # Define the parameters from your description
        super().__init__()
        self.render_mode = render_mode
        self.desc = np.asarray(self.MAP, dtype="c")
        self.track_height = 30
        self.track_width = 25
        self.infield_height = 20
        self.infield_width = 15
        self.infield_y_start = 5
        self.infield_x_start = 5
        self.limit_steps = 1000
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.track_width, self.track_height]),
            shape=(2,),
            dtype=np.int32,
        )
        self.agent_pos = [2, 2 + self.track_width // 2]
        self.agent_velocity = (0, 0)
        self.agent_max_velocity = 1  # Set your desired maximum velocity here
        self.wall_penalty = 1  # Define the penalty for colliding with a wall
        self.window_surface = None
        self.clock = None
        self.cab_image = None
        self.track_image = None
        self.wall_horizontal_image = None
        self.wall_vertical_image = None
        self.window_size = (27 * 20, 32 * 20)
        self.steps_taken = 0
        self.checkpoints = np.array([False, False, False, False], dtype=bool)
        self.cell_size = (
            self.window_size[0] * (32 / 27) / self.desc.shape[0],
            self.window_size[1] * (27 / 32) / self.desc.shape[1],
        )

        # Reward Settings
        self.had_collision = False
        self.last_potential = 0
        self.map_visited = np.zeros((self.track_height, self.track_width), dtype=bool)
        self.cumulative_reward_info = {
            "reward_objective": 0,
            "reward_collision": 0,
            "reward_potential": 0,
            "reward_time": 0,
            "Original_reward": 0,
        }

    def _get_state(self):
        # Return the state of the agent as a tuple of (x, y, vx, vy)
        return np.array(
            [
                self.agent_pos[1],
                self.agent_pos[0],
                # 5 + self.agent_velocity[1],
                # 5 + self.agent_velocity[0],
            ],
            dtype=np.int32,
        )

    def _handle_wall_collision(self, new_pos, new_velocity):
        # Handle collisions with walls and adjust position and velocity
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.track_height
            or new_pos[1] < 0
            or new_pos[1] >= self.track_width
        ):
            new_pos = (
                max(0, min(new_pos[0], self.track_height - 1)),
                max(0, min(new_pos[1], self.track_width - 1)),
            )
            new_velocity = (0, 0)
            self.had_collision = True
        else:
            self.had_collision = False
        return new_pos, new_velocity

    def _handle_infield_collision(self, new_pos, new_velocity):
        # Handle collisions with infield and adjust position and velocity
        # if the agent is in the infield put it back on the track parralel to the wall
        if (
            new_pos[0] >= self.infield_y_start
            and new_pos[0] < self.infield_y_start + self.infield_height
            and new_pos[1] >= self.infield_x_start
            and new_pos[1] < self.infield_x_start + self.infield_width
        ):
            meiota = (self.agent_pos[0] >= 5 or self.agent_pos[0] < 25) and (
                self.agent_pos[1] >= 20 or self.agent_pos[1] < 5
            )
            if not meiota:
                # If comming from up
                if self.agent_pos[0] < new_pos[0] and self.agent_pos[1] == new_pos[1]:
                    new_pos[0] = self.infield_y_start - 1
                # If comming from down
                elif self.agent_pos[0] > new_pos[0] and self.agent_pos[1] == new_pos[1]:
                    new_pos[0] = self.infield_y_start + self.infield_height
                # If comming from up-left
                elif self.agent_pos[0] < new_pos[0] and self.agent_pos[1] < new_pos[1]:
                    new_pos[0] = self.infield_y_start - 1
                    new_pos[1] = self.agent_pos[1]
                # If comming from up-right
                elif self.agent_pos[0] < new_pos[0] and self.agent_pos[1] > new_pos[1]:
                    new_pos[0] = self.infield_y_start - 1
                    new_pos[1] = self.agent_pos[1]
                # If comming from down-left
                elif self.agent_pos[0] > new_pos[0] and self.agent_pos[1] < new_pos[1]:
                    new_pos[0] = self.infield_y_start + self.infield_height
                    new_pos[1] = self.agent_pos[1]
                # If comming from down-right
                elif self.agent_pos[0] > new_pos[0] and self.agent_pos[1] > new_pos[1]:
                    new_pos[0] = self.infield_y_start + self.infield_height
                    new_pos[1] = self.agent_pos[1]
            else:
                # If comming from left
                if self.agent_pos[1] < new_pos[1] and self.agent_pos[0] == new_pos[0]:
                    new_pos[1] = self.infield_x_start - 1
                # If comming from right
                elif self.agent_pos[1] > new_pos[1] and self.agent_pos[0] == new_pos[0]:
                    new_pos[1] = self.infield_x_start + self.infield_width
                # If comming from up-left
                elif self.agent_pos[0] < new_pos[0] and self.agent_pos[1] < new_pos[1]:
                    new_pos[0] = self.agent_pos[0]
                    new_pos[1] = self.infield_x_start - 1
                # If comming from up-right
                elif self.agent_pos[0] < new_pos[0] and self.agent_pos[1] > new_pos[1]:
                    new_pos[0] = self.agent_pos[0]
                    new_pos[1] = self.infield_x_start + self.infield_width
                # If comming from down-left
                elif self.agent_pos[0] > new_pos[0] and self.agent_pos[1] < new_pos[1]:
                    new_pos[0] = self.agent_pos[0]
                    new_pos[1] = self.infield_x_start - 1
                # If comming from down-right
                elif self.agent_pos[0] > new_pos[0] and self.agent_pos[1] > new_pos[1]:
                    new_pos[0] = self.agent_pos[0]
                    new_pos[1] = self.infield_x_start + self.infield_width

            new_velocity = (0, 0)
            self.had_collision = True
        else:
            self.had_collision = False
        return new_pos, new_velocity

    def _handle_start_collision(self, new_pos, new_velocity):
        agent_y, agent_x = self.agent_pos
        new_y, new_x = new_pos
        up = agent_y < self.infield_y_start
        mid_right = agent_x > self.track_width // 2
        if up and mid_right:
            if new_x == 1 + self.track_width // 2:
                new_pos = (new_y, 2 + self.track_width // 2)
                new_velocity = (0, 0)
                self.had_collision = True
        return new_pos, new_velocity

    def _do_action(self, action):
        # Calculate the new velocity based on the action
        action_idx = (
            action // 3
        )  # Convert action to an index for horizontal velocity change (-1, 0, 1)
        action_idy = (
            action % 3
        )  # Convert action to an index for vertical velocity change (-1, 0, 1)
        new_velocity = [
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
        ]

        # Update agent's position based on velocity
        new_pos = [
            self.agent_pos[0] + new_velocity[0],
            self.agent_pos[1] + new_velocity[1],
        ]

        new_pos, new_velocity = self._handle_wall_collision(new_pos, new_velocity)
        new_pos, new_velocity = self._handle_infield_collision(new_pos, new_velocity)
        new_pos, new_velocity = self._handle_start_collision(new_pos, new_velocity)

        # Update agent's position and velocity
        self.agent_pos = new_pos
        self.agent_velocity = new_velocity

    def _randomize_action(self, action):
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
        return action

    def _potential_reward(self):
        reward = 0
        agent_y, agent_x = self.agent_pos
        up = agent_y < self.infield_y_start
        down = agent_y >= self.infield_y_start + self.infield_height
        right = agent_x >= self.infield_x_start + self.infield_width
        left = agent_x < self.infield_x_start
        tile_check = self.map_visited[agent_y, agent_x]
        if up:
            if tile_check:
                reward = -1
            else:
                reward = 1
                self.map_visited[: self.infield_y_start, agent_x] = True
                self.map_visited[:agent_y, 1 + self.track_width // 2 :] = True
        elif down:
            if tile_check:
                reward = -1
            else:
                reward = 1
                self.map_visited[
                    self.infield_y_start + self.infield_height :, agent_x
                ] = True
                self.map_visited[agent_y:, agent_x:] = True
        elif right:
            if tile_check:
                reward = -1
            else:
                reward = 1
                self.map_visited[
                    agent_y, self.infield_x_start + self.infield_width :
                ] = True
        elif left:
            if tile_check:
                reward = -1
            else:
                reward = 1
                self.map_visited[agent_y, : self.infield_x_start] = True
        return reward

    def _check_lap_finished(self):
        agent_y, agent_x = self.agent_pos
        if agent_x == -1 + self.track_width // 2 and agent_y < self.infield_y_start:
            return True
        return False

    def _calculate_reward_done(self):
        # Rewards: [Lap finished, Collision, Velocity towards goal]
        reward = np.array([0, 0, 0, 0], dtype=np.float32)
        finished_lap = self._check_lap_finished()
        if finished_lap:
            print(utils.colorize("OH YEAH", "blue", highlight=True))
        potential_reward = self._potential_reward()
        penalty = self.wall_penalty if self.had_collision else 0
        reward[0] = 1 if finished_lap else 0
        self.cumulative_reward_info["reward_objective"] += reward[0]
        reward[1] = -penalty / 100
        self.cumulative_reward_info["reward_collision"] += penalty
        reward[2] = potential_reward / 1000
        self.cumulative_reward_info["reward_potential"] += potential_reward
        reward[3] = -1e-3 if not finished_lap else 0
        self.cumulative_reward_info["reward_time"] += -1e3 * reward[3]
        return reward, finished_lap

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
        # action = self._randomize_action(action)
        self._do_action(action)
        self.steps_taken += 1
        state = self._get_state()
        reward, done = self._calculate_reward_done()
        done = done or self.steps_taken > self.limit_steps
        self.cumulative_reward_info.update({"Original_reward": reward.sum()})
        if self.render_mode == "human":
            self.render()
        return (
            state,
            reward,
            done,
            self.steps_taken > self.limit_steps,
            self.cumulative_reward_info,
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)
        self.agent_pos = [2, 2 + self.track_width // 2]
        self.agent_velocity = (0, 0)
        # Reward Settings
        self.had_collision = False
        self.previous_pos = self.agent_pos
        self.steps_taken = 0
        self.checkpoints = np.array([False, False, False, False], dtype=bool)
        self.last_potential = (
            np.abs(np.array(self.agent_pos) - np.array([15, 22])).sum() / 21
        )
        self.cumulative_reward_info = {
            "reward_objective": 0,
            "reward_collision": 0,
            "reward_potential": 0,
            "reward_time": 0,
            "Original_reward": 0,
        }
        if self.render_mode == "human":
            self.render()

        return self._get_state(), self.cumulative_reward_info

    def render(self):
        row, col = self.agent_pos
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row + 1][col + 1] = "5"
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.render_mode == "ansi":
            return self._render_text(desc)
        if self.render_mode in ["human", "rgb_array"]:
            return self._render_gui(desc)

    def _render_gui(self, desc):
        if self.window_surface is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Racetrack")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.cab_image is None:
            self.cab_image = pygame.transform.scale(
                pygame.image.load(
                    path.join(path.dirname(__file__), "img/cab_right.png")
                ),
                self.cell_size,
            )
        if self.track_image is None:
            self.track_image = pygame.transform.scale(
                pygame.image.load(path.join(path.dirname(__file__), "img/track.png")),
                self.cell_size,
            )
        if self.wall_horizontal_image is None:
            self.wall_horizontal_image = pygame.transform.scale(
                pygame.image.load(
                    path.join(path.dirname(__file__), "img/wall_horizontal.png")
                ),
                self.cell_size,
            )
        if self.wall_vertical_image is None:
            self.wall_vertical_image = pygame.transform.scale(
                pygame.image.load(
                    path.join(path.dirname(__file__), "img/wall_vertical.png")
                ),
                self.cell_size,
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

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _render_text(self, desc):
        outfile = outfile = StringIO()
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
        with closing(outfile):
            return outfile.getvalue()


if __name__ == "__main__":
    env = RacetrackEnv(render_mode="human")
    env.reset()
    epi_reward = 0
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
        epi_reward += r
        print("state: ", s)
        print("reward: ", r)
        # print("done: ", d)
        # print("info: ", info)
        if d:
            print(epi_reward)
            print(epi_reward.sum())
            break
