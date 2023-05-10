import os
from contextlib import closing
from io import StringIO

import gym
import numpy as np
import pygame
from gym.spaces import Box, Discrete
from gymnasium import utils

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


class FrozenLakeMod(gym.Env):
    """
    Grid environment for Reinforcement Learning
    The environment is a 11x11 grid with 4 actions: up, down, left, right
    The agent starts at the center of the grid and it has to reach two objectives.
    The first objective is 4 steps on left of the center and the second objective is 4 steps on right of the center.
    """

    metadata = {
        "render.modes": ["human", "ansi", "rgb_array"],
        "render.fps": 4,
    }

    desc = np.full((11, 11), "F", dtype="U1")

    def __init__(self, stratified=False):
        super().__init__()
        self.stratified = stratified
        self.num_rewards = 3
        self.ori_weights = np.ones(self.num_rewards)

        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0,
            high=self.desc.shape[0] * self.desc.shape[1],
            shape=(3,),
            dtype=np.int32,
        )
        self.agent_pos = 60

        self.objectives = np.array([56, 64])
        self.objective_count = 0

        self.obstacle_pos = 58
        # gaussian for static obstacle reward calculation
        self.obstacle_max_punish = 20
        self.obstacle_gauss_xvar = 0.2
        self.obstacle_gauss_xycov = 0
        self.obstacle_gauss_yxcov = 0
        self.obstacle_gauss_yvar = 0.2

        self.max_dist = np.sqrt(self.desc.shape[0] ** 2 + self.desc.shape[1] ** 2)
        self.last_dist_objective = 4 / self.max_dist
        self.last_dist_obstacle = 2 / self.max_dist

        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_obstacle": 0,
            "reward_objective": 0,
            "Original_reward": 0,
        }

        # pygame utils
        self.last_action = None
        self.window_size = (
            min(64 * self.desc.shape[0], 512),
            min(64 * self.desc.shape[1], 512),
        )
        self.cell_size = (
            self.window_size[0] // self.desc.shape[0],
            self.window_size[1] // self.desc.shape[1],
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

    def reset(self):
        self.last_action = None
        self.objective_count = 0
        self.agent_pos = 60
        self.objectives = np.array([56, 64])
        self.obstacle_pos = 58
        self.last_dist_objective = 4
        self.last_dist_obstacle = 2

        self.desc = np.full((self.desc.shape[0], self.desc.shape[1]), "F", dtype="U1")
        self.desc[self.agent_pos // self.desc.shape[0]][
            self.agent_pos % self.desc.shape[1]
        ] = "S"
        self.desc[self.objectives[0] // self.desc.shape[0]][
            self.objectives[0] % self.desc.shape[1]
        ] = "G"
        self.desc[self.objectives[1] // self.desc.shape[0]][
            self.objectives[1] % self.desc.shape[1]
        ] = "G"
        self.desc[self.obstacle_pos // self.desc.shape[0]][
            self.obstacle_pos % self.desc.shape[1]
        ] = "H"

        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_obstacle": 0,
            "reward_objective": 0,
            "Original_reward": 0,
        }
        return self._get_obs()

    def min_max_norm(self, val, min, max):
        return (val - min) / (max - min)

    def _dist_reward(self):
        agent_x = self.agent_pos % self.desc.shape[0]
        agent_y = self.agent_pos // self.desc.shape[1]
        objective_x = self.objectives[self.objective_count] % self.desc.shape[0]
        objective_y = self.objectives[self.objective_count] // self.desc.shape[1]
        dist = np.sqrt((agent_x - objective_x) ** 2 + (agent_y - objective_y) ** 2)
        dist = self.min_max_norm(dist, 0, self.max_dist)
        dist_reward = self.last_dist_objective - dist
        self.last_dist_objective = dist
        return dist_reward

    def _obstacle_reward(self):
        agent_x = self.agent_pos % self.desc.shape[0]
        agent_y = self.agent_pos // self.desc.shape[1]
        obstacle_x = self.obstacle_pos % self.desc.shape[0]
        obstacle_y = self.obstacle_pos // self.desc.shape[1]
        dist = np.sqrt((agent_x - obstacle_x) ** 2 + (agent_y - obstacle_y) ** 2)
        dist = self.min_max_norm(dist, 0, self.max_dist)
        dist_obstacle = -(self.last_dist_obstacle - dist)
        self.last_dist_obstacle = dist
        return dist_obstacle

    def _do_action(self, action):
        if action == LEFT:
            if self.agent_pos % self.desc.shape[1] != 0:
                self.agent_pos -= 1
        elif action == DOWN:
            if self.agent_pos > 10:
                self.agent_pos -= self.desc.shape[0]
        elif action == RIGHT:
            if self.agent_pos % self.desc.shape[1] != 10:
                self.agent_pos += 1
        elif action == UP:
            if self.agent_pos < 110:
                self.agent_pos += self.desc.shape[0]

    def _get_obs(self):
        return np.array(
            [self.agent_pos, self.obstacle_pos, self.objectives[self.objective_count]]
        )

    def step(self, action):
        self.last_action = action
        self._do_action(action)

        reward = np.zeros(self.num_rewards)
        done = False
        if self.agent_pos == self.obstacle_pos:
            done = True
            reward[2] += -200
        if self.agent_pos == self.objectives[1]:
            done = True
            if self.objective_count == 0:
                reward[2] -= 20
            else:
                reward[2] += 200
        elif self.agent_pos == self.objectives[0] and self.objective_count == 0:
            reward[2] += 50
            self.objective_count += 1

        reward[0] = self._dist_reward()
        reward[1] = self._obstacle_reward()
        self.cumulative_reward_info["reward_dist"] += reward[0]
        self.cumulative_reward_info["reward_obstacle"] += reward[1]
        self.cumulative_reward_info["reward_objective"] += reward[2]
        self.cumulative_reward_info["Original_reward"] += reward.sum()

        if not self.stratified:
            reward = (reward * self.ori_weights).sum()

        return self._get_obs(), reward, done, self.cumulative_reward_info

    def render(self, render_mode):
        if render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(render_mode)

    def _render_gui(self, mode="human"):
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Grid Environment")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                os.path.join(os.path.dirname(__file__), "img/elf_left.png"),
                os.path.join(os.path.dirname(__file__), "img/elf_down.png"),
                os.path.join(os.path.dirname(__file__), "img/elf_right.png"),
                os.path.join(os.path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]
        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.desc.shape[0]):
            for x in range(self.desc.shape[1]):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == "H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == "G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == "S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = (
            self.agent_pos // self.desc.shape[0],
            self.agent_pos % self.desc.shape[1],
        )
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.last_action if self.last_action is not None else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == "H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = (
            self.agent_pos // self.desc.shape[0],
            self.agent_pos % self.desc.shape[1],
        )
        desc = [[c for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.last_action is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.last_action]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


# class HierarchicalFrozenLakeMod(FrozenLakeMod):
#     """
#     Grid environment for Hierarchical Reinforcement Learning
#     The environment is a 11x11 grid with 4 actions: up, down, left, right
#     The agent starts at the center of the grid and it has to reach two objectives.
#     The first objective is 4 steps on left of the center and the second objective is 4 steps on right of the center.
#     """

#     def __init__(self) -> None:
#         super().__init__()
#         self.worker_action_space = Discrete(4)
#         self.worker_observation_space = Discrete(121)
#         self.manager_action_space = Discrete(121)
#         self.manager_observation_space = Discrete(121)

#         self.worker_state = 60
#         self.manager_last_action = 60
