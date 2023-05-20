import os
from contextlib import closing
from io import StringIO

import numpy as np
import pygame
from colorama import Fore, Style
from gym.spaces import Box, Dict, Discrete
from gymnasium import utils

from envs.frozen_lake.frozen_lake import FrozenLakeMod


class HierarchicalFrozenLakeMod(FrozenLakeMod):
    """
    Grid environment for Hierarchical Reinforcement Learning
    The environment is a 11x11 grid with 4 actions: up, down, left, right
    The agent starts at the center of the grid and it has to reach two objectives.
    The first objective is 4 steps on left of the center and the second objective is 4 steps on right of the center.
    """

    def __init__(self, worker_stratified=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.worker_action_space = Discrete(4)
        self.worker_observation_space = Box(
            low=0,
            high=self.desc.shape[0],
            shape=(12,),
            dtype=np.int32,
        )
        self.manager_action_space = Discrete(self.desc.shape[0] * self.desc.shape[1])
        self.manager_observation_space = Box(
            low=0,
            high=self.desc.shape[0],
            shape=(10,),
            dtype=np.int32,
        )
        self.manager_last_action = self.agent_pos
        self.last_man_dist_objective = self.agent_pos - self.objectives[0]
        self.observation_space = Dict(
            {
                "worker": self.worker_observation_space,
                "manager": self.manager_observation_space,
            }
        )
        self.action_space = Dict(
            {"worker": self.worker_action_space, "manager": self.manager_action_space}
        )
        self.worker_stratifed = worker_stratified
        self.worker_weights = np.array([1, 1])
        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_obstacle": 0,
            "reward_objective": 0,
            "reward_subobjective": 0,
            "reward_manager": 0,
            "reward_worker_sum": 0,
            "worker_done": False,
        }
        self.sub_goal_img = None
        self.steps_count = 0
        self.worker_last_count = 0

    def reset(self):
        _ = super().reset()
        self.steps_count = 0
        self.worker_last_count = 0
        self.agent_pos = self.ori_agent_pos
        self.manager_last_action = self.agent_pos
        self.last_man_dist_objective = self.agent_pos - self.objectives[0]

        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_obstacle": 0,
            "reward_objective": 0,
            "reward_subobjective": 0,
            "reward_manager": 0,
            "reward_worker_sum": 0,
            "worker_done": False,
        }
        return self._get_obs()

    def _get_obs(self):
        manager_obs = super()._get_obs()
        manager_target_x, manager_target_y = (
            self.manager_last_action % self.desc.shape[0],
            self.manager_last_action // self.desc.shape[1],
        )
        worker_obs = np.concatenate([manager_obs, (manager_target_x, manager_target_y)])
        observation = {"worker": worker_obs, "manager": manager_obs}
        return observation

    def _manager_act(self, action):
        self.manager_last_action = action

    def _worker_reward(self):
        reward = np.zeros(2)
        dist = self._dist_reward(objective_pos=self.manager_last_action)
        reward[0] = -10 - dist
        reward[1] = self._obstacle_reward()
        if self.hit_wall:
            reward[0] = -200
            self.cumulative_reward_info["worker_done"] = True
            self.hit_wall = False
        elif dist == 0:
            self.cumulative_reward_info["worker_done"] = True
            self.cumulative_reward_info["reward_subobjective"] += 1
        self.cumulative_reward_info["reward_dist"] += reward[0]
        self.cumulative_reward_info["reward_obstacle"] += reward[1]
        self.cumulative_reward_info["reward_worker_sum"] += reward.sum()
        if not self.worker_stratifed:
            reward = reward.sum()
        return reward

    def _manager_reward(self):
        reward = 0
        dist_objective1 = self._dist_reward(objective_pos=self.objectives[0])
        dist_objective2 = self._dist_reward(objective_pos=self.objectives[1])
        dist_agent = self._dist_reward(objective_pos=self.manager_last_action)
        if dist_agent > 1:
            reward = -10
        else:
            reward = -4
        if dist_agent == 0:
            self.worker_last_count = self.steps_count + 1
        if dist_objective1 == 0 and not self.reached_objectives[0]:
            reward = 2
        if dist_objective2 == 0 and not self.reached_objectives[1]:
            reward = 2
        return reward

    def step(self, action):
        reward = {"worker": 0, "manager": 0}
        done = False
        self.last_action = action["worker"]
        self._do_action(action["worker"])
        reward["worker"] = self._worker_reward()

        self._manager_act(action["manager"])
        reward["manager"] = self._manager_reward()
        self.steps_count += 1

        if self.steps_count - self.worker_last_count > 20:
            print(Fore.RED + "Worker Failure" + Style.RESET_ALL)
            self.cumulative_reward_info["worker_done"] = True
        if self.agent_pos == self.objectives[0] and not self.reached_objectives[0]:
            print(Fore.GREEN + "objective 1 reached" + Style.RESET_ALL)
            self.objective_count += 1
            self.reached_objectives[0] = True
        elif self.agent_pos == self.objectives[1] and not self.reached_objectives[1]:
            self.reached_objectives[1] = True
            print(Fore.GREEN + "objective 2 reached" + Style.RESET_ALL)
            self.objective_count += 1
        if self.reached_objectives[0] and self.reached_objectives[1]:
            done = True
            reward["manager"] = 10
            print(Fore.CYAN + "objective 1 and 2 reached" + Style.RESET_ALL)

        self.cumulative_reward_info["reward_objective"] = self.objective_count
        self.cumulative_reward_info["reward_manager"] += reward["manager"]

        return self._get_obs(), reward, done, self.cumulative_reward_info

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
        if self.sub_goal_img is None:
            file_name = os.path.join(os.path.dirname(__file__), "img/cookie.png")
            self.sub_goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                os.path.join(os.path.dirname(__file__), "img/elf_up.png"),
                os.path.join(os.path.dirname(__file__), "img/elf_down.png"),
                os.path.join(os.path.dirname(__file__), "img/elf_right.png"),
                os.path.join(os.path.dirname(__file__), "img/elf_left.png"),
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

        manager_row, manager_col = (
            self.manager_last_action // self.desc.shape[0],
            self.manager_last_action % self.desc.shape[1],
        )
        manager_cell_rect = (
            manager_col * self.cell_size[0],
            manager_row * self.cell_size[1],
        )
        self.window_surface.blit(self.sub_goal_img, manager_cell_rect)

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
        manager_row, manager_col = (
            self.manager_last_action // self.desc.shape[0],
            self.manager_last_action % self.desc.shape[1],
        )
        desc = [[c for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        desc[manager_row][manager_col] = utils.colorize(
            desc[manager_row][manager_col], "green", highlight=True
        )
        if self.last_action is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.last_action]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()
