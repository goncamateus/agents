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

    def __init__(self) -> None:
        super().__init__()
        self.worker_action_space = Discrete(4)
        self.worker_observation_space = Box(
            low=0,
            high=self.desc.shape[0] * self.desc.shape[1],
            shape=(3,),
            dtype=np.int32,
        )
        self.manager_action_space = Discrete(121)
        self.manager_observation_space = Box(
            low=0,
            high=self.desc.shape[0] * self.desc.shape[1],
            shape=(4,),
            dtype=np.int32,
        )
        self.manager_last_action = 56
        self.observation_space = Dict(
            {
                "worker": self.worker_observation_space,
                "manager": self.manager_observation_space,
            }
        )
        self.action_space = Dict(
            {"worker": self.worker_action_space, "manager": self.manager_action_space}
        )
        self.steps_count = 0
        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_obstacle": 0,
            "reward_objective": 0,
            "reward_manager": 0,
            "Original_reward": 0,
        }

    def reset(self):
        _ = super().reset()
        self.steps_count = 0
        self.agent_pos = 60
        self.manager_last_action = 56

        self.desc[self.agent_pos // self.desc.shape[0]][
            self.agent_pos % self.desc.shape[1]
        ] = "F"
        self.desc[self.manager_last_action // self.desc.shape[0]][
            self.manager_last_action % self.desc.shape[1]
        ] = "S"

        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_obstacle": 0,
            "reward_objective": 0,
            "reward_manager": 0,
        }
        return self._get_obs()

    def _get_obs(self):
        manager_obs = np.array(
            [self.agent_pos, self.obstacle_pos, self.objectives[0], self.objectives[1]]
        )
        worker_obs = np.array(
            [self.agent_pos, self.obstacle_pos, self.manager_last_action]
        )
        observation = {"worker": worker_obs, "manager": manager_obs}
        return observation

    def _manager_act(self, action):
        self.desc[self.manager_last_action // self.desc.shape[0]][
            self.manager_last_action % self.desc.shape[1]
        ] = "F"
        self.manager_last_action = action
        self.desc[self.manager_last_action // self.desc.shape[0]][
            self.manager_last_action % self.desc.shape[1]
        ] = "S"

    def _worker_reward(self):
        reward = np.zeros(2)
        reward[0] = self._dist_reward(obejctive_pos=self.manager_last_action)
        reward[1] = self._obstacle_reward()
        self.cumulative_reward_info["reward_dist"] += reward[0]
        self.cumulative_reward_info["reward_obstacle"] += reward[1]
        return reward

    def _manager_reward(self):
        reward = np.zeros(1)
        objective_x = self.objectives[self.objective_count] % self.desc.shape[0]
        objective_y = self.objectives[self.objective_count] // self.desc.shape[1]
        action_x = self.manager_last_action % self.desc.shape[0]
        action_y = self.manager_last_action // self.desc.shape[1]
        dist = np.sqrt((action_x - objective_x) ** 2 + (action_y - objective_y) ** 2)
        manhattan_dist = abs(action_x - objective_x) + abs(action_y - objective_y)
        if dist > 2:
            reward[0] = -1
        elif dist <= 2:
            reward[0] = -0.4
        if action_x == objective_x or action_y == objective_y:
            reward[0] = 0.2
        if dist == 0:
            reward[0] = 1
        if manhattan_dist > 10:
            reward[0] = -1
        self.cumulative_reward_info["reward_manager"] += reward[0]
        return reward

    def step(self, action):
        reward = {"worker": 0, "manager": 0}
        self.last_action = action["worker"]
        self._do_action(action["worker"])
        reward["worker"] = self._worker_reward()
        self.steps_count += 1

        if self.steps_count % 10 == 0:
            self._manager_act(action["manager"])
            reward["manager"] = self._manager_reward()

        done = False
        if self.agent_pos == self.obstacle_pos:
            print(Fore.RED + "Failure" + Style.RESET_ALL)
            done = True
            self.cumulative_reward_info["reward_objective"] += -1
        if self.agent_pos == self.objectives[1]:
            done = True
            if self.objective_count != 0:
                self.cumulative_reward_info["reward_objective"] += 1
                print(Fore.CYAN + "objective 1 and 2 reached" + Style.RESET_ALL)
        elif self.agent_pos == self.objectives[0] and self.objective_count == 0:
            print(Fore.GREEN + "objective 1 reached" + Style.RESET_ALL)
            self.objective_count += 1
            self.cumulative_reward_info["reward_objective"] += 0.5
            agent_x = self.agent_pos % self.desc.shape[0]
            agent_y = self.agent_pos // self.desc.shape[1]
            action_x = self.manager_last_action % self.desc.shape[0]
            action_y = self.manager_last_action // self.desc.shape[1]
            self.last_dist_objective = np.sqrt(
                (action_x - agent_x) ** 2 + (action_y - agent_y) ** 2
            )

        return self._get_obs(), reward, done, self.cumulative_reward_info

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
