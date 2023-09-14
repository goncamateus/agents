from typing import Optional

import numpy as np
from gymnasium.envs.toy_text.taxi import TaxiEnv
from gymnasium.spaces import Box, Dict


class StratTaxiEnv(TaxiEnv):
    metadata = {
        "render_fps": 30,
        "render_modes": ["human", "rgb_array", "ansi"],
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode=render_mode)
        self.reward_space = Dict(
            {
                "decomposed": Box(-1, 1, shape=(3,)),
                "original": Box(-1, 1, shape=()),
            }
        )
        self.reward_dim = 3
        num_states = 500
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        num_actions = 6
        self.initial_state_distrib = np.zeros(num_states)
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(self.locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(self.locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx:
                            self.initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = {
                                "decomposed": np.array([-1 / 200, 0, 0]),
                                "original": -1,
                            }
                            terminated = False
                            taxi_loc = (row, col)

                            if action == 0:
                                new_row = min(row + 1, max_row)
                            elif action == 1:
                                new_row = max(row - 1, 0)
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
                                    new_pass_idx = 4
                                else:  # passenger not at location
                                    reward = {
                                        "decomposed": np.array([0, 0, -1 / 10]),
                                        "original": -10,
                                    }
                            elif action == 5:  # dropoff
                                if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = dest_idx
                                    terminated = True
                                    reward = {
                                        "decomposed": np.array([0, 1, 0]),
                                        "original": 20,
                                    }
                                elif (taxi_loc in self.locs) and pass_idx == 4:
                                    new_pass_idx = self.locs.index(taxi_loc)
                                else:  # dropoff at wrong location
                                    reward = {
                                        "decomposed": np.array([0, 0, -1 / 10]),
                                        "original": -10,
                                    }
                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx
                            )
                            self.P[state][action].append(
                                (1.0, new_state, reward, terminated)
                            )

        self.initial_state_distrib /= self.initial_state_distrib.sum()

    def step(self, action):
        state, reward, done, truncated, info = super().step(action)
        info.update({"Original_reward": reward["original"]})
        return state, reward["decomposed"], done, truncated, info
