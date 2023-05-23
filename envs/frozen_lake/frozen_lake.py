import os
from contextlib import closing
from io import StringIO

import gym
import numpy as np
import pygame
from colorama import Fore, Style
from gym.spaces import Box, Discrete
from gymnasium import utils
from scipy.stats import multivariate_normal

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3


class FrozenLakeMod(gym.Env):
    """
    Grid environment for Reinforcement Learning
    The environment is a 11x11 grid with 4 actions: up, down, left, right
    The agent starts at the center of the grid and it has to reach two objectives.
    The first objective is 5 steps on left of the center and the second objective is 4 steps on right of the center.
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render.modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
        "render.fps": 4,
    }

    def __init__(self, stratified=False, **kwargs):
        super().__init__()
        self.stratified = stratified
        self.num_rewards = 2
        self.ori_weights = np.array([1, 1])

        self.desc = np.full(kwargs["desc_shape"], "F", dtype="U1")
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0,
            high=self.desc.shape[0],
            shape=(12,),
            dtype=np.int32,
        )
        self.agent_pos = kwargs["agent_pos"]
        self.ori_agent_pos = kwargs["agent_pos"]

        self.objectives = np.array([kwargs["objective_0"], kwargs["objective_1"]])
        self.objective_count = 0
        self.last_fifty_objective_count = []

        self.obstacle_pos = kwargs["obstacle_pos"]
        # gaussian for static obstacle reward calculation
        self.obstacle_max_punish = 20
        self.obstacle_gauss_xvar = 0.2
        self.obstacle_gauss_xycov = 0
        self.obstacle_gauss_yxcov = 0
        self.obstacle_gauss_yvar = 0.2

        self.max_dist = np.sqrt(self.desc.shape[0] ** 2 + self.desc.shape[1] ** 2)
        self.last_dist_objective = self.max_dist

        self.reached_objectives = [False, False]

        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_obstacle": 0,
            "reward_objective": 0,
            "reward_success_rate": 0,
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
        self.man_objective = self.objectives[0]
        self.steps_to_reach = 20

    def reset(self):
        self.reached_objectives = [False, False]
        self.last_action = None
        self.objective_count = 0
        self.steps_to_reach = 20
        self.agent_pos = np.random.choice(self.desc.shape[0] * self.desc.shape[1])
        dist1 = self._dist_reward(objective_pos=self.objectives[0])
        dist2 = self._dist_reward(objective_pos=self.objectives[1])
        if dist1 < dist2:
            self.last_dist_objective = -dist1
            self.man_objective = self.objectives[0]
        else:
            self.last_dist_objective = -dist2
            self.man_objective = self.objectives[1]
        man_objective_x, man_objective_y = (
            self.agent_pos % self.desc.shape[0],
            self.agent_pos // self.desc.shape[1],
        )
        obs = self.get_obs(np.array([man_objective_x, man_objective_y]))
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
            "reward_success_rate": 0,
            "Original_reward": 0,
        }
        self.hit_wall = False
        return obs

    def min_max_norm(self, val, min, max):
        return (val - min) / (max - min)

    def _dist_reward(self, objective_pos):
        agent_x = self.agent_pos % self.desc.shape[0]
        agent_y = self.agent_pos // self.desc.shape[1]
        objective_x = objective_pos % self.desc.shape[0]
        objective_y = objective_pos // self.desc.shape[1]
        dist = abs(agent_x - objective_x) + abs(agent_y - objective_y)
        return dist

    def _obstacle_reward(self):
        agent_x = self.agent_pos % self.desc.shape[0]
        agent_y = self.agent_pos // self.desc.shape[1]
        obstacle_x = self.obstacle_pos % self.desc.shape[0]
        obstacle_y = self.obstacle_pos // self.desc.shape[1]
        activation = self.gaussian_activation(
            x=agent_x,
            y=agent_y,
            xmean=obstacle_x,
            ymean=obstacle_y,
            x_var=self.obstacle_gauss_xvar,
            xy_cov=self.obstacle_gauss_xycov,
            yx_cov=self.obstacle_gauss_yxcov,
            y_var=self.obstacle_gauss_yvar,
        )
        normed_act = self.min_max_norm(
            activation,
            min=0,
            max=self.gaussian_activation(
                x=0,
                y=0,
                xmean=0,
                ymean=0,
                x_var=self.obstacle_gauss_xvar,
                xy_cov=self.obstacle_gauss_xycov,
                yx_cov=self.obstacle_gauss_yxcov,
                y_var=self.obstacle_gauss_yvar,
            ),
        )
        obstacle_punishment = self.obstacle_max_punish * normed_act
        return -obstacle_punishment

    def _do_action(self, action):
        pos_before = self.agent_pos
        x = self.agent_pos % self.desc.shape[0]
        y = self.agent_pos // self.desc.shape[1]

        if action == LEFT and x > 0:
            self.agent_pos -= 1
        elif action == DOWN and y < self.desc.shape[0] - 1:
            self.agent_pos += self.desc.shape[1]
        elif action == RIGHT and x < self.desc.shape[1] - 1:
            self.agent_pos += 1
        elif action == UP and y > 0:
            self.agent_pos -= self.desc.shape[1]
        else:
            self.hit_wall = True
            return

        if self.agent_pos == self.obstacle_pos:
            self.agent_pos = pos_before
            
    def get_obs(self, cat_vec):
        obs = self._get_obs()
        obs = np.concatenate((obs, cat_vec))
        return obs

    def _get_obs(self):
        agent_x, agent_y = (
            self.agent_pos % self.desc.shape[0],
            self.agent_pos // self.desc.shape[1],
        )

        obstacle_x, obstacle_y = (
            self.obstacle_pos % self.desc.shape[0],
            self.obstacle_pos // self.desc.shape[1],
        )

        objective1_x, objective1_y = (
            self.objectives[0] % self.desc.shape[0],
            self.objectives[0] // self.desc.shape[1],
        )

        objective2_x, objective2_y = (
            self.objectives[1] % self.desc.shape[0],
            self.objectives[1] // self.desc.shape[1],
        )
        return np.array(
            [
                int(self.reached_objectives[0]),
                int(self.reached_objectives[1]),
                agent_x,
                agent_y,
                obstacle_x,
                obstacle_y,
                objective1_x,
                objective1_y,
                objective2_x,
                objective2_y,
            ]
        )

    def step(self, action):
        done = False
        self.last_action = action
        self._do_action(action)
        self.steps_to_reach -= 1

        reward = np.zeros(self.num_rewards)
        dist = self._dist_reward(self.man_objective)
        reward[0] += self.last_dist_objective - dist
        self.last_dist_objective = -dist
        reward[1] = self._obstacle_reward()
        if self.hit_wall or self.steps_to_reach < 0:
            reward[0] -= 200
            done = True
            self.last_fifty_objective_count.append(0)
            self.last_fifty_objective_count = self.last_fifty_objective_count[-50:]
        elif dist == 0:
            self.steps_to_reach = 20
            objs = self.objectives == self.man_objective
            which_objective = objs.argmax()
            next_objective = objs.argmin()
            self.man_objective = self.objectives[next_objective]
            self.last_dist_objective = -self._dist_reward(
                objective_pos=self.man_objective
            )
            self.reached_objectives[which_objective] = True
            print(
                Fore.GREEN
                + f"objective {which_objective + 1} reached"
                + Style.RESET_ALL
            )
            self.objective_count += 1
        if self.objective_count == 2:
            done = True
            self.cumulative_reward_info["reward_objective"] += 1
            self.last_fifty_objective_count.append(1)
            self.last_fifty_objective_count = self.last_fifty_objective_count[-50:]
            print(Fore.CYAN + "objective 1 and 2 reached" + Style.RESET_ALL)

        self.cumulative_reward_info["reward_dist"] += reward[0]
        self.cumulative_reward_info["reward_obstacle"] += reward[1]
        self.cumulative_reward_info["reward_success_rate"] = np.mean(
            self.last_fifty_objective_count
        )
        self.cumulative_reward_info["Original_reward"] += reward.sum()
        man_objective_x, man_objective_y = (
            self.agent_pos % self.desc.shape[0],
            self.agent_pos // self.desc.shape[1],
        )

        if not self.stratified:
            reward = (reward * self.ori_weights).sum()
        return self.get_obs(np.array([man_objective_x, man_objective_y])), reward, done, self.cumulative_reward_info

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

    def gaussian_activation(
        self, x, y, xmean, ymean, x_var=1, xy_cov=0, yx_cov=0, y_var=1
    ):
        """
        Return the value for a 2d gaussian distribution with mean at [xmean, ymean] and the covariance matrix based on
        [[x_var, xy_cov],[yx_cov, y_var]].
        """
        means = [xmean, ymean]
        cov_mat = [[x_var, xy_cov], [yx_cov, y_var]]

        rv = multivariate_normal(means, cov_mat)

        return rv.pdf([x, y])

    def min_max_norm(self, val, min, max):
        return (val - min) / (max - min)

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
