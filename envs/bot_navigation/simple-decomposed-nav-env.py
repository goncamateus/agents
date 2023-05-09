import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style
from gym import spaces
from pyrep import PyRep
from pyrep.const import JointMode
from pyrep.objects import Object
from pyrep.robots.mobiles.youbot import YouBot
from scipy.stats import multivariate_normal


def min_max_norm(val, min, max):
    return (val - min) / (max - min)


def gaussian_activation(x, y, xmean, ymean, x_var=1, xy_cov=0, yx_cov=0, y_var=1):
    """
    Return the value for a 2d gaussian distribution with mean at [xmean, ymean] and the covariance matrix based on
    [[x_var, xy_cov],[yx_cov, y_var]].
    """
    means = [xmean, ymean]
    cov_mat = [[x_var, xy_cov], [yx_cov, y_var]]

    rv = multivariate_normal(means, cov_mat)

    return rv.pdf([x, y])


class CoppeliaYouBotNavEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, headless=False):
        super(CoppeliaYouBotNavEnv, self).__init__()
        self.headless = headless
        self.pr = PyRep()
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, f"YouBotNavigationScene.ttt")
        self.scene_file = filename
        self.pr.launch(filename, headless=headless)
        self.agent = YouBot()
        self.agent.set_joint_mode(JointMode.FORCE)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.staring_pos = self.agent.get_position()
        self.starting_ori = self.agent.get_orientation()

        self.right_goal = Object.get_object("LeftGoal")
        self.left_goal = Object.get_object("RightGoal")
        self.obstacle = Object.get_object("Obstacle")
        self.vision_sensor = Object.get_object("Vision_sensor")
        self.goal_indicator = Object.get_object("GoalIndicator")
        self.nav_targets = [self.left_goal, self.right_goal]

        self.action_space = spaces.Dict(
            {
                "atomic_action_space": spaces.Discrete(4),  # N,S,E,W
                "meta_action_space": spaces.Discrete(
                    121
                ),  # arena space as 11x11 grid (-5, ... 0, ... 5)
            }
        )

        self.atomic_action_names = [
            "North",
            "South",
            "East",
            "West",
        ]
        self.observation_space = spaces.Dict(
            {
                "atomic_observation_space": spaces.Box(
                    low=np.array(
                        [
                            0,  # first goal reached
                            0,  # second goal reached
                            -5,  # agent xy
                            -5,
                            -5,  # obstacle xy
                            -5,
                            -5,  # left goal xy
                            -5,
                            -5,  # right goal xy
                            -5,
                            -5,  # hierarchical goal xy
                            -5,
                        ],
                        dtype=np.float64,  # just so that gym logger doesn't give warning about casting to float32...
                    ),
                    high=np.array(
                        [1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.float64
                    ),
                    dtype=np.float64,
                ),
                "meta_observation_space": spaces.Box(
                    low=np.array(
                        [
                            0,  # first goal reached
                            0,  # second goal reached
                            -5,  # agent xy
                            -5,
                            -5,  # obstacle xy
                            -5,
                            -5,  # left goal xy
                            -5,
                            -5,  # right goal xy
                            -5,
                        ],
                        dtype=np.float64,
                    ),
                    high=np.array([1, 1, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.float64),
                    dtype=np.float64,
                ),
            }
        )

        # have to redefine those with the new atomic obseravtion
        eval_state_0 = [0, 0, 2.99998045, -1.99999964, 0, -1.5, 0, 3, 0, -3, 2, -2]
        eval_state_1 = [0, 0, -1.00002074, 1.00000215, 0, -1.5, 0, 3, 0, -3, -2, 3]
        eval_state_1_1 = [1, 0, -1.00002074, 1.00000215, 0, -1.5, 0, 3, 0, -3, -2, 3]
        eval_state_2 = [0, 0, 0.99998063, -0.99999934, 0, -1.5, 0, 3, 0, -3, -1, -2]
        eval_state_3 = [
            0,
            0,
            -4.47773709e-05,
            -1.49999881e00,
            0,
            -1.5,
            0,
            3,
            0,
            -3,
            2,
            3,
        ]
        eval_state_4 = [
            0,
            0,
            -1.99094902e-05,
            2.00000048e00,
            0,
            -1.50000000e00,
            0,
            3,
            0,
            -3,
            0,
            -3,
        ]
        eval_state_5 = [
            0,
            0,
            -7.39204843e-05,
            -4.99996006e-01,
            0,
            -1.50000000e00,
            0,
            3.00000000e00,
            0,
            -3,
            0,
            -1.50000000e00,
        ]
        eval_state_6 = [0, 0, 0.99997997, 1.00000072, 0, -1.5, 0, 3, 0, -3, 0, -1.5]
        self.eval_states = [
            eval_state_0,
            eval_state_1,
            eval_state_1_1,
            eval_state_2,
            eval_state_3,
            eval_state_4,
            eval_state_5,
            eval_state_6,
        ]
        self.eval_state_preds = None

        self.done_thresh = 0.1
        self.episode_step_limit = {"atomic": 20, "meta": 20}
        self.n_reward_components = {"atomic": 2, "meta": 1}
        self.component_names = {
            "atomic": ["Navigation", "Obstacle"],
            "meta": ["Baseline"],
        }
        self.arena_size = 6
        self.coordinates = np.arange(-5, 6, 1)

        self.cumulative_reward_info = {
            "reward_navigation": 0,
            "reward_obstacle": 0,
            "reward_baseline": 0,
        }

        # gaussian for static obstacle reward calculation
        self.obstacle_max_punish = 20
        self.obstacle_gauss_xvar = 0.2
        self.obstacle_gauss_xycov = 0
        self.obstacle_gauss_yxcov = 0
        self.obstacle_gauss_yvar = 0.2
        self.obstacle_activation_visualization_thresh = (
            5.0  # when ba punishment fac is greater than this we change ba color
        )

        # episode lvl vars (have to be reset each episode)
        self.atomic_episode_step_counter = None
        self.meta_episode_step_counter = None
        self.on_obstacle_counter = None
        self.first_goal_done = None
        self.second_goal_done = None
        self.first_goal_meta_bonus = None
        self.second_goal_meta_bonus = None
        self.atomic_episode_success = None
        self.crash_prevented = None

        # helpers
        self.atomic_reset_counter = 0
        self.meta_reset_counter = 0

    def get_atomic_obs(self):
        state = np.array([self.first_goal_done, self.second_goal_done])

        for pr_obj in [self.agent, self.obstacle, self.left_goal, self.right_goal]:
            obj_pos = pr_obj.get_position()[0:2]
            # obj_vec_rep = get_goal_vec_rep(two_d_goal=obj_pos, youbot=self.agent)

            state = np.append(state, obj_pos)

        target_pos = self.goal_indicator.get_position()[0:2]
        state = np.append(state, target_pos)

        return state

    def get_meta_obs(self):
        state = np.array([self.first_goal_done, self.second_goal_done])

        for pr_obj in [self.agent, self.obstacle, self.left_goal, self.right_goal]:
            obj_pos = pr_obj.get_position()[0:2]

            state = np.append(state, obj_pos)

        return state

    def _calc_goal_dist_reward(self):
        return -10 - self.agent.check_distance(self.goal_indicator)

    def _calc_obstacle_reward(self):
        agent_x, agent_y = self.agent.get_position()[0:2]

        obstacle_x, obstacle_y = self.obstacle.get_position()[0:2]

        activation = gaussian_activation(
            x=agent_x,
            y=agent_y,
            xmean=obstacle_x,
            ymean=obstacle_y,
            x_var=self.obstacle_gauss_xvar,
            xy_cov=self.obstacle_gauss_xycov,
            yx_cov=self.obstacle_gauss_yxcov,
            y_var=self.obstacle_gauss_yvar,
        )

        normed_act = min_max_norm(
            activation,
            min=0,
            max=gaussian_activation(
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

        # just visualization and printing...
        if obstacle_punishment > self.obstacle_activation_visualization_thresh:
            self.on_obstacle_counter += 1
        else:
            if self.on_obstacle_counter != 0:
                self.pretty_print(
                    f"Left bad area after {self.on_obstacle_counter} successive steps!",
                    Fore.YELLOW,
                )

                self.on_obstacle_counter = 0

        return -1 * obstacle_punishment

    def _calc_atomic_reward(self):
        nav_reward = self._calc_goal_dist_reward()
        obs_reward = self._calc_obstacle_reward()

        return np.array([nav_reward, obs_reward])

    @staticmethod
    def _calc_meta_reward(self):
        reward_vec = [-10]
        # order of ifs matters: Goal bonus overwrites just the reach target
        if self.atomic_episode_success:
            reward_vec[0] = -4

        if self.first_goal_done and not self.first_goal_meta_bonus:
            reward_vec[0] = 2

        if self.second_goal_done and not self.second_goal_meta_bonus:
            reward_vec[0] = 2

        if self.first_goal_done and self.second_goal_done:
            reward_vec[0] = 10

        return reward_vec

    def _do_atomic_action(self, action):
        step_size = 0.025  # agent moves by this much if we move with velocity 1 for ten sim steps (old setup)
        step_size = 0.25  # increasing to make it easier to solve!
        pos = self.agent.get_position()

        if action == 0:
            # north
            new_pos = pos - [step_size, 0, 0]
            if abs(new_pos[0]) < 5.5:
                self.agent.set_position(new_pos)
            else:
                self.pretty_print(f"Preventing move out of bounds!", Fore.YELLOW)
                self.crash_prevented = True
        elif action == 1:
            # south
            new_pos = pos + [step_size, 0, 0]
            if abs(new_pos[0]) < 5.5:
                self.agent.set_position(new_pos)
            else:
                self.pretty_print(f"Preventing move out of bounds!", Fore.YELLOW)
                self.crash_prevented = True
        elif action == 2:
            # east
            new_pos = pos + [0, step_size, 0]
            if abs(new_pos[1]) < 5.5:
                self.agent.set_position(new_pos)
            else:
                self.pretty_print(f"Preventing move out of bounds!", Fore.YELLOW)
                self.crash_prevented = True
        elif action == 3:
            # west
            new_pos = pos - [0, step_size, 0]
            if abs(new_pos[1]) < 5.5:
                self.agent.set_position(new_pos)
            else:
                self.pretty_print(f"Preventing move out of bounds!", Fore.YELLOW)
                self.crash_prevented = True
        else:
            raise ValueError(f"do_action for {action} undefined!")

        self.atomic_episode_step_counter += 1
        self.pr.step()

    def _do_meta_action(self, action):
        # conver action (0, 100] to x y index
        x_index = min(
            int(action / 11), 10
        )  # ugly, but 121 / 11 = 11 and the last index for our 2d array is 10,10...
        y_index = int(action % 11)

        x_coord = self.coordinates[x_index]
        y_coord = self.coordinates[y_index]

        self.goal_indicator.set_position(
            [x_coord, y_coord, self.goal_indicator.get_position()[-1]]
        )

        self.meta_episode_step_counter += 1

        self.pr.step()

    def _check_atomic_done(self, reward_vec):
        done = False
        success = False
        info = ""

        dist = self.agent.check_distance(self.goal_indicator)
        vec = self.agent.get_position()[0:2] - self.goal_indicator.get_position()[0:2]
        vec_dist = np.linalg.norm(vec)

        # if dist < self.done_thresh:
        if vec_dist < 0.4:
            info = f"Atomic success, reached goal indicator"
            self.pretty_print(info, style=Fore.GREEN)
            success = True
            done = True
            self.atomic_episode_success = True

            for idx, goal in enumerate(self.nav_targets):
                dist = self.agent.check_distance(goal)
                if dist < self.done_thresh:
                    if idx == 0 and not self.first_goal_done:
                        self.first_goal_done = True
                        self.pretty_print("Reached first goal!")
                    elif idx == 1 and not self.second_goal_done:
                        self.second_goal_done = True
                        self.pretty_print("Reached second goal!")
        elif self.crash_prevented:
            reward_vec[0] = -200
            info = f"Atomic fail, crashed into border"
            self.pretty_print(info, Fore.RED)
            done = True

        elif self.atomic_episode_step_counter > self.episode_step_limit["atomic"]:
            info = f"Atomic fail, limit was exceeded"
            self.pretty_print(info, Fore.RED)
            done = True

        return done, info, success, reward_vec

    def _check_meta_done(self):
        done = False
        success = False
        info = ""

        # order of ifs matters: Goal bonus overwrites just the reach target
        if self.first_goal_done and not self.first_goal_meta_bonus:
            self.first_goal_meta_bonus = True

        if self.second_goal_done and not self.second_goal_meta_bonus:
            self.second_goal_meta_bonus = True

        if self.first_goal_done and self.second_goal_done:
            info = f"Meta Success, both goals reached"
            self.pretty_print(info, style=Fore.CYAN)
            success = True
            done = True

        if self.meta_episode_step_counter > self.episode_step_limit["meta"]:
            info = f"Meta fail, limit was exceeded"
            self.pretty_print(info, Fore.RED)
            done = True

        return done, success, info

    @staticmethod
    def _get_info():
        return ""

    def step(self, action):
        pass

    def atomic_step(self, action):
        self._do_atomic_action(action)

        atomic_reward = self._calc_atomic_reward()
        atomic_done, done_msg, atomic_success, atomic_reward = self._check_atomic_done(
            atomic_reward
        )  # check done and potentially give reward bonus
        observation = self.get_atomic_obs()
        info_msg = self._get_info()

        msg_dict = {"done_msg": done_msg, "info_msg": info_msg}
        info = {}

        return observation, atomic_reward, atomic_done, msg_dict, atomic_success

    def meta_step(self, action):
        self._do_meta_action(action)
        meta_reward = self._calc_meta_reward()
        meta_done, meta_success, info = self._check_meta_done()
        observation = self.get_meta_obs()

        return observation, meta_reward, meta_done, meta_success, info

    def atomic_reset(self):
        self.atomic_reset_counter += 1

        # reset episode vars
        self.atomic_episode_step_counter = 0
        self.on_obstacle_counter = 0
        self.atomic_episode_success = False
        self.crash_prevented = False

        return self.get_atomic_obs()

    def meta_reset(self):
        self.pr.stop()
        self.pr.start()

        self.meta_reset_counter += 1

        self.agent.set_joint_mode(JointMode.FORCE)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent.set_orientation(self.starting_ori)

        random_x, random_y = np.random.uniform(-3.5, 3.5, 2)
        self.agent.set_position([random_x, random_y, self.staring_pos[-1]])
        self.pr.step()

        # reset episode vars
        self.first_goal_done = 0
        self.second_goal_done = 0
        self.meta_episode_step_counter = 0
        self.first_goal_meta_bonus = False
        self.second_goal_meta_bonus = False

        return self.get_meta_obs()

    def render(self, mode="human"):
        pass

    def close(self):
        self.pr.stop()
        self.pr.shutdown()

    def restart_coppelia(self):
        self.close()
        self.pr.launch(self.scene_file, headless=self.headless)
        self.pr.start()

    def _set_youbot_position(self, x, y):
        self.agent.set_position([x, y, self.agent.get_position()[-1]])
        self.pr.step()

    def pretty_print(self, msg, style=Fore.CYAN):
        print(
            style + f"ME {self.meta_reset_counter} "
            f"MS: {self.meta_episode_step_counter} "
            f"AE: {self.meta_reset_counter} "
            f"AS: {self.atomic_episode_step_counter} | " + msg + Style.RESET_ALL
        )


if __name__ == "__main__":

    def describe_current_state(env, num):
        a_state = env.get_atomic_obs()
        print(a_state)
        img = env.vision_sensor.capture_rgb()
        plt.imsave(f"evalState{num}.png", img)

    env = CoppeliaYouBotNavEnv(headless=False)
    dists = []
    _ = env.meta_reset()
    for a in range(60, 121):
        env.do_meta_action(a)
        dist = env.agent.check_distance(env.goal_indicator)
        print(dist)
        dists.append(dist)
        time.sleep(0.25)
