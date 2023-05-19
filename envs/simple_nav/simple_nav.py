import os

import gym
import numpy as np
from colorama import Fore, Style
from gym.spaces import Box, Discrete
from pyrep import PyRep
from pyrep.const import JointMode
from pyrep.objects import Object
from pyrep.robots.mobiles.youbot import YouBot

from envs.simple_nav.utils import gaussian_activation, min_max_norm

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3


class SimpleNav(gym.Env):
    """
    Grid environment for Reinforcement Learning
    The environment is a 11x11 grid with 4 actions: up, down, left, right
    The agent starts at the center of the grid and it has to reach two objectives.
    The first objective is 4 steps on left of the center and the second objective is 4 steps on right of the center.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, stratified=False, headless=False, **kwargs):
        super().__init__()
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

        self.stratified = stratified
        self.num_rewards = 2
        self.ori_weights = np.array([1, 1])

        self.action_space = Discrete(4)
        # goal1_reached, goal2_reached, agent_xy, obstacle_xy, goal1_xy, goal2_xy
        self.observation_space = Box(
            low=np.array(
                [0, 0, -5, -5, -5, -5, -5, -5, -5, -5],
                dtype=np.float64,
            ),
            high=np.array([1, 1, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.float64),
            dtype=np.float64,
        )
        self.agent_pos = self.staring_pos[:2]
        self.ori_agent_pos = self.staring_pos[:2]

        self.obstacle_pos = self.obstacle.get_position()[:2]
        self.objectives = np.array(
            [
                self.left_goal.get_position()[:2],
                self.right_goal.get_position()[:2],
            ]
        )
        self.objective_count = 0

        # gaussian for static obstacle reward calculation
        self.obstacle_max_punish = 20
        self.obstacle_gauss_xvar = 0.2
        self.obstacle_gauss_xycov = 0
        self.obstacle_gauss_yxcov = 0
        self.obstacle_gauss_yvar = 0.2
        
        self.done_thresh = 0.1

        self.reached_objectives = [False, False]

        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_obstacle": 0,
            "reward_objective": 0,
            "Original_reward": 0,
        }

    def reset(self):
        self.pr.stop()
        self.pr.start()
        self.agent.set_joint_mode(JointMode.FORCE)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent.set_orientation(self.starting_ori)

        random_x, random_y = np.random.uniform(-3.5, 3.5, 2)
        self.agent.set_position([random_x, random_y, self.staring_pos[-1]])
        self.pr.step()

        self.reached_objectives = [False, False]
        self.objective_count = 0
        self.agent_pos = self.agent.get_position()[:2]
        self.obstacle_pos = self.obstacle.get_position()[:2]
        self.objectives = np.array(
            [
                self.left_goal.get_position()[:2],
                self.right_goal.get_position()[:2],
            ]
        )

        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_obstacle": 0,
            "reward_objective": 0,
            "Original_reward": 0,
        }
        self.hit_wall = False
        return self._get_obs()

    def _do_action(self, action):
        step_size = 0.25
        pos = self.agent.get_position()

        if action == UP:
            new_pos = pos - [step_size, 0, 0]
        elif action == DOWN:
            new_pos = pos + [step_size, 0, 0]
        elif action == RIGHT:
            new_pos = pos + [0, step_size, 0]
        elif action == LEFT:
            new_pos = pos - [0, step_size, 0]
        else:
            raise ValueError(f"do_action for {action} undefined!")

        if abs(new_pos[0]) < 5.5 and abs(new_pos[1]) < 5.5:
            self.agent.set_position(new_pos)
        else:
            self.hit_wall = True

        self.pr.step()
        self.agent_pos = self.agent.get_position()[:2]

    def _get_obs(self):
        agent_x, agent_y = self.agent_pos
        obstacle_x, obstacle_y = self.obstacle_pos
        objective1_x, objective1_y = self.objectives[0]
        objective2_x, objective2_y = self.objectives[1]
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

    def _dist_reward(self, objective):
        return self.agent.check_distance(objective)

    def _obstacle_reward(self):
        agent_x, agent_y = self.agent_pos
        obstacle_x, obstacle_y = self.obstacle_pos
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
        return -obstacle_punishment

    def step(self, action):
        self._do_action(action)

        reward = np.zeros(self.num_rewards)
        reward[0] = -10
        reward[0] -= (1 - int(self.reached_objectives[0])) * self._dist_reward(
            objective=self.left_goal
        )
        reward[0] -= (1 - int(self.reached_objectives[1])) * self._dist_reward(
            objective=self.right_goal
        )
        if self.hit_wall:
            reward[0] -= 200
            done = True
        reward[1] = self._obstacle_reward()
        done = False
        dist_to_obj1 = self._dist_reward(self.left_goal)
        dist_to_obj2 = self._dist_reward(self.right_goal)
        if dist_to_obj1 < self.done_thresh and not self.reached_objectives[0]:
            print(Fore.GREEN + "objective 1 reached" + Style.RESET_ALL)
            self.objective_count += 1
            self.reached_objectives[0] = True
        elif dist_to_obj2 < self.done_thresh and not self.reached_objectives[1]:
            self.reached_objectives[1] = True
            print(Fore.GREEN + "objective 2 reached" + Style.RESET_ALL)
            self.objective_count += 1
        if self.reached_objectives[0] and self.reached_objectives[1]:
            done = True
            print(Fore.CYAN + "objective 1 and 2 reached" + Style.RESET_ALL)

        self.cumulative_reward_info["reward_dist"] += reward[0]
        self.cumulative_reward_info["reward_obstacle"] += reward[1]
        self.cumulative_reward_info["reward_objective"] = self.objective_count
        self.cumulative_reward_info["Original_reward"] += reward.sum()

        if not self.stratified:
            reward = (reward * self.ori_weights).sum()
        return self._get_obs(), reward, done, self.cumulative_reward_info

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
