import numpy as np
from colorama import Fore, Style
from gym.spaces import Box, Dict, Discrete

from envs.simple_nav.simple_nav import SimpleNav


class HierarchicalSimpleNav(SimpleNav):
    """
    Grid environment for Hierarchical Reinforcement Learning
    The environment is a 11x11 grid with 4 actions: up, down, left, right
    The agent starts at the center of the grid and it has to reach two objectives.
    The first objective is 4 steps on left of the center and the second objective is 4 steps on right of the center.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, worker_stratified=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.worker_action_space = Discrete(4)
        self.worker_observation_space = Box(
            low=np.array(
                [0, 0, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5],
                dtype=np.float64,
            ),
            high=np.array([1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.float64),
            dtype=np.float64,
        )
        self.manager_action_space = Discrete(121)
        self.manager_observation_space = Box(
            low=np.array(
                [0, 0, -5, -5, -5, -5, -5, -5, -5, -5],
                dtype=np.float64,
            ),
            high=np.array([1, 1, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.float64),
            dtype=np.float64,
        )
        self.observation_space = Dict(
            {
                "worker": self.worker_observation_space,
                "manager": self.manager_observation_space,
            }
        )
        self.action_space = Dict(
            {"worker": self.worker_action_space, "manager": self.manager_action_space}
        )
        self.worker_stratified = worker_stratified
        self.worker_weights = np.array([1, 1])
        self.reached_before = [False, False]
        self.episode_step_limit = {"worker": 20, "manager": 20}
        self.hcumulative_reward_info = {
            "reward_manager": 0,
            "reward_objective": 0,
            "reward_dist": 0,
            "reward_obstacle": 0,
            "reward_subobjective": 0,
            "reward_worker_sum": 0,
            "worker_done": False,
        }
        self.sub_goal_img = None
        self.steps_count = 0
        self.worker_steps_count = 0
        self.manager_steps_count = 0

    def reset(self):
        _ = super().reset()
        self.reached_before = [False, False]
        self.steps_count = 0
        self.manager_steps_count = 0

        self.hcumulative_reward_info.update(
            {
                "reward_objective": 0,
                "reward_manager": 0,
            }
        )

        return self._get_obs()

    def _worker_reset(self):
        self.hit_wall = False
        self.worker_steps_count = 0
        self.hcumulative_reward_info.update(
            {
                "reward_dist": 0,
                "reward_obstacle": 0,
                "reward_subobjective": 0,
                "reward_worker_sum": 0,
                "worker_done": False,
            }
        )

    def _get_obs(self):
        manager_obs = super()._get_obs()
        manager_target_x, manager_target_y = self.goal_indicator.get_position()[:2]
        worker_obs = np.concatenate([manager_obs, (manager_target_x, manager_target_y)])
        observation = {"worker": worker_obs, "manager": manager_obs}
        return observation

    def _do_action(self, action):
        super()._do_action(action)
        self.worker_steps_count += 1

    def _manager_act(self, action):
        x_index, y_index = action // 11, action % 11
        action_x = self.coordinates[x_index]
        action_y = self.coordinates[y_index]
        self.goal_indicator.set_position(
            [action_x, action_y, self.goal_indicator.get_position()[-1]]
        )
        self.pr.step()
        self.manager_steps_count += 1

    def _worker_reward(self):
        reward = np.zeros(2)
        dist = self._dist_reward(objective=self.goal_indicator)
        reward[0] = -10 - dist
        reward[1] = self._obstacle_reward()
        vec = self.agent.get_position()[:2] - self.goal_indicator.get_position()[:2]
        vec_dist = np.linalg.norm(vec)

        if vec_dist < 0.4:
            self.hcumulative_reward_info["worker_done"] = True
            self.hcumulative_reward_info["reward_subobjective"] += 1
            for idx, goal in enumerate(self.objectives):
                dist = self.agent.check_distance(goal)
                if dist < self.done_thresh:
                    if idx == 0 and not self.reached_objectives[0]:
                        self.reached_objectives[0] = True
                        print(Fore.GREEN + "objective 1 reached" + Style.RESET_ALL)
                    elif idx == 1 and not self.reached_objectives[1]:
                        self.reached_objectives[1] = True
                        print(Fore.GREEN + "objective 2 reached" + Style.RESET_ALL)
        elif self.hit_wall:
            reward[0] = -200
            self.hcumulative_reward_info["worker_done"] = True
        elif self.worker_steps_count > self.episode_step_limit["worker"]:
            self.hcumulative_reward_info["worker_done"] = True
        self.hcumulative_reward_info["reward_dist"] += reward[0]
        self.hcumulative_reward_info["reward_obstacle"] += reward[1]
        self.hcumulative_reward_info["reward_worker_sum"] += reward.sum()
        if not self.worker_stratified:
            reward = reward.sum()
        return reward

    def _manager_reward(self):
        reward = -10
        if self.hcumulative_reward_info["reward_subobjective"] > 0:
            reward = -4
        if self.reached_objectives[0] and not self.reached_before[0]:
            reward = 2
            self.reached_before[0] = True
            self.objective_count += 1
        if self.reached_objectives[1] and not self.reached_before[1]:
            reward = 2
            self.reached_before[1] = True
            self.objective_count += 1
        if self.reached_objectives[0] and self.reached_objectives[1]:
            reward = 10
            self.reached_before[1] = True

        return reward

    def step(self, action):
        reward = {"worker": 0, "manager": 0}
        done = False
        self.last_action = action["worker"]
        self._do_action(action["worker"])
        reward["worker"] = self._worker_reward()

        if action["manager"]:
            self._manager_act(action["manager"])
        reward["manager"] = self._manager_reward()
        self.steps_count += 1

        if self.hcumulative_reward_info["worker_done"]:
            self._worker_reset()

        if self.reached_objectives[0] and self.reached_objectives[1]:
            done = True
            reward["manager"] = 10
            print(Fore.CYAN + "objective 1 and 2 reached" + Style.RESET_ALL)

        self.hcumulative_reward_info["reward_objective"] = int(
            self.objective_count == 2
        )
        self.hcumulative_reward_info["reward_manager"] += reward["manager"]

        return self._get_obs(), reward, done, self.hcumulative_reward_info
