import numpy as np
from rsoccer_gym.ssl.ssl_path_planning.navigation import (
    Point2D,
    abs_smallest_angle_diff,
    dist_to,
)
from rsoccer_gym.ssl.ssl_path_planning.ssl_path_planning import (
    ANGLE_TOLERANCE,
    SSLPathPlanningEnv,
)


class SSLGoToStrat(SSLPathPlanningEnv):
    """The SSL robot needs to reach the target point with a given angle"""

    def __init__(self, field_type=1, stratified=False):
        super().__init__(
            field_type=field_type,
        )

        self.stratified = stratified
        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_angle": 0,
            "reward_objective": 0,
            "Original_reward": 0,
        }

        print("Environment initialized")

    def reset(self):
        self.cumulative_reward_info = {
            "reward_dist": 0,
            "reward_angle": 0,
            "reward_objective": 0,
            "Original_reward": 0,
        }
        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info.update(self.cumulative_reward_info)
        return observation, reward, done, info

    def reward_function(
        self,
        robot_pos: Point2D,
        last_robot_pos: Point2D,
        robot_vel: Point2D,
        last_robot_vel: Point2D,
        robot_angle: float,
        target_pos: Point2D,
        target_angle: float,
        target_vel: Point2D,
    ):
        WEIGHT_DIST = 0.5
        WEIGHT_ANGLE = 0.5
        WEIGHT_OBJECTIVE = 0.5
        reward = np.zeros(3)
        done = False

        last_dist_robot_to_target = dist_to(target_pos, last_robot_pos)
        dist_robot_to_target = dist_to(target_pos, robot_pos)

        dist_rw = (last_dist_robot_to_target - dist_robot_to_target) / self.max_v

        last_robot_angle = np.deg2rad(self.last_frame.robots_blue[0].theta)

        angle_dist_rw = (
            abs_smallest_angle_diff(last_robot_angle, target_angle)
            - abs_smallest_angle_diff(robot_angle, target_angle)
        ) / self.max_w
        angle_ok = abs_smallest_angle_diff(robot_angle, target_angle) < ANGLE_TOLERANCE

        if dist_robot_to_target < 0.2 and angle_ok:
            done = True
            reward[2] += 1
            self.cumulative_reward_info["reward_objective"] += 1

        reward[0] = dist_rw
        angle_ok = 1 if angle_ok else -1
        reward[1] = angle_dist_rw

        self.cumulative_reward_info["reward_dist"] += dist_rw
        self.cumulative_reward_info["reward_angle"] += angle_dist_rw
        self.cumulative_reward_info["Original_reward"] += (
            WEIGHT_DIST * dist_rw
            + WEIGHT_ANGLE * angle_dist_rw
            + WEIGHT_OBJECTIVE * reward[2]
        )
        if not self.stratified:
            reward = (
                WEIGHT_DIST * dist_rw
                + WEIGHT_ANGLE * angle_dist_rw
                + WEIGHT_OBJECTIVE * reward[2]
            )
        return reward, done
