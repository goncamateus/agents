import math

import numpy as np
from gym import spaces
from gym.envs.box2d.lunar_lander import (
    FPS,
    LEG_DOWN,
    MAIN_ENGINE_POWER,
    SCALE,
    SIDE_ENGINE_AWAY,
    SIDE_ENGINE_HEIGHT,
    SIDE_ENGINE_POWER,
    VIEWPORT_H,
    VIEWPORT_W,
    LunarLander,
)


class LunarLanderStratV2(
    LunarLander
):  # no need for EzPickle, it's already in LunarLander
    """
    ## Description
    Multi-objective version of the LunarLander environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/box2d/lunar_lander/) for more information.

    ## Reward Space
    The reward is 4-dimensional:
    - 0: -100 if crash, +100 if lands successfully
    - 1: Shaping reward
    - 2: Fuel cost (main engine)
    - 3: Fuel cost (side engine)
    """

    def __init__(self, stratified=True, **kwargs):
        super().__init__(**kwargs)
        self.cumulative_reward_info = {
            "reward_Shaping": 0,
            "reward_Power_linear": 0,
            "reward_Power_angular": 0,
            "reward_Goal": 0,
            "Original_reward": 0,
        }
        self.stratified = stratified
        self.num_rewards = 4

    def reset(self, **kwargs):
        self.cumulative_reward_info = {
            "reward_Shaping": 0,
            "reward_Power_linear": 0,
            "reward_Power_angular": 0,
            "reward_Goal": 0,
            "Original_reward": 0,
        }
        return super().reset(**kwargs)

    def step(self, action):
        # Update wind
        assert self.lander is not None, "You forgot to call reset()"
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            # 4 is move a bit downwards, +-2 for randomness
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(
                3.5,  # 3.5 is here to make particle speed adequate
                impulse_pos[0],
                impulse_pos[1],
                m_power,
            )  # particles are just a decoration
            p.ApplyLinearImpulse(
                (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(
                (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        vector_reward = np.zeros(4, dtype=np.float32)
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
            vector_reward[1] = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
            m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        vector_reward[2] = -m_power
        reward -= s_power * 0.03
        vector_reward[3] = -s_power

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
            vector_reward[0] = -100
        if not self.lander.awake:
            terminated = True
            reward = +100
            vector_reward[0] = +100

        self.cumulative_reward_info["reward_Goal"] += vector_reward[0]
        self.cumulative_reward_info["reward_Shaping"] += vector_reward[1]
        self.cumulative_reward_info["reward_Power_linear"] += vector_reward[2]
        self.cumulative_reward_info["reward_Power_angular"] += vector_reward[3]
        self.cumulative_reward_info["Original_reward"] += reward

        if not self.stratified:
            vector_reward = reward
        return (
            np.array(state, dtype=np.float32),
            vector_reward,
            terminated,
            self.cumulative_reward_info,
        )


class LunarLanderContinuousStratV2(LunarLanderStratV2):
    continuous = True
