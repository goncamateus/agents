from gym.envs.registration import register

register(
    id="LunarLanderStrat-v0", max_episode_steps=1000, reward_threshold=200,
)
