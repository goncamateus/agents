from gym.envs.registration import register


register(
    id="LunarLanderStrat-v0",
    entry_point="envs.lunar_lander_strat:LunarLanderStrat",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderOri-v0",
    entry_point="envs.lunar_lander_strat:LunarLanderStrat",
    kwargs={"stratified": False},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuousStrat-v0",
    entry_point="envs.lunar_lander_strat:LunarLanderContinuousStrat",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuousOri-v0",
    entry_point="envs.lunar_lander_strat:LunarLanderContinuousStrat",
    kwargs={"stratified": False},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="HalfCheetahStrat-v0",
    entry_point="envs.half_cheetah_strat:HalfCheetahStratEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetahOri-v0",
    entry_point="envs.half_cheetah_strat:HalfCheetahStratEnv",
    kwargs={"stratified": False},
    max_episode_steps=1000,
    reward_threshold=4800.0,
)