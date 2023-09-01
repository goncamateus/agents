from gym.envs.registration import register
import gymnasium.envs.registration as mod_reg

mod_reg.register(
    id="TaxiStrat-v0",
    entry_point="envs.taxi:StratTaxiEnv",
    max_episode_steps=200,
    reward_threshold=8,
)

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
    id="LunarLanderStrat-v1",
    entry_point="envs.lunar_lander_strat_v1:LunarLanderStratV1",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderOri-v1",
    entry_point="envs.lunar_lander_strat_v1:LunarLanderStratV1",
    kwargs={"stratified": False},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuousStrat-v1",
    entry_point="envs.lunar_lander_strat_v1:LunarLanderContinuousStratV1",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuousOri-v1",
    entry_point="envs.lunar_lander_strat_v1:LunarLanderContinuousStratV1",
    kwargs={"stratified": False},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderStrat-v2",
    entry_point="envs.lunar_lander_strat_v2:LunarLanderStratV2",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderOri-v2",
    entry_point="envs.lunar_lander_strat_v2:LunarLanderStratV2",
    kwargs={"stratified": False},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuousStrat-v2",
    entry_point="envs.lunar_lander_strat_v2:LunarLanderContinuousStratV2",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuousOri-v2",
    entry_point="envs.lunar_lander_strat_v2:LunarLanderContinuousStratV2",
    kwargs={"stratified": False},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuousStrat-v3",
    entry_point="envs.lunar_lander_strat_v2:LunarLanderContinuousStratV3",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuous-v3",
    entry_point="envs.lunar_lander_strat_v2:LunarLanderContinuousStratV3",
    kwargs={"stratified": False},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuousModStrat-v0",
    entry_point="envs.lunar_lander_strat_v2:LunarLanderContinuousMod",
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
register(
    id="HalfCheetahOri-v1",
    entry_point="envs.half_cheetah_strat:HalfCheetahStratEnv",
    kwargs={
        "stratified": False,
        "forward_reward_weight": 1,
        "ctrl_cost_weight": 1e-9,
    },
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
register(
    id="HalfCheetahOri-v2",
    entry_point="envs.half_cheetah_strat:HalfCheetahStratEnv",
    kwargs={
        "stratified": False,
        "forward_reward_weight": .5,
        "ctrl_cost_weight": .5,
    },
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HumanoidStrat-v0",
    entry_point="envs.humanoid_strat:HumanoidStratEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidOri-v0",
    entry_point="envs.humanoid_strat:HumanoidStratEnv",
    kwargs={"stratified": False},
    max_episode_steps=1000,
)

register(
    id="VSS-v1",
    entry_point="envs.vss:VSSStratEnv",
    kwargs={"stratified": False},
    max_episode_steps=1200,
)

register(
    id="VSSStrat-v0",
    entry_point="envs.vss:VSSStratEnv",
    kwargs={"stratified": True},
    max_episode_steps=1200,
)

register(
    id="SSLGoToOri-v0",
    entry_point="envs.ssl_goto:SSLGoToStrat",
    kwargs={"stratified": False},
    max_episode_steps=1200,
)

register(
    id="SSLGoToMod-v0",
    entry_point="envs.ssl_goto:SSLGoToStratMod",
    kwargs={"stratified": False},
    max_episode_steps=1200,
)

register(
    id="SSLGoToStrat-v0",
    entry_point="envs.ssl_goto:SSLGoToStrat",
    kwargs={"stratified": True},
    max_episode_steps=1200,
)

register(
    id="SSLGoToStratMod-v0",
    entry_point="envs.ssl_goto:SSLGoToStratMod",
    kwargs={"stratified": True},
    max_episode_steps=1200,
)

register(
    id="FrozenLake11x11-v0",
    entry_point="envs.frozen_lake.frozen_lake:FrozenLakeMod",
    kwargs={
        "stratified": False,
        "desc_shape": (11, 11),
        "agent_pos": 60,
        "objective_0": 56,
        "objective_1": 64,
        "obstacle_pos": 58,
    },
    max_episode_steps=100,
)

register(
    id="FrozenLake13x13-v0",
    entry_point="envs.frozen_lake.frozen_lake:FrozenLakeMod",
    kwargs={
        "stratified": False,
        "desc_shape": (13, 13),
        "agent_pos": 84,
        "objective_0": 78,
        "objective_1": 90,
        "obstacle_pos": 80,
    },
    max_episode_steps=100,
)


register(
    id="FrozenLake11x11Strat-v0",
    entry_point="envs.frozen_lake.frozen_lake:FrozenLakeMod",
    kwargs={
        "stratified": True,
        "desc_shape": (11, 11),
        "agent_pos": 60,
        "objective_0": 56,
        "objective_1": 64,
        "obstacle_pos": 58,
    },
    max_episode_steps=100,
)

register(
    id="FrozenLake13x13Strat-v0",
    entry_point="envs.frozen_lake.frozen_lake:FrozenLakeMod",
    kwargs={
        "stratified": True,
        "desc_shape": (13, 13),
        "agent_pos": 84,
        "objective_0": 78,
        "objective_1": 90,
        "obstacle_pos": 80,
    },
    max_episode_steps=100,
)

register(
    id="HierarchicalFrozenLake11x11-v0",
    entry_point="envs.frozen_lake.hierarchical_frozen_lake:HierarchicalFrozenLakeMod",
    max_episode_steps=100,
    kwargs={
        "desc_shape": (11, 11),
        "agent_pos": 60,
        "objective_0": 56,
        "objective_1": 64,
        "obstacle_pos": 58,
    },
)

register(
    id="HierarchicalFrozenLake13x13-v0",
    entry_point="envs.frozen_lake.hierarchical_frozen_lake:HierarchicalFrozenLakeMod",
    max_episode_steps=100,
    kwargs={
        "desc_shape": (13, 13),
        "agent_pos": 84,
        "objective_0": 78,
        "objective_1": 90,
        "obstacle_pos": 80,
    },
)

register(
    id="HierarchicalFrozenLake11x11Strat-v0",
    entry_point="envs.frozen_lake.hierarchical_frozen_lake:HierarchicalFrozenLakeMod",
    kwargs={
        "worker_stratified": True,
        "desc_shape": (11, 11),
        "agent_pos": 60,
        "objective_0": 56,
        "objective_1": 64,
        "obstacle_pos": 58,
    },
    max_episode_steps=100,
)

register(
    id="HierarchicalFrozenLake13x13Strat-v0",
    entry_point="envs.frozen_lake.hierarchical_frozen_lake:HierarchicalFrozenLakeMod",
    max_episode_steps=100,
    kwargs={
        "worker_stratified": True,
        "desc_shape": (13, 13),
        "agent_pos": 84,
        "objective_0": 78,
        "objective_1": 90,
        "obstacle_pos": 80,
    },
)

register(
    id="SimpleNav-v0",
    entry_point="envs.simple_nav.simple_nav:SimpleNav",
    kwargs={
        "stratified": False,
        "headless": True,
    },
    max_episode_steps=100,
)

register(
    id="SimpleNavStrat-v0",
    entry_point="envs.simple_nav.simple_nav:SimpleNav",
    kwargs={
        "stratified": True,
        "headless": True,
    },
    max_episode_steps=100,
)

register(
    id="SimpleNav-v1",
    entry_point="envs.simple_nav.simple_nav:SimpleNav",
    kwargs={
        "stratified": False,
        "headless": False,
    },
    max_episode_steps=100,
)

register(
    id="SimpleNavStrat-v1",
    entry_point="envs.simple_nav.simple_nav:SimpleNav",
    kwargs={
        "stratified": True,
        "headless": False,
    },
    max_episode_steps=100,
)

register(
    id="HierarchicalSimpleNav-v0",
    entry_point="envs.simple_nav.hierarchical_simple_nav:HierarchicalSimpleNav",
    kwargs={
        "stratified": False,
        "headless": True,
    },
)

register(
    id="HierarchicalSimpleNav-v1",
    entry_point="envs.simple_nav.hierarchical_simple_nav:HierarchicalSimpleNav",
    kwargs={
        "stratified": False,
        "headless": False,
    },
)
