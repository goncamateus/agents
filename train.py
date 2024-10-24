import gymnasium as gym

sync_env = gym.make_vec("LunarLander-v3", 3, "sync")
async_env = gym.make_vec("LunarLander-v3", 3, "async")

sync_env.reset()
async_env.reset()
for _ in range(1000):
    _, _, sync_done, sync_truncated, _ = sync_env.step(sync_env.action_space.sample())
    _, _, async_done, async_truncated, _ = async_env.step(
        async_env.action_space.sample()
    )
    print("Sync")
    print(sync_done, sync_truncated)
    print("Async")
    print(async_done, async_truncated)
