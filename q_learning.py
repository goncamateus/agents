import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def set_plots(q_table, image=None):
    _, (ax1, ax2) = plt.subplots(1, 2)
    # Plot the Q-table
    im1 = ax1.imshow(q_table, cmap="hot")

    # Customize the plot
    ax1.set_xticks(np.arange(q_table.shape[1]))
    ax1.set_yticks(np.arange(q_table.shape[0]))
    ax1.set_xticklabels(np.arange(q_table.shape[1]))
    ax1.set_yticklabels(np.arange(q_table.shape[0]))
    ax1.set_xlabel("Action")
    ax1.set_ylabel("State")
    ax1.set_title("Q-Table")

    # Add colorbar
    cbar1 = ax1.figure.colorbar(im1, ax=ax1)

    # Plot the initial RGB array
    im2 = ax2.imshow(image)
    plt.subplots_adjust(wspace=0.4)
    plt.show()
    return im1, cbar1, im2


def update_plots(im1, cbar1, im2, q_table, image):
    # Update the Q-table plot
    im1.set_data(q_table)
    im1.autoscale()

    # Update the RGB array plot
    im2.set_data(image)
    im2.autoscale()

    # Redraw the colorbars
    cbar1.update_normal(im1)

    # Pause to show the updated plot
    plt.pause(1e-12)


class QLearningAgent:
    def __init__(self, observation_space, action_space, alpha=1e-1, gamma=0.99):
        self.obs_size = observation_space.n
        self.action_size = action_space.n
        self.q_table = np.zeros((self.obs_size, self.action_size))
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, observation):
        # check if array has the same value
        if np.all(self.q_table[observation] == self.q_table[observation][0]):
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.q_table[observation])
        return action

    def update_policy(self, observation, action, reward, next_obs):
        update = reward + self.gamma * (
            self.q_table[next_obs].max() - self.q_table[observation][action]
        )
        self.q_table[observation][action] = (
            self.q_table[observation][action] + self.alpha * update
        )


def main():
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
    agent = QLearningAgent(env.observation_space, env.action_space)
    plt.ion()
    obs, info = env.reset()
    screen = env.render()
    im1, cbar1, im2 = set_plots(agent.q_table, screen)
    reward = 0

    for episodes in range(1000):
        obs, info = env.reset()
        done = False
        while not done:
            if np.random.random() < 0.85:
                action = agent.get_action(obs)
            else:
                action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)
            if done:
                next_obs = obs
            screen = env.render()
            agent.update_policy(obs, action, reward, next_obs)
            update_plots(im1, cbar1, im2, agent.q_table, screen)
            obs = next_obs

    # Wait until the plot window is closed
    plt.ioff()
    env.close()


if __name__ == "__main__":
    main()
