import numpy as np

from agents.methods.value_based.q_learning.q_learning import QLearning


class NumpyQLearning(QLearning):
    def set_table(self):
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def get_output(self, observation):
        if np.all(self.q_table[observation] == self.q_table[observation][0]):
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q_table[observation])
        return action

    def epsilon_greedy(self, observation):
        if np.random.random() < 1 - self.epsilon:
            action = self.get_output(observation)
        else:
            action = np.random.randint(self.num_actions)
        return action
