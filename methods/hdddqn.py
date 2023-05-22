import random
from copy import copy

import numpy as np
import torch

from methods.rainbow import RainbowDQNAgent


class HDDDQN:
    def __init__(
        self,
        arguments,
        observation_space,
        action_space,
    ):
        self.arguments = arguments
        self.manager_obs_space = observation_space["manager"]
        self.manager_action_space = action_space["manager"]
        self.worker_obs_space = observation_space["worker"]
        self.worker_action_space = action_space["worker"]
        worker_arguments = copy(arguments)
        worker_arguments.batch_size = arguments.worker_batch_size
        worker_arguments.gamma = arguments.worker_gamma
        self.workers = [
            RainbowDQNAgent(
                worker_arguments, self.worker_obs_space, self.worker_action_space
            )
        ] * 2
        manager_arguments = copy(arguments)
        manager_arguments.batch_size = arguments.manager_batch_size
        manager_arguments.gamma = arguments.manager_gamma
        manager_arguments.target_network_frequency = (
            arguments.manager_target_update_freq
        )
        self.manager = RainbowDQNAgent(
            manager_arguments, self.manager_obs_space, self.manager_action_space
        )
        self.env_steps = 0
        self.worker_updates = 0
        self.manager_updates = 0
        self.pre_train_steps = self.arguments.pre_train_steps
        self.epsilon_decay = self.arguments.eps_greedy_decay
        self.epsilon_min = 0.1
        self.epsilon_max = 1
        self.randomness_rate_worker = 0
        self.manager_action_count = 0
        self.max_steps = 20
        self.num_episodes = {
            "worker": 0,
            "manager": 0,
        }
        self.worker_epsilons = self.epsilon_vals(
            self.arguments.worker_max_episodes + self.arguments.pre_train_steps
        )
        self.worker_epsilon = self.worker_epsilons[0]
        self.manager_epsilons = self.epsilon_vals(self.arguments.manager_max_steps)
        self.manager_epsilon = self.manager_epsilons[0]

    def epsilon_vals(self, total_steps):
        epsilon_decay_steps = int(total_steps * self.epsilon_decay)
        epsilon_decay_vals = np.linspace(
            start=self.epsilon_max,
            stop=self.epsilon_min,
            num=epsilon_decay_steps,
        )
        all_vals = np.append(
            epsilon_decay_vals,
            np.array(
                [self.epsilon_min]
                * int(total_steps * (1 - self.epsilon_decay) + self.max_steps)
            ),
        )

        return all_vals

    def store_worker_transition(self, transition):
        for i in range(2):
            tran = copy(transition["worker"])
            tran[2] = transition["worker"][2][i]
            self.workers[i].transition = tran
            self.workers[i].memory.store(*self.workers[i].transition)
            self.workers[i].beta = 0.4

    def store_manager_transition(self, transition):
        if transition["manager"][1] is not None:
            self.manager.transition = transition["manager"]
            self.manager.memory.store(*self.manager.transition)
            self.manager.beta = 0.4

    def store_transition(self, transition, manager_store=False):
        self.store_worker_transition(transition)
        if self.pre_train_steps <= 0 and manager_store:
            self.store_manager_transition(transition)

    def get_action(
        self, state: np.ndarray, global_step: int, resample_manager=False
    ) -> np.ndarray:
        """Select an action from the input state."""
        manager_action = None
        if self.pre_train_steps > 0 and resample_manager:
            manager_action = random.randrange(self.manager_action_space.n)
        elif resample_manager:
            manager_action = random.randrange(self.manager_action_space.n)
            self.manager_epsilon = self.manager_epsilons[self.manager_action_count]
            if random.random() > self.manager_epsilon:
                manager_action = self.manager.get_action(state["manager"])
            self.manager_action_count += 1

        try:
            self.worker_epsilon = self.worker_epsilons[global_step]
        except IndexError:
            self.worker_epsilon = self.worker_epsilons[-1]
        if random.random() > self.worker_epsilon:
            worker_action = random.randrange(self.worker_action_space.n)
            self.randomness_rate_worker += 1
        else:
            stacked_pred = None
            for worker in self.workers:
                worker.dqn.eval()
                pred = worker.get_action(state["worker"])
                if stacked_pred is None:
                    stacked_pred = pred
                else:
                    stacked_pred = np.vstack((stacked_pred, pred))
                worker.dqn.train()
            worker_action = stacked_pred.argmax()  
        self.pre_train_steps -= 1
        action = {
            "manager": manager_action,
            "worker": worker_action,
        }
        return action

    def update(self, global_step, writer, update_manager=False) -> torch.Tensor:
        log = {}
        if len(self.workers[0].memory) > self.arguments.worker_batch_size:
            self.worker_updates += 1
            for i in range(2):
                worker_loss = self.workers[i].update()
                # if hard update is needed
                if self.worker_updates % self.workers[i].target_update == 0:
                    self.workers[i]._target_hard_update()

                # logging losses
                log.update({f"losses/worker_{i}": worker_loss})
                writer.add_scalar(f"losses/worker_{i}", worker_loss, global_step)

        if (
            len(self.manager.memory) > self.arguments.manager_batch_size
            and update_manager
        ):
            for _ in range(self.arguments.manager_updates):
                manager_loss = self.manager.update()
                self.manager_updates += 1
                # if hard update is needed
                if self.manager_updates % self.manager.target_update == 0:
                    self.manager._target_hard_update()
            # logging losses
            log.update({"losses/manager": manager_loss})
            writer.add_scalar("losses/manager", manager_loss, global_step)

        return log
