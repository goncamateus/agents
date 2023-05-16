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
        self.worker = RainbowDQNAgent(
            worker_arguments, self.worker_obs_space, self.worker_action_space
        )
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
        self.pre_train_steps = 150000
        self.epsilon_decay = self.arguments.eps_greedy_decay
        self.epsilon_min = 0.1
        self.worker_epsilon = 1
        self.manager_epsilon = 1

    def store_worker_transition(self, transition, global_step):
        self.worker.transition = transition["worker"]
        # N-step transition
        if self.worker.use_n_step:
            one_step_transition = self.worker.memory_n.store(*self.worker.transition)
        # 1-step transition
        else:
            one_step_transition = self.worker.transition

        # add a single step transition
        if one_step_transition:
            self.worker.memory.store(*one_step_transition)

        # PER: increase beta
        fraction = min(global_step / self.arguments.total_timesteps, 1.0)
        self.worker.beta = self.worker.beta + fraction * (1.0 - self.worker.beta)

    def store_manager_transition(self, transition, global_step):
        self.manager.transition = transition["manager"]
        # N-step transition
        if self.manager.use_n_step:
            one_step_transition = self.manager.memory_n.store(*self.manager.transition)
        # 1-step transition
        else:
            one_step_transition = self.manager.transition

        # add a single step transition
        if one_step_transition:
            self.manager.memory.store(*one_step_transition)

        # PER: increase beta
        fraction = min(
            (global_step - self.pre_train_steps) / self.arguments.total_timesteps, 1.0
        )
        self.manager.beta = self.manager.beta + fraction * (1.0 - self.manager.beta)

    def store_transition(self, transition, global_step):
        self.store_worker_transition(transition, global_step)
        if self.env_steps % 10 == 0 and self.worker_updates > self.pre_train_steps:
            self.store_manager_transition(transition, global_step)

    def get_action(self, state: np.ndarray, global_step: int) -> np.ndarray:
        """Select an action from the input state."""
        if self.worker_updates < self.pre_train_steps:
            manager_action = self.manager_action_space.sample()
        else:
            eps_step = global_step - self.pre_train_steps
            self.manager_epsilon = self.epsilon_decay ** (eps_step / 100)
            self.manager_epsilon = max(self.manager_epsilon, self.epsilon_min)
            if np.random.uniform() < self.manager_epsilon:
                manager_action = self.manager_action_space.sample()
            else:
                manager_action = self.manager.get_action(state["manager"])

        self.worker_epsilon = self.epsilon_decay ** (global_step / 100)
        self.worker_epsilon = max(self.worker_epsilon, self.epsilon_min)
        if np.random.uniform() < self.worker_epsilon:
            worker_action = self.worker_action_space.sample()
        else:
            worker_action = self.worker.get_action(state["worker"])
        action = {
            "manager": manager_action,
            "worker": worker_action,
        }
        return action

    def update(self, global_step, writer) -> torch.Tensor:
        log = {}
        if len(self.worker.memory) > self.arguments.worker_batch_size:
            worker_loss = self.worker.update()
            self.worker_updates += 1
            # if hard update is needed
            if self.worker_updates % self.worker.target_update == 0:
                self.worker._target_hard_update()

            # logging losses
            log.update({"losses/worker": worker_loss})
            writer.add_scalar("losses/worker", worker_loss, global_step)

        if (
            len(self.manager.memory) > self.arguments.manager_batch_size
            and self.worker_updates % self.arguments.manager_update_freq == 0
        ):
            manager_loss = self.manager.update()
            self.manager_updates += 1
            # if hard update is needed
            if self.manager_updates % self.manager.target_update == 0:
                self.worker._target_hard_update()
            # logging losses
            log.update({"losses/manager": manager_loss})
            writer.add_scalar("losses/manager", manager_loss, global_step)

        return log
