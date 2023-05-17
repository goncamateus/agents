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
        self.pre_train_steps = 150000
        self.epsilon_decay = self.arguments.eps_greedy_decay
        self.epsilon_min = 0.03
        self.worker_epsilon = 1
        self.manager_epsilon = 1
        self.last_manager_action = None
        self.randomness_rate_worker = 0

    def store_worker_transition(self, transition, global_step):
        for i in range(2):
            tran = copy(transition["worker"])
            tran[2] = transition["worker"][2][i]
            self.workers[i].transition = tran
            # N-step transition
            if self.workers[i].use_n_step:
                one_step_transition = self.workers[i].memory_n.store(
                    *self.workers[i].transition
                )
            # 1-step transition
            else:
                one_step_transition = self.workers[i].transition

            # add a single step transition
            if one_step_transition:
                self.workers[i].memory.store(*one_step_transition)

            # PER: increase beta
            fraction = min(global_step / self.arguments.total_timesteps, 1.0)
            self.workers[i].beta = self.workers[i].beta + fraction * (
                1.0 - self.workers[i].beta
            )

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
        fraction = min(global_step / self.arguments.total_timesteps, 1.0)
        self.manager.beta = self.manager.beta + fraction * (1.0 - self.manager.beta)

    def store_transition(self, transition, global_step):
        self.store_worker_transition(transition, global_step)
        if self.env_steps % 5 == 0 and self.worker_updates > self.pre_train_steps:
            self.store_manager_transition(transition, global_step)

    def get_action(self, state: np.ndarray, global_step: int) -> np.ndarray:
        """Select an action from the input state."""
        if self.worker_updates < self.pre_train_steps:
            manager_action = self.last_manager_action
            if self.last_manager_action is None:
                self.last_manager_action = self.manager_action_space.sample()
                manager_action = self.last_manager_action
        else:
            manager_action = self.manager.get_action(state["manager"])
            
        self.worker_epsilon = self.epsilon_decay ** (global_step / 100)
        self.worker_epsilon = max(self.worker_epsilon, self.epsilon_min)
        if np.random.uniform() < self.worker_epsilon:
            worker_action = self.worker_action_space.sample()
            self.randomness_rate_worker += 1
        else:
            values = 0
            for i in range(2):
                values += (
                    self.workers[i]
                    .dqn(torch.FloatTensor(state["worker"]).to(self.workers[i].device))
                    .detach()
                    .cpu()
                    .numpy()
                )
            worker_action = values.argmax()
        action = {
            "manager": manager_action,
            "worker": worker_action,
        }
        return action

    def update(self, global_step, writer) -> torch.Tensor:
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
            and self.worker_updates % self.arguments.manager_update_freq == 0
        ):
            manager_loss = self.manager.update()
            self.manager_updates += 1
            # if hard update is needed
            if self.manager_updates % self.manager.target_update == 0:
                self.manager._target_hard_update()
            # logging losses
            log.update({"losses/manager": manager_loss})
            writer.add_scalar("losses/manager", manager_loss, global_step)

        return log
