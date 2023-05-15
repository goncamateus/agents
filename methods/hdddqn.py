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
        self.manager_obs_dim = observation_space["manager"]
        self.manager_action_dim = action_space["manager"]
        self.worker_obs_dim = observation_space["worker"]
        self.worker_action_dim = action_space["worker"]
        self.worker = RainbowDQNAgent(
            arguments, self.worker_obs_dim, self.worker_action_dim
        )
        self.manager = RainbowDQNAgent(
            arguments, self.manager_obs_dim, self.manager_action_dim
        )
        self.env_steps = 0
        self.worker_updates = 0
        self.manager_updates = 0
        self.pre_train_steps = 100000

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
        fraction = min(global_step / self.arguments.total_timesteps, 1.0)
        self.manager.beta = self.manager.beta + fraction * (1.0 - self.manager.beta)

    def store_transition(self, transition, global_step):
        self.store_worker_transition(transition, global_step)
        if self.env_steps % 10 == 0 and self.worker_updates > self.pre_train_steps:
            self.store_manager_transition(transition, global_step)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        if self.worker_updates < self.pre_train_steps:
            manager_action = np.random.randint(self.manager_action_dim.n)
        else:
            manager_action = self.manager.get_action(state["manager"])
        action = {
            "manager": manager_action,
            "worker": self.worker.get_action(state["worker"]),
        }
        return action

    def update(self, global_step, writer) -> torch.Tensor:
        log = {}
        if global_step > self.arguments.learning_starts:
            worker_loss = self.worker.update()
            self.worker_updates += 1
            # if hard update is needed
            if self.worker_updates % self.worker.target_update == 0:
                self.worker._target_hard_update()

            # logging losses
            log.update({"losses/worker": worker_loss})
            writer.add_scalar("losses/worker", worker_loss, global_step)

        if len(self.manager.memory) > self.arguments.learning_starts:
            manager_loss = self.worker.update()
            self.worker_updates += 1
            # if hard update is needed
            if self.worker_updates % self.worker.target_update == 0:
                self.worker._target_hard_update()

            # logging losses
            log.update({"losses/manager": manager_loss})
            writer.add_scalar("losses/manager", manager_loss, global_step)

        return log
