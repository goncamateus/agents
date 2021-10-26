import torch
from utils.buffer import ReplayBuffer
from utils.experiment import HyperParameters, soft_update

from methods.networks import Actor, Critic, TargetActor, TargetCritic


class DDPGAgent:
    def __init__(self, hp: HyperParameters):
        self.state_size = hp.N_OBS
        self.action_size = hp.N_ACTS
        self.buffer_size = hp.REPLAY_SIZE
        self.batch_size = hp.BATCH_SIZE
        self.gamma = hp.GAMMA
        self.lr_actor = hp.LEARNING_RATE
        self.lr_critic = hp.LEARNING_RATE
        self.device = hp.DEVICE

        # Actor Networks
        self.actor = Actor(hp.N_OBS, hp.N_ACTS).to(hp.DEVICE)
        self.actor_target = Actor(hp.N_OBS, hp.N_ACTS).to(hp.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=hp.LEARNING_RATE
        )

        # Critic Networks
        self.critic = Critic(hp.N_OBS, hp.N_ACTS).to(hp.DEVICE)
        self.critic_target = Critic(hp.N_OBS, hp.N_ACTS).to(hp.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_loss_func = torch.nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=hp.LEARNING_RATE
        )

        self.replay_buffer = ReplayBuffer(hp.REPLAY_SIZE)

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action = self.actor(state).cpu().detach().numpy()
        return action

    def train(self):
        train_logs = {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = self.actor_target(next_states)
        next_Qs = self.critic_target(next_states, next_actions)

        # Compute Q targets for current states
        next_Qs[dones] = 0.0
        Q_targets = rewards + self.gamma * next_Qs.detach()

        # Compute critic loss
        actual_Qs = self.critic(states, actions)
        critic_loss = self.critic_loss_func(actual_Qs, Q_targets)
        train_logs["train/loss_Q"] = critic_loss.item()

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor(states)
        critic_evaluation = self.critic(states, actions_pred)
        actor_loss = -critic_evaluation.mean()
        train_logs["train/loss_pi"] = actor_loss.item()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        soft_update(self.actor_target, self.actor, 1 - 1e-3)
        soft_update(self.critic_target, self.critic, 1 - 1e-3)

        return train_logs

