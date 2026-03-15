import numpy as np
import torch
import torch.nn.functional as F
from TD3PG.actor_critic import Actor, Critic
from TD3PG.replay_buffer import ReplayBuffer
from AlgorithmImports import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SharedTD3:
    """
    One shared Actor + Critic trained across ALL universe symbols.
    sym_idx routes the per-symbol embedding so the network learns
    both cross-stock patterns and per-stock behaviour.
    """

    def __init__(self, algorithm, state_dim, action_dim, max_action, seed,
                 n_symbols=50, embed_dim=8,
                 h1_units=64, h2_units=32,
                 dropout=0.2, weight_decay=1e-4):
        self.algorithm = algorithm
        self.max_action = max_action
        self.is_trained = False

        def actor():
            return Actor(state_dim, action_dim, max_action, seed,
                         n_symbols, embed_dim, h1_units, h2_units, dropout).to(device)

        def critic():
            return Critic(state_dim, action_dim, seed,
                          n_symbols, embed_dim, h1_units, h2_units).to(device)

        self.actor = actor()
        self.actor_target = actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4, weight_decay=weight_decay)

        self.critic = critic()
        self.critic_target = critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4, weight_decay=weight_decay * 2)

        self.replay_buffer = ReplayBuffer(algorithm)

    # ──────────────────────────────────────────────────────────────────────

    def select_action(self, state, sym_idx, noise=0.0):
        """Inference — eval() disables dropout for deterministic output."""
        self.actor.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state.reshape(1, -1)).to(device)
            si = torch.LongTensor([sym_idx]).to(device)
            act = self.actor(s, si).cpu().numpy().flatten()
        self.actor.train()
        if noise != 0:
            act = act + np.random.normal(0, noise, size=act.shape)
        return act.clip(-self.max_action, self.max_action)

    # ──────────────────────────────────────────────────────────────────────

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99,
              tau=0.005, policy_noise=0.1, noise_clip=0.3, policy_freq=2):
        if replay_buffer.cntr < batch_size:
            return

        for it in range(iterations):
            s, s2, a, r, d, si = replay_buffer.sample(batch_size)

            state = torch.FloatTensor(s.reshape(batch_size, -1)).to(device)
            next_state = torch.FloatTensor(
                s2.reshape(batch_size, -1)).to(device)
            action = torch.FloatTensor(a).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            sym_idx = torch.LongTensor(si).to(device)

            with torch.no_grad():
                noise_t = torch.clamp(
                    torch.randn_like(action) * policy_noise,
                    -noise_clip, noise_clip)
                self.actor_target.eval()
                next_act = torch.clamp(
                    self.actor_target(next_state, sym_idx) + noise_t,
                    -self.max_action, self.max_action)
                self.actor_target.train()
                tQ1, tQ2 = self.critic_target(next_state, next_act, sym_idx)
                target_Q = reward + done * discount * torch.min(tQ1, tQ2)

            cur_Q1, cur_Q2 = self.critic(state, action, sym_idx)
            critic_loss = (F.mse_loss(cur_Q1, target_Q) +
                           F.mse_loss(cur_Q2, target_Q))

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                actions = self.actor(state, sym_idx)
                # 0.01 action penalty keeps actor from saturating at ±1
                actor_loss = (-self.critic.Q1(state, actions, sym_idx).mean()
                              # reduced from 0.01 — was creating gravity well at 0
                              + 0.0 * (actions ** 2).mean())  # L2 removed — caused action collapse to -epsilon

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()

                for p, tp in zip(self.critic.parameters(),
                                 self.critic_target.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
                for p, tp in zip(self.actor.parameters(),
                                 self.actor_target.parameters()):
                    tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    # ──────────────────────────────────────────────────────────────────────

    def save(self, filename="shared_best"):
        torch.save(self.actor.state_dict(),  f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")

    def load(self, filename="shared_best"):
        self.actor.load_state_dict(torch.load(f"{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{filename}_critic.pth"))