# region imports
from AlgorithmImports import *
# endregion
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Symbol-aware actor.  A learned embedding (n_symbols × embed_dim) is
    looked up by sym_idx and concatenated to the observation before the
    FC layers.  This lets the network learn both cross-stock patterns
    (shared weights) and per-stock behaviour (unique embedding vector).

    Dropout on hidden layers only — disabled at inference via model.eval().
    """

    def __init__(self, state_dim, action_dim, max_action, seed,
                 n_symbols=50, embed_dim=8,
                 h1_units=64, h2_units=32, dropout=0.2):
        super().__init__()
        torch.manual_seed(seed)
        self.max_action = max_action
        self.embedding = nn.Embedding(n_symbols, embed_dim)
        in_dim = state_dim + embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1_units), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1_units, h2_units), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2_units, action_dim), nn.Tanh(),
        )

    def forward(self, state, sym_idx):
        # sym_idx: (batch,) LongTensor
        emb = self.embedding(sym_idx)           # (batch, embed_dim)
        x = torch.cat([state, emb], dim=1)   # (batch, state_dim+embed_dim)
        return self.max_action * self.net(x)


class Critic(nn.Module):
    """
    Twin Q-networks, also symbol-aware via the same embedding approach.
    No dropout — we want stable Q-estimates.
    """

    def __init__(self, state_dim, action_dim, seed,
                 n_symbols=50, embed_dim=8, h1_units=64, h2_units=32):
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = nn.Embedding(n_symbols, embed_dim)
        in_dim = state_dim + embed_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, h1_units), nn.ReLU(),
            nn.Linear(h1_units, h2_units), nn.ReLU(),
            nn.Linear(h2_units, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, h1_units), nn.ReLU(),
            nn.Linear(h1_units, h2_units), nn.ReLU(),
            nn.Linear(h2_units, 1),
        )

    def _augment(self, state, action, sym_idx):
        emb = self.embedding(sym_idx)
        return torch.cat([state, emb, action], dim=1)

    def forward(self, state, action, sym_idx):
        x = self._augment(state, action, sym_idx)
        return self.q1(x), self.q2(x)

    def Q1(self, state, action, sym_idx):
        return self.q1(self._augment(state, action, sym_idx))