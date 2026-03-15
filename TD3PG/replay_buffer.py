import numpy as np
from AlgorithmImports import *


class ReplayBuffer:
    """
    Shared replay buffer storing transitions from ALL universe symbols.
    Each tuple includes sym_idx so the shared network knows which
    embedding to look up during training.
    """

    def __init__(self, algorithm, max_size=200_000):
        self.algorithm = algorithm
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.cntr = 0

    def add(self, data):
        # data = (obs, next_obs, action, reward, done, sym_idx)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
        self.cntr += 1

    def sample(self, batch_size):
        index = np.random.randint(0, len(self.storage), size=batch_size)
        s, s2, a, r, d, si = [], [], [], [], [], []
        for i in index:
            t = self.storage[i]
            s.append(np.array(t[0], copy=False))
            s2.append(np.array(t[1], copy=False))
            a.append(np.array(t[2], copy=False))
            r.append(np.array(t[3], copy=False))
            d.append(np.array(t[4], copy=False))
            si.append(int(t[5]))
        return (np.array(s), np.array(s2), np.array(a),
                np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1),
                np.array(si, dtype=np.int64))