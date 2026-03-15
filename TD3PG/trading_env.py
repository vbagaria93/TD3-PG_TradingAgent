import gym
from gym import spaces
from AlgorithmImports import *
import numpy as np
from enum import Enum


class Actions(Enum):
    Hold = 0
    Buy  = 1
    Exit = 2


class Positions(Enum):
    Null = 0
    Long = 1


class TradingEnv(gym.Env):
    """
    Long-only trading environment for SharedTD3.
    Negative action = close/stay flat. Positive action = enter long.

    Reward (asymmetric v4):
      Open long        → -TRANSACTION_COST
      Close long       → -TRANSACTION_COST
      Long, winning    → ret + MOMENTUM_BONUS  (0.0015 — 3x hold penalty)
      Long, losing     → ret
      Flat             → -HOLD_PENALTY         (0.0005)
    """

    TRANSACTION_COST = 0.001
    HOLD_PENALTY     = 0.0005
    MOMENTUM_BONUS   = 0.0015   # raised 0.0008→0.0015: unambiguous positive signal

    def __init__(self, symbol_data=None, window_size=10,
                 start_tick=0, end_tick=None, buy_threshold=0.0):

        self.window_size   = window_size
        self.symbol_data   = symbol_data
        self.buy_threshold = buy_threshold
        self.action_space  = spaces.Box(-1, +1, (1,), dtype=np.float32)

        self._start_tick = start_tick + self.window_size
        self._end_tick   = (symbol_data.length() - 1
                            if end_tick is None else end_tick - 1)

        self._done             = None
        self._current_tick     = None
        self._position         = None
        self._position_history = None
        self._total_reward     = None
        self._entry_log_price  = None
        self.history           = None

    def reset(self):
        self._done             = False
        self._current_tick     = self._start_tick
        self._position         = Positions.Null
        self._position_history = ([None] * self.window_size) + [self._position]
        self._total_reward     = 0.0
        self._entry_log_price  = None
        self.history           = {}
        obs = self._get_observation()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs.shape[0],), dtype=np.float32)
        return obs

    def step(self, action):
        af  = float(action[0]) if hasattr(action, '__len__') else float(action)
        buy = af >= self.buy_threshold

        self._current_tick += 1
        self._done = (self._current_tick >= self._end_tick)

        prev_pos = self._position
        reward   = 0.0

        if buy and prev_pos == Positions.Null:
            self._position        = Positions.Long
            self._entry_log_price = self._log_price()
            reward -= self.TRANSACTION_COST
        elif not buy and prev_pos == Positions.Long:
            self._position        = Positions.Null
            self._entry_log_price = None
            reward -= self.TRANSACTION_COST

        ret = float(np.clip(self._bar_return(), -0.05, 0.05))

        if self._position == Positions.Long:
            reward += ret
            if self._entry_log_price is not None:
                if self._log_price() - self._entry_log_price > 0:
                    reward += self.MOMENTUM_BONUS
        else:
            reward -= self.HOLD_PENALTY

        self._total_reward += reward
        self._position_history.append(self._position)
        obs  = self._get_observation()
        info = dict(total_reward=self._total_reward, position=self._position.value)
        self._update_history(info)
        return obs, reward, self._done, info

    def _log_price(self):
        cols = self.symbol_data.signal_features.columns.tolist()
        col  = 'ret_c' if 'ret_c' in cols else 'close'
        ci   = cols.index(col)
        val  = float(self.symbol_data.signal_features.iat[self._current_tick, ci])
        if col == 'close':
            return np.log(val) if val > 0 else 0.0
        vals = self.symbol_data.signal_features.iloc[
            self._start_tick:self._current_tick + 1, ci].values
        return float(np.sum(vals))

    def _bar_return(self):
        cols = self.symbol_data.signal_features.columns.tolist()
        col  = 'ret_c' if 'ret_c' in cols else 'close'
        ci   = cols.index(col)
        if self._current_tick <= self._start_tick:
            return 0.0
        curr = float(self.symbol_data.signal_features.iat[self._current_tick, ci])
        if col == 'close':
            prev = float(self.symbol_data.signal_features.iat[self._current_tick - 1, ci])
            return (curr - prev) / prev if prev != 0 else 0.0
        return curr

    def _get_observation(self):
        return self.symbol_data.get_observation(self._current_tick)

    def _update_history(self, info):
        if not self.history:
            self.history = {k: [] for k in info}
        for k, v in info.items():
            self.history[k].append(v)

    def length(self):
        return self._end_tick - self._start_tick