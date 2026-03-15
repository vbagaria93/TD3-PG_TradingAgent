from QuantConnect.Indicators import *
from AlgorithmImports import *
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from TD3PG.td3_agent import SharedTD3
from TD3PG.trading_env import TradingEnv
from TD3PG.trainer import SharedRunner


class RLAlphaModel(AlphaModel):
    """
    Shared-policy alpha model.

    One SharedTD3 agent is shared across all universe symbols.
    Each symbol gets a stable integer index used for embedding lookup.
    When universe changes, the agent retrains on ALL ready symbols together,
    warm-starting from previous weights to preserve prior knowledge.
    """

    def __init__(self):
        self.symbols = {}    # symbol → SymbolData
        self.signals = {}    # symbol → float action
        self.sym_idx_map = {}    # symbol → stable int index
        self.shared_agent = None
        self._next_idx = 0
        self.Name = "RLAlphaModel"

    def _assign_idx(self, symbol):
        if symbol not in self.sym_idx_map:
            self.sym_idx_map[symbol] = self._next_idx
            self._next_idx += 1
        return self.sym_idx_map[symbol]

    def Update(self, algorithm, data):
        if self.shared_agent is None or not self.shared_agent.is_trained:
            return []

        for symbol, sd in self.symbols.items():
            if not sd.is_ready:
                continue
            if not data.ContainsKey(symbol) or data[symbol] is None:
                continue
            bar = data[symbol]
            if bar.EndTime == sd.last_bar_time:
                continue
            sd.last_bar_time = bar.EndTime
            sd.update_live(bar)
            if sd.signal_features.shape[0] < sd.window_size:
                continue
            try:
                obs = sd.get_observation()
                sym_idx = self.sym_idx_map[symbol]
                action = self.shared_agent.select_action(obs, sym_idx, noise=0)
                af = float(action[0])
                self.signals[symbol] = af
                algorithm.Debug(
                    f"[Live] {symbol} idx={sym_idx} action={af:.4f} "
                    f"obs_mean={obs.mean():.3f} obs_std={obs.std():.3f}")
            except Exception as e:
                algorithm.Debug(f"[Live] Inference error {symbol}: {e}")
        return []

    def OnSecuritiesChanged(self, algorithm, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.symbols:
                self._assign_idx(symbol)
                self.symbols[symbol] = SymbolData(algorithm, symbol)

        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            self.symbols.pop(symbol, None)
            self.signals.pop(symbol, None)

        # Retrain shared agent on ALL ready symbols
        ready = {s: sd for s, sd in self.symbols.items() if sd.is_ready}
        if not ready:
            return

        algorithm.Debug(
            f"[SharedPolicy] Retraining on {len(ready)} symbols: "
            f"{[str(s) for s in ready]}")

        train_envs, test_envs = [], []
        state_dim = None

        for symbol, sd in ready.items():
            n = sd.signal_features.shape[0]
            t_end = round(n * 0.8)
            ws = sd.window_size
            if t_end <= ws or (n - t_end) <= ws:
                algorithm.Debug(
                    f"[SharedPolicy] {symbol} too few rows ({n}), skipping")
                continue

            si = self.sym_idx_map[symbol]
            state_dim = sd.FEAT_WIDTH * ws

            te = TradingEnv(symbol_data=sd, window_size=ws,
                            start_tick=0, end_tick=t_end, buy_threshold=0.0)
            ve = TradingEnv(symbol_data=sd, window_size=ws,
                            start_tick=t_end, end_tick=None, buy_threshold=0.0)
            te.reset()
            ve.reset()
            train_envs.append((str(symbol), te, si))
            test_envs.append((str(symbol),  ve, si))

        if not train_envs or state_dim is None:
            return

        n_symbols = max(self.sym_idx_map.values()) + 1

        if self.shared_agent is None:
            self.shared_agent = SharedTD3(
                algorithm, state_dim, 1,
                max_action=1, seed=42,
                n_symbols=max(n_symbols, 50),
                embed_dim=8, h1_units=64, h2_units=32,
                dropout=0.2, weight_decay=1e-4)
            algorithm.Debug("[SharedPolicy] New SharedTD3 created")
            is_warm_start = False
        else:
            self._maybe_grow_embedding(algorithm, n_symbols)
            algorithm.Debug(
                "[SharedPolicy] Warm-starting from previous weights")
            is_warm_start = True

        # Cold start: 30 episodes to initialise policy from scratch.
        # Warm start: 10 episodes — weights already good, just adapt to new symbols.
        # LEAN kills any time step exceeding 10 minutes; fewer episodes on
        # warm-start retrains keeps each monthly retrain well within that limit.
        n_eps = 10 if is_warm_start else 30

        SharedRunner(
            algorithm, self.shared_agent,
            n_episodes=n_eps, batch_size=64,
            gamma=0.99, tau=0.005,
            noise=0.1, noise_clip=0.3,
            explore_noise=0.20, policy_frequency=2
        ).train(train_envs, test_envs)

        # Post-train sanity check
        for symbol, sd in ready.items():
            si = self.sym_idx_map[symbol]
            obs = sd.get_observation(sd.window_size * 2)
            post = [round(float(
                self.shared_agent.select_action(obs, si, noise=0)[0]), 3)
                for _ in range(5)]
            algorithm.Debug(
                f"[SharedPolicy] {symbol} post-train actions: {post}")

    def _maybe_grow_embedding(self, algorithm, n_symbols_needed):
        import torch
        current = self.shared_agent.actor.embedding.num_embeddings
        if n_symbols_needed <= current:
            return
        new_size = max(n_symbols_needed, 50)
        embed_dim = self.shared_agent.actor.embedding.embedding_dim
        algorithm.Debug(
            f"[SharedPolicy] Growing embedding {current}→{new_size}")
        for net in [self.shared_agent.actor, self.shared_agent.actor_target,
                    self.shared_agent.critic, self.shared_agent.critic_target]:
            old_w = net.embedding.weight.data
            new_emb = torch.nn.Embedding(new_size, embed_dim)
            new_emb.weight.data[:current] = old_w
            net.embedding = new_emb.to(old_w.device)


# ══════════════════════════════════════════════════════════════════════════

class SymbolData:
    """
    Builds and caches normalised features for one symbol.
    Does NOT own an agent — inference goes through the shared agent
    in RLAlphaModel, keyed by sym_idx.
    """

    IND_COLS = ['macd', 'rsi', 'adx', 'cci']
    RAW_COLS = ['endTime', 'open', 'high', 'low', 'close'] + IND_COLS
    SIGNAL_COLS = ['endTime', 'ret_o', 'ret_h', 'ret_l', 'ret_c',
                   'pca1', 'pca2', 'pca3']

    N_PCA = 3
    FEAT_WIDTH = 7
    WARMUP_BARS = 350
    MIN_BARS = 80
    WINDOW_SIZE = 10
    OBS_CLIP = 4.0

    def __init__(self, algorithm, symbol):
        self.algorithm = algorithm
        self.symbol = symbol
        self.window_size = self.WINDOW_SIZE
        self.raw_features = pd.DataFrame(columns=self.RAW_COLS)
        self.signal_features = pd.DataFrame(columns=self.SIGNAL_COLS)
        self.pca = None
        self.scaler_ind = None
        self.is_ready = False
        self.last_bar_time = None
        self._prev_close = None

        D = algorithm.Debug
        self._macd = MovingAverageConvergenceDivergence(
            12, 26, 9, MovingAverageType.Exponential)
        self._rsi = RelativeStrengthIndex(14, MovingAverageType.Simple)
        self._adx = AverageDirectionalIndex(14)
        self._cci = CommodityChannelIndex(14, MovingAverageType.Simple)

        history = algorithm.History(symbol, self.WARMUP_BARS, Resolution.Daily)
        if history.empty:
            D(f"[Init] {symbol}: no history. Skipping.")
            return
        D(f"[Init] {symbol}: history rows={len(history)}")

        for idx, row in history.iterrows():
            t = idx[1]
            tb = TradeBar(t, symbol, row['open'], row['high'],
                          row['low'], row['close'], row['volume'])
            self._update_indicators(t, tb)
            self._collect_raw(tb)

        n_raw = self.raw_features.shape[0]
        D(f"[Init] {symbol}: raw_features={n_raw} rows")
        if n_raw < self.MIN_BARS:
            D(f"[Init] {symbol}: only {n_raw} bars (need {self.MIN_BARS}). Skipping.")
            return

        closes = self.raw_features['close'].values.astype(float)
        prev_c = np.roll(closes, 1)
        prev_c[0] = closes[0]

        def safe_log(num, den):
            return np.log(np.maximum(num / np.maximum(den, 1e-9), 1e-9))

        ind_data = self.raw_features[self.IND_COLS].values.astype(float)
        self.scaler_ind = StandardScaler().fit(ind_data)
        ind_scaled = self.scaler_ind.transform(ind_data)
        D(f"[Init] {symbol}: ind_scaled mean={ind_scaled.mean():.3f} "
          f"std={ind_scaled.std():.3f} min={ind_scaled.min():.3f} max={ind_scaled.max():.3f}")

        try:
            self.pca = PCA(n_components=self.N_PCA)
            pca_v = self.pca.fit_transform(ind_scaled)
            D(f"[Init] {symbol}: PCA explained variance: "
              f"{[round(v, 3) for v in self.pca.explained_variance_ratio_]}")
        except Exception as e:
            D(f"[Init] {symbol}: PCA failed: {e}")
            return

        self.signal_features = pd.DataFrame({
            'endTime': self.raw_features['endTime'].reset_index(drop=True),
            'ret_o':   safe_log(self.raw_features['open'].values.astype(float), prev_c),
            'ret_h':   safe_log(self.raw_features['high'].values.astype(float), prev_c),
            'ret_l':   safe_log(self.raw_features['low'].values.astype(float),  prev_c),
            'ret_c':   safe_log(closes, prev_c),
            'pca1':    pca_v[:, 0], 'pca2': pca_v[:, 1], 'pca3': pca_v[:, 2],
        })
        self.signal_features.columns = self.SIGNAL_COLS
        self._prev_close = float(self.raw_features['close'].iloc[-1])

        sample = self.get_observation(self.window_size * 2)
        D(f"[Init] {symbol}: sample_obs mean={sample.mean():.3f} "
          f"std={sample.std():.3f} min={sample.min():.3f} max={sample.max():.3f}")

        self.is_ready = True  # features ready; agent training deferred to RLAlphaModel

    def _update_indicators(self, t, tb):
        self._macd.Update(t, tb.Close)
        self._rsi.Update(t,  tb.Close)
        self._adx.Update(tb)
        self._cci.Update(tb)

    def _collect_raw(self, bar):
        if not (self._macd.IsReady and self._rsi.IsReady and
                self._adx.IsReady and self._cci.IsReady):
            return
        row = pd.DataFrame([[
            bar.EndTime,
            float(bar.Open), float(bar.High), float(bar.Low), float(bar.Close),
            float(self._macd.Current.Value), float(self._rsi.Current.Value),
            float(self._adx.Current.Value),  float(self._cci.Current.Value),
        ]], columns=self.RAW_COLS)
        self.raw_features = pd.concat(
            [self.raw_features, row], ignore_index=True)

    def update_live(self, bar):
        t = bar.EndTime
        tb = TradeBar(t, self.symbol, bar.Open, bar.High,
                      bar.Low, bar.Close, bar.Volume)
        self._update_indicators(t, tb)
        if not (self._macd.IsReady and self._rsi.IsReady and
                self._adx.IsReady and self._cci.IsReady):
            return
        if not self.is_ready or self._prev_close is None:
            return
        pc = self._prev_close
        eps = 1e-9
        def lr(x): return float(np.log(max(x / max(pc, eps), eps)))
        ind_raw = np.array([[float(self._macd.Current.Value),
                             float(self._rsi.Current.Value),
                             float(self._adx.Current.Value),
                             float(self._cci.Current.Value)]])
        ind_s = self.scaler_ind.transform(ind_raw)[0]
        pca_v = self.pca.transform(ind_s.reshape(1, -1))[0]
        row = pd.DataFrame([[bar.EndTime,
                             lr(float(bar.Open)),  lr(float(bar.High)),
                             lr(float(bar.Low)),   lr(float(bar.Close)),
                             pca_v[0], pca_v[1], pca_v[2]]],
                           columns=self.SIGNAL_COLS)
        self.signal_features = pd.concat(
            [self.signal_features, row], ignore_index=True)
        self._prev_close = float(bar.Close)

    def get_observation(self, current_tick=None):
        if current_tick is None:
            frame = self.signal_features.tail(self.window_size).iloc[:, 1:]
        else:
            frame = self.signal_features.iloc[
                (current_tick - self.window_size): current_tick, 1:]
        return np.clip(
            frame.values.flatten().astype(np.float32),
            -self.OBS_CLIP, self.OBS_CLIP)

    def length(self):
        return self.signal_features.shape[0]

    def width(self):
        return self.FEAT_WIDTH