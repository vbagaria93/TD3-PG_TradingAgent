from TD3PG.universe_selector import LiquidValueUniverseSelectionModel
from TD3PG.alpha_model import RLAlphaModel
from AlgorithmImports import *
import numpy as np
from scipy.stats import norm, skew, kurtosis as sp_kurtosis


class RLAlgorithm(QCAlgorithm):

    STOP_LOSS_PCT    = 0.12
    MAX_WEIGHT       = 0.33
    BUY_THRESHOLD    = 0.0
    EXIT_THRESHOLD   = -0.05
    COOLDOWN_DAYS    = 5
    N_TRIALS         = 22
    SPY_SR_BENCHMARK = 0.64   # SPY Sharpe over 2017-2020 (LEAN's actual benchmark)

    def Initialize(self):
        self.SetStartDate(2017, 1, 1)
        self.SetEndDate(2020, 1, 1)
        self.SetCash(100_000)

        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage,
                               AccountType.Cash)

        self.SetUniverseSelection(LiquidValueUniverseSelectionModel())
        self.UniverseSettings.Resolution = Resolution.Daily

        self.alpha_model = RLAlphaModel()
        self.SetAlpha(self.alpha_model)
        self.SetPortfolioConstruction(NullPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())

        self._entry_prices    = {}
        self._cooldown        = {}
        self._any_exit_today  = False
        self._prev_equity     = 100_000.0
        self._daily_returns   = []
        self._attempted_buys  = set()

    def OnSecuritiesChanged(self, changes):
        if changes.AddedSecurities:
            self._attempted_buys.clear()

    def OnData(self, data):
        curr_equity = float(self.Portfolio.TotalPortfolioValue)
        if self._prev_equity > 0:
            self._daily_returns.append(
                (curr_equity - self._prev_equity) / self._prev_equity)
        self._prev_equity = curr_equity

        signals = self.alpha_model.signals
        if not signals:
            return

        for sym in list(self._cooldown.keys()):
            self._cooldown[sym] -= 1
            if self._cooldown[sym] <= 0:
                del self._cooldown[sym]
                self.Debug(f"[Cooldown] {sym} expired")

        self._any_exit_today = False

        # ── Step 1: Hard stop-loss ────────────────────────────────────────
        for symbol, entry in list(self._entry_prices.items()):
            if not self.Securities.ContainsKey(symbol):
                continue
            price = float(self.Securities[symbol].Price)
            if price <= 0 or entry <= 0:
                continue
            loss_pct = (entry - price) / entry
            if loss_pct >= self.STOP_LOSS_PCT:
                self.Liquidate(symbol)
                self._entry_prices.pop(symbol, None)
                self._cooldown[symbol] = self.COOLDOWN_DAYS
                self._attempted_buys.discard(symbol)
                self._any_exit_today   = True
                self.Debug(f"[StopLoss] {symbol} loss={loss_pct*100:.1f}%")

        # ── Step 2: Signal exit ───────────────────────────────────────────
        for symbol in list(self.Portfolio.Keys):
            if not self.Portfolio[symbol].Invested:
                continue
            af = signals.get(symbol, None)
            if af is None or af < self.EXIT_THRESHOLD:
                self.Liquidate(symbol)
                self._entry_prices.pop(symbol, None)
                self._attempted_buys.discard(symbol)
                self._any_exit_today = True
                self.Debug(f"[Exit] {symbol} signal={'none' if af is None else f'{af:.3f}'}")

        # ── Step 3: New entries ───────────────────────────────────────────
        if self._any_exit_today:
            return

        buy_symbols = [
            s for s, af in signals.items()
            if af >= self.BUY_THRESHOLD
            and s not in self._cooldown
            and not self.Portfolio[s].Invested
            and s not in self._attempted_buys
        ]
        if not buy_symbols:
            return
        weight = min(1.0 / len(buy_symbols), self.MAX_WEIGHT)

        for symbol in buy_symbols:
            if not self.Securities.ContainsKey(symbol):
                continue
            sec = self.Securities[symbol]
            if not sec.IsTradable or sec.Price <= 0:
                continue
            self.SetHoldings(symbol, weight)
            self._entry_prices[symbol] = float(sec.Price)
            self._attempted_buys.add(symbol)
            self.Debug(f"[Entry] {symbol} @ {sec.Price:.2f} "
                       f"action={signals[symbol]:.3f} weight={weight:.2f}")

    def OnEndOfAlgorithm(self):
        rets = np.array(self._daily_returns)
        if len(rets) < 30:
            self.Debug("[Metrics] Not enough return data.")
            return

        T   = len(rets)
        mu  = float(np.mean(rets))
        sig = float(np.std(rets, ddof=1))

        ann_return = (1 + mu) ** 252 - 1
        ann_vol    = sig * np.sqrt(252)
        sharpe     = (mu / sig) * np.sqrt(252) if sig > 0 else 0.0

        down     = rets[rets < 0]
        down_std = float(np.std(down, ddof=1)) if len(down) > 1 else 1e-9
        sortino  = (mu / down_std) * np.sqrt(252)

        cumulative  = np.cumprod(1 + rets)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns   = (cumulative - rolling_max) / rolling_max
        max_dd      = float(np.min(drawdowns))
        calmar      = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        gamma3 = float(skew(rets))
        gamma4 = float(sp_kurtosis(rets) + 3)

        sr_std = np.sqrt(
            (1 - gamma3 * sharpe + (gamma4 - 1) / 4 * sharpe ** 2) / (T - 1)
        ) if T > 1 else 1e-9

        psr_vs_zero = float(norm.cdf(sharpe / sr_std)) if sr_std > 0 else 0.0
        psr_vs_spy  = float(norm.cdf(
            (sharpe - self.SPY_SR_BENCHMARK) / sr_std)) if sr_std > 0 else 0.0

        N = self.N_TRIALS
        euler_gamma = 0.5772156649
        e_max_sr = (
            (1 - euler_gamma) * norm.ppf(1 - 1.0 / N)
            + euler_gamma     * norm.ppf(1 - 1.0 / (N * np.e))
        )
        dsr = float(norm.cdf(
            (sharpe - e_max_sr) / sr_std)) if sr_std > 0 else 0.0

        self.Debug("=" * 55)
        self.Debug("[Metrics] === END-OF-BACKTEST PERFORMANCE ===")
        self.Debug(f"[Metrics] Observations (days)  : {T}")
        self.Debug(f"[Metrics] Ann. Return           : {ann_return*100:+.2f}%")
        self.Debug(f"[Metrics] Ann. Volatility       : {ann_vol*100:.2f}%")
        self.Debug(f"[Metrics] Sharpe Ratio          : {sharpe:.4f}")
        self.Debug(f"[Metrics] Sortino Ratio         : {sortino:.4f}")
        self.Debug(f"[Metrics] Max Drawdown          : {max_dd*100:.2f}%")
        self.Debug(f"[Metrics] Calmar Ratio          : {calmar:.4f}")
        self.Debug(f"[Metrics] Skewness              : {gamma3:.4f}")
        self.Debug(f"[Metrics] Kurtosis (full)       : {gamma4:.4f}")
        self.Debug(f"[Metrics] Sharpe std error      : {sr_std:.4f}")
        self.Debug(f"[Metrics] PSR(SR>0)             : {psr_vs_zero*100:.3f}%")
        self.Debug(f"[Metrics] PSR(SR>SPY={self.SPY_SR_BENCHMARK}) : {psr_vs_spy*100:.3f}%  (matches LEAN)")
        self.Debug(f"[Metrics] DSR (N={N} trials)    : {dsr*100:.3f}%")
        self.Debug(f"[Metrics] E[max SR]             : {e_max_sr:.4f}")
        self.Debug(f"[Metrics] DSR interpretation    : "
                   f"{'Likely genuine' if dsr > 0.95 else 'Likely selection bias' if dsr < 0.5 else 'Ambiguous'}")
        self.Debug("=" * 55)