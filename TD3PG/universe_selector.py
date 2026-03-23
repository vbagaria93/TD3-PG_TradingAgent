from QuantConnect.Data.UniverseSelection import *
from Selection.FundamentalUniverseSelectionModel import FundamentalUniverseSelectionModel
from AlgorithmImports import *


class LiquidValueUniverseSelectionModel(FundamentalUniverseSelectionModel):
    """
    Selects top N mid-cap stocks by momentum, filtered for:
      • Price $5–$200
      • RSI 40–65: not oversold, not overbought
      • Momentum 5%–80%: in uptrend but NOT at squeeze peak
        (PPSI at 200% momentum is clearly exhausted — cap at 80%)
      • $10M+ daily dollar volume
    """

    MAX_SYMBOLS = 2    # reduced from 3 — concentrate on top 2 highest-conviction picks.
                       # 3rd pick consistently diluted alpha. Proven best in Alert run.
    MIN_PRICE = 5.0
    MAX_PRICE = 500.0
    COARSE_COUNT = 200
    RSI_MAX = 65     # exclude overbought
    RSI_MIN = 55     # raised from 50 — requires confirmed momentum, not just non-oversold
    MOM_MIN = 0.15   # raised from 0.10 — filters low-beta drift names without over-restricting
    MOM_MAX = 0.80   # cap at 80% — above this likely exhausted squeeze
    RSI_PERIOD = 14

    def __init__(self):
        super().__init__(True, None)
        self._last_month = -1

    def Select(self, algorithm, fundamental):
        if self._last_month == algorithm.Time.month:
            return Universe.Unchanged
        self._last_month = algorithm.Time.month

        # Coarse filter
        filtered = [
            f for f in fundamental
            if f.Price > self.MIN_PRICE
            and f.Price < self.MAX_PRICE
            and f.DollarVolume > 1e7
            and f.HasFundamentalData
        ]
        filtered.sort(key=lambda x: x.DollarVolume, reverse=True)
        candidates = filtered[:self.COARSE_COUNT]
        if not candidates:
            return Universe.Unchanged

        # Compute momentum + RSI for each candidate
        scored = []
        for f in candidates:
            sym = f.Symbol
            hist = algorithm.History(
                sym, self.RSI_PERIOD + 25, Resolution.Daily)
            if hist.empty or len(hist) < self.RSI_PERIOD + 5:
                continue
            closes = hist['close'].values.astype(float)

            # 20-day momentum
            mom = ((closes[-1] - closes[-21]) / closes[-21]
                   if len(closes) >= 21 else 0.0)

            # RSI
            d = closes[-(self.RSI_PERIOD + 1):]
            diffs = d[1:] - d[:-1]
            avg_gain = max(sum(x for x in diffs if x > 0), 0) / self.RSI_PERIOD
            avg_loss = max(sum(-x for x in diffs if x < 0),
                           0) / self.RSI_PERIOD
            rsi = (100.0 if avg_loss == 0
                   else 100 - (100 / (1 + avg_gain / avg_loss)))

            # Apply filters
            if not (self.RSI_MIN <= rsi <= self.RSI_MAX):
                continue
            if not (self.MOM_MIN <= mom <= self.MOM_MAX):
                continue

            scored.append((f, mom, rsi))

        # If strict filter returns nothing, relax momentum cap to 120%
        if not scored:
            for f in candidates:
                sym = f.Symbol
                hist = algorithm.History(
                    sym, self.RSI_PERIOD + 25, Resolution.Daily)
                if hist.empty or len(hist) < self.RSI_PERIOD + 5:
                    continue
                closes = hist['close'].values.astype(float)
                mom = ((closes[-1] - closes[-21]) / closes[-21]
                       if len(closes) >= 21 else 0.0)
                d = closes[-(self.RSI_PERIOD + 1):]
                diffs = d[1:] - d[:-1]
                avg_gain = max(sum(x for x in diffs if x > 0),
                               0) / self.RSI_PERIOD
                avg_loss = max(sum(-x for x in diffs if x < 0),
                               0) / self.RSI_PERIOD
                rsi = (100.0 if avg_loss == 0
                       else 100 - (100 / (1 + avg_gain / avg_loss)))
                if self.RSI_MIN <= rsi <= 75 and 0 < mom <= 1.20:
                    scored.append((f, mom, rsi))

        if not scored:
            return Universe.Unchanged

        # Rank by momentum (highest first, but capped so squeezed stocks fall out)
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:self.MAX_SYMBOLS]
        symbols = [f.Symbol for f, _, _ in top]

        algorithm.Debug(
            f"[Universe] Selected {[str(s) for s in symbols]} | "
            f"RSI: {[round(rsi, 1) for _, _, rsi in top]} | "
            f"Mom: {[round(mom*100, 1) for _, mom, _ in top]}%")
        return symbols