# TD3-PG_TradingAgent
Shared Twin-Delayed Deep Deterministic Policy Gradient Trading Agent

# TD3PG — Shared-Policy TD3 Equity Trading System

A reinforcement learning trading system built on [QuantConnect LEAN](https://www.quantconnect.com/lean), implementing a shared Twin Delayed Deep Deterministic Policy Gradient (TD3) agent that learns a single policy across a dynamically-selected universe of momentum equities.

---

## Overview

Most RL trading systems train one agent per symbol. This system trains **one shared agent across all symbols simultaneously**, using per-symbol learned embeddings to differentiate instruments while sharing actor-critic weights. This gives cross-symbol gradient diversity, more training signal per episode, and natural transfer learning when the universe rotates.

**Key design decisions:**
- Continuous action space: action ∈ [-1, +1], positive = long, negative/zero = flat (long-only)
- Asymmetric reward shaping: momentum bonus for holding profitable positions, discouraging premature exits
- Persistent replay buffer across all monthly retrains — accumulates 200k+ transitions over the full backtest
- Monthly universe reselection via RSI + momentum filters; warm-start retraining on universe changes

**Backtest window:** 2017–2020 (3-year in-sample, covers bull market, Q4 2018 correction, 2019 recovery)

**Best in-sample result:** +52.7% net profit, PSR 73%, Max Drawdown 17.4%

> ⚠️ All results are **in-sample**. The 2017–2020 window was used for iterative development. Out-of-sample validation on 2022–2024 is the intended next step.

---

## Architecture

```
TD3PG/
├── main.py                  # QCAlgorithm entry point — deployment logic, metrics
├── alpha_model.py           # RLAlphaModel — signal generation, monthly retraining
├── universe_selector.py     # Momentum + RSI universe filter
├── data_consolidator.py     # Daily bar consolidation for LEAN
├── td3_agent.py             # SharedTD3 — actor-critic with per-symbol embeddings
├── actor_critic.py          # Actor and Critic network definitions (PyTorch)
├── trainer.py               # SharedRunner — training loop, episode management
├── trading_env.py           # TradingEnv — gym-compatible environment, reward shaping
└── replay_buffer.py         # Persistent replay buffer with cross-symbol sampling
```

### Agent: SharedTD3

The actor takes `[observation, embedding(symbol_idx)]` as input and outputs a scalar action. The critic takes `[observation, action, embedding(symbol_idx)]`. Embeddings are learned end-to-end and grow dynamically as new symbols enter the universe.

```
Input:  obs (PCA-compressed indicators) + symbol_embedding(idx)
Actor:  Linear → ReLU → Linear → ReLU → Linear → tanh  →  action ∈ [-1, +1]
Critic: [obs, action, embedding] → Linear → ReLU → ... → Q-value (×2, clipped)
```

TD3-specific components: target policy smoothing, clipped double Q, delayed policy updates (`policy_freq=2`).

### Feature Pipeline

Each symbol's observation is a 3-component PCA projection of four technical indicators:

| Indicator | Period | Notes |
|-----------|--------|-------|
| MACD | 12/26/9 | Momentum signal |
| RSI | 14 | Overbought/oversold filter |
| ADX | 14 | Trend strength |
| CCI | 14 | Commodity channel index |

OBV was dropped — its unbounded cumulative values dominated PC1, making the other indicators statistically invisible. Log-returns are appended as a fifth feature before PCA.

### Reward Function

```python
if position == Long:
    reward = clip(bar_return, -5%, +5%)
    if unrealised_pnl > 0:
        reward += MOMENTUM_BONUS   # 0.0015/bar — run winners
else:
    reward -= HOLD_PENALTY         # 0.0005/bar — stay active
# On open/close: reward -= TRANSACTION_COST (0.001)
```

The momentum bonus asymmetry directly addresses the win/loss dollar asymmetry common in momentum strategies — training the agent to hold profitable positions rather than exiting early.

---

## Universe Selection

Monthly reselection using `LiquidValueUniverseSelectionModel`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `MAX_SYMBOLS` | 3 | LEAN 10-min compute budget |
| `RSI_MIN` | 50 | Exclude weakening trends (e.g. RSI=40 names showed −0.6 ep_r) |
| `RSI_MAX` | 65 | Exclude overbought |
| `MOM_MIN` | 10% | 10–80% trailing momentum |
| `MOM_MAX` | 80% | Cap exhausted squeezes |
| `MAX_PRICE` | $500 | Allow high-beta names |

Selected symbols are scored and ranked; the top 3 by combined RSI + momentum score proceed to training.

---

## Deployment Logic (`main.py`)

```
OnData() each bar:
  1. Tick down cooldowns
  2. Hard stop-loss check (12%) → liquidate + 5-day cooldown
  3. Signal exit: action < -0.05 → liquidate
  4. New entries: action ≥ 0.0, not invested, not attempted this period
     → SetHoldings(symbol, weight) where weight = min(1/n, 0.33)
```

Key safeguards:
- **T+1 cash settlement guard**: no new entries on any bar where an exit fires
- **`_attempted_buys` set**: prevents daily retry churn when orders fail due to buying power constraints; cleared on each universe change
- **Exit hysteresis** (`EXIT_THRESHOLD = -0.05`): prevents flip-flopping near the zero boundary

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes (cold start) | 30 |
| Episodes (warm-start) | 10 |
| Batch size | 64 |
| Discount (γ) | 0.99 |
| Tau (soft update) | 0.005 |
| Policy noise | 0.1 |
| Noise clip | 0.3 |
| Explore noise | 0.20 |
| Policy frequency | 2 |
| L2 action penalty | 0.0 (removed) |
| Replay buffer size | 1,000,000 |

**L2 penalty rationale:** Set to zero after observing systematic action collapse to `-epsilon`. TD3's target policy noise and tanh output activation provide sufficient regularisation without L2. When L2 > 0, the gravity well at zero competed with the momentum bonus, causing the policy to collapse to flat whenever early training episodes produced negative Q-signal.

---

## Setup

### Requirements

```
quantconnect-lean>=2.5
torch>=1.13
numpy
pandas
scikit-learn
scipy
gym
```

### QuantConnect Cloud

1. Create a new project in [QuantConnect Cloud](https://www.quantconnect.com)
2. Upload all files in `TD3PG/` to the project directory
3. Set `main.py` as the algorithm entry point
4. Set backtest dates: 2017-01-01 to 2020-01-01
5. Starting cash: $100,000
6. Brokerage: Interactive Brokers (Cash account)

### Local LEAN (CLI)

```bash
# Install LEAN CLI
pip install lean

# Initialise workspace
lean init

# Create project
lean project-create "TD3PG"

# Copy files
cp TD3PG/* <workspace>/TD3PG/

# Run backtest
lean backtest "TD3PG"
```

### File Structure in LEAN Project

```
<project_root>/
├── main.py              # Algorithm entry point (import from TD3PG.*)
└── TD3PG/
    ├── __init__.py      # Empty — marks TD3PG as a Python package
    ├── main.py          # (contents placed in project root main.py)
    ├── alpha_model.py
    ├── universe_selector.py
    ├── data_consolidator.py
    ├── td3_agent.py
    ├── actor_critic.py
    ├── trainer.py
    ├── trading_env.py
    └── replay_buffer.py
```

## Limitations and Future Work

- **In-sample only**: all development was on 2017–2020. Out-of-sample (2022–2024) validation pending.
- **Long-only**: no short positions; strategy is directionally biased and underperforms in sustained bear markets.
- **Training stochasticity**: Sharpe variance of ~0.3 between identical runs due to random TD3 weight initialisation. Multiple seeds + ensemble averaging would reduce this.
- **Kurtosis**: daily return kurtosis ~9–12 from 33% single-position concentration inflates Sharpe standard error and suppresses PSR. Position sizing with Kelly or volatility targeting could address this.
- **DSR**: with 21+ backtest iterations on the same window, the deflated Sharpe Ratio is near zero. A clean out-of-sample run is the only path to meaningful DSR.

---

## References

- Fujimoto et al. (2018) — [Addressing Function Approximation Error in Actor-Critic Methods (TD3)](https://arxiv.org/abs/1802.09477)
- Bailey & López de Prado (2012) — [The Sharpe Ratio Efficient Frontier](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643)
- QuantConnect LEAN — [https://github.com/QuantConnect/Lean](https://github.com/QuantConnect/Lean)
