import numpy as np
from AlgorithmImports import *
from TD3PG.replay_buffer import ReplayBuffer


class SharedRunner:
    """
    Trains ONE shared agent across ALL symbol environments simultaneously.

    Each episode does a full round-robin pass over all symbols:
      ep1: STOCK_A full episode → STOCK_B full episode → STOCK_C ...
      ep2: repeat

    All transitions go into one shared replay buffer so every gradient
    step samples experience from multiple stocks — giving 3× the diversity
    per update vs per-symbol training.

    Best model saved on mean eval_r across ALL test envs.
    Warmup episodes skipped from best-model tracking to avoid saving
    the random-init policy that happens to score well on ep1.
    """

    WARMUP_EPISODES = 1  # reduced from 3 — was discarding training budget

    def __init__(self, algorithm, agent, n_episodes=20, batch_size=64,
                 gamma=0.99, tau=0.005, noise=0.1, noise_clip=0.3,
                 explore_noise=0.05, policy_frequency=2):
        self.algorithm = algorithm
        self.agent = agent
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.noise_clip = noise_clip
        self.explore_noise = explore_noise
        self.policy_frequency = policy_frequency
        self.replay_buffer = agent.replay_buffer

    def _eval_all(self, test_envs):
        total = 0.0
        for _, test_env, sym_idx in test_envs:
            obs = test_env.reset()
            done = False
            ep_r = 0.0
            while not done:
                act = self.agent.select_action(obs, sym_idx, noise=0)
                obs, r, done, _ = test_env.step(act)
                ep_r += r
            total += ep_r
        return total / max(len(test_envs), 1)

    def train(self, sym_train_envs, sym_test_envs):
        """
        sym_train_envs / sym_test_envs: list of (sym_name, env, sym_idx)
        """
        D = self.algorithm.Debug
        best_reward = -np.inf
        n_syms = len(sym_train_envs)

        D(f"[SharedRunner] symbols={n_syms} episodes={self.n_episodes} "
          f"batch={self.batch_size} explore={self.explore_noise} "
          f"warmup_eps={self.WARMUP_EPISODES}")
        for sn, env, si in sym_train_envs:
            D(f"[SharedRunner]   {sn} idx={si} "
              f"train {env._start_tick}→{env._end_tick}")

        for ep in range(1, self.n_episodes + 1):
            ep_stats = []

            for sym_name, train_env, sym_idx in sym_train_envs:
                obs = train_env.reset()
                done = False
                ep_r = 0.0
                a_buf = []

                while not done:
                    act = self.agent.select_action(
                        np.array(obs), sym_idx, noise=self.explore_noise)
                    new_obs, reward, done, _ = train_env.step(act)
                    self.replay_buffer.add(
                        (obs, new_obs, act, reward, done, sym_idx))
                    obs = new_obs
                    ep_r += reward
                    a_buf.append(float(act[0]))

                    self.agent.train(
                        self.replay_buffer, iterations=2,
                        batch_size=self.batch_size, discount=self.gamma,
                        tau=self.tau, policy_noise=self.noise,
                        noise_clip=self.noise_clip,
                        policy_freq=self.policy_frequency)

                a = np.array(a_buf)
                ep_stats.append((sym_name, ep_r, a.mean(), a.std()))

            stats_str = "  ".join(
                f"{sn}:ep_r={er:.2f}|mean={am:.2f}|std={as_:.2f}"
                for sn, er, am, as_ in ep_stats)
            D(f"[Ep {ep:02d}] buf={self.replay_buffer.cntr}  {stats_str}")

            mean_eval = self._eval_all(sym_test_envs)
            D(f"[Ep {ep:02d}] mean_eval_r={mean_eval:.3f}")

            if ep <= self.WARMUP_EPISODES:
                D(f"[Ep {ep:02d}] warmup — not tracking best yet")
                continue

            if mean_eval > best_reward:
                best_reward = mean_eval
                self.agent.save("shared_best")
                D(f"[Ep {ep:02d}] ★ best={best_reward:.3f} saved")

        if best_reward == -np.inf:
            self.agent.save("shared_best")
            D("[SharedRunner] No post-warmup improvement — saving final policy")
        else:
            D(f"[SharedRunner] Done. best_reward={best_reward:.3f}")

        self.agent.is_trained = True