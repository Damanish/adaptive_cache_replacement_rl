"""
Gym-style environment: one episode = one (multi-phase) trace.

    obs_t   = [trace statistics of window t-1, own hit-rate stats, cache stats, prev action]
    a_t     in {0: LRU, 1: LFU, 2: FIFO [, 3: MRU]} -> victim rule for all misses in window t
    r_t     = (1/W) * sum_{i in window t} ( +1[hit_i] - 1[miss_i] )      ("hitmiss")
            or latency based:  -AMAT_t / L_miss                         ("latency")
            or "relative":  hitmiss_t - hitmiss_t(shadow LRU cache)
              The shadow term depends only on the request stream (never on
              the agent's actions), so the optimal policy is unchanged, but
              it removes the large phase-driven swings in hit rate from the
              TD targets (variance reduction).
    done    when the trace is exhausted
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .cache import MultiPolicyCache
from .features import OBS_DIM, TRACE_FEATURES, PreparedTrace

N_PREV = 4   # prev-action one-hot always has 4 slots so obs layout is fixed


class CacheEnv:
    def __init__(self, traces: List[PreparedTrace], reward: str = "hitmiss",
                 hit_latency: float = 1.0, miss_latency: float = 100.0,
                 switch_cost: float = 0.0, ema: float = 0.8, seed: int = 0,
                 n_actions: int = 3):
        assert traces, "need at least one prepared trace"
        assert n_actions in (3, 4)
        self.traces = traces
        self.reward_type = reward
        self.hl, self.ml = hit_latency, miss_latency
        self.switch_cost = switch_cost
        self.ema = ema
        self.rng = np.random.default_rng(seed)
        self.obs_dim, self.n_actions = OBS_DIM, n_actions
        self.cache: Optional[MultiPolicyCache] = None

    # ------------------------------------------------------------------ #
    def reset(self, idx: Optional[int] = None) -> np.ndarray:
        if idx is None:
            idx = int(self.rng.integers(len(self.traces)))
        self.pt = self.traces[idx]
        if self.cache is None or self.cache.C != self.pt.cache_size:
            self.cache = MultiPolicyCache(self.pt.cache_size)
        self.cache.reset()
        self.w = 0
        self.own_hr = 0.0
        self.own_ema = 0.0
        self.prev = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        tf = self.pt.feats[self.w - 1] if self.w > 0 else np.zeros(len(TRACE_FEATURES), np.float32)
        onehot = np.zeros(N_PREV, np.float32)
        onehot[self.prev] = 1.0
        return np.concatenate([tf, [self.own_hr, self.own_ema, self.cache.frac_freq_one()], onehot]
                              ).astype(np.float32)

    def step(self, action: int):
        W = self.pt.window
        s = self.w * W
        keys = self.pt.trace.keys[s:s + W]
        hits = self.cache.run(keys, int(action))
        h = hits / W
        if self.reward_type == "latency":
            r = -(h * self.hl + (1 - h) * self.ml) / self.ml
        elif self.reward_type == "relative":
            r = 2.0 * (h - self.pt.feats[self.w, 0])     # feats[w,0] = shadow-LRU hit rate of window w
        else:
            r = 2.0 * h - 1.0
        if action != self.prev:
            r -= self.switch_cost
        self.own_hr = h
        self.own_ema = h if self.w == 0 else self.ema * self.own_ema + (1 - self.ema) * h
        self.prev = int(action)
        self.w += 1
        done = self.w >= self.pt.n_windows
        return self._obs(), float(r), done, {"hits": hits, "hit_rate": h}


# ---------------------------------------------------------------------- #
#  Run any obs->action selector over a prepared trace (used for baselines
#  and for evaluating trained agents).
# ---------------------------------------------------------------------- #
def rollout(env: CacheEnv, idx: int, selector: Callable[[np.ndarray], int]):
    obs = env.reset(idx)
    actions, rates = [], []
    done = False
    while not done:
        a = selector(obs)
        obs, _, done, info = env.step(a)
        actions.append(a)
        rates.append(info["hit_rate"])
    return np.array(actions), np.array(rates)
