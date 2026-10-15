"""
Baselines.

Window-level selectors (obs -> action), run through the same env as the agent:
    Fixed-LRU / Fixed-LFU / Fixed-FIFO (/ Fixed-MRU in the 4-action setting)
    Random      - uniformly random policy each window
    FTL         - "follow the leader": pick the policy whose shadow cache had the
                  best hit rate in the previous window (a strong hand-written
                  adaptive heuristic using exactly the agent's information)
    Phase-Oracle- knows the true phase boundaries and uses, in each phase, the
                  fixed policy that is best for that phase in hindsight

Standalone algorithms (not window selectors):
    ARC         - Adaptive Replacement Cache (classical adaptive policy)
    Belady      - offline optimum (upper bound)
"""
from __future__ import annotations

import numpy as np

from .cache import arc_hits, belady_hits
from .env import CacheEnv, rollout
from .features import TRACE_FEATURES, PreparedTrace

_G = TRACE_FEATURES.index("ghost_lru")


def fixed(p: int):
    return lambda obs: p


def ftl_factory(n_actions: int = 3):
    def ftl(obs: np.ndarray) -> int:
        g = obs[_G:_G + n_actions]
        if g.sum() == 0:           # first window: no information yet
            return 0
        return int(np.argmax(g))
    return ftl


ftl = ftl_factory(3)


def phase_oracle(env: CacheEnv, pt: PreparedTrace):
    W = pt.window
    n = env.n_actions
    best = []
    for (s, e, _) in pt.trace.phases:
        best.append((s, e, int(pt.ghost_hits[:n, s:e].mean(axis=1).argmax())))

    def sel(obs):
        t = env.w * W + W // 2
        for s, e, b in best:
            if s <= t < e:
                return b
        return best[-1][2]
    return sel


def window_selector_names(n_actions: int = 3):
    fixed_names = ["Fixed-LRU", "Fixed-LFU", "Fixed-FIFO", "Fixed-MRU"][:n_actions]
    return fixed_names + ["Random", "FTL", "Phase-Oracle"]


def run_selector_baseline(name: str, env: CacheEnv, idx: int):
    pt = env.traces[idx]
    n = env.n_actions
    sel = {
        "Fixed-LRU": fixed(0), "Fixed-LFU": fixed(1), "Fixed-FIFO": fixed(2), "Fixed-MRU": fixed(3),
        "FTL": ftl_factory(n),
    }.get(name)
    if name == "Random":
        rng = np.random.default_rng(idx)
        sel = lambda obs: int(rng.integers(n))  # noqa: E731
    if name == "Phase-Oracle":
        sel = phase_oracle(env, pt)
    return rollout(env, idx, sel)


def windowed(hits: np.ndarray, W: int) -> np.ndarray:
    nw = len(hits) // W
    return hits[: nw * W].reshape(nw, W).mean(axis=1)


def run_standalone(name: str, pt: PreparedTrace) -> np.ndarray:
    keys = pt.trace.keys
    h = arc_hits(keys, pt.cache_size) if name == "ARC" else belady_hits(keys, pt.cache_size)
    return windowed(h, pt.window)
