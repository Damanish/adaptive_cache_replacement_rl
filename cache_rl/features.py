"""
Observation features.

The trace is split into decision windows of W requests. At the start of window
t the agent observes statistics computed **only from windows < t** (strictly
causal, no look-ahead) and picks the eviction policy used for every miss in
window t.

Agent-independent statistics (depend only on the request stream) are
pre-computed once per trace for speed; agent-dependent ones (own hit rate,
cache-content statistics, previous action) are added online by the env.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import FIFO, LFU, LRU, fixed_policy_hits
from .workloads import Trace

TRACE_FEATURES = (
    "ghost_lru", "ghost_lfu", "ghost_fifo",            # shadow-cache hit rates (last window)
    "ghost_lru_ema", "ghost_lfu_ema", "ghost_fifo_ema",  # same, long-horizon EMA
    "unique_ratio",                                     # distinct keys / W
    "seq_ratio",                                        # P(key_t = key_{t-1}+1)
    "new_ratio",                                        # first-ever references / W
    "reuse_short",                                      # reuse interval <= C
    "reuse_mid",                                        # C < reuse interval <= 4C
    "entropy",                                          # normalised popularity entropy
    "jaccard",                                          # key-set overlap with previous window
)
AGENT_FEATURES = ("own_hr", "own_hr_ema", "frac_freq1", "prev_LRU", "prev_LFU", "prev_FIFO")
OBS_DIM = len(TRACE_FEATURES) + len(AGENT_FEATURES)


@dataclass
class PreparedTrace:
    trace: Trace
    window: int
    cache_size: int
    feats: np.ndarray        # (n_windows, len(TRACE_FEATURES)) stats *of* window w
    ghost_hits: np.ndarray   # (3, n) per-request hits of fixed LRU/LFU/FIFO

    @property
    def n_windows(self) -> int:
        return len(self.trace.keys) // self.window


def prepare_trace(trace: Trace, cache_size: int, window: int, ema: float = 0.8) -> PreparedTrace:
    keys = trace.keys
    n = len(keys)
    nw = n // window
    C = cache_size
    ghosts = np.stack([fixed_policy_hits(keys, C, p) for p in (LRU, LFU, FIFO)])

    # reuse interval (time since previous reference to same key)
    last_seen = {}
    gap = np.full(n, -1, dtype=np.int64)
    for i, k in enumerate(keys.tolist()):
        j = last_seen.get(k)
        if j is not None:
            gap[i] = i - j
        last_seen[k] = i

    F = np.zeros((nw, len(TRACE_FEATURES)), dtype=np.float32)
    g_ema = None
    prev_set = None
    for w in range(nw):
        s, e = w * window, (w + 1) * window
        kw = keys[s:e]
        g = ghosts[:, s:e].mean(axis=1)
        g_ema = g if g_ema is None else ema * g_ema + (1 - ema) * g
        uniq, counts = np.unique(kw, return_counts=True)
        p = counts / counts.sum()
        ent = float(-(p * np.log(p)).sum() / np.log(window))
        gw = gap[s:e]
        cur_set = set(uniq.tolist())
        jac = 0.0 if prev_set is None else len(cur_set & prev_set) / max(1, len(cur_set | prev_set))
        prev_set = cur_set
        F[w] = [
            *g, *g_ema,
            len(uniq) / window,
            float((np.diff(kw) == 1).mean()),
            float((gw < 0).mean()),
            float(((gw > 0) & (gw <= C)).mean()),
            float(((gw > C) & (gw <= 4 * C)).mean()),
            ent,
            jac,
        ]
    return PreparedTrace(trace, window, cache_size, F, ghosts)
