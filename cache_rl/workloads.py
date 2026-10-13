"""
Synthetic block-access trace generators with *non-stationary* behaviour.

Each "regime" is designed so that a different classical eviction policy is the
best choice, which is exactly the situation in which a learned policy selector
should pay off:

    zipf      - stationary skewed popularity           -> LFU is best
    drift     - hot working set that keeps relocating  -> LRU is best, LFU fails
    scan      - small hot set + long one-shot scans    -> LFU protects hot set
    stack     - strong temporal locality (stack model) -> LRU / FIFO good
    loop      - cyclic sweep slightly larger than cache -> all recency policies thrash

A full trace is a concatenation of phases; phase types, lengths and key regions
are randomised, so the agent can never memorise *when* the workload changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

REGIMES = ("zipf", "drift", "scan", "stack", "loop")


@dataclass
class Trace:
    keys: np.ndarray                     # int64 array of requested block ids
    phases: List[Tuple[int, int, str]] = field(default_factory=list)  # (start, end, regime)
    name: str = "trace"

    def __len__(self) -> int:
        return len(self.keys)


# --------------------------------------------------------------------------- #
#  Individual regimes. Every generator returns keys in [0, universe) which the
#  caller then shifts into a fresh key region so phases do not share blocks.
# --------------------------------------------------------------------------- #
def _zipf_probs(n: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    p = ranks ** (-alpha)
    return p / p.sum()


def gen_zipf(rng, n, cache_size, alpha=None):
    universe = cache_size * 20
    alpha = alpha if alpha is not None else rng.uniform(0.9, 1.2)
    perm = rng.permutation(universe)
    ranks = rng.choice(universe, size=n, p=_zipf_probs(universe, alpha))
    return perm[ranks]


def gen_drift(rng, n, cache_size):
    """Hot set of ~0.8C blocks that jumps to a brand-new location every few
    hundred requests. Fresh blocks arrive with frequency 1, so an in-cache LFU
    keeps the stale-but-once-popular blocks and evicts the new hot ones."""
    ws = int(cache_size * rng.uniform(0.6, 0.9))
    shift_every = int(rng.integers(3, 7) * cache_size)
    out = np.empty(n, dtype=np.int64)
    base = 0
    p = _zipf_probs(ws, 0.6)
    for s in range(0, n, shift_every):
        e = min(n, s + shift_every)
        out[s:e] = base + rng.choice(ws, size=e - s, p=p)
        base += ws
    return out


def gen_scan(rng, n, cache_size):
    """Hot set (~0.6C) interleaved with long sequential scans of cold blocks.
    Scan blocks are never reused -> recency policies get polluted."""
    hot = int(cache_size * rng.uniform(0.5, 0.7))
    scan_frac = rng.uniform(0.5, 0.7)
    out = np.empty(n, dtype=np.int64)
    scan_ptr = hot
    i = 0
    while i < n:
        if rng.random() < scan_frac:
            run = int(rng.integers(cache_size // 2, 2 * cache_size))
            run = min(run, n - i)
            out[i:i + run] = np.arange(scan_ptr, scan_ptr + run)
            scan_ptr += run
        else:
            run = int(rng.integers(cache_size // 2, cache_size))
            run = min(run, n - i)
            out[i:i + run] = rng.integers(0, hot, size=run)
        i += run
    return out


def gen_stack(rng, n, cache_size):
    """LRU stack-distance model: with prob p re-reference the block at a
    geometrically distributed depth of the recency stack, else a new block."""
    p_reuse = rng.uniform(0.75, 0.9)
    mean_depth = cache_size * rng.uniform(0.2, 0.5)
    stack: List[int] = []
    nxt = 0
    out = np.empty(n, dtype=np.int64)
    depths = rng.geometric(1.0 / mean_depth, size=n) - 1
    coin = rng.random(n)
    for i in range(n):
        if stack and coin[i] < p_reuse:
            d = min(int(depths[i]), len(stack) - 1)
            k = stack.pop(-1 - d)
        else:
            k = nxt
            nxt += 1
        stack.append(k)
        if len(stack) > 4 * cache_size:
            stack.pop(0)
        out[i] = k
    return out


def gen_loop(rng, n, cache_size):
    """Cyclic loop over L > C blocks plus a small frequently used set.
    LRU/FIFO thrash on the loop, frequency keeps the small set resident."""
    L = int(cache_size * rng.uniform(1.1, 1.5))
    small = max(4, cache_size // 8)
    out = np.empty(n, dtype=np.int64)
    pos = 0
    coin = rng.random(n)
    for i in range(n):
        if coin[i] < 0.3:
            out[i] = L + rng.integers(0, small)
        else:
            out[i] = pos
            pos = (pos + 1) % L
    return out


GENERATORS = {
    "zipf": gen_zipf,
    "drift": gen_drift,
    "scan": gen_scan,
    "stack": gen_stack,
    "loop": gen_loop,
}


# --------------------------------------------------------------------------- #
#  Composite traces
# --------------------------------------------------------------------------- #
def make_trace(regimes: List[str], lengths: List[int], cache_size: int,
               seed: int, name: str = "trace") -> Trace:
    """Concatenate the given regimes; each phase lives in its own key region."""
    rng = np.random.default_rng(seed)
    parts, phases = [], []
    offset, t = 0, 0
    for reg, n in zip(regimes, lengths):
        k = GENERATORS[reg](rng, n, cache_size).astype(np.int64)
        parts.append(k + offset)
        offset += int(k.max()) + 1
        phases.append((t, t + n, reg))
        t += n
    return Trace(np.concatenate(parts), phases, name)


def random_trace(cache_size: int, seed: int, n_phases: int = 6,
                 min_len: int = 3000, max_len: int = 8000,
                 regimes=REGIMES) -> Trace:
    """Random sequence of phases (no two identical consecutive regimes)."""
    rng = np.random.default_rng(seed + 10_007)
    seq, prev = [], None
    for _ in range(n_phases):
        choices = [r for r in regimes if r != prev]
        prev = str(rng.choice(choices))
        seq.append(prev)
    lengths = [int(rng.integers(min_len, max_len)) for _ in seq]
    return make_trace(seq, lengths, cache_size, seed, name=f"random_{seed}")


SCENARIOS = {
    # hand-crafted traces used for the "behaviour after a workload change" study
    "zipf_scan_drift": (["zipf", "scan", "drift", "zipf"], [6000, 6000, 6000, 6000]),
    "drift_loop_stack": (["drift", "loop", "stack", "scan"], [6000, 6000, 6000, 6000]),
    "fast_switch": (["zipf", "drift", "zipf", "drift", "scan", "drift"],
                    [2500, 2500, 2500, 2500, 2500, 2500]),
}


def scenario_trace(name: str, cache_size: int, seed: int = 0) -> Trace:
    regs, lens = SCENARIOS[name]
    return make_trace(regs, lens, cache_size, seed, name=name)
