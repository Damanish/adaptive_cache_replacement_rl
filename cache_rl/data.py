"""Trace-set construction shared by training and evaluation scripts."""
from __future__ import annotations

import os
import pickle
from typing import Iterable, List

from .features import PreparedTrace, prepare_trace
from .workloads import random_trace, scenario_trace

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")
FEATURE_VERSION = "v2"   # bump whenever features.py / workloads.py change

TRAIN_SEEDS = range(0, 64)
VAL_SEEDS = range(500, 508)
TEST_SEEDS = range(1000, 1020)


def build_random_set(seeds: Iterable[int], cache_size: int, window: int,
                     n_phases: int = 6, use_cache: bool = True) -> List[PreparedTrace]:
    seeds = list(seeds)
    tag = f"{FEATURE_VERSION}_rand_C{cache_size}_W{window}_P{n_phases}_{seeds[0]}-{seeds[-1]}_{len(seeds)}.pkl"
    path = os.path.join(CACHE_DIR, tag)
    if use_cache and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    out = [prepare_trace(random_trace(cache_size, s, n_phases), cache_size, window) for s in seeds]
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(out, f)
    return out


def build_scenarios(names: Iterable[str], cache_size: int, window: int, seed: int = 0):
    return [prepare_trace(scenario_trace(n, cache_size, seed), cache_size, window) for n in names]
