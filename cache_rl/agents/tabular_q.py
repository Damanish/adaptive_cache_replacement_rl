"""Tabular Q-learning over a hand-discretised version of the observation."""
from __future__ import annotations

import numpy as np

from ..features import TRACE_FEATURES

_G = TRACE_FEATURES.index("ghost_lru")
_J = TRACE_FEATURES.index("jaccard")
_OWN = len(TRACE_FEATURES)            # own_hr index
_PREV = len(TRACE_FEATURES) + 3       # start of prev-action one-hot (4 slots)

MARGIN_BINS = (0.02, 0.08)            # gap between best and 2nd-best shadow cache
HR_BINS = (0.2, 0.45, 0.7)            # own hit-rate buckets
JAC_BIN = 0.3                         # low overlap with previous window => change


def n_states(n_actions: int) -> int:
    # best shadow policy x margin bin x own-hit-rate bin x previous action x change flag
    return n_actions * 3 * 4 * n_actions * 2


def discretise(obs: np.ndarray, n_actions: int) -> int:
    g = obs[_G:_G + n_actions]
    order = np.argsort(-g)
    best = int(order[0])
    margin = int(np.digitize(g[order[0]] - g[order[1]], MARGIN_BINS))
    hr = int(np.digitize(obs[_OWN], HR_BINS))
    prev = int(np.argmax(obs[_PREV:_PREV + n_actions]))
    change = int(obs[_J] < JAC_BIN)
    return ((((best * 3 + margin) * 4 + hr) * n_actions + prev) * 2) + change


class TabularQAgent:
    def __init__(self, n_actions: int = 3, alpha: float = 0.1, gamma: float = 0.9,
                 eps_start: float = 1.0, eps_end: float = 0.03, eps_decay_steps: int = 30_000,
                 reward_scale: float = 10.0, seed: int = 0):
        self.Q = np.zeros((n_states(n_actions), n_actions))
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_steps
        self.reward_scale = reward_scale
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def epsilon(self) -> float:
        f = min(1.0, self.steps / self.eps_decay)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_actions))
        q = self.Q[discretise(obs, self.n_actions)]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def observe(self, o, a, r, o2, d):
        s, s2 = discretise(o, self.n_actions), discretise(o2, self.n_actions)
        target = r * self.reward_scale + (0.0 if d else self.gamma * self.Q[s2].max())
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        self.steps += 1

    def save(self, path: str):
        np.save(path, self.Q)

    @classmethod
    def load(cls, path: str) -> "TabularQAgent":
        Q = np.load(path)
        a = cls(n_actions=Q.shape[1])
        a.Q = Q
        return a
