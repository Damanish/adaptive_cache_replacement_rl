"""Double DQN agent (written from scratch in PyTorch)."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.body(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.cap = capacity
        self.o = np.zeros((capacity, obs_dim), np.float32)
        self.o2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.i = 0
        self.full = False

    def add(self, o, a, r, o2, d):
        self.o[self.i], self.a[self.i], self.r[self.i] = o, a, r
        self.o2[self.i], self.d[self.i] = o2, d
        self.i = (self.i + 1) % self.cap
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, n: int):
        idx = np.random.randint(0, len(self), size=n)
        return (torch.from_numpy(self.o[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.o2[idx]),
                torch.from_numpy(self.d[idx]))


@dataclass
class DQNConfig:
    gamma: float = 0.9
    lr: float = 5e-4
    batch_size: int = 128
    buffer_size: int = 50_000
    warmup: int = 1_000
    tau: float = 0.01             # soft target update
    eps_start: float = 1.0
    eps_end: float = 0.03
    eps_decay_steps: int = 30_000
    hidden: int = 128
    reward_scale: float = 10.0    # window rewards are small; scale for stable TD targets


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig = DQNConfig(), seed: int = 0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.cfg = cfg
        self.n_actions = n_actions
        self.q = QNet(obs_dim, n_actions, cfg.hidden)
        self.q_tgt = QNet(obs_dim, n_actions, cfg.hidden)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.buffer_size, obs_dim)
        self.steps = 0

    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.steps / c.eps_decay_steps)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        return int(self.q(torch.from_numpy(obs).unsqueeze(0)).argmax(1).item())

    def observe(self, o, a, r, o2, d):
        self.buf.add(o, a, r * self.cfg.reward_scale, o2, d)
        self.steps += 1
        if len(self.buf) >= self.cfg.warmup:
            return self._learn()
        return None

    def _learn(self) -> float:
        c = self.cfg
        o, a, r, o2, d = self.buf.sample(c.batch_size)
        q = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a2 = self.q(o2).argmax(1, keepdim=True)              # Double DQN: online selects,
            q2 = self.q_tgt(o2).gather(1, a2).squeeze(1)         # target evaluates
            target = r + c.gamma * (1 - d) * q2
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_tgt.parameters()):
                pt.mul_(1 - c.tau).add_(c.tau * p)
        return float(loss.item())

    def save(self, path: str):
        torch.save({"state_dict": self.q.state_dict(), "cfg": self.cfg.__dict__}, path)

    @classmethod
    def load(cls, path: str, obs_dim: int, n_actions: int) -> "DQNAgent":
        ck = torch.load(path, map_location="cpu")
        agent = cls(obs_dim, n_actions, DQNConfig(**ck["cfg"]))
        agent.q.load_state_dict(ck["state_dict"])
        agent.q.eval()
        return agent
