"""Unit tests verifying all components of the cache_rl package."""
import os
import unittest
import numpy as np
import torch

from cache_rl.workloads import make_trace, REGIMES, random_trace
from cache_rl.cache import MultiPolicyCache, fixed_policy_hits, arc_hits, belady_hits
from cache_rl.features import prepare_trace, OBS_DIM
from cache_rl.env import CacheEnv, rollout
from cache_rl.agents import TabularQAgent, DQNAgent, DQNConfig
from cache_rl.baselines import ftl_factory, window_selector_names, run_selector_baseline


class TestWorkloads(unittest.TestCase):
    def test_regimes(self):
        for reg in REGIMES:
            tr = make_trace([reg], [500], cache_size=50, seed=42)
            self.assertEqual(len(tr.keys), 500)
            self.assertEqual(len(tr.phases), 1)

    def test_random_trace(self):
        tr = random_trace(cache_size=50, seed=123, n_phases=4, min_len=200, max_len=400)
        self.assertGreaterEqual(len(tr.keys), 800)
        self.assertEqual(len(tr.phases), 4)


class TestCacheSimulator(unittest.TestCase):
    def test_multipolicy_hits(self):
        keys = np.array([1, 2, 3, 1, 2, 4, 1, 5, 2, 3, 4], dtype=np.int64)
        c = MultiPolicyCache(capacity=3)
        hits = c.run(keys, policy=0)  # LRU
        self.assertGreater(hits, 0)
        self.assertLessEqual(hits, len(keys))

    def test_arc_hits(self):
        keys = np.array([1, 2, 3, 1, 2, 4, 1, 5, 2, 3, 4], dtype=np.int64)
        h = arc_hits(keys, capacity=3)
        self.assertEqual(len(h), len(keys))

    def test_belady_hits(self):
        keys = np.array([1, 2, 3, 1, 2, 4, 1, 5, 2, 3, 4], dtype=np.int64)
        h = belady_hits(keys, capacity=3)
        self.assertEqual(len(h), len(keys))
        # Belady must be >= any heuristic
        h_lru = fixed_policy_hits(keys, 3, 0)
        self.assertGreaterEqual(h.sum(), h_lru.sum())


class TestFeaturesAndEnv(unittest.TestCase):
    def setUp(self):
        self.tr = random_trace(cache_size=30, seed=7, n_phases=3, min_len=300, max_len=500)
        self.pt = prepare_trace(self.tr, cache_size=30, window=50)

    def test_feature_dimensions(self):
        from cache_rl.features import TRACE_FEATURES
        self.assertEqual(self.pt.feats.shape[1], len(TRACE_FEATURES))
        self.assertEqual(self.pt.ghost_hits.shape[0], 4)

    def test_env_3_actions(self):
        env = CacheEnv([self.pt], reward="relative", n_actions=3)
        self.assertEqual(env.obs_dim, OBS_DIM)
        obs = env.reset(0)
        self.assertEqual(len(obs), OBS_DIM)
        obs2, r, done, info = env.step(1)
        self.assertEqual(len(obs2), OBS_DIM)
        self.assertIn("hit_rate", info)

    def test_env_4_actions(self):
        env = CacheEnv([self.pt], reward="relative", n_actions=4)
        obs = env.reset(0)
        obs2, r, done, info = env.step(3)  # MRU
        self.assertEqual(len(obs2), OBS_DIM)


class TestAgents(unittest.TestCase):
    def setUp(self):
        self.tr = random_trace(cache_size=30, seed=9, n_phases=3, min_len=300, max_len=500)
        self.pt = prepare_trace(self.tr, cache_size=30, window=50)
        self.env = CacheEnv([self.pt], n_actions=3)

    def test_tabular_agent(self):
        ag = TabularQAgent(n_actions=3, seed=0)
        obs = self.env.reset(0)
        a = ag.act(obs)
        self.assertIn(a, [0, 1, 2])
        obs2, r, done, _ = self.env.step(a)
        ag.observe(obs, a, r, obs2, done)
        self.assertEqual(ag.steps, 1)

    def test_dqn_agent(self):
        ag = DQNAgent(self.env.obs_dim, 3, DQNConfig(batch_size=4, warmup=5, hidden=32), seed=0)
        obs = self.env.reset(0)
        a = ag.act(obs)
        self.assertIn(a, [0, 1, 2])
        obs2, r, done, _ = self.env.step(a)
        loss = ag.observe(obs, a, r, obs2, done)
        self.assertEqual(ag.steps, 1)

    def test_model_loading(self):
        if os.path.exists("models/qlearn_a3.npy"):
            ag_q = TabularQAgent.load("models/qlearn_a3.npy")
            self.assertEqual(ag_q.n_actions, 3)
        if os.path.exists("models/dqn_a3.pt"):
            ag_dqn = DQNAgent.load("models/dqn_a3.pt", OBS_DIM, 3)
            self.assertEqual(ag_dqn.n_actions, 3)


if __name__ == "__main__":
    unittest.main()
