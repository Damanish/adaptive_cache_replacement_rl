"""
Train an RL policy-selector for adaptive cache replacement.

Examples
--------
    python train.py --algo dqn     --episodes 300
    python train.py --algo qlearn  --episodes 300
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch

from cache_rl.agents import DQNAgent, DQNConfig, TabularQAgent
from cache_rl.baselines import ftl_factory
from cache_rl.data import TRAIN_SEEDS, VAL_SEEDS, build_random_set
from cache_rl.env import CacheEnv, rollout

torch.set_num_threads(2)   # tiny MLP: more threads only add overhead


def evaluate(agent, env: CacheEnv) -> float:
    return float(np.mean([rollout(env, i, lambda o: agent.act(o, greedy=True))[1].mean()
                          for i in range(len(env.traces))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "qlearn"], default="dqn")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--cache-size", type=int, default=100)
    ap.add_argument("--window", type=int, default=100, help="requests per decision")
    ap.add_argument("--reward", choices=["hitmiss", "latency", "relative"], default="relative")
    ap.add_argument("--actions", type=int, choices=[3, 4], default=3,
                    help="3 = {LRU,LFU,FIFO} (proposed), 4 = + MRU (extension)")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models")
    ap.add_argument("--log", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.log, exist_ok=True)

    t0 = time.time()
    train = build_random_set(TRAIN_SEEDS, args.cache_size, args.window)
    val = build_random_set(VAL_SEEDS, args.cache_size, args.window)
    print(f"prepared {len(train)} train / {len(val)} val traces in {time.time()-t0:.1f}s")

    env = CacheEnv(train, reward=args.reward, seed=args.seed, n_actions=args.actions)
    val_env = CacheEnv(val, reward=args.reward, n_actions=args.actions)
    steps_per_ep = int(np.mean([pt.n_windows for pt in train]))
    decay = int(0.6 * args.episodes * steps_per_ep)
    tag = f"{args.algo}_a{args.actions}"

    if args.algo == "dqn":
        agent = DQNAgent(env.obs_dim, env.n_actions, DQNConfig(eps_decay_steps=decay), seed=args.seed)
        ckpt = os.path.join(args.out, f"{tag}.pt")
    else:
        agent = TabularQAgent(n_actions=args.actions, eps_decay_steps=decay, seed=args.seed)
        ckpt = os.path.join(args.out, f"{tag}.npy")

    ftl = ftl_factory(args.actions)
    ftl_val = float(np.mean([rollout(val_env, i, ftl)[1].mean() for i in range(len(val))]))
    print(f"FTL validation hit-rate (reference): {ftl_val:.4f}")

    log_path = os.path.join(args.log, f"train_log_{tag}.csv")
    best = -1.0
    with open(log_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["episode", "steps", "epsilon", "train_hit_rate", "return", "val_hit_rate"])
        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            done, ret, hits = False, 0.0, []
            while not done:
                a = agent.act(obs)
                obs2, r, done, info = env.step(a)
                agent.observe(obs, a, r, obs2, float(done))
                obs = obs2
                ret += r
                hits.append(info["hit_rate"])
            val_hr = ""
            if ep % args.eval_every == 0 or ep == args.episodes:
                val_hr = evaluate(agent, val_env)
                if val_hr > best:
                    best = val_hr
                    agent.save(ckpt)
                print(f"ep {ep:4d} | eps {agent.epsilon():.3f} | train hr {np.mean(hits):.4f} "
                      f"| val hr {val_hr:.4f} (best {best:.4f}) | {time.time()-t0:.0f}s", flush=True)
            wr.writerow([ep, agent.steps, round(agent.epsilon(), 4), round(float(np.mean(hits)), 5),
                         round(ret, 4), val_hr if val_hr == "" else round(val_hr, 5)])
    print(f"saved best checkpoint ({best:.4f}) -> {ckpt}")


if __name__ == "__main__":
    main()
