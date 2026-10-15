"""
Evaluate trained agents against all baselines on held-out traces.

    python evaluate.py                      # default: C=100, W=100, 3 actions, 20 test traces
    python evaluate.py --actions 4          # extended action set (+MRU)
    python evaluate.py --cache-sizes 50 100 200   # generalisation across cache sizes

Writes CSVs to results/ (consumed by plots.py).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from cache_rl.agents import DQNAgent, TabularQAgent
from cache_rl.baselines import run_selector_baseline, run_standalone, window_selector_names
from cache_rl.data import TEST_SEEDS, build_random_set, build_scenarios
from cache_rl.env import CacheEnv, rollout
from cache_rl.workloads import SCENARIOS

HIT_LAT, MISS_LAT = 1.0, 100.0          # latency model (arbitrary units, e.g. us)
POST_CHANGE_WINDOWS = 10


def amat(h):
    return h * HIT_LAT + (1 - h) * MISS_LAT


def load_agents(model_dir, obs_dim, n_actions):
    agents = {}
    p = os.path.join(model_dir, f"qlearn_a{n_actions}.npy")
    if os.path.exists(p):
        agents["Q-Learning"] = TabularQAgent.load(p)
    p = os.path.join(model_dir, f"dqn_a{n_actions}.pt")
    if os.path.exists(p):
        agents["DQN"] = DQNAgent.load(p, obs_dim, n_actions)
    return agents


def run_all(traces, agents, n_actions):
    """Returns {method: [ (window_hit_rates, actions_or_None) per trace ]}."""
    env = CacheEnv(traces, n_actions=n_actions)
    out = {}
    for name in window_selector_names(n_actions):
        out[name] = [run_selector_baseline(name, env, i)[::-1] for i in range(len(traces))]
    for name in ["ARC", "Belady"]:
        out[name] = [(run_standalone(name, pt), None) for pt in traces]
    for name, ag in agents.items():
        out[name] = [rollout(env, i, lambda o: ag.act(o, greedy=True))[::-1]
                     for i in range(len(traces))]
    return out


def window_regimes(pt):
    W = pt.window
    labels = []
    for w in range(pt.n_windows):
        mid = w * W + W // 2
        labels.append(next((r for s, e, r in pt.trace.phases if s <= mid < e), pt.trace.phases[-1][2]))
    return np.array(labels)


def summarise(traces, res, cache_size):
    rows, reg_rows, pc_rows, act_rows = [], [], [], []
    for m, runs in res.items():
        hr = np.array([r[0].mean() for r in runs])
        pcs = []
        curve = np.zeros(POST_CHANGE_WINDOWS)
        cnt = 0
        for pt, (rates, acts) in zip(traces, runs):
            regs = window_regimes(pt)
            for reg in np.unique(regs):
                reg_rows.append({"method": m, "regime": reg, "cache_size": cache_size,
                                 "hit_rate": rates[regs == reg].mean()})
            for (s, _, _) in pt.trace.phases[1:]:
                w0 = s // pt.window + 1
                seg = rates[w0:w0 + POST_CHANGE_WINDOWS]
                if len(seg) == POST_CHANGE_WINDOWS:
                    pcs.append(seg.mean())
                    curve += seg
                    cnt += 1
            if acts is not None and m in ("DQN", "Q-Learning", "FTL"):
                for reg in np.unique(regs):
                    a = acts[regs == reg]
                    act_rows.append({"method": m, "regime": reg, "cache_size": cache_size,
                                     **{p: float((a == i).mean()) for i, p in enumerate(["LRU", "LFU", "FIFO", "MRU"])}})
        rows.append({"method": m, "cache_size": cache_size,
                     "hit_rate": hr.mean(), "hit_rate_std": hr.std(),
                     "amat": amat(hr).mean(), "post_change_hit_rate": float(np.mean(pcs))})
        for k in range(POST_CHANGE_WINDOWS):
            pc_rows.append({"method": m, "cache_size": cache_size, "window_after_change": k + 1,
                            "hit_rate": curve[k] / max(cnt, 1)})
    return (pd.DataFrame(rows), pd.DataFrame(reg_rows).groupby(["method", "regime", "cache_size"],
            as_index=False).mean(), pd.DataFrame(pc_rows), pd.DataFrame(act_rows).groupby(
            ["method", "regime", "cache_size"], as_index=False).mean() if act_rows else pd.DataFrame())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-sizes", type=int, nargs="+", default=[100])
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--actions", type=int, choices=[3, 4], default=3)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n = args.actions

    from cache_rl.features import OBS_DIM
    agents = load_agents(args.models, OBS_DIM, n)
    print("loaded agents:", list(agents))

    summ, regs, pcs, acts = [], [], [], []
    for C in args.cache_sizes:
        t0 = time.time()
        test = build_random_set(TEST_SEEDS, C, args.window)
        res = run_all(test, agents, n)
        s, r, p, a = summarise(test, res, C)
        summ.append(s), regs.append(r), pcs.append(p), acts.append(a)
        print(f"\n=== {n} actions | cache size {C}  ({len(test)} test traces, {time.time()-t0:.0f}s) ===")
        print(s.sort_values("hit_rate", ascending=False).to_string(index=False, float_format="%.4f"))

        if C == args.cache_sizes[0] and len(args.cache_sizes) == 1:
            # timeline data for hand-crafted scenarios
            sc = build_scenarios(SCENARIOS.keys(), C, args.window)
            sres = run_all(sc, agents, n)
            tl = {}
            for i, pt in enumerate(sc):
                tl[pt.trace.name] = {
                    "window": pt.window, "phases": pt.trace.phases,
                    "methods": {m: {"hit_rate": runs[i][0].tolist(),
                                    "actions": None if runs[i][1] is None else runs[i][1].tolist()}
                                for m, runs in sres.items()},
                }
            with open(os.path.join(args.out, f"scenarios_a{n}.json"), "w") as f:
                json.dump(tl, f)

    tag = f"_a{n}" + ("" if len(args.cache_sizes) == 1 else "_sizes")
    pd.concat(summ).to_csv(os.path.join(args.out, f"summary{tag}.csv"), index=False)
    pd.concat(regs).to_csv(os.path.join(args.out, f"per_regime{tag}.csv"), index=False)
    pd.concat(pcs).to_csv(os.path.join(args.out, f"post_change{tag}.csv"), index=False)
    pd.concat(acts).to_csv(os.path.join(args.out, f"actions{tag}.csv"), index=False)
    print(f"\nresults written to {args.out}/")


if __name__ == "__main__":
    main()
