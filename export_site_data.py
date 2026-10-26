"""
Bundle results/ and the experiment configuration into docs/data/results.json
for the interactive results page (docs/index.html, served by GitHub Pages).

    python export_site_data.py

Run after evaluate.py; everything on the page is read from this file.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import pandas as pd
import torch

from cache_rl.agents.dqn import DQNConfig
from cache_rl.agents.tabular_q import HR_BINS, JAC_BIN, MARGIN_BINS, n_states
from cache_rl.data import TEST_SEEDS, TRAIN_SEEDS, VAL_SEEDS
from cache_rl.features import AGENT_FEATURES, OBS_DIM, TRACE_FEATURES
from cache_rl.workloads import REGIMES, SCENARIOS

RES = "results"
OUT = os.path.join("docs", "data", "results.json")


def _csv(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p).round(5).to_dict(orient="records") if os.path.exists(p) else []


def _json(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def _train_log(tag):
    p = os.path.join(RES, f"train_log_{tag}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    val = df.dropna(subset=["val_hit_rate"])
    return {"episode": df.episode.tolist(), "return": df["return"].round(3).tolist(),
            "epsilon": df.epsilon.tolist(),
            "val_episode": val.episode.tolist(), "val_hit_rate": val.val_hit_rate.round(5).tolist()}


def _dqn_cfg():
    # the config actually used for training is stored in the checkpoint
    p = os.path.join("models", "dqn_a3.pt")
    if os.path.exists(p):
        return torch.load(p, map_location="cpu")["cfg"]
    return asdict(DQNConfig())


def main():
    data = {
        "summary": {"a3": _csv("summary_a3.csv"), "a4": _csv("summary_a4.csv"),
                    "a3_sizes": _csv("summary_a3_sizes.csv")},
        "per_regime": {"a3": _csv("per_regime_a3.csv"), "a4": _csv("per_regime_a4.csv")},
        "post_change": {"a3": _csv("post_change_a3.csv"), "a4": _csv("post_change_a4.csv")},
        "actions": {"a3": _csv("actions_a3.csv"), "a4": _csv("actions_a4.csv")},
        "scenarios": {"a3": _json("scenarios_a3.json"), "a4": _json("scenarios_a4.json")},
        "training": {t: _train_log(t) for t in ("dqn_a3", "dqn_a4", "qlearn_a3", "qlearn_a4")},
        "config": {
            "obs_dim": OBS_DIM,
            "trace_features": list(TRACE_FEATURES),
            "agent_features": list(AGENT_FEATURES),
            "regimes": list(REGIMES),
            "scenarios": {k: {"regimes": v[0], "lengths": v[1]} for k, v in SCENARIOS.items()},
            "seeds": {"train": [TRAIN_SEEDS.start, TRAIN_SEEDS.stop - 1],
                      "val": [VAL_SEEDS.start, VAL_SEEDS.stop - 1],
                      "test": [TEST_SEEDS.start, TEST_SEEDS.stop - 1]},
            "dqn": _dqn_cfg(),
            "tabular": {"alpha": 0.1, "gamma": 0.9, "states": {"a3": n_states(3), "a4": n_states(4)},
                        "margin_bins": MARGIN_BINS, "hr_bins": HR_BINS, "jaccard_bin": JAC_BIN},
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
