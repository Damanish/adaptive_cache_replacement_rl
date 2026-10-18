"""Generate all figures from the CSV/JSON files written by train.py / evaluate.py."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES, FIG = "results", "figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

COL = {"Fixed-LRU": "#4C72B0", "Fixed-LFU": "#55A868", "Fixed-FIFO": "#8172B2", "Fixed-MRU": "#937860",
       "Random": "#AAAAAA", "FTL": "#CCB974", "ARC": "#C44E52", "Phase-Oracle": "#64B5CD",
       "Q-Learning": "#DD8452", "DQN": "#2CA02C", "Belady": "#333333"}
ACOL = ["#4C72B0", "#55A868", "#8172B2", "#937860"]
ANAMES = ["LRU", "LFU", "FIFO", "MRU"]


def _read(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def fig_training():
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.4))
    for tag, lbl, c in [("dqn_a3", "DQN (3 act.)", "#2CA02C"), ("qlearn_a3", "Q-Learning (3 act.)", "#DD8452"),
                        ("dqn_a4", "DQN (4 act.)", "#1B5E20"), ("qlearn_a4", "Q-Learning (4 act.)", "#A0522D")]:
        df = _read(f"train_log_{tag}.csv")
        if df is None:
            continue
        ls = "-" if tag.endswith("a3") else "--"
        ax[0].plot(df.episode, df["return"].rolling(15, min_periods=1).mean(), ls, color=c, label=lbl, lw=1.2)
        v = df.dropna(subset=["val_hit_rate"])
        ax[1].plot(v.episode, v.val_hit_rate, ls, marker="o", ms=2.5, color=c, label=lbl, lw=1.2)
    ax[0].set(xlabel="episode", ylabel="episode return (smoothed)", title="Training return")
    ax[1].set(xlabel="episode", ylabel="hit rate", title="Validation hit rate (greedy)")
    ax[1].legend(fontsize=7, frameon=False)
    fig.savefig(os.path.join(FIG, "training_curves.png"))
    plt.close(fig)


def fig_overall():
    s3, s4 = _read("summary_a3.csv"), _read("summary_a4.csv")
    if s3 is None:
        return
    rows = [(m, r.hit_rate, r.hit_rate_std, r.amat, COL.get(m, "#777")) for m, r in s3.set_index("method").iterrows()]
    if s4 is not None:
        for m in ["Fixed-MRU", "FTL", "Phase-Oracle", "Q-Learning", "DQN"]:
            if m in set(s4.method):
                r = s4.set_index("method").loc[m]
                name = m if m == "Fixed-MRU" else f"{m} (+MRU)"
                rows.append((name, r.hit_rate, r.hit_rate_std, r.amat, COL.get(m, "#777")))
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    y = np.arange(len(rows))
    bars = ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], color=[r[4] for r in rows],
                   error_kw={"lw": 0.6, "capsize": 1.5})
    for i, r in enumerate(rows):
        if "(+MRU)" in r[0]:
            bars[i].set_hatch("///")
            bars[i].set_edgecolor("white")
        ax.text(r[1] + r[2] + 0.005, i, f"{r[1]:.3f} | {r[3]:.1f}", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel("hit rate (mean ± std over 20 test traces)")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.25)
    ax.set_title("Overall: hit rate | AMAT (units)", fontsize=9)
    fig.savefig(os.path.join(FIG, "overall_hit_rate.png"))
    plt.close(fig)


def fig_regime(n=3):
    df = _read(f"per_regime_a{n}.csv")
    if df is None:
        return
    methods = [m for m in ["Fixed-LRU", "Fixed-LFU", "Fixed-FIFO", "Fixed-MRU", "ARC", "FTL", "Q-Learning", "DQN", "Belady"]
               if m in set(df.method)]
    regs = ["zipf", "drift", "scan", "stack", "loop"]
    piv = df.pivot_table(index="regime", columns="method", values="hit_rate").loc[regs, methods]
    fig, ax = plt.subplots(figsize=(7, 2.3))
    w = 0.85 / len(methods)
    x = np.arange(len(regs))
    for i, m in enumerate(methods):
        ax.bar(x + i * w - 0.425 + w / 2, piv[m], w, label=m, color=COL[m])
    ax.set_xticks(x)
    ax.set_xticklabels(regs)
    ax.set_ylabel("hit rate")
    ax.set_title(f"Hit rate per workload regime ({n}-action setting)")
    ax.legend(ncol=len(methods), fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(os.path.join(FIG, f"per_regime_a{n}.png"))
    plt.close(fig)


def fig_timeline(n=3, scenario="zipf_scan_drift", smooth=5):
    p = os.path.join(RES, f"scenarios_a{n}.json")
    if not os.path.exists(p):
        return
    sc = json.load(open(p))[scenario]
    W = sc["window"]
    meths = [m for m in ["Fixed-LRU", "Fixed-LFU", "ARC", "FTL", "DQN"] if m in sc["methods"]]
    fig, (ax, axa) = plt.subplots(2, 1, figsize=(7, 2.6), sharex=True, gridspec_kw={"height_ratios": [5, 1]})
    k = np.ones(smooth) / smooth
    for m in meths:
        h = np.array(sc["methods"][m]["hit_rate"])
        x = (np.arange(len(h)) + 0.5) * W
        ax.plot(x, np.convolve(h, k, mode="same"), color=COL[m], lw=1.6 if m == "DQN" else 1.0, label=m)
    for s, e, r in sc["phases"]:
        ax.axvline(s, color="k", lw=0.5, ls=":")
        ax.text((s + e) / 2, 1.0, r, ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_ylabel("hit rate")
    ax.legend(ncol=len(meths), fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.07))
    if "DQN" in sc["methods"]:
        a = np.array(sc["methods"]["DQN"]["actions"])
        for i, ai in enumerate(a):
            axa.axvspan(i * W, (i + 1) * W, color=ACOL[ai], lw=0)
        axa.set_yticks([])
        axa.set_ylabel("DQN\naction", fontsize=7, rotation=0, ha="right", va="center")
        handles = [plt.Rectangle((0, 0), 1, 1, color=ACOL[i]) for i in range(n)]
        axa.legend(handles, ANAMES[:n], ncol=n, fontsize=6.5, frameon=False, loc="upper right",
                   bbox_to_anchor=(1.0, -0.4))
    axa.set_xlabel("request index")
    fig.savefig(os.path.join(FIG, f"timeline_{scenario}_a{n}.png"))
    plt.close(fig)


def fig_post_change(n=3):
    df = _read(f"post_change_a{n}.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for m in ["Fixed-LRU", "Fixed-LFU", "ARC", "FTL", "Q-Learning", "DQN"]:
        d = df[df.method == m]
        if len(d):
            ax.plot(d.window_after_change, d.hit_rate, marker="o", ms=2.5, color=COL[m], label=m,
                    lw=1.6 if m == "DQN" else 1.0)
    ax.set(xlabel="decision window after a workload change", ylabel="hit rate",
           title="Recovery after workload changes")
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.savefig(os.path.join(FIG, f"post_change_a{n}.png"))
    plt.close(fig)


def fig_actions(n=3):
    df = _read(f"actions_a{n}.csv")
    if df is None or not len(df):
        return
    d = df[df.method == "DQN"].set_index("regime").loc[["zipf", "drift", "scan", "stack", "loop"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    bottom = np.zeros(len(d))
    for i, a in enumerate(ANAMES[:n]):
        ax.bar(d.index, d[a], bottom=bottom, color=ACOL[i], label=a)
        bottom += d[a].values
    ax.set(ylabel="fraction of windows", title=f"DQN policy choice per regime ({n} actions)")
    ax.legend(fontsize=7, frameon=False, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(os.path.join(FIG, f"actions_a{n}.png"))
    plt.close(fig)


def fig_sizes(n=3):
    df = _read(f"summary_a{n}_sizes.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for m in ["Fixed-LRU", "Fixed-LFU", "ARC", "FTL", "DQN", "Belady"]:
        d = df[df.method == m].sort_values("cache_size")
        if len(d):
            ax.plot(d.cache_size, d.hit_rate, marker="o", ms=3, color=COL[m], label=m)
    ax.set(xlabel="cache size (blocks)", ylabel="hit rate", xscale="log",
           title="Generalisation to unseen cache sizes")
    ax.set_xticks(sorted(df.cache_size.unique()))
    ax.set_xticklabels([str(c) for c in sorted(df.cache_size.unique())])
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.savefig(os.path.join(FIG, f"cache_sizes_a{n}.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_training()
    fig_overall()
    for n in (3, 4):
        fig_regime(n)
        fig_post_change(n)
        fig_actions(n)
        fig_sizes(n)
        for sc in ("zipf_scan_drift", "drift_loop_stack", "fast_switch"):
            fig_timeline(n, sc)
    print("figures written to", FIG)
