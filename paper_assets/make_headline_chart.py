"""
Headline figure: 12 grouped bars (4 tasks x 3 victims), four bars per group:
No defence / MACABEU offline / MACABEU online (oracle) / MACABEU online
(estimated) --- all attacked by XARELLO. Lower BODEGA = stronger defence.

Reads result_*.txt files from the xarello repo. Each file has lines like:
    BODEGA score: 0.2909870845247931

Missing cells (e.g. MACABEU-estimated on GEMMA, not run within the revision
window) render as an "n/a" tick on the x-axis for that bar.

Usage:
    python paper_assets/make_headline_chart.py \
        --xarello_root ../xarello/results \
        --out paper_assets/fig_headline.pdf
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TASKS = ["PR2", "FC", "HN", "RD"]
VICTIMS = ["BiLSTM", "BERT", "GEMMA"]


def read_bodega(path: Path):
    if not path.exists():
        return None
    for line in path.open():
        m = re.match(r"BODEGA score:\s*([\d.]+)", line.strip())
        if m:
            return float(m.group(1))
    return None


def collect(root: Path):
    nodef, off, on_oracle, on_est = {}, {}, {}, {}
    nodef_root = root / "xarello_vs_static"
    off_root = root / "xarello_vs_macabeu"
    on_oracle_root = root / "xarello_vs_macabeu_online"
    on_est_root = root / "xarello_vs_macabeu_online_true_hard"
    for victim in VICTIMS:
        off_dir = off_root if victim == "BiLSTM" else off_root / victim
        on_oracle_dir = on_oracle_root if victim == "BiLSTM" else on_oracle_root / victim
        # Estimated uses per-victim subdir for all three victims (different layout).
        on_est_dir = on_est_root / victim
        for task in TASKS:
            nodef[(victim, task)] = read_bodega(
                nodef_root / f"results_{task}_True_XARELLO_{victim}.txt"
            )
            off[(victim, task)] = read_bodega(
                off_dir / f"results_{task}_True_XARELLO_{victim}_macabeu.txt"
            )
            on_oracle[(victim, task)] = read_bodega(
                on_oracle_dir / f"results_{task}_True_XARELLO_{victim}_macabeu_online.txt"
            )
            on_est[(victim, task)] = read_bodega(
                on_est_dir / f"results_{task}_True_XARELLO_{victim}_macabeu_online.txt"
            )
    return nodef, off, on_oracle, on_est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xarello_root", default="../xarello/results")
    ap.add_argument("--out", default="paper_assets/fig_headline.pdf")
    args = ap.parse_args()

    nodef, off, on_oracle, on_est = collect(Path(args.xarello_root))

    n_groups = len(VICTIMS) * len(TASKS)
    labels = [f"{v}\n{t}" for v in VICTIMS for t in TASKS]
    nodef_vals = [nodef.get((v, t)) for v in VICTIMS for t in TASKS]
    off_vals = [off.get((v, t)) for v in VICTIMS for t in TASKS]
    oracle_vals = [on_oracle.get((v, t)) for v in VICTIMS for t in TASKS]
    est_vals = [on_est.get((v, t)) for v in VICTIMS for t in TASKS]

    x = np.arange(n_groups)
    # Four bars per group. Slight gap for legibility.
    w = 0.21

    fig, ax = plt.subplots(figsize=(11.0, 3.6))

    def as_plot(vals):
        return [v if v is not None else 0.0 for v in vals]

    ax.bar(x - 1.5 * w, as_plot(nodef_vals),  w, label="No defence",
           color="#fdae6b", edgecolor="black", linewidth=0.5)
    ax.bar(x - 0.5 * w, as_plot(off_vals),    w, label="MACABEU-off",
           color="#bcbddc", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.5 * w, as_plot(oracle_vals), w, label="MACABEU-oracle",
           color="#3182bd", edgecolor="black", linewidth=0.5)
    ax.bar(x + 1.5 * w, as_plot(est_vals),    w, label="MACABEU-estimated",
           color="#31a354", edgecolor="black", linewidth=0.5)

    # "n/a" marker where the underlying result file is missing.
    for i, v in enumerate(nodef_vals):
        if v is None:
            ax.text(i - 1.5 * w, 0.01, "n/a", ha="center", fontsize=6,
                    color="grey", rotation=90)
    for i, v in enumerate(off_vals):
        if v is None:
            ax.text(i - 0.5 * w, 0.01, "n/a", ha="center", fontsize=6,
                    color="grey", rotation=90)
    for i, v in enumerate(oracle_vals):
        if v is None:
            ax.text(i + 0.5 * w, 0.01, "n/a", ha="center", fontsize=6,
                    color="grey", rotation=90)
    for i, v in enumerate(est_vals):
        if v is None:
            ax.text(i + 1.5 * w, 0.01, "n/a", ha="center", fontsize=6,
                    color="grey", rotation=90)

    for k in range(1, len(VICTIMS)):
        ax.axvline(k * len(TASKS) - 0.5, color="grey",
                   linestyle=":", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("BODEGA score (lower = better defence)")
    all_vals = [v for v in
                as_plot(nodef_vals) + as_plot(off_vals)
                + as_plot(oracle_vals) + as_plot(est_vals) if v > 0]
    ax.set_ylim(0, max(all_vals) * 1.10)
    ax.legend(loc="upper right", frameon=False, ncol=4, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")
    for name, vals in [("no-defence", nodef_vals), ("off", off_vals),
                       ("oracle", oracle_vals), ("estimated", est_vals)]:
        n_missing = sum(1 for v in vals if v is None)
        if n_missing:
            print(f"Note: {n_missing}/12 {name} cells missing")


if __name__ == "__main__":
    main()
