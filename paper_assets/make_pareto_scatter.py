"""
Pareto scatter: mean clean accuracy (x) vs. mean BODEGA (y) per defence.

Lower BODEGA + higher accuracy = top-right corner is best (Pareto-dominant).
Visualises that MACABEU-off pays no clean-accuracy cost yet sits well below
the static defences in BODEGA.

x-axis: mean clean accuracy averaged over tasks x victims, read from the same
  clean_accuracy/clean_accuracy_<TASK>_<VICTIM>_rl.txt files used by
  make_clean_accuracy_table.py.

y-axis: mean BODEGA averaged over standard attackers (BERTattack, PWWS,
  DeepWordBug, Genetic) and over tasks x victims, computed via the same
  paths as make_results_table.py. XARELLO is excluded so that the y-axis
  is comparable across defences (XARELLO is only run on 3/8 static rows).

Usage:
    python paper_assets/make_pareto_scatter.py \
        --bodega_root results/experiment-7_bleurt \
        --macabeu_root ../macabeu/results \
        --xarello_root ../xarello/results \
        --out paper_assets/fig_pareto.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_clean_accuracy_table import parse_file as parse_clean_file, parse_online_file  # noqa: E402
from make_results_table import (  # noqa: E402
    DEFENCES as RESULTS_DEFENCES,
    TASKS, VICTIMS,
    path_for, read_bodega,
)

STD_ATTACKERS = ["BERTattack", "PWWS", "DeepWordBug", "Genetic"]

# (display_label, clean-acc lookup key, bodega-row defence_key)
# Clean-acc keys starting with "__online:" are read from clean_accuracy_online/
# (label_source, reward_mode) pattern instead of the offline eval file.
POINTS = [
    ("none",              "none",              ""),
    ("spellcheck",        "spellcheck",        "_spellcheck"),
    ("unicode",           "unicode",           "_unicode"),
    ("discretize",        "discretize",        "_discretize"),
    ("MV@3",              "majority_vote@3",   "_majority_vote_3.0"),
    ("MV@7",              "majority_vote@7",   "_majority_vote_7.0"),
    ("sc_mv@3",           "spellcheck_mv@3",   "_spellcheck_mv_3.0"),
    ("char_noise@0.1",    "char_noise@0.1",    "_char_noise_0.1"),
    ("MACABEU-off",       "__macabeu_off__",   "macabeu_off"),
    ("MACABEU-oracle",    "__online:oracle_hard__",  "macabeu_oracle"),
    ("MACABEU-estimated", "__online:mv7_hard__",     "macabeu_on_hard"),
]


def mean_clean_acc(clean_root: Path, key: str,
                   online_root: Path = None) -> float:
    accs = []
    if key.startswith("__online:"):
        # e.g. "__online:oracle_hard__" -> ("oracle", "hard")
        payload = key[len("__online:"):].rstrip("_")
        lsrc, rmode = payload.split("_", 1)
        for task in TASKS:
            for vic in VICTIMS:
                v = parse_online_file(
                    online_root /
                    f"clean_accuracy_{task}_{vic}_online_{lsrc}_{rmode}.txt")
                if v is not None:
                    accs.append(v)
        return float(np.mean(accs)) if accs else float("nan")

    for task in TASKS:
        for vic in VICTIMS:
            base, rl, static = parse_clean_file(
                clean_root / f"clean_accuracy_{task}_{vic}_rl.txt")
            if key == "none":
                v = base
            elif key == "__macabeu_off__":
                v = rl
            else:
                v = static.get(key)
            if v is not None:
                accs.append(v)
    return float(np.mean(accs)) if accs else float("nan")


def mean_bodega(bodega_root: Path, mac_root: Path, xar_root: Path,
                defence_key: str) -> float:
    vals = []
    for atk in STD_ATTACKERS:
        for task in TASKS:
            for vic in VICTIMS:
                v = read_bodega(path_for(bodega_root, mac_root, xar_root,
                                         defence_key, atk, task, vic))
                if not np.isnan(v):
                    vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodega_root", default="results/experiment-7_bleurt")
    ap.add_argument("--macabeu_root", default="../macabeu/results")
    ap.add_argument("--xarello_root", default="../xarello/results")
    ap.add_argument("--out", default="paper_assets/fig_pareto.pdf")
    args = ap.parse_args()

    clean_root = Path(args.macabeu_root) / "clean_accuracy"
    online_clean_root = Path(args.macabeu_root) / "clean_accuracy_online"
    bodega_root = Path(args.bodega_root)
    mac_root = Path(args.macabeu_root)
    xar_root = Path(args.xarello_root)

    rows = []
    for label, clean_key, bodega_key in POINTS:
        x = mean_clean_acc(clean_root, clean_key, online_clean_root)
        y = mean_bodega(bodega_root, mac_root, xar_root, bodega_key)
        rows.append((label, x, y))
        print(f"  {label:<18s}  acc={x:.3f}  BODEGA={y:.3f}")

    # Coordinated ColorBrewer-ish palette: cool/muted blues+greens for the
    # static defences (best-to-worst gradient), a warm orange highlight for
    # MACABEU-off, and two distinct colours for the online MACABEU variants.
    style = {
        "MV@3":              ("#08519c", "^"),
        "sc_mv@3":           ("#3182bd", "v"),
        "MV@7":              ("#6baed6", "<"),
        "char_noise@0.1":    ("#9ecae1", ">"),
        "MACABEU-off":       ("#d94801", "D"),
        "MACABEU-oracle":    ("#54278f", "*"),
        "MACABEU-estimated": ("#807dba", "P"),
        "discretize":        ("#74c476", "h"),
        "spellcheck":        ("#41ab5d", "p"),
        "unicode":           ("#238b45", "o"),
        "none":              ("#525252", "s"),
    }
    # Plot order: ascending BODEGA so the legend reads best -> worst.
    rows_sorted = sorted(rows, key=lambda r: r[2])

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for label, x, y in rows_sorted:
        color, marker = style[label]
        is_highlight = label in {"MACABEU-off", "MACABEU-oracle",
                                  "MACABEU-estimated"}
        size = 95 if is_highlight else 50
        lw = 0.9 if is_highlight else 0.5
        ax.scatter([x], [y], c=color, marker=marker, s=size,
                   edgecolors="black", linewidths=lw, zorder=4,
                   label=label)

    ax.set_xlabel("Mean clean accuracy", fontsize=8)
    ax.set_ylabel("Mean BODEGA", fontsize=8)
    ax.set_xlim(0.700, 0.800)
    ax.set_ylim(0.070, 0.410)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    # Pareto-friendly orientation: top-right is best.
    ax.invert_yaxis()

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, frameon=False, handletextpad=0.4,
              borderaxespad=0.0, labelspacing=0.6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
