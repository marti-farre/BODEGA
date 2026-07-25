"""
Clean accuracy table on unattacked data: defenses (rows) x tasks (cols).
Each cell = mean accuracy over the three victims (BiLSTM, BERT, GEMMA).

Reads:
  - ../macabeu/results/clean_accuracy/clean_accuracy_<TASK>_<VICTIM>_rl.txt
      Offline eval: static defenses + MACABEU-off frozen policy.
  - ../macabeu/results/clean_accuracy_online/clean_accuracy_<TASK>_<VICTIM>_online_<label>_<mode>.txt
      Online eval (adaptive on clean traffic): MACABEU-Oracle + MACABEU-est.

Usage:
    python paper_assets/make_clean_accuracy_table.py \
        --root ../macabeu/results/clean_accuracy \
        --online_root ../macabeu/results/clean_accuracy_online \
        --out paper_assets/tab_clean_accuracy.tex
"""
import argparse
import re
from pathlib import Path

import numpy as np

TASKS = ["PR2", "FC", "HN", "RD"]
VICTIMS = ["BiLSTM", "BERT", "GEMMA"]

# (display_label, key_in_file)
DEFENSES = [
    ("none",                       "none"),
    ("spellcheck",                 "spellcheck"),
    ("unicode",                    "unicode"),
    ("discretize",                 "discretize"),
    ("MV@3",                       "majority_vote@3"),
    ("MV@7",                       "majority_vote@7"),
    ("sc\\_mv@3",                  "spellcheck_mv@3"),
    ("char\\_noise@0.1",           "char_noise@0.1"),
    ("MACABEU-off",                "__macabeu_off__"),
    ("MACABEU-oracle",             "__macabeu_online_oracle_hard__"),
    ("MACABEU-estimated",          "__macabeu_online_mv7_hard__"),
]


def parse_file(path: Path):
    """Return (baseline_acc, rl_acc, static_accs dict)."""
    if not path.exists():
        return None, None, {}
    baseline_acc, rl_acc, static_accs = None, None, {}
    in_baseline = in_rl = in_static = False
    for line in path.open():
        s = line.strip()
        if s.startswith("# Baseline"):
            in_baseline, in_rl, in_static = True, False, False
            continue
        if s.startswith("# RL Defense"):
            in_baseline, in_rl, in_static = False, True, False
            continue
        if s.startswith("# Static Defenses"):
            in_baseline, in_rl, in_static = False, False, True
            continue
        if s.startswith("#"):
            in_baseline = in_rl = in_static = False
            continue
        if in_baseline:
            m = re.match(r"Accuracy:\s*([\d.]+)", s)
            if m:
                baseline_acc = float(m.group(1))
        elif in_rl:
            m = re.match(r"Accuracy:\s*([\d.]+)", s)
            if m:
                rl_acc = float(m.group(1))
        elif in_static:
            m = re.match(r"([^:]+?)\s*:\s*acc=([\d.]+)", s)
            if m:
                static_accs[m.group(1).strip()] = float(m.group(2))
    return baseline_acc, rl_acc, static_accs


def parse_online_file(path: Path):
    """Parse the online-adaptive clean-accuracy file. Returns online_acc or None.
    File format written by runs/eval_clean_accuracy_online.py."""
    if not path.exists():
        return None
    in_online = False
    for line in path.open():
        s = line.strip()
        if s.startswith("# Online RL Defense"):
            in_online = True
            continue
        if s.startswith("#"):
            in_online = False
            continue
        if in_online:
            m = re.match(r"Accuracy:\s*([\d.]+)", s)
            if m:
                return float(m.group(1))
    return None


# Map the placeholder keys to (label_source, reward_mode) suffix used by the
# online eval script's filename.
ONLINE_KEY_SUFFIX = {
    "__macabeu_online_oracle_hard__": ("oracle", "hard"),
    "__macabeu_online_mv7_hard__":    ("mv7",    "hard"),
    "__macabeu_online_mv7_soft__":    ("mv7",    "soft"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../macabeu/results/clean_accuracy")
    ap.add_argument("--online_root", default="../macabeu/results/clean_accuracy_online")
    ap.add_argument("--out", default="paper_assets/tab_clean_accuracy.tex")
    args = ap.parse_args()

    root = Path(args.root)
    online_root = Path(args.online_root)
    # matrix[defense, task, victim]
    cube = np.full((len(DEFENSES), len(TASKS), len(VICTIMS)), np.nan)
    for ti, task in enumerate(TASKS):
        for vi, victim in enumerate(VICTIMS):
            path = root / f"clean_accuracy_{task}_{victim}_rl.txt"
            base_acc, rl_acc, static_accs = parse_file(path)
            for di, (label, key) in enumerate(DEFENSES):
                if key == "__macabeu_off__":
                    val = rl_acc
                elif key in ONLINE_KEY_SUFFIX:
                    lsrc, rmode = ONLINE_KEY_SUFFIX[key]
                    opath = online_root / (
                        f"clean_accuracy_{task}_{victim}_online_"
                        f"{lsrc}_{rmode}.txt")
                    val = parse_online_file(opath)
                elif key == "none":
                    val = base_acc
                else:
                    val = static_accs.get(key)
                if val is not None:
                    cube[di, ti, vi] = val

    matrix = np.nanmean(cube, axis=2)  # defense x task, mean over victims
    row_mean = np.nanmean(matrix, axis=1)

    # Baseline is the `none` row (DEFENSES[0]); other rows render as
    # "abs (Δ%)" relative to baseline.
    baseline_row = matrix[0]              # per-task baseline accuracy
    baseline_mean = row_mean[0]           # baseline mean
    deltas = (matrix - baseline_row) / baseline_row * 100.0
    delta_means = (row_mean - baseline_mean) / baseline_mean * 100.0

    # Per-column best non-baseline absolute (highest accuracy).
    best_per_col = []
    for ci in range(len(TASKS)):
        col = matrix[1:, ci]
        col_max = np.nanmax(col)
        best_per_col.append(
            {i + 1 for i, v in enumerate(col)
             if not np.isnan(v) and abs(v - col_max) < 5e-4}
        )
    mean_col = row_mean[1:]
    mean_max = np.nanmax(mean_col)
    best_mean = {i + 1 for i, v in enumerate(mean_col)
                 if not np.isnan(v) and abs(v - mean_max) < 5e-4}

    def fmt_cell(absval, delta, bold=False):
        if np.isnan(absval):
            return "--"
        delta_s = f"{delta:+.1f}\\%".replace("-", "$-$")
        s = f"{absval:.3f} ({delta_s})"
        return rf"\textbf{{{s}}}" if bold else s

    out = [
        r"% Auto-generated by paper_assets/make_clean_accuracy_table.py",
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{l" + "c" * (len(TASKS) + 1) + r"}",
        r"\toprule",
        (r"\textbf{Defence} & "
         + " & ".join(rf"\textbf{{{t}}}" for t in TASKS)
         + r" & \textbf{Mean} \\"),
        r"\midrule",
    ]
    for di, (label, _) in enumerate(DEFENSES):
        if di == 0:
            cells = " & ".join(f"{v:.3f}" if not np.isnan(v) else "--"
                               for v in matrix[di])
            out.append(f"{label} & {cells} & {row_mean[di]:.3f} \\\\")
        else:
            cells = " & ".join(
                fmt_cell(matrix[di, ci], deltas[di, ci],
                         bold=(di in best_per_col[ci]))
                for ci in range(len(TASKS))
            )
            mean_cell = fmt_cell(row_mean[di], delta_means[di],
                                 bold=(di in best_mean))
            out.append(f"{label} & {cells} & {mean_cell} \\\\")
        # Divider before the block of adaptive-defence rows (MACABEU-off + online).
        if DEFENSES[di][1] == "char_noise@0.1":
            out.append(r"\midrule")

    out += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Clean accuracy per (defence, task), averaged over 3 "
         r"victims. Each non-baseline cell: absolute accuracy and \% change "
         r"vs.\ \texttt{none}. Best non-baseline per column in \textbf{bold}.}"),
        r"\label{tab:clean-accuracy}",
        r"\end{table*}",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    if np.isnan(cube).any():
        print(f"Note: {int(np.isnan(cube).sum())}/{cube.size} sub-cells missing.")


if __name__ == "__main__":
    main()
