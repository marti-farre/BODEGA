"""
Aggregated results table: defences (rows) x attackers (cols).

Each cell = mean BODEGA score over {4 tasks x 3 victims} = 12 sub-cells, or
NaN if the (defence, attacker) pair wasn't evaluated.

Reads BODEGA score lines from three result roots:
  - bodega_root  : static defences vs standard attackers
  - macabeu_root : MACABEU-off / MACABEU-oracle / MACABEU-on (mv7, soft/hard)
  - xarello_root : XARELLO vs {none, sc_mv@3, MACABEU-off, MACABEU-*}

Usage:
    python paper_assets/make_results_table.py \
        --bodega_root results/experiment-7_bleurt \
        --macabeu_root ../macabeu/results \
        --xarello_root ../xarello/results \
        --out paper_assets/tab_results_main.tex
"""
import argparse
import re
from pathlib import Path

import numpy as np

TASKS = ["PR2", "FC", "HN", "RD"]
VICTIMS = ["BiLSTM", "BERT", "GEMMA"]
ATTACKERS = ["BERTattack", "PWWS", "DeepWordBug", "Genetic", "XARELLO"]

# (display_label, lookup_key) for static defences. The lookup_key drives the
# filename suffix in `bodega_root` (suffix=="" means undefended).
STATIC_DEFENCES = [
    ("none",            ""),
    ("spellcheck",      "_spellcheck"),
    ("unicode",         "_unicode"),
    ("discretize",      "_discretize"),
    ("MV@3",            "_majority_vote_3.0"),
    ("MV@7",            "_majority_vote_7.0"),
    ("sc\\_mv@3",       "_spellcheck_mv_3.0"),
    ("char\\_noise@0.1", "_char_noise_0.1"),
]
MACABEU_ROWS = [
    ("MACABEU-off",                 "macabeu_off"),
    ("MACABEU-oracle",              "macabeu_oracle"),
    ("MACABEU-estimated (soft)",    "macabeu_on_soft"),
    ("MACABEU-estimated",           "macabeu_on_hard"),
]
DEFENCES = STATIC_DEFENCES + MACABEU_ROWS

BODEGA_RE = re.compile(r"^BODEGA score:\s*([0-9.eE+-]+)")


def read_bodega(path: Path):
    if not path.exists():
        return np.nan
    for line in path.open():
        m = BODEGA_RE.match(line)
        if m:
            return float(m.group(1))
    return np.nan


def xarello_subdir(root: Path, subdir: str, victim: str) -> Path:
    """xarello vs_macabeu / vs_macabeu_online: BiLSTM at top level, BERT and
    GEMMA in a per-victim subdirectory (legacy oracle layout)."""
    base = root / subdir
    return base if victim == "BiLSTM" else base / victim


def xarello_true_subdir(root: Path, subdir: str, victim: str) -> Path:
    """xarello_vs_macabeu_online_true_*: all three victims sit in a per-victim
    subdirectory, unlike the legacy oracle layout above."""
    return root / subdir / victim


# Map macabeu-on flavours to their result-directory names.
MACABEU_ON_MACROOT = {
    "macabeu_oracle":  "online",
    "macabeu_on_soft": "online_true_soft",
    "macabeu_on_hard": "online_true_hard",
}
MACABEU_ON_XARROOT = {
    "macabeu_oracle":  ("xarello_vs_macabeu_online",           xarello_subdir),
    "macabeu_on_soft": ("xarello_vs_macabeu_online_true_soft", xarello_true_subdir),
    "macabeu_on_hard": ("xarello_vs_macabeu_online_true_hard", xarello_true_subdir),
}


def path_for(bodega_root: Path, mac_root: Path, xar_root: Path,
             defence_key: str, attacker: str, task: str, victim: str) -> Path:
    """Return the result file path for one (defence, attacker, task, victim)
    cell. The file may or may not exist; the caller handles missing files."""
    if attacker == "XARELLO":
        if defence_key == "":
            return (xar_root / "xarello_vs_static"
                    / f"results_{task}_True_XARELLO_{victim}.txt")
        if defence_key == "_spellcheck_mv_3.0":
            return (xar_root / "xarello_vs_static"
                    / f"results_{task}_True_XARELLO_{victim}"
                      f"_spellcheck_mv_3.0.txt")
        if defence_key == "macabeu_off":
            return (xarello_subdir(xar_root, "xarello_vs_macabeu", victim)
                    / f"results_{task}_True_XARELLO_{victim}_macabeu.txt")
        if defence_key in MACABEU_ON_XARROOT:
            subdir_name, layout_fn = MACABEU_ON_XARROOT[defence_key]
            return (layout_fn(xar_root, subdir_name, victim)
                    / f"results_{task}_True_XARELLO_{victim}"
                      f"_macabeu_online.txt")
        # XARELLO not evaluated against the other static defences.
        return Path("/nonexistent")

    # Standard attackers (BERTattack/PWWS/DeepWordBug/Genetic).
    if defence_key == "macabeu_off":
        return (mac_root / "offline"
                / f"results_{task}_False_{attacker}_{victim}_rl_defense.txt")
    if defence_key in MACABEU_ON_MACROOT:
        return (mac_root / MACABEU_ON_MACROOT[defence_key]
                / f"results_{task}_False_{attacker}_{victim}_online_rl.txt")
    return (bodega_root
            / f"results_{task}_False_{attacker}_{victim}{defence_key}.txt")


def build_cube(bodega_root: Path, mac_root: Path, xar_root: Path):
    n_def, n_atk, n_task, n_vic = (len(DEFENCES), len(ATTACKERS),
                                   len(TASKS), len(VICTIMS))
    cube = np.full((n_def, n_atk, n_task, n_vic), np.nan)
    for di, (_, key) in enumerate(DEFENCES):
        for ai, atk in enumerate(ATTACKERS):
            for ti, task in enumerate(TASKS):
                for vi, vic in enumerate(VICTIMS):
                    cube[di, ai, ti, vi] = read_bodega(
                        path_for(bodega_root, mac_root, xar_root,
                                 key, atk, task, vic))
    return cube


def fmt(v):
    return "--" if np.isnan(v) else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodega_root", default="results/experiment-7_bleurt")
    ap.add_argument("--macabeu_root", default="../macabeu/results")
    ap.add_argument("--xarello_root", default="../xarello/results")
    ap.add_argument("--out", default="paper_assets/tab_results_main.tex")
    args = ap.parse_args()

    cube = build_cube(Path(args.bodega_root), Path(args.macabeu_root),
                      Path(args.xarello_root))
    matrix = np.nanmean(cube, axis=(2, 3))  # def x atk

    # Per-attacker best (lowest) score over static + MACABEU rows.
    best_di = np.nanargmin(matrix, axis=0)

    col_spec = "l" + "c" * len(ATTACKERS) + "|c"
    out = [
        r"% Auto-generated by paper_assets/make_results_table.py",
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{6pt}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        (r"\textbf{Defence} & "
         + " & ".join(rf"\textbf{{{a}}}" for a in ATTACKERS)
         + r" & \textbf{Mean} \\"),
        r"\midrule",
    ]

    def row(di, label):
        row_vals = matrix[di]
        cells = []
        for ai, v in enumerate(row_vals):
            s = fmt(v)
            if best_di[ai] == di and not np.isnan(v):
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        row_mean = np.nanmean(row_vals)
        return f"{label} & " + " & ".join(cells) + f" & {fmt(row_mean)} \\\\"

    for di, (label, _) in enumerate(STATIC_DEFENCES):
        out.append(row(di, label))
    out.append(r"\midrule")
    for offset, (label, _) in enumerate(MACABEU_ROWS):
        out.append(row(len(STATIC_DEFENCES) + offset, label))

    out.append(r"\midrule")
    col_means = np.nanmean(matrix, axis=0)
    out.append(
        r"\textbf{Mean} & "
        + " & ".join(rf"\textbf{{{fmt(v)}}}" for v in col_means)
        + rf" & \textbf{{{fmt(np.nanmean(matrix))}}} \\"
    )

    out += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Mean BODEGA per (defence, attacker), averaged over 4 "
         r"tasks $\times$ 3 victims. Lower = stronger; best per column in "
         r"\textbf{bold}. `--' = pair not evaluated.}"),
        r"\label{tab:results-main}",
        r"\end{table*}",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    n_missing = int(np.isnan(cube).sum())
    # XARELLO is only evaluated for a subset of defence rows; count those gaps.
    xarello_eval_keys = {"", "_spellcheck_mv_3.0", "macabeu_off",
                         "macabeu_oracle", "macabeu_on_soft",
                         "macabeu_on_hard"}
    n_def_with_xar = sum(1 for _, k in DEFENCES if k in xarello_eval_keys)
    expected_missing = (
        (len(DEFENCES) - n_def_with_xar) * len(TASKS) * len(VICTIMS)
    )
    print(f"Note: {n_missing}/{cube.size} sub-cells NaN "
          f"({expected_missing} of those are by-design XARELLO gaps).")


if __name__ == "__main__":
    main()
