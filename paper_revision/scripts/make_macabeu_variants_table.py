"""
Appendix table: MACABEU variants comparison (off / oracle / estimated) on
BiLSTM + BERT only, excluding GEMMA. Answers the reviewer's ``how does the
realistic MACABEU-estimated compare to MACABEU-oracle?'' question while
staying within the sub-set of cells we could run to completion.

Rows: 5 attackers (BERTattack, PWWS, DeepWordBug, Genetic, XARELLO) + Mean.
Cols: MACABEU-off, MACABEU-oracle, MACABEU-estimated + delta (est vs oracle).
Each cell = mean BODEGA over 4 tasks x 2 victims (BiLSTM + BERT).

Reads exactly the same paths as paper_assets/make_results_table.py.

Usage:
    python paper_revision/scripts/make_macabeu_variants_table.py \
        --macabeu_root ../../../macabeu/results \
        --xarello_root ../../../xarello/results \
        --out ../manuscript/figs/tab_macabeu_variants.tex
"""
import argparse
import re
from pathlib import Path

import numpy as np

TASKS = ["PR2", "FC", "HN", "RD"]
VICTIMS = ["BiLSTM", "BERT"]        # GEMMA excluded by design
ATTACKERS = ["BERTattack", "PWWS", "DeepWordBug", "Genetic", "XARELLO"]

VARIANTS = [
    ("MACABEU-off",       "macabeu_off"),
    ("MACABEU-oracle",    "macabeu_oracle"),
    ("MACABEU-estimated", "macabeu_estimated"),
]

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
    """xarello vs_macabeu / vs_macabeu_online: BiLSTM at top level, BERT in
    per-victim subdirectory (legacy oracle layout)."""
    base = root / subdir
    return base if victim == "BiLSTM" else base / victim


def xarello_true_subdir(root: Path, subdir: str, victim: str) -> Path:
    """xarello_vs_macabeu_online_true_*: victim always in subdirectory."""
    return root / subdir / victim


def path_for(mac_root: Path, xar_root: Path, variant_key: str,
             attacker: str, task: str, victim: str) -> Path:
    if attacker == "XARELLO":
        if variant_key == "macabeu_off":
            return (xarello_subdir(xar_root, "xarello_vs_macabeu", victim)
                    / f"results_{task}_True_XARELLO_{victim}_macabeu.txt")
        if variant_key == "macabeu_oracle":
            return (xarello_subdir(xar_root, "xarello_vs_macabeu_online",
                                   victim)
                    / f"results_{task}_True_XARELLO_{victim}"
                      f"_macabeu_online.txt")
        if variant_key == "macabeu_estimated":
            return (xarello_true_subdir(
                        xar_root, "xarello_vs_macabeu_online_true_hard",
                        victim)
                    / f"results_{task}_True_XARELLO_{victim}"
                      f"_macabeu_online.txt")
    else:
        # Standard attackers (BERTattack/PWWS/DeepWordBug/Genetic)
        if variant_key == "macabeu_off":
            return (mac_root / "offline"
                    / f"results_{task}_False_{attacker}_{victim}_rl_defense.txt")
        if variant_key == "macabeu_oracle":
            return (mac_root / "online"
                    / f"results_{task}_False_{attacker}_{victim}_online_rl.txt")
        if variant_key == "macabeu_estimated":
            return (mac_root / "online_true_hard"
                    / f"results_{task}_False_{attacker}_{victim}_online_rl.txt")
    return Path("/nonexistent")


def build_matrix(mac_root: Path, xar_root: Path):
    n_atk, n_var = len(ATTACKERS), len(VARIANTS)
    # attacker x variant x task x victim
    cube = np.full((n_atk, n_var, len(TASKS), len(VICTIMS)), np.nan)
    for ai, atk in enumerate(ATTACKERS):
        for vi, (_, key) in enumerate(VARIANTS):
            for ti, task in enumerate(TASKS):
                for vic_i, victim in enumerate(VICTIMS):
                    cube[ai, vi, ti, vic_i] = read_bodega(
                        path_for(mac_root, xar_root, key, atk, task, victim))
    # attacker x variant, averaged over tasks + victims
    return np.nanmean(cube, axis=(2, 3)), cube


def fmt_cell(v, is_min=False):
    if np.isnan(v):
        return "--"
    s = f"{v:.3f}"
    return rf"\textbf{{{s}}}" if is_min else s


def fmt_delta(est, oracle):
    if np.isnan(est) or np.isnan(oracle):
        return "--"
    d = (est - oracle) / oracle * 100.0
    sign = "+" if d >= 0 else "$-$"
    return f"{sign}{abs(d):.1f}\\%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--macabeu_root", default="../../../macabeu/results")
    ap.add_argument("--xarello_root", default="../../../xarello/results")
    ap.add_argument("--out",
                    default="../manuscript/figs/tab_macabeu_variants.tex")
    args = ap.parse_args()

    matrix, cube = build_matrix(Path(args.macabeu_root),
                                Path(args.xarello_root))
    # matrix[attacker, variant]
    means = np.nanmean(matrix, axis=0)  # per variant

    # For bolding: lowest BODEGA in each row (across the 3 variants).
    row_min_idx = np.nanargmin(matrix, axis=1)

    out = [
        r"% Auto-generated by paper_revision/scripts/make_macabeu_variants_table.py",
        r"\begin{table}[!htbp]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        (r"\textbf{Attacker} & \textbf{off} & \textbf{oracle} "
         r"& \textbf{estimated} & \textbf{est.\ vs.\ oracle} \\"),
        r"\midrule",
    ]
    for ai, atk in enumerate(ATTACKERS):
        vals = matrix[ai]
        cells = [
            fmt_cell(vals[vi], is_min=(row_min_idx[ai] == vi))
            for vi in range(len(VARIANTS))
        ]
        delta = fmt_delta(vals[2], vals[1])  # est vs oracle
        out.append(f"{atk} & {cells[0]} & {cells[1]} & {cells[2]} & {delta} \\\\")

    out.append(r"\midrule")
    # Mean row: mean over attackers (already stored in `means`).
    # Bold every cell (this is the summary row); underline the minimum.
    row_min = int(np.nanargmin(means))
    def bold_underline(v, is_min):
        if np.isnan(v):
            return r"\textbf{--}"
        s = rf"\textbf{{{v:.3f}}}"
        return rf"\underline{{{s}}}" if is_min else s
    m0 = bold_underline(means[0], row_min == 0)
    m1 = bold_underline(means[1], row_min == 1)
    m2 = bold_underline(means[2], row_min == 2)
    mean_delta = fmt_delta(means[2], means[1])
    out.append(
        rf"\textbf{{Mean}} & {m0} & {m1} & {m2} & \textbf{{{mean_delta}}} \\"
    )

    out += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Comparison of MACABEU variants (mean BODEGA on BiLSTM + "
         r"BERT, averaged over 4 tasks). Lower = stronger defence; best per "
         r"row in \textbf{bold}. The last column reports the relative change "
         r"of the deployment-realistic \textbf{MACABEU-estimated} variant "
         r"against \textbf{MACABEU-oracle} (which requires the gold label). "
         r"GEMMA is excluded because the estimated variant did not finish on "
         r"the GEMMA sub-grid within the revision cycle; the full grid is "
         r"reported in Table~\ref{tab:results-main} for MACABEU-off and "
         r"MACABEU-oracle.}"),
        r"\label{tab:macabeu-variants}",
        r"\end{table}",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    n_missing = int(np.isnan(cube).sum())
    if n_missing:
        print(f"Note: {n_missing}/{cube.size} sub-cells NaN.")


if __name__ == "__main__":
    main()
