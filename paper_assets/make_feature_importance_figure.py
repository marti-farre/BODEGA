"""
Feature-importance heatmaps from analyze_feature_importance.py's CSV.

Emits two PDFs:
  fig_feature_importance.pdf       — main heatmap: 10 features x 3 variants,
                                     aggregated over all (task, victim, attacker).
  fig_feature_importance_by_atk.pdf — per-attacker breakdown: three side-by-side
                                     heatmaps (offline / oracle / est_hard),
                                     each 10 features x N_attackers.

Usage:
    python paper_assets/make_feature_importance_figure.py \
        --csv ../macabeu/results/feature_importance.csv \
        --out_dir paper_assets
"""
import argparse
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


VARIANT_ORDER = ["offline", "oracle", "est_hard"]
VARIANT_LABELS = {
    "offline":  "MACABEU-off",
    "oracle":   "MACABEU-\noracle",
    "est_hard": "MACABEU-\nestimated",
}

FEATURE_ORDER = [
    "text_length", "word_count", "avg_word_length",
    "oov_ratio", "non_ascii_ratio", "uppercase_ratio",
    "punctuation_ratio", "digit_ratio", "repeated_char_ratio",
    "char_entropy",
]
FEATURE_LABELS = {
    "text_length":         "text length",
    "word_count":          "word count",
    "avg_word_length":     "avg word len",
    "oov_ratio":           "OOV ratio",
    "non_ascii_ratio":     "non-ASCII ratio",
    "uppercase_ratio":     "uppercase ratio",
    "punctuation_ratio":   "punct. ratio",
    "digit_ratio":         "digit ratio",
    "repeated_char_ratio": "repeat-char ratio",
    "char_entropy":        "char entropy",
}


def load_csv(path: Path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["flip_rate"] = float(r["flip_rate"])
            rows.append(r)
    return rows


def aggregate(rows, keys):
    """Return dict {(*key_values,): mean_flip_rate} averaged over remaining dims."""
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in rows:
        k = tuple(r[k] for k in keys)
        bucket[k].append(r["flip_rate"])
    return {k: float(np.mean(v)) for k, v in bucket.items()}


def draw_heatmap(ax, mat, row_labels, col_labels, title,
                 cmap="Reds", vmin=0.0, vmax=None,
                 annotate=True):
    if vmax is None:
        vmax = float(np.nanmax(mat)) * 1.02 if np.nanmax(mat) > 0 else 1.0
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                colour = "white" if v > (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color=colour)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../macabeu/results/feature_importance.csv")
    ap.add_argument("--out_dir", default="paper_assets")
    args = ap.parse_args()

    rows = load_csv(Path(args.csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Aggregate: (variant, feature) -> mean flip rate over task/victim/attacker
    agg = aggregate(rows, ["policy_type", "feature"])
    mat = np.array([
        [agg.get((v, f), np.nan) for v in VARIANT_ORDER]
        for f in FEATURE_ORDER
    ])
    fig, ax = plt.subplots(figsize=(3.2, 4.2))
    im = draw_heatmap(
        ax, mat,
        row_labels=[FEATURE_LABELS[f] for f in FEATURE_ORDER],
        col_labels=[VARIANT_LABELS[v] for v in VARIANT_ORDER],
        title="Feature flip rate (mean over policies)",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("flip rate", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    out_main = out_dir / "fig_feature_importance.pdf"
    fig.savefig(out_main, bbox_inches="tight")
    print(f"Wrote {out_main}")
    plt.close(fig)

    # --- Per-attacker: three side-by-side panels (one per variant),
    #     rows = features, cols = attackers ("N/A" for offline).
    online_rows = [r for r in rows if r["policy_type"] != "offline"]
    attackers = sorted({r["attacker"] for r in online_rows})
    offline_rows = [r for r in rows if r["policy_type"] == "offline"]

    # Build a per-variant matrix.
    def variant_matrix(variant, atk_list, only_offline=False):
        if only_offline:
            agg_v = aggregate(offline_rows, ["feature"])
            m = np.array([[agg_v.get((f,), np.nan)] for f in FEATURE_ORDER])
            return m, ["mean"]
        sub = [r for r in rows if r["policy_type"] == variant]
        agg_v = aggregate(sub, ["attacker", "feature"])
        m = np.array([
            [agg_v.get((a, f), np.nan) for a in atk_list]
            for f in FEATURE_ORDER
        ])
        return m, atk_list

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 4.4),
                             gridspec_kw={"width_ratios": [1, 4, 4]})
    vmax = max(
        np.nanmax(variant_matrix("offline", [], only_offline=True)[0]),
        np.nanmax(variant_matrix("oracle",   attackers)[0]),
        np.nanmax(variant_matrix("est_hard", attackers)[0]),
    )
    for ax, variant in zip(axes, VARIANT_ORDER):
        if variant == "offline":
            m, cols = variant_matrix(variant, [], only_offline=True)
            show_ylabels = True
        else:
            m, cols = variant_matrix(variant, attackers)
            show_ylabels = False
        im = draw_heatmap(
            ax, m,
            row_labels=[FEATURE_LABELS[f] for f in FEATURE_ORDER] if show_ylabels else [""] * len(FEATURE_ORDER),
            col_labels=cols,
            title=VARIANT_LABELS[variant].replace("\n", " "),
            vmin=0.0, vmax=vmax,
        )
        if not show_ylabels:
            ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("flip rate", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    out_atk = out_dir / "fig_feature_importance_by_atk.pdf"
    fig.savefig(out_atk, bbox_inches="tight")
    print(f"Wrote {out_atk}")
    plt.close(fig)


if __name__ == "__main__":
    main()
