"""report_generator.py — All figures for UL Report.
Pure white bg (#FFFFFF), seaborn whitegrid, denim palette.
Mirrors SL/OL report_generator style exactly.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import kurtosis as sp_kurtosis

from .styles import (
    apply_denim_theme, denim_fig,
    DENIM, DENIM_DARK, DENIM_MID, DENIM_BRT,
    AMBER, PLUM, TEAL, SIENNA, GREEN, TEXT_DARK, WHITE, GRID_CLR,
    CT_PALETTE, DR_COLORS
)

FIGDIR = None  # set by run_pipeline


def _savefig(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


def _autocontrast_heatmap(ax):
    """Force dark text on light cells, white text on dark cells."""
    for text in ax.texts:
        try:
            bg = text.get_backgroundcolor()
        except Exception:
            bg = None
        # Try to find cell color from collections
        pass
    # safer: just read each annotation's position → get cell rgba
    # We iterate over all Text objects in the axes
    cmap_ax = ax.collections[0] if ax.collections else None
    if cmap_ax is None:
        return
    norm = cmap_ax.norm
    cmap_fn = cmap_ax.cmap
    for text in ax.texts:
        try:
            val = float(text.get_text().replace("−", "-"))
        except Exception:
            continue
        rgba = cmap_fn(norm(val))
        lum  = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        text.set_color("#FFFFFF" if lum < 0.35 else TEXT_DARK)


# ── FIG 1: Class distribution ─────────────────────────────────────────────────

def fig_class_distribution(y):
    apply_denim_theme()
    fig, ax = denim_fig(figsize=(7, 3.5))
    classes = np.arange(1, 8)
    counts  = [int(np.sum(y == c)) for c in classes]
    bars = ax.bar(classes, counts,
                  color=[CT_PALETTE[i] for i in range(7)],
                  edgecolor="none", width=0.65)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c + 50, str(c),
                ha="center", va="bottom", fontsize=8, color=TEXT_DARK)
    ax.set_xlabel("Cover Type", color=TEXT_DARK)
    ax.set_ylabel("Count", color=TEXT_DARK)
    ax.set_title("Class Distribution — 20k Covertype Sample", color=DENIM_DARK,
                 fontweight="bold")
    ax.set_xticks(classes)
    ax.set_xticklabels([f"CT{c}" for c in classes])
    fig.tight_layout()
    return _savefig(fig, "fig_class_distribution.pdf")


# ── FIG 2: Clustering sweep (inertia/BIC + metrics) ──────────────────────────

def fig_clustering_sweep(km_sweep, gmm_sweep):
    apply_denim_theme()
    fig, axes = denim_fig(1, 3, figsize=(12, 3.8))

    ks     = [r["k"]         for r in km_sweep]
    inert  = [r["inertia"]   for r in km_sweep]
    sil_km = [r["silhouette"]for r in km_sweep]
    ari_km = [r["ari"]       for r in km_sweep]

    ns     = [r["n"]         for r in gmm_sweep]
    bic    = [r["bic"]       for r in gmm_sweep]
    sil_gm = [r["silhouette"]for r in gmm_sweep]
    ari_gm = [r["ari"]       for r in gmm_sweep]

    # Panel 1: Inertia & BIC (elbow)
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(ks,  inert, "o-", color=DENIM,  label="K-Means inertia")
    ax2.plot(ns, bic,   "s--", color=AMBER, label="GMM BIC")
    ax.axvline(7, color=SIENNA, ls=":", lw=1.2, label="k=7 (n_labels)")
    ax.set_xlabel("k / n"); ax.set_ylabel("Inertia",     color=DENIM)
    ax2.set_ylabel("BIC",  color=AMBER)
    ax.set_title("Elbow / BIC", color=DENIM_DARK, fontweight="bold", fontsize=10)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, lab1+lab2, fontsize=7, loc="upper right")

    # Panel 2: Silhouette
    ax = axes[1]
    ax.plot(ks, sil_km, "o-", color=DENIM, label="K-Means")
    ax.plot(ns, sil_gm, "s--", color=AMBER, label="GMM")
    ax.axvline(7, color=SIENNA, ls=":", lw=1.2)
    ax.set_xlabel("k / n"); ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette", color=DENIM_DARK, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)

    # Panel 3: ARI
    ax = axes[2]
    ax.plot(ks, ari_km, "o-", color=DENIM, label="K-Means")
    ax.plot(ns, ari_gm, "s--", color=AMBER, label="GMM")
    ax.axvline(7, color=SIENNA, ls=":", lw=1.2)
    ax.set_xlabel("k / n"); ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("ARI vs Cover Type", color=DENIM_DARK, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return _savefig(fig, "fig_clustering_sweep.pdf")


# ── FIG 3: Contingency heatmaps ───────────────────────────────────────────────

def fig_contingency(km_final, gmm_final):
    apply_denim_theme()
    fig, axes = denim_fig(1, 2, figsize=(13, 4.5))

    for ax, rec, title in [
        (axes[0], km_final,  "K-Means (k=7) × Cover Type"),
        (axes[1], gmm_final, "GMM (n=7) × Cover Type"),
    ]:
        ct  = np.array(rec["contingency"])  # (k, 7)
        row_sums = ct.sum(axis=1, keepdims=True)
        ct_norm  = ct / (row_sums + 1e-9)

        cmap = sns.color_palette("Blues", as_cmap=True)
        sns.heatmap(ct_norm, ax=ax, cmap=cmap, annot=True, fmt=".2f",
                    linewidths=0.4, linecolor=GRID_CLR,
                    xticklabels=[f"CT{i+1}" for i in range(7)],
                    yticklabels=[f"Cl{j+1}" for j in range(ct.shape[0])],
                    annot_kws={"size": 7})
        _autocontrast_heatmap(ax)
        ax.set_title(title, color=DENIM_DARK, fontweight="bold", fontsize=10)
        ax.set_xlabel("Cover Type"); ax.set_ylabel("Cluster")

    fig.tight_layout()
    return _savefig(fig, "fig_contingency.pdf")


# ── FIG 4: PCA explained variance ────────────────────────────────────────────

def fig_pca_variance(evr, cum_var, pca_n_final):
    apply_denim_theme()
    fig, axes = denim_fig(1, 2, figsize=(10, 3.8))

    comps = np.arange(1, len(evr)+1)

    ax = axes[0]
    ax.bar(comps[:20], evr[:20], color=DENIM, edgecolor="none", width=0.75)
    ax.set_xlabel("Component"); ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA — Individual EVR (top 20)", color=DENIM_DARK, fontweight="bold")

    ax = axes[1]
    ax.plot(comps, cum_var, color=DENIM, lw=1.8)
    ax.axhline(0.90, color=AMBER, ls="--", lw=1.2, label="90%")
    ax.axhline(0.95, color=SIENNA, ls="--", lw=1.2, label="95%")
    ax.axvline(pca_n_final, color=PLUM, ls=":", lw=1.4,
               label=f"n={pca_n_final}")
    ax.set_xlabel("n Components"); ax.set_ylabel("Cumulative EVR")
    ax.set_title("PCA — Cumulative Variance", color=DENIM_DARK, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(1, len(evr))

    fig.tight_layout()
    return _savefig(fig, "fig_pca_variance.pdf")


# ── FIG 5: ICA kurtosis ───────────────────────────────────────────────────────

def fig_ica_kurtosis(kurt_vals, seed_sensitivity):
    apply_denim_theme()
    fig, axes = denim_fig(1, 2, figsize=(10, 3.8))

    n = len(kurt_vals)
    comps = np.arange(1, n+1)
    colors = [DENIM if k > 0 else SIENNA for k in kurt_vals]

    ax = axes[0]
    ax.bar(comps, np.abs(kurt_vals), color=colors, edgecolor="none", width=0.7)
    ax.set_xlabel("ICA Component"); ax.set_ylabel("|Kurtosis| (excess)")
    ax.set_title("ICA — Non-Gaussianity per Component",
                 color=DENIM_DARK, fontweight="bold")
    ax.axhline(3.0, color=AMBER, ls="--", lw=1.2, label="kurtosis=3 threshold")
    ax.legend(fontsize=8)

    ax = axes[1]
    seeds = seed_sensitivity["seeds"]
    mk    = seed_sensitivity["mean_abs_kurtosis_per_seed"]
    ax.plot(seeds, mk, "o-", color=PLUM, lw=1.8, ms=6)
    ax.set_xlabel("Seed"); ax.set_ylabel("Mean |Kurtosis| across components")
    ax.set_title("ICA — Seed Sensitivity", color=DENIM_DARK, fontweight="bold")

    fig.tight_layout()
    return _savefig(fig, "fig_ica_kurtosis.pdf")


# ── FIG 6: RP distance preservation ──────────────────────────────────────────

def fig_rp_distortion(rp_results):
    apply_denim_theme()
    fig, ax = denim_fig(figsize=(7, 3.8))

    seeds     = [r["seed"]            for r in rp_results]
    means     = [r["dist_ratio_mean"] for r in rp_results]
    stds      = [r["dist_ratio_std"]  for r in rp_results]
    p5        = [r["dist_ratio_p5"]   for r in rp_results]
    p95       = [r["dist_ratio_p95"]  for r in rp_results]

    xs = np.arange(len(seeds))
    ax.bar(xs, means, color=TEAL, edgecolor="none", width=0.5, label="Mean ratio")
    ax.errorbar(xs, means, yerr=stds, fmt="none", color=DENIM_DARK, capsize=4, lw=1.5)
    ax.plot(xs, p5,  "v", color=SIENNA, ms=6, label="5th pct")
    ax.plot(xs, p95, "^", color=AMBER,  ms=6, label="95th pct")
    ax.axhline(1.0, color=DENIM, ls="--", lw=1.2, label="Perfect preservation")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s={s}" for s in seeds], fontsize=8)
    ax.set_ylabel("D_proj / D_orig (scaled)")
    ax.set_title("RP — Pairwise Distance Preservation (5 seeds)",
                 color=DENIM_DARK, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _savefig(fig, "fig_rp_distortion.pdf")


# ── FIG 7: DR + Clustering comparison ────────────────────────────────────────

def fig_dr_clustering_comparison(orig_km, orig_gm,
                                 dr_km_dict, dr_gm_dict):
    """Bar chart: silhouette and ARI for all DR × algo combos."""
    apply_denim_theme()
    fig, axes = denim_fig(1, 2, figsize=(11, 4))

    combos  = ["original", "PCA", "ICA", "RP"]
    keys    = ["original", "pca", "ica", "rp"]
    km_sil  = [orig_km["silhouette"]] + [dr_km_dict[k]["silhouette"] for k in ["pca","ica","rp"]]
    gm_sil  = [orig_gm["silhouette"]] + [dr_gm_dict[k]["silhouette"] for k in ["pca","ica","rp"]]
    km_ari  = [orig_km["ari"]]        + [dr_km_dict[k]["ari"]        for k in ["pca","ica","rp"]]
    gm_ari  = [orig_gm["ari"]]        + [dr_gm_dict[k]["ari"]        for k in ["pca","ica","rp"]]

    x = np.arange(len(combos))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, km_sil, w, color=DENIM, label="K-Means", edgecolor="none")
    ax.bar(x + w/2, gm_sil, w, color=AMBER, label="GMM",     edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(combos)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette: Original vs. DR", color=DENIM_DARK, fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - w/2, km_ari, w, color=DENIM, label="K-Means", edgecolor="none")
    ax.bar(x + w/2, gm_ari, w, color=AMBER, label="GMM",     edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(combos)
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("ARI: Original vs. DR", color=DENIM_DARK, fontweight="bold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return _savefig(fig, "fig_dr_clustering_comparison.pdf")


# ── FIG 8: NN after DR metric table ──────────────────────────────────────────

def fig_nn_dr_comparison(nn_results):
    """Bar chart of Macro-F1 + runtime for original/PCA/ICA/RP."""
    apply_denim_theme()
    fig, axes = denim_fig(1, 2, figsize=(10, 4))

    tags   = ["original", "pca", "ica", "rp"]
    labels = ["Original", "PCA", "ICA", "RP"]
    colors = [DR_COLORS[t] for t in tags]
    f1s    = [nn_results[t]["test_metrics"]["macro_f1"] for t in tags]
    walls  = [nn_results[t]["wall_train_s"] for t in tags]

    x = np.arange(len(tags))
    ax = axes[0]
    bars = ax.bar(x, f1s, color=colors, edgecolor="none", width=0.55)
    for b, v in zip(bars, f1s):
        ax.text(b.get_x()+b.get_width()/2, v+0.003, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8, color=TEXT_DARK)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("NN — Test Macro-F1 by Feature Space",
                 color=DENIM_DARK, fontweight="bold")
    ax.set_ylim(0, max(f1s)*1.10)

    ax = axes[1]
    bars2 = ax.bar(x, walls, color=colors, edgecolor="none", width=0.55)
    for b, v in zip(bars2, walls):
        ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=8, color=TEXT_DARK)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Training Time (s)")
    ax.set_title("NN — Training Runtime by Feature Space",
                 color=DENIM_DARK, fontweight="bold")

    fig.tight_layout()
    return _savefig(fig, "fig_nn_dr_comparison.pdf")


# ── FIG 9: 2D projection (PCA + UMAP) colored by CT and cluster ──────────────

def fig_2d_visualization(X_pca2, y, km_labels, umap_result=None):
    """2D scatter: PCA-2 and (optionally) UMAP-2, colored by CT and cluster."""
    apply_denim_theme()
    n_panels = 4 if umap_result is not None else 2
    fig, axes = denim_fig(1, n_panels, figsize=(5*n_panels, 4.5))
    if n_panels == 2:
        axes = list(axes)

    rng = np.random.RandomState(0)
    idx = rng.choice(len(y), min(3000, len(y)), replace=False)

    def _scatter(ax, Xp, color_ids, cmap_vals, title, cbar_label):
        sc = ax.scatter(Xp[idx, 0], Xp[idx, 1],
                        c=cmap_vals[idx], cmap="tab10",
                        s=4, alpha=0.55, linewidths=0)
        ax.set_title(title, color=DENIM_DARK, fontweight="bold", fontsize=9)
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        ax.set_facecolor(WHITE)
        plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.75)

    _scatter(axes[0], X_pca2, y, y.astype(float),
             "PCA-2 — colored by Cover Type", "Cover Type")
    _scatter(axes[1], X_pca2, km_labels, km_labels.astype(float),
             "PCA-2 — colored by K-Means cluster", "Cluster")

    if umap_result is not None and umap_result[0] is not None:
        Xu = umap_result[0]
        _scatter(axes[2], Xu, y, y.astype(float),
                 "UMAP-2 — colored by Cover Type", "Cover Type")
        _scatter(axes[3], Xu, km_labels, km_labels.astype(float),
                 "UMAP-2 — colored by K-Means cluster", "Cluster")

    fig.tight_layout()
    return _savefig(fig, "fig_2d_visualization.pdf")


# ── FIG 10: NN learning curves (best DR method) ───────────────────────────────

def fig_nn_learning_curves(nn_results):
    apply_denim_theme()
    fig, axes = denim_fig(1, 2, figsize=(10, 4))

    tags   = ["original", "pca", "ica", "rp"]
    labels = ["Original", "PCA", "ICA", "RP"]
    colors = [DR_COLORS[t] for t in tags]

    ax = axes[0]
    for tag, lbl, clr in zip(tags, labels, colors):
        tl = nn_results[tag]["train_loss"]
        ax.plot(range(1, len(tl)+1), tl, color=clr, lw=1.4, label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss — All Feature Spaces",
                 color=DENIM_DARK, fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1]
    for tag, lbl, clr in zip(tags, labels, colors):
        vl = nn_results[tag]["val_loss"]
        ax.plot(range(1, len(vl)+1), vl, color=clr, lw=1.4, label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss — All Feature Spaces",
                 color=DENIM_DARK, fontweight="bold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return _savefig(fig, "fig_nn_learning_curves.pdf")
