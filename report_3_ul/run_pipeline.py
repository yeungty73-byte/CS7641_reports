#!/usr/bin/env python3
"""run_pipeline.py — CS7641 UL Report full pipeline.

Runs in order:
  1. Load data + screening
  2. Part 1: Clustering on original features (KMeans + GMM sweep + final)
  3. Part 2: DR — PCA full diagnostics, ICA kurtosis + seed sensitivity,
             RP distance preservation across 5 seeds
  4. Part 3: Clustering after PCA / ICA / RP
  5. Part 4: NN after DR (leakage-safe: DR fit on train only)
  Extra credit: UMAP (visualization only)
  6. Generate all figures + save JSONs

Usage:
    python run_pipeline.py [--no-umap] [--fast]
"""
import os, sys, json, time, logging, argparse
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data_loader      import load_and_split, load_full_scaled
from src.clustering       import (run_kmeans_sweep, run_kmeans_final,
                                   run_gmm_sweep,   run_gmm_final)
from src.dr               import (fit_pca_full, pca_diagnostics, apply_pca,
                                   fit_ica, ica_seed_sensitivity,
                                   run_rp_sweep, apply_rp, apply_umap)
from src.nn_after_dr      import run_nn_after_dr
import src.report_generator as rg

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(ROOT, "results", "logs", "pipeline.log"),
                            mode="w"),
    ]
)
log = logging.getLogger(__name__)


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    log.info(f"  JSON → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-umap", action="store_true",
                        help="Skip UMAP extra credit")
    parser.add_argument("--fast", action="store_true",
                        help="Smaller sweeps for quick dev test")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg_path = os.path.join(ROOT, "configs", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed      = int(cfg["experiment"]["seed"])
    seeds_rp  = cfg["experiment"]["seeds_rp"]
    LOGDIR    = os.path.join(ROOT, cfg["paths"]["logs"])
    FIGDIR    = os.path.join(ROOT, cfg["paths"]["figures"])
    TABDIR    = os.path.join(ROOT, cfg["paths"]["tables"])
    for d in [LOGDIR, FIGDIR, TABDIR]:
        os.makedirs(d, exist_ok=True)

    rg.FIGDIR = FIGDIR

    if args.fast:
        cfg["clustering"]["k_range"]    = [3, 5, 7, 10]
        cfg["clustering"]["gmm_n_range"] = [3, 5, 7, 10]
        seeds_rp = [seed, 42]
        log.info("FAST mode: reduced sweeps")

    # ── 0. Load data ──────────────────────────────────────────────────────────
    log.info("=== LOADING DATA ===")
    (X_train, X_val, X_test,
     y_train, y_val,  y_test,
     feat_names, scaler) = load_and_split(cfg, log)
    X_full, y_full, _ = load_full_scaled(cfg, log)

    log.info(f"Shapes — train:{X_train.shape} val:{X_val.shape} "
             f"test:{X_test.shape} full:{X_full.shape}")

    # EDA / class distribution
    rg.fig_class_distribution(y_full)
    _save_json({"n_train": len(y_train), "n_val": len(y_val),
                "n_test": len(y_test),  "n_full": len(y_full),
                "n_features": X_full.shape[1],
                "class_counts": {int(c): int(np.sum(y_full==c))
                                 for c in range(1, 8)}},
               os.path.join(LOGDIR, "eda_summary.json"))

    # ── 1. PART 1: Clustering on original features ────────────────────────────
    log.info("=== PART 1: CLUSTERING (ORIGINAL) ===")
    k_range   = cfg["clustering"]["k_range"]
    gmm_range = cfg["clustering"]["gmm_n_range"]
    k_final   = int(cfg["clustering"]["k_final"])
    gmm_n_fin = int(cfg["clustering"]["gmm_n_final"])
    cov_type  = cfg["clustering"]["gmm_covariance"]

    log.info("K-Means sweep...")
    km_sweep = run_kmeans_sweep(X_full, y_full, k_range, seed, log)
    _save_json(km_sweep, os.path.join(LOGDIR, "kmeans_sweep.json"))

    log.info(f"K-Means final (k={k_final})...")
    km_final, km_labels, km_model = run_kmeans_final(
        X_full, y_full, k_final, seed, log)
    _save_json(km_final, os.path.join(LOGDIR, "kmeans_final.json"))

    log.info("GMM sweep...")
    gmm_sweep = run_gmm_sweep(X_full, y_full, gmm_range, cov_type, seed, log)
    _save_json(gmm_sweep, os.path.join(LOGDIR, "gmm_sweep.json"))

    log.info(f"GMM final (n={gmm_n_fin})...")
    gmm_final, gmm_labels, gmm_model = run_gmm_final(
        X_full, y_full, gmm_n_fin, cov_type, seed, log)
    _save_json(gmm_final, os.path.join(LOGDIR, "gmm_final.json"))

    rg.fig_clustering_sweep(km_sweep, gmm_sweep)
    rg.fig_contingency(km_final, gmm_final)

    # ── 2. PART 2: Dimensionality Reduction ───────────────────────────────────
    log.info("=== PART 2: DIMENSIONALITY REDUCTION ===")

    # PCA
    log.info("PCA full diagnostic...")
    pca_full, evr, cum_var = fit_pca_full(X_full, seed, log)
    pca_n_final = int(cfg["dr"]["pca_n_final"])

    diag_ns = [5, 10, 15, 20, 25, 30, 40, 54]
    pca_diag = pca_diagnostics(X_full, seed, diag_ns)
    _save_json({"evr": evr, "cum_var": cum_var, "diagnostics": pca_diag,
                "n_final": pca_n_final},
               os.path.join(LOGDIR, "pca_diagnostics.json"))
    rg.fig_pca_variance(evr, cum_var, pca_n_final)

    X_pca_full, _   = apply_pca(X_full, pca_n_final, seed)
    X_pca2,    _    = apply_pca(X_full, 2, seed)   # for 2D viz

    # ICA
    ica_n_final = int(cfg["dr"]["ica_n_final"])
    log.info(f"ICA n={ica_n_final}...")
    ica_model, X_ica_full, kurt_vals, ica_wall = fit_ica(X_full, ica_n_final, seed, log)
    ica_seeds = [seed, 42, 123, 7, 2024]
    ica_sens  = ica_seed_sensitivity(X_full, ica_n_final, ica_seeds, log)
    _save_json({"n_components": ica_n_final, "kurtosis": kurt_vals,
                "seed_sensitivity": ica_sens, "wall_s": ica_wall},
               os.path.join(LOGDIR, "ica_diagnostics.json"))
    rg.fig_ica_kurtosis(kurt_vals, ica_sens)

    # RP
    rp_n_final = int(cfg["dr"]["rp_n_final"])
    log.info(f"RP distance preservation (n={rp_n_final}, seeds={seeds_rp})...")
    rp_results = run_rp_sweep(X_full, rp_n_final, seeds_rp, log)
    _save_json({"n_components": rp_n_final, "results": rp_results},
               os.path.join(LOGDIR, "rp_diagnostics.json"))
    rg.fig_rp_distortion(rp_results)
    X_rp_full, _ = apply_rp(X_full, rp_n_final, seed)

    # UMAP (extra credit)
    umap_result = (None, None, 0.0)
    if not args.no_umap:
        log.info("UMAP (extra credit, visualization only)...")
        umap_n    = int(cfg["dr"]["umap_n_components"])
        umap_nn   = int(cfg["dr"]["umap_n_neighbors"])
        umap_md   = float(cfg["dr"]["umap_min_dist"])
        umap_result = apply_umap(X_full, umap_n, umap_nn, umap_md, seed, log)
        if umap_result[0] is not None:
            _save_json({"n_components": umap_n, "wall_s": umap_result[2],
                        "n_neighbors": umap_nn, "min_dist": umap_md},
                       os.path.join(LOGDIR, "umap_diagnostics.json"))

    # 2D visualization
    rg.fig_2d_visualization(X_pca2, y_full, km_labels, umap_result)

    # ── 3. PART 3: Clustering after DR ────────────────────────────────────────
    log.info("=== PART 3: CLUSTERING AFTER DR ===")
    dr_km_dict  = {}
    dr_gm_dict  = {}

    for dr_name, X_dr in [("pca", X_pca_full),
                            ("ica", X_ica_full),
                            ("rp",  X_rp_full)]:
        log.info(f"  K-Means after {dr_name.upper()}...")
        rec_km, _, _ = run_kmeans_final(X_dr, y_full, k_final, seed, log)
        dr_km_dict[dr_name] = rec_km
        _save_json(rec_km, os.path.join(LOGDIR, f"kmeans_after_{dr_name}.json"))

        log.info(f"  GMM after {dr_name.upper()}...")
        rec_gm, _, _ = run_gmm_final(X_dr, y_full, gmm_n_fin, cov_type, seed, log)
        dr_gm_dict[dr_name] = rec_gm
        _save_json(rec_gm, os.path.join(LOGDIR, f"gmm_after_{dr_name}.json"))

    rg.fig_dr_clustering_comparison(km_final, gmm_final, dr_km_dict, dr_gm_dict)

    # ── 4. PART 4: NN after DR (leakage-safe) ─────────────────────────────────
    log.info("=== PART 4: NN AFTER DR ===")
    nn_results = run_nn_after_dr(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        cfg, log)
    _save_json(nn_results, os.path.join(LOGDIR, "nn_after_dr.json"))

    rg.fig_nn_dr_comparison(nn_results)
    rg.fig_nn_learning_curves(nn_results)

    # ── Summary table ─────────────────────────────────────────────────────────
    import pandas as pd
    rows = []
    for tag in ["original", "pca", "ica", "rp"]:
        r = nn_results[tag]
        rows.append({
            "Features":    tag,
            "n_dim":       r["in_dim"],
            "Macro-F1":    round(r["test_metrics"]["macro_f1"],   4),
            "Accuracy":    round(r["test_metrics"]["accuracy"],   4),
            "Bal. Acc.":   round(r["test_metrics"]["balanced_acc"], 4),
            "Train time(s)": round(r["wall_train_s"], 1),
            "Epochs":      r["n_epochs_run"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABDIR, "nn_dr_summary.csv"), index=False)
    log.info("Saved nn_dr_summary.csv")

    # Clustering summary
    clust_rows = []
    for dr_name, km_rec, gm_rec in [
        ("original", km_final, gmm_final),
        ("pca",   dr_km_dict["pca"],  dr_gm_dict["pca"]),
        ("ica",   dr_km_dict["ica"],  dr_gm_dict["ica"]),
        ("rp",    dr_km_dict["rp"],   dr_gm_dict["rp"]),
    ]:
        clust_rows.append({
            "Space": dr_name, "Algo": "KMeans",
            "Sil": round(km_rec["silhouette"], 4),
            "DB":  round(km_rec["davies_bouldin"], 4),
            "ARI": round(km_rec["ari"], 4),
            "NMI": round(km_rec["nmi"], 4),
        })
        clust_rows.append({
            "Space": dr_name, "Algo": "GMM",
            "Sil": round(gm_rec["silhouette"], 4),
            "DB":  round(gm_rec["davies_bouldin"], 4),
            "ARI": round(gm_rec["ari"], 4),
            "NMI": round(gm_rec["nmi"], 4),
        })
    pd.DataFrame(clust_rows).to_csv(
        os.path.join(TABDIR, "clustering_summary.csv"), index=False)
    log.info("Saved clustering_summary.csv")

    log.info("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
