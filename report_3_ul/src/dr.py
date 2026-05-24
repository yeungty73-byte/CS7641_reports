"""dr.py — Dimensionality reduction: PCA, ICA, RP (+ optional UMAP).

For exploratory analysis (clustering), fit on full working sample.
For downstream NN, fit on X_train only (leakage-safe).
"""
import time
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis


# ── PCA ──────────────────────────────────────────────────────────────────────

def fit_pca_full(X, seed, logger=None):
    """Fit PCA on X (full), return (pca, explained_var_ratio, cum_var)."""
    pca = PCA(random_state=seed)
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    if logger:
        n90 = int(np.searchsorted(cum, 0.90)) + 1
        n95 = int(np.searchsorted(cum, 0.95)) + 1
        logger.info(f"PCA: n90%={n90}  n95%={n95}  "
                    f"top-1={evr[0]:.3f}  eff_rank~{1/np.sum(evr**2):.1f}")
    return pca, evr.tolist(), cum.tolist()


def pca_diagnostics(X, seed, n_components_list):
    """Reconstruction error at various component counts."""
    pca_full = PCA(random_state=seed)
    pca_full.fit(X)
    out = {}
    for n in n_components_list:
        Xr = pca_full.inverse_transform(
             pca_full.transform(X)[:, :n] @
             np.eye(pca_full.n_components_)[:n, :])
        # use explicit truncated reconstruction
        pca_n = PCA(n_components=n, random_state=seed)
        Xt = pca_n.fit_transform(X)
        Xrec = pca_n.inverse_transform(Xt)
        rmse = float(np.sqrt(np.mean((X - Xrec) ** 2)))
        out[str(n)] = {"rmse": rmse,
                       "cum_var": float(np.sum(
                           pca_full.explained_variance_ratio_[:n]))}
    return out


def apply_pca(X, n, seed, pca_fitted=None):
    """Fit PCA(n) on X (or reuse fitted) and return transformed X."""
    if pca_fitted is None:
        pca = PCA(n_components=n, random_state=seed)
        pca.fit(X)
    else:
        pca = pca_fitted
    return pca.transform(X), pca


# ── ICA ──────────────────────────────────────────────────────────────────────

def fit_ica(X, n, seed, logger=None):
    """Fit FastICA on X; return (ica, transformed X, kurtosis per component)."""
    # Whiten first (recommended) — PCA-whiten then ICA
    t0 = time.perf_counter()
    ica = FastICA(n_components=n, random_state=seed,
                  max_iter=500, tol=1e-4, whiten="unit-variance")
    Xt = ica.fit_transform(X)
    wall = time.perf_counter() - t0
    kurt = [float(kurtosis(Xt[:, i], fisher=True)) for i in range(n)]
    if logger:
        logger.info(f"ICA n={n}  converged  wall={wall:.1f}s  "
                    f"mean_kurtosis={np.mean(np.abs(kurt)):.2f}")
    return ica, Xt, kurt, wall


def ica_seed_sensitivity(X, n, seeds, logger=None):
    """Run ICA across multiple seeds; return kurtosis variability."""
    kurt_matrix = []
    for s in seeds:
        ica = FastICA(n_components=n, random_state=s,
                      max_iter=500, tol=1e-4, whiten="unit-variance")
        try:
            Xt = ica.fit_transform(X)
            kurt = [float(kurtosis(Xt[:, i], fisher=True)) for i in range(n)]
        except Exception:
            kurt = [0.0] * n
        kurt_matrix.append(kurt)
    kmat = np.array(kurt_matrix)  # (n_seeds, n_comp)
    return {
        "seeds": seeds,
        "mean_abs_kurtosis_per_seed": np.mean(np.abs(kmat), axis=1).tolist(),
        "std_kurtosis_across_seeds":  float(np.std(kmat)),
    }


# ── Randomized Projections ────────────────────────────────────────────────────

def run_rp_sweep(X, n, seeds, logger=None):
    """Run RP across multiple seeds; evaluate pairwise-distance preservation."""
    rng = np.random.RandomState(0)
    idx_sample = rng.choice(len(X), min(500, len(X)), replace=False)
    X_sub = X[idx_sample]
    from sklearn.metrics import pairwise_distances
    D_orig = pairwise_distances(X_sub, metric="euclidean")

    results = []
    for s in seeds:
        t0 = time.perf_counter()
        rp = GaussianRandomProjection(n_components=n, random_state=s)
        Xt = rp.fit_transform(X)
        wall = time.perf_counter() - t0
        Xt_sub = Xt[idx_sample]
        D_proj = pairwise_distances(Xt_sub, metric="euclidean")

        # JL distortion: max ratio deviation from 1.0
        scale = np.mean(D_proj) / (np.mean(D_orig) + 1e-9)
        D_ratio = D_proj / (D_orig * scale + 1e-9)
        upper = np.triu_indices_from(D_ratio, k=1)
        ratios = D_ratio[upper]
        results.append({
            "seed": s,
            "wall_s": wall,
            "dist_ratio_mean": float(np.mean(ratios)),
            "dist_ratio_std":  float(np.std(ratios)),
            "dist_ratio_p5":   float(np.percentile(ratios, 5)),
            "dist_ratio_p95":  float(np.percentile(ratios, 95)),
        })
        if logger:
            logger.info(f"  RP seed={s}  ratio_mean={results[-1]['dist_ratio_mean']:.3f}"
                        f"  ±{results[-1]['dist_ratio_std']:.3f}")
    return results


def apply_rp(X, n, seed):
    """Fit RP on X and return (transformed X, fitted rp)."""
    rp = GaussianRandomProjection(n_components=n, random_state=seed)
    Xt = rp.fit_transform(X)
    return Xt, rp


# ── UMAP (extra credit, visualization only) ───────────────────────────────────

def apply_umap(X, n_components, n_neighbors, min_dist, seed, logger=None):
    """Fit UMAP on full X for visualization (extra credit).
    Not used for downstream NN — visualization only.
    """
    try:
        import umap
        t0 = time.perf_counter()
        um = umap.UMAP(n_components=n_components,
                       n_neighbors=n_neighbors,
                       min_dist=min_dist,
                       random_state=seed)
        Xt = um.fit_transform(X)
        wall = time.perf_counter() - t0
        if logger:
            logger.info(f"UMAP n={n_components}  wall={wall:.1f}s")
        return Xt, um, wall
    except ImportError:
        if logger: logger.warning("umap-learn not installed; skipping UMAP.")
        return None, None, 0.0
