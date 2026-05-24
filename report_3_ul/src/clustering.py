"""clustering.py — K-Means and GMM/EM on original and DR-reduced features.

Part 1: Clustering on original feature space.
Part 3: Clustering after PCA / ICA / RP.
"""
import json
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
)


def _internal_metrics(X, labels):
    """Compute internal clustering quality metrics."""
    # subsample for silhouette (expensive at n=20k)
    rng = np.random.RandomState(0)
    n_sil = min(3000, len(labels))
    idx = rng.choice(len(labels), n_sil, replace=False)
    sil = float(silhouette_score(X[idx], labels[idx]))
    db  = float(davies_bouldin_score(X, labels))
    ch  = float(calinski_harabasz_score(X, labels))
    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}


def _external_metrics(y_true, labels):
    """Label-alignment metrics (computed after fitting)."""
    return {
        "ari":         float(adjusted_rand_score(y_true, labels)),
        "nmi":         float(normalized_mutual_info_score(y_true, labels)),
        "homogeneity": float(homogeneity_score(y_true, labels)),
        "completeness":float(completeness_score(y_true, labels)),
        "v_measure":   float(v_measure_score(y_true, labels)),
    }


def _contingency(y_true, labels, k):
    """Cluster × Cover-Type contingency table as nested list."""
    ct = np.zeros((k, 7), dtype=int)
    for lb, gt in zip(labels, y_true):
        ct[int(lb), int(gt)-1] += 1
    return ct.tolist()


def run_kmeans_sweep(X, y, k_range, seed, logger=None):
    """Sweep k values; return list of result dicts."""
    results = []
    for k in k_range:
        t0 = time.perf_counter()
        km = KMeans(n_clusters=k, n_init=20, max_iter=500,
                    random_state=seed)
        labels = km.fit_predict(X)
        wall   = time.perf_counter() - t0

        rec = {"k": k,
               "inertia": float(km.inertia_),
               "wall_s":  wall}
        rec.update(_internal_metrics(X, labels))
        rec.update(_external_metrics(y, labels))
        if logger: logger.info(f"  KMeans k={k}  sil={rec['silhouette']:.4f}  "
                               f"ari={rec['ari']:.4f}")
        results.append(rec)
    return results


def run_kmeans_final(X, y, k, seed, logger=None):
    """Run final K-Means and return full result dict with labels."""
    t0 = time.perf_counter()
    km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=seed)
    labels = km.fit_predict(X)
    wall   = time.perf_counter() - t0

    rec = {"k": k, "inertia": float(km.inertia_), "wall_s": wall,
           "cluster_sizes": [int(x) for x in np.bincount(labels)],
           "contingency": _contingency(y, labels, k)}
    rec.update(_internal_metrics(X, labels))
    rec.update(_external_metrics(y, labels))
    return rec, labels, km


def run_gmm_sweep(X, y, n_range, cov_type, seed, logger=None):
    """Sweep component counts; return list of result dicts."""
    results = []
    for n in n_range:
        t0 = time.perf_counter()
        gm = GaussianMixture(n_components=n, covariance_type=cov_type,
                             n_init=5, max_iter=200, random_state=seed)
        gm.fit(X)
        labels = gm.predict(X)
        wall   = time.perf_counter() - t0

        rec = {"n": n,
               "aic":    float(gm.aic(X)),
               "bic":    float(gm.bic(X)),
               "log_likelihood": float(gm.score(X) * len(X)),
               "converged": bool(gm.converged_),
               "wall_s": wall}
        rec.update(_internal_metrics(X, labels))
        rec.update(_external_metrics(y, labels))
        if logger: logger.info(f"  GMM n={n}  bic={rec['bic']:.1f}  "
                               f"ari={rec['ari']:.4f}")
        results.append(rec)
    return results


def run_gmm_final(X, y, n, cov_type, seed, logger=None):
    """Run final GMM and return full result dict with labels."""
    t0 = time.perf_counter()
    gm = GaussianMixture(n_components=n, covariance_type=cov_type,
                         n_init=5, max_iter=200, random_state=seed)
    gm.fit(X)
    labels = gm.predict(X)
    wall   = time.perf_counter() - t0

    rec = {"n": n,
           "aic": float(gm.aic(X)),
           "bic": float(gm.bic(X)),
           "log_likelihood": float(gm.score(X) * len(X)),
           "converged": bool(gm.converged_),
           "wall_s": wall,
           "cluster_sizes": [int(x) for x in np.bincount(labels)],
           "contingency": _contingency(y, labels, n)}
    rec.update(_internal_metrics(X, labels))
    rec.update(_external_metrics(y, labels))
    return rec, labels, gm
