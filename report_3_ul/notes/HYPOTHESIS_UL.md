# HYPOTHESIS_UL.md — Locked Before Running Experiments
# CS7641 UL Report — Timothy Leung (tleung37) | seed 7641
# Date locked: 2026-05-23

## H1 — Clustering Structure (locked)

**Claim:** K-Means will partition the Covertype feature space primarily along
elevation and wilderness-area boundaries rather than ecological cover-type labels,
yielding only moderate ARI (< 0.15) with k = 7.

**Rationale:** Elevation and Hillshade variables dominate Euclidean distances
(range ~2,000 m vs. 0/1 for binary indicators). K-Means minimises variance,
not class separation. GMM/EM with full covariance should achieve slightly higher
NMI than K-Means because Covertype's class distributions are elongated in PCA space.

**Testable:** Compare ARI, NMI, silhouette for K-Means (k=7) vs. GMM (n=7).

---

## H2 — PCA Dominance for NN (locked)

**Claim:** PCA will preserve downstream MLP Macro-F1 within 0.02 of the
original-feature baseline at ≤ 30 components (capturing ≥ 90 % variance),
outperforming ICA and RP on Macro-F1 because Covertype's continuous cartographic
features are strongly correlated and PCA captures that covariance efficiently.
ICA will show higher kurtosis components but lower NN performance than PCA because
it maximises non-Gaussianity rather than class-relevant variance.
RP will show the highest seed-to-seed variability in clustering ARI.

**Testable:** Compare {PCA, ICA, RP} × NN Macro-F1, and compare RP clustering
ARI standard deviation across ≥ 5 seeds.

---

## H3 — Dimensionality Reduction + Clustering (locked)

**Claim:** PCA will modestly improve K-Means silhouette over original-space by
removing sparse binary indicator noise, while ICA will produce less stable clusters
across seeds. GMM AIC/BIC will favour fewer components in the reduced spaces
(≤ 10) because DR compresses the effective rank of the covariance matrix.

**Testable:** Compare silhouette and ARI before/after each DR method.
