# CS7641 OL Report — Timothy Leung (tleung37)
## Optimization & Uncertainty in Learning | Summer 2026

**Dataset:** UCI Forest Cover Type (Covertype) — same stratified 20k subsample as the SL Report (seed 7641, 60/20/20 stratified split).  
**Backbone:** PyTorch MLP from the SL Report, fixed. FC(256)→BN→ReLU→FC(128)→BN→ReLU→FC(7), He init, SGD-only in SL, freely optimized here.  
**Goal of this codebase:** reproduce Parts 1–3 (RO, Adam ablations, regularization) plus the R audit of the Python results — end-to-end from raw data to bound PDFs.

---

## Layout

```
report_2_ol/
├── requirements.txt              # pinned Python deps (torch 2.3.1, sklearn 1.5.0, …)
├── configs/
│   └── config.yaml               # single source of truth: seeds, HP grids, paths
├── notes/
│   └── HYPOTHESIS_OL.md          # H1–H3 locked before any Part 2/3 experiment ran
├── src/
│   ├── __init__.py
│   ├── styles.py                 # denim palette + seaborn whitegrid theme
│   ├── data_loader.py            # Covertype fetch/cache, stratified splits, DataLoaders
│   ├── data_screening.py         # EDA confirmation, class distribution
│   ├── stat_learners.py          # CovertypeNet backbone, freeze/unfreeze, metrics
│   ├── optimizer.py              # Parts 1–3: RHC/SA/GA, all Adam variants, reg study
│   └── report_generator.py       # all denim-themed figures and tables (PDF output)
├── run_pipeline.py               # main entry; runs all four parts in order
├── data/
│   └── README.md                 # Covertype fetched via sklearn; see run_pipeline.py
├── results/
│   ├── figures/                  # PDF figures consumed by report/OL_Report.tex
│   ├── tables/                   # CSV ground truth (final_metric_table.csv)
│   └── logs/                     # JSON provenance: one file per run × seed
├── report/
│   ├── OL_Report.tex             # IEEE-conference, denim aesthetic, hypothesis-first
│   └── OL_Report.pdf             # 6 pages, compiled with pdflatex
├── repro/
│   ├── REPRO_OL.tex              # reproducibility sheet
│   └── REPRO_OL.pdf
└── rsanity/
    └── sanity_check_OL.Rmd       # bookdown::pdf_document2, denim-themed
                                  # R nnet baseline vs Python SL SGD + log audit
```

---

## How to re-run end-to-end

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Run all four experimental parts (~35–45 min on CPU; ~8 min with CUDA)
python run_pipeline.py

# 3. Re-compile the IEEE report PDF
cd report && pdflatex OL_Report.tex && pdflatex OL_Report.tex

# 4. Re-knit the R sanity check (~5–8 min)
cd ../rsanity && Rscript -e 'bookdown::render_book("sanity_check_OL.Rmd")'
```

`python run_pipeline.py` covers the full sequence: SL baseline → Part 1 (RHC/SA/GA) → Part 2 (7 Adam variants × 3 seeds + heatmap) → Part 3 (9 reg configs × 3 seeds) → all figures.

---

## Reproducibility

- All randomness keyed to `seed: 7641` (course number, same as SL Report) in `configs/config.yaml`.
- Multi-seed stability runs use seeds `{42, 123, 7}`.
- Hypotheses H1–H3 locked in `notes/HYPOTHESIS_OL.md` before any Part 2 or Part 3 experiment ran.
- Covertype sample, split, preprocessing, and backbone are **identical** to the SL Report — no changes, no corrections needed.
- The R sanity check (`rsanity/`) loads the same split at the same seed and compares `nnet` (BFGS) against the Python SL SGD baseline; a gap of 0.03–0.10 is the known BFGS-vs-SGD library difference, consistent with the SL sanity check's NN gap of −0.108.

---

## What to wire up before submission

1. **Enterprise GT GitHub repo** — push `report_2_ol/` minus `.cache/` (trivially regenerable via sklearn).
2. **Enterprise GT Overleaf project** — upload `report/OL_Report.tex` plus all PDFs in `results/figures/`. The `.tex` expects figures at `../results/figures/`; mirror the same relative path in Overleaf or update the `\includegraphics` lines if Overleaf flattens the tree.
3. **Fill in the placeholders in `repro/REPRO_OL.tex`:**
   - `PLACEHOLDER_OVERLEAF_LINK` → the read-only Overleaf share URL
   - `PLACEHOLDER_COMMIT_HASH` → the final `git rev-parse HEAD` SHA

---

## Headline numbers (test set, n=4,000, seed=42)

| Part | Method | Val Loss | Acc | Macro-F1 | Bal. Acc |
|---|---|---|---|---|---|
| SL baseline | PyTorch SGD (no mom.) | 0.538 | 0.753 | 0.590 | 0.526 |
| Part 1 | RHC (best RO) | 0.563 | — | 0.594 | 0.530 |
| Part 2 | Adam | 0.489 | 0.787 | 0.657 | — |
| Part 2 | AdamW | 0.495 | 0.775 | **0.671** | — |
| Part 3 | L2 λ=1e-4 (best single) | 0.497 | 0.779 | 0.653 | — |
| Part 3 | Best combo | 0.511 | 0.760 | 0.653 | **0.588** |

R sanity check confirms Python pipeline: `nnet` (BFGS) vs. SL SGD baseline within the expected library-difference band; all 34 Part 2/3 JSON logs pass internal consistency checks (Macro-F1 ∈ [0,1], seed IQR < 0.015).
