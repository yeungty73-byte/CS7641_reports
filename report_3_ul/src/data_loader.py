"""data_loader.py — Load and split Covertype for UL Report.
Identical split strategy to SL/OL Reports: same 20k stratified sample,
same 60/20/20 train/val/test, same seed 7641.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


CONT_COLS = list(range(10))   # cols 0-9: continuous cartographic features
BIN_COLS  = list(range(10, 54))  # cols 10-53: binary wilderness/soil


def load_and_split(cfg, logger=None):
    """Return (X_train, X_val, X_test, y_train, y_val, y_test, feat_names).

    Preprocessing:
    - Continuous cols (0-9): StandardScaler fit on train only.
    - Binary cols (10-53): left as 0/1 (no scaling).
    This matches the decision justified in the SL Report and mirrors the
    covtype_25k.csv.gz already in data/.
    """
    seed        = int(cfg["experiment"]["seed"])
    n_sample    = int(cfg["data"]["sample_size"])   # 20000
    test_frac   = float(cfg["data"]["test_size"])   # 0.20
    val_frac    = float(cfg["data"]["val_size"])     # 0.20

    cache_csv = os.path.join(
        os.path.dirname(__file__), "..", "data", "covtype_25k.csv.gz")

    if os.path.exists(cache_csv):
        df = pd.read_csv(cache_csv)
        if logger: logger.info(f"Loaded covtype from cache: {cache_csv}")
    else:
        # Try inheriting from report_2_ol data dir first
        parent_csv = os.path.join(
            os.path.dirname(__file__), "..", "..", "report_2_ol",
            "data", "covtype_25k.csv.gz")
        if os.path.exists(parent_csv):
            df = pd.read_csv(parent_csv)
        else:
            if logger: logger.info("Fetching Covertype from sklearn...")
            cov = fetch_covtype(as_frame=True)
            df  = cov.frame
        # Stratified subsample to n_sample
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=n_sample, random_state=seed)
        idx, _ = next(sss.split(df, df["Cover_Type"]))
        df = df.iloc[idx].reset_index(drop=True)
        os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
        df.to_csv(cache_csv, index=False, compression="gzip")
        if logger: logger.info(f"Saved covtype sample → {cache_csv}")

    feat_names = [c for c in df.columns if c != "Cover_Type"]
    X = df[feat_names].values.astype(float)
    y = df["Cover_Type"].values.astype(int)   # 1-7

    # --- train / val / test split (same as SL/OL) ----------------------------
    rng = np.random.RandomState(seed)
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(X, y))

    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    val_frac_of_tv = val_frac / (1.0 - test_frac)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_of_tv, random_state=seed)
    train_idx, val_idx = next(sss_val.split(X_tv, y_tv))

    X_train, y_train = X_tv[train_idx], y_tv[train_idx]
    X_val,   y_val   = X_tv[val_idx],   y_tv[val_idx]

    # --- Scale continuous cols fit on train only -----------------------------
    scaler = StandardScaler()
    X_train[:, CONT_COLS] = scaler.fit_transform(X_train[:, CONT_COLS])
    X_val[:,   CONT_COLS] = scaler.transform(X_val[:,   CONT_COLS])
    X_test[:,  CONT_COLS] = scaler.transform(X_test[:,  CONT_COLS])

    if logger:
        logger.info(f"Split sizes — train:{len(y_train)} val:{len(y_val)} "
                    f"test:{len(y_test)}")

    return (X_train, X_val, X_test,
            y_train, y_val,  y_test,
            feat_names, scaler)


def load_full_scaled(cfg, logger=None):
    """Return (X_scaled, y) for exploratory unsupervised analysis.
    Scales all continuous cols using StandardScaler fit on the full sample
    (leakage-safe for unsupervised; downstream NN uses split version).
    """
    (X_tr, X_val, X_te,
     y_tr, y_val, y_te,
     feat_names, scaler) = load_and_split(cfg, logger)

    import numpy as np
    X_full = np.vstack([X_tr, X_val, X_te])
    y_full = np.concatenate([y_tr, y_val, y_te])
    return X_full, y_full, feat_names
