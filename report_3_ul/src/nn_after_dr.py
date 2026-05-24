"""nn_after_dr.py — Rerun SL/OL best MLP on DR-reduced features.

Best justified recipe from SL+OL:
  - Architecture: [256, 128] hidden, ReLU, BatchNorm (SL best NN)
  - Optimizer: Adam no-bias-correction (OL Part2 best: macro_f1=0.6695)
  - Regularization: L2 weight_decay=1e-4 (OL Part3 best single regularizer)
  - 40 epochs, batch 256, lr=1e-3
  - Early stopping patience=10 (OL Part3)
  - Leakage-safe: DR fit on X_train only
"""
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler


N_CLASSES = 7


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128), n_classes=7):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)
        # Kaiming init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class AdamNoBias(optim.Adam):
    """Adam without bias-correction (matches OL best optimizer)."""
    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg']    = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    b1, b2 = group['betas']
                    state['step'] += 1
                    exp_avg.mul_(b1).add_(grad, alpha=1 - b1)
                    exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    # No bias correction — raw m/v
                    step_size = group['lr']
                    if group['weight_decay'] != 0:
                        p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return None


def _to_tensor(X, y=None, device="cpu"):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    if y is not None:
        yt = torch.tensor(y - 1, dtype=torch.long, device=device)  # 1-7 → 0-6
        return Xt, yt
    return Xt


def train_eval_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
                   cfg, tag="original", logger=None):
    """Train MLP and return results dict."""
    seed   = int(cfg["experiment"]["seed"])
    epochs = int(cfg["training"]["n_epochs"])
    bs     = int(cfg["training"]["batch_size"])
    lr     = float(cfg["training"]["lr"])
    wd     = float(cfg["training"]["weight_decay"])
    pat    = int(cfg["training"]["early_stopping_patience"])

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    in_dim = X_train.shape[1]
    model  = MLP(in_dim, hidden=(256, 128), n_classes=N_CLASSES).to(device)
    opt    = AdamNoBias(model.parameters(), lr=lr, weight_decay=wd)
    crit   = nn.CrossEntropyLoss()

    Xtr, ytr = _to_tensor(X_train, y_train, device)
    Xvl, yvl = _to_tensor(X_val,   y_val,   device)
    Xte, yte  = _to_tensor(X_test,  y_test,  device)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)

    best_val_loss = float("inf")
    best_weights  = None
    patience_ctr  = 0
    train_losses, val_losses = [], []
    t_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for Xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(yb)
        ep_loss /= len(ytr)

        model.eval()
        with torch.no_grad():
            vl = crit(model(Xvl), yvl).item()
        train_losses.append(ep_loss)
        val_losses.append(vl)

        if vl < best_val_loss - 1e-5:
            best_val_loss = vl
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= pat:
                if logger: logger.info(f"  [{tag}] early-stop ep={ep}")
                break

    wall_train = time.perf_counter() - t_start

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        preds = model(Xte).argmax(dim=1).cpu().numpy() + 1  # back to 1-7
        wall_pred = time.perf_counter() - t0

    acc   = float(accuracy_score(y_test, preds))
    f1    = float(f1_score(y_test, preds, average="macro", zero_division=0))
    bal   = float(balanced_accuracy_score(y_test, preds))
    cm    = confusion_matrix(y_test, preds, labels=list(range(1, 8))).tolist()

    if logger:
        logger.info(f"  [{tag}] acc={acc:.4f}  macro_f1={f1:.4f}  "
                    f"bal={bal:.4f}  wall_train={wall_train:.1f}s")

    return {
        "tag":            tag,
        "in_dim":         in_dim,
        "n_epochs_run":   len(train_losses),
        "best_val_loss":  float(best_val_loss),
        "train_loss":     train_losses,
        "val_loss":       val_losses,
        "wall_train_s":   wall_train,
        "wall_pred_s":    wall_pred,
        "test_metrics": {
            "accuracy":     acc,
            "macro_f1":     f1,
            "balanced_acc": bal,
            "confusion_matrix": cm,
        }
    }


def run_nn_after_dr(X_train, X_val, X_test, y_train, y_val, y_test,
                    cfg, logger=None):
    """Run MLP on original + PCA + ICA + RP features. Returns dict of results."""
    results = {}
    seed      = int(cfg["experiment"]["seed"])
    pca_n     = int(cfg["dr"]["pca_n_final"])
    ica_n     = int(cfg["dr"]["ica_n_final"])
    rp_n      = int(cfg["dr"]["rp_n_final"])

    # ── Original features ────────────────────────────────────────────────
    if logger: logger.info("[NN] Original features")
    results["original"] = train_eval_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg,
        tag="original", logger=logger)

    # ── PCA ──────────────────────────────────────────────────────────────
    if logger: logger.info(f"[NN] PCA n={pca_n}")
    pca = PCA(n_components=pca_n, random_state=seed)
    Xtr_pca  = pca.fit_transform(X_train)
    Xvl_pca  = pca.transform(X_val)
    Xte_pca  = pca.transform(X_test)
    results["pca"] = train_eval_mlp(
        Xtr_pca, y_train, Xvl_pca, y_val, Xte_pca, y_test, cfg,
        tag="pca", logger=logger)

    # ── ICA ──────────────────────────────────────────────────────────────
    if logger: logger.info(f"[NN] ICA n={ica_n}")
    ica = FastICA(n_components=ica_n, random_state=seed,
                  max_iter=500, tol=1e-4, whiten="unit-variance")
    try:
        Xtr_ica = ica.fit_transform(X_train)
        Xvl_ica = ica.transform(X_val)
        Xte_ica = ica.transform(X_test)
    except Exception as e:
        if logger: logger.warning(f"ICA failed: {e}; using PCA init")
        Xtr_ica, Xvl_ica, Xte_ica = Xtr_pca[:, :ica_n], Xvl_pca[:, :ica_n], Xte_pca[:, :ica_n]
    results["ica"] = train_eval_mlp(
        Xtr_ica, y_train, Xvl_ica, y_val, Xte_ica, y_test, cfg,
        tag="ica", logger=logger)

    # ── RP ────────────────────────────────────────────────────────────────
    if logger: logger.info(f"[NN] RP n={rp_n}")
    rp = GaussianRandomProjection(n_components=rp_n, random_state=seed)
    Xtr_rp  = rp.fit_transform(X_train)
    Xvl_rp  = rp.transform(X_val)
    Xte_rp  = rp.transform(X_test)
    results["rp"] = train_eval_mlp(
        Xtr_rp, y_train, Xvl_rp, y_val, Xte_rp, y_test, cfg,
        tag="rp", logger=logger)

    return results
