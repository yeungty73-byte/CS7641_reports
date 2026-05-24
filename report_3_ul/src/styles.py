"""styles.py — Denim palette and ggplot-like theme for UL Report figures.
Mirrors report_1_sl and report_2_ol styles exactly.
Pure white background (#FFFFFF) — never parchment.
"""
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ── Denim palette ────────────────────────────────────────────────────────────
DENIM       = "#2C6FBB"
DENIM_DARK  = "#1B3A5C"
DENIM_MID   = "#3A6B8C"
DENIM_BRT   = "#5B9BD5"
AMBER       = "#D4A017"
PLUM        = "#7B2D8E"
TEAL        = "#4C9E82"
SIENNA      = "#B8552E"
GREEN       = "#408E2D"
TEXT_DARK   = "#1A1A2E"
WHITE       = "#FFFFFF"
GRID_CLR    = "#D0D8E8"

# 7-class palette (one per Cover Type) — denim family + complements
CT_PALETTE  = [DENIM, AMBER, TEAL, PLUM, SIENNA, GREEN, DENIM_BRT]

# Ordered optimizer colors reused in DR comparison
DR_COLORS   = {"original": DENIM_DARK, "pca": DENIM, "ica": AMBER,
               "rp": TEAL, "umap": PLUM}

def apply_denim_theme():
    """Apply seaborn whitegrid + denim aesthetics globally."""
    sns.set_theme(style="whitegrid", font_scale=0.92)
    plt.rcParams.update({
        "figure.facecolor":  WHITE,
        "axes.facecolor":    WHITE,
        "axes.edgecolor":    DENIM_DARK,
        "axes.labelcolor":   DENIM_DARK,
        "xtick.color":       DENIM_DARK,
        "ytick.color":       DENIM_DARK,
        "axes.titlecolor":   DENIM_DARK,
        "axes.grid":         True,
        "grid.color":        GRID_CLR,
        "grid.linewidth":    0.4,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "savefig.facecolor": WHITE,
        "savefig.bbox":      "tight",
        "font.family":       "sans-serif",
    })

def denim_fig(nrows=1, ncols=1, **kwargs):
    """Create a figure with white background."""
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    fig.patch.set_facecolor(WHITE)
    return fig, axes
