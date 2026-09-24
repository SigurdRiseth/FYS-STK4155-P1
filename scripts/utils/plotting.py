"""Shared matplotlib style and figure output for report figures.

Call `set_style()` once near the top of any script or notebook that produces a
figure for the report, then `save_figure()` to write it under `docs/figures/`.
Using both from every figure-producing script/notebook keeps fonts, sizes, and
output paths consistent no matter where a given figure was generated.
"""

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

FIGURE_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures"

# Figure sizes (inches) matched to the IEEE two-column layout of the report:
# COLUMN figures are included at \linewidth (3.5 in), WIDE ones in a figure*
# at \textwidth (7.16 in), so text is printed at its nominal size (8 pt).
COLUMN = (3.4, 2.3)
COLUMN_TALL = (3.4, 3.4)
WIDE = (7.0, 2.5)
WIDE_TALL = (7.0, 4.2)
# Backwards-compatible aliases.
FIGSIZE_SINGLE = COLUMN
FIGSIZE_WIDE = WIDE

# One color per method/estimator/optimizer, used by every figure (tab10 based,
# colorblind-distinguishable in the combinations used).
COLORS = {
    "ols": "#1f77b4",
    "ridge": "#ff7f0e",
    "lasso": "#2ca02c",
    "train": "#7f7f7f",
    "test": "#1f77b4",
    "bootstrap": "#000000",
    "cv5": "#1f77b4",
    "cv10": "#ff7f0e",
    "single_split": "#9467bd",
    "oracle": "#d62728",
    "plain": "#7f7f7f",
    "momentum": "#1f77b4",
    "adagrad": "#9467bd",
    "rmsprop": "#8c564b",
    "adam": "#d62728",
    "sklearn": "#000000",
}
LABELS = {
    "bootstrap": "bootstrap",
    "cv5": "5-fold CV",
    "cv10": "10-fold CV",
    "single_split": "single split",
    "oracle": "true error",
    "plain": "plain GD",
    "momentum": "momentum",
    "adagrad": "AdaGrad",
    "rmsprop": "RMSprop",
    "adam": "Adam",
}

_RC_PARAMS = {
    # Figure and output
    "figure.figsize": COLUMN,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    # Typography: STIX ships with matplotlib and matches the report's Times font,
    # so figures look the same on every machine.
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    # Axes
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    # Legend
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    # Lines and markers
    "lines.linewidth": 1.0,
    "lines.markersize": 2.5,
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "figure.constrained_layout.use": True,
}


def set_style() -> None:
    """Apply the shared report figure style to matplotlib's rcParams."""
    # matplotlib's stubs type rcParams.update() against a Literal union of every
    # known key, which a plain dict[str, object] never satisfies; the runtime
    # behavior (setting arbitrary known rcParam keys) is exactly what we want.
    plt.rcParams.update(cast(Any, _RC_PARAMS))


def save_figure(fig: Figure, name: str, subdir: str | None = None) -> Path:
    """Save a figure as a PDF under docs/figures/, creating directories as needed.

    PDF (vector) rather than a rasterized format so figures stay sharp at any
    zoom/print size and text embeds as real, selectable glyphs when included in
    the LaTeX report.

    Args:
        fig: The figure to save.
        name: File stem, without extension (e.g. "ridge_trace_degree15").
        subdir: Optional subdirectory under docs/figures/ (e.g. "ridge").

    Returns:
        The path the figure was written to.
    """
    out_dir = FIGURE_DIR / subdir if subdir else FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.pdf"
    fig.savefig(out_path)
    return out_path
