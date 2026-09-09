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

_RC_PARAMS = {
    # Figure and output
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    # Typography
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    # Axes
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    # Legend
    "legend.fontsize": 8,
    # Lines and markers
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.3,
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
