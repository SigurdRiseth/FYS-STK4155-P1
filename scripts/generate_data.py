"""
Generate the reference Runge-function data set (settings from utils/config.py).

Usage: uv run python scripts/generate_data.py [--pdf]

LLM-assisted: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
wrote/rewrote this file (level 4) to use the shared settings in utils/config.py.
TODO(author): describe your review/changes.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from utils import config as C
from utils.plotting import COLORS, save_figure, set_style

from fys_stk4155_p1.data.runge import generate_runge_data, runge_function

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _save_runge_data_figure(x: NDArray[np.float64], y: NDArray[np.float64], noise: float) -> Path:
    """Plot noisy samples against the ground-truth Runge function and save as PDF."""
    set_style()

    x_plot = np.linspace(-1, 1, 500)

    fig, ax = plt.subplots()
    ax.plot(x_plot, runge_function(x_plot), color="black", lw=1.2, label=r"$f(x)=1/(1+25x^2)$")
    ax.scatter(x, y, s=6, alpha=0.7, color=COLORS["ols"], lw=0, label=rf"data ($\sigma = {noise}$)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper left")

    path = save_figure(fig, "runge_data", subdir="data")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=C.N, help="number of sample points")
    parser.add_argument("--noise", type=float, default=C.NOISE, help="Gaussian noise std")
    parser.add_argument("--seed", type=int, default=C.SEED, help="RNG seed")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "runge.npz")
    parser.add_argument("--pdf", action="store_true", help="save a PDF figure to docs/figures/data")
    args = parser.parse_args()

    x, y = generate_runge_data(args.n, args.noise, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, x=x, y=y, n=args.n, noise=args.noise, seed=args.seed)
    print(f"Wrote {args.n} samples (seed={args.seed}, noise={args.noise}) to {args.out}")

    if args.pdf:
        path = _save_runge_data_figure(x, y, args.noise)
        print(f"Wrote figure to {path}")


if __name__ == "__main__":
    main()
