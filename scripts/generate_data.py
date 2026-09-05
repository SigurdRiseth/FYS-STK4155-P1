"""
Generate the Franke-function dataset used for OLS/Ridge/LASSO experiments.

Usage: uv run python scripts/generate_data.py [--n 100] [--noise 0.1] [--seed 42]
"""

import argparse
from pathlib import Path

import numpy as np

from fys_stk4155_p1.runge import generate_runge_data

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="number of sample points")
    parser.add_argument("--noise", type=float, default=0.1, help="Gaussian noise std")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "franke.npz")
    args = parser.parse_args()

    x, y = generate_runge_data(args.n, args.noise, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, x=x, y=y, n=args.n, noise=args.noise, seed=args.seed)
    print(f"Wrote {args.n} samples (seed={args.seed}, noise={args.noise}) to {args.out}")


if __name__ == "__main__":
    main()
