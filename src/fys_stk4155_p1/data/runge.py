"""The Runge function"""

import numpy as np
from numpy.typing import NDArray


def runge_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Runge's function

    Runge's function, defined as f(x) = 1 / (1 + 25 x^2) over the interval [-1, 1].

    Args:
        x: x-coordinates, in [-1, 1].

    Returns:
        y-coordinates, same shape as x.
    """
    assert np.all((x >= -1) & (x <= 1))
    return 1 / (1 + 25 * x**2)


def generate_runge_data(
    n: int, noise_std: float, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample a noisy Runge-function dataset.

    Draws n points uniformly at random from [-1, 1], evaluates the Runge
    function at each, and adds i.i.d. Gaussian noise. Deterministic for a
    given seed.

    Args:
        n: number of points to sample.
        noise_std: standard deviation of the additive Gaussian noise
            (0.0 for the noise-free surface).
        seed: RNG seed for reproducibility.

    Returns:
        x: Sampled x-coordinates, shape (n,).
        y: Corresponding noisy Runge-function values, shape (n,).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = runge_function(x) + rng.normal(0, noise_std, n)
    return x, y
