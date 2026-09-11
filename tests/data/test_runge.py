import numpy as np

from fys_stk4155_p1.data.runge import generate_runge_data, runge_function


def test_runge_function_at_zero_is_one() -> None:
    assert runge_function(np.array([0.0])) == 1.0


def test_runge_function_matches_definition() -> None:
    x = np.array([-1.0, -0.5, 0.2, 1.0])
    expected = 1.0 / (1.0 + 25.0 * x**2)
    np.testing.assert_allclose(runge_function(x), expected)


def test_generate_runge_data_shapes_and_bounds() -> None:
    x, y = generate_runge_data(n=50, noise_std=0.1, seed=0)
    assert x.shape == (50,)
    assert y.shape == (50,)
    assert np.all((x >= -1) & (x <= 1))


def test_generate_runge_data_deterministic_for_same_seed() -> None:
    x1, y1 = generate_runge_data(n=20, noise_std=0.1, seed=42)
    x2, y2 = generate_runge_data(n=20, noise_std=0.1, seed=42)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_runge_data_differs_across_seeds() -> None:
    x1, _ = generate_runge_data(n=20, noise_std=0.1, seed=1)
    x2, _ = generate_runge_data(n=20, noise_std=0.1, seed=2)
    assert not np.array_equal(x1, x2)


def test_generate_runge_data_no_noise_matches_function() -> None:
    x, y = generate_runge_data(n=10, noise_std=0.0, seed=0)
    np.testing.assert_allclose(y, runge_function(x))
