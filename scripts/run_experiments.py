"""
Numerical experiments behind the numbers quoted in the report.

Each subcommand writes one JSON file to data/results/ (regenerate with this
script or `make experiments`). All settings come from `utils/config.py`; all
randomness is seeded. Repeated experiments use the data seeds
`config.seeds()` (0..N_SEEDS-1 plus the reference seed).

    e1  OLS: which degree do the bootstrap, k-fold CV and a single split select,
        and how does that compare to the true test error (known f)?
    e2  Model selection OLS vs Ridge vs Lasso (own subgradient-descent solver):
        K_SELECT-fold CV on the training split, refit, evaluate on the held-out
        split and against the known f.                          [slow: ~1 h CPU]
    e3  Lasso diagnostics at the reference seed's selected (degree, lambda):
        our subgradient descent with several optimizers vs scikit-learn.
        Needs e2.
    e4  Optimizers: iterations to reach the closed-form OLS/Ridge solution,
        learning rate tuned per optimizer.
    e5  Analytical vs automatic-differentiation gradients.

Usage:
    uv run python scripts/run_experiments.py {e1,e2,e3,e4,e5} [--workers 4]

LLM-assisted: Claude (claude-opus-5-5, Claude Cowork desktop app, September 2026)
wrote/rewrote this file (level 4) to use the shared settings in utils/config.py.
TODO(author): describe your review/changes.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso as SkLasso
from utils import config as C
from utils.common import RESULTS_DIR, data, design, oracle_mse

from fys_stk4155_p1.data.runge import runge_function
from fys_stk4155_p1.regression.cost import lasso_cost
from fys_stk4155_p1.regression.lasso import Lasso
from fys_stk4155_p1.regression.ordinary_least_squares import OLS
from fys_stk4155_p1.regression.ridge import Ridge
from fys_stk4155_p1.resampling.bootstrap import bootstrap_bias_variance_sweep
from fys_stk4155_p1.resampling.cross_validation import kfold_mse_degree_sweep


# ----------------------------------------------------------------------------- helpers
def _data(seed: int) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    return data(seed)


def _fit_predict(degree: int, model: Any, x_tr: NDArray, y_tr: NDArray, *x_eval: NDArray) -> list:
    X_tr, *Xs = design(x_tr, degree, *x_eval)
    model.fit(X_tr, y_tr)
    return [model.predict(X) for X in Xs]


def _oracle_mse(pred_grid: NDArray) -> float:
    return oracle_mse(pred_grid)


def _ridge(lam: float) -> Ridge:
    return Ridge(lam=lam, fit_intercept_column=True)


def _lasso(lam: float, **overrides: Any) -> Lasso:
    kwargs = dict(C.LASSO_GD) | overrides
    return Lasso(lam=lam, fit_intercept_column=True, **kwargs)


def _sk_lasso(lam: float) -> SkLasso:
    # sklearn: (1/2n)||y - Xw - b||^2 + alpha ||w||_1; ours: (1/n)||.||^2 + lam ||w||_1
    # -> alpha = lam / 2. The all-ones column is centred to zero inside sklearn, so its
    # coefficient stays 0 and sklearn's own (unpenalized) intercept takes its place.
    return SkLasso(alpha=lam / 2, fit_intercept=True, max_iter=1_000_000, tol=1e-10)


def _to_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_json(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        return float(obj) if np.isfinite(obj) else None
    return obj


def _config() -> dict[str, Any]:
    return {k: v for k, v in vars(C).items() if k.isupper()}


def _save(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(_to_json({"config": _config()} | payload), indent=1))
    print(f"wrote {path}")
    return path


def load(name: str) -> dict:
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


def _map(fn: Any, items: list, workers: int) -> list:
    if workers <= 1:
        return [fn(it) for it in items]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, items)


def _mean_se(v: Any) -> list[float]:
    v = np.asarray(v, dtype=float)
    return [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))]


# ----------------------------------------------------------------------------- e1
def _e1_seed(seed: int) -> dict:
    x, y, x_tr, x_te, y_tr, y_te = _data(seed)
    boot = bootstrap_bias_variance_sweep(
        x,
        y,
        C.DEGREES,
        OLS,
        n_bootstraps=C.N_BOOTSTRAPS,
        test_size=C.TEST_SIZE,
        seed=seed,
        f_true=runge_function,
    )
    cv = {k: kfold_mse_degree_sweep(x_tr, y_tr, C.DEGREES, OLS, k=k, seed=seed) for k in C.K_VALUES}
    split_mse, oracle = [], []
    for d in C.DEGREES:
        p_te, p_grid = _fit_predict(d, OLS(), x_tr, y_tr, x_te, C.X_GRID)
        split_mse.append(float(np.mean((y_te - p_te) ** 2)))
        oracle.append(_oracle_mse(p_grid))
    curves = {"bootstrap": boot["mse_test"]} | {f"cv{k}": cv[k]["mse_mean"] for k in C.K_VALUES}
    curves |= {"single_split": np.array(split_mse), "oracle": np.array(oracle)}
    best = {name: C.DEGREES[int(np.argmin(c))] for name, c in curves.items()}
    return {
        "seed": seed,
        "curves": curves,
        "cv_folds": {f"cv{k}": cv[k]["mse_folds"] for k in C.K_VALUES},
        "best_degree": best,
        "oracle_at_selected": {name: oracle[C.DEGREES.index(d)] for name, d in best.items()},
        "bias2": boot["bias2"],
        "bias2_f": boot["bias2_f"],
        "variance": boot["variance"],
    }


def run_e1(a: argparse.Namespace) -> None:
    per_seed = _map(_e1_seed, C.seeds(), a.workers)
    ref = next(r for r in per_seed if r["seed"] == C.SEED)
    var = np.asarray(ref["variance"])
    b = ref["best_degree"]["bootstrap"]
    summary: dict[str, Any] = {
        "ref_seed_best_degree": ref["best_degree"],
        "ref_seed_variance_ratio": {f"var[{d}]/var[{b}]": var[d] / var[b] for d in (10, 15, 20)},
    }
    for name in ref["best_degree"]:
        ds = [r["best_degree"][name] for r in per_seed]
        summary[f"{name}.best_degree_counts"] = {d: ds.count(d) for d in sorted(set(ds))}
        summary[f"{name}.best_degree_median"] = float(np.median(ds))
        o = [r["oracle_at_selected"][name] for r in per_seed]
        summary[f"{name}.oracle_at_selected"] = _mean_se(o) + [float(np.median(o))]
    _save("e1_ols_degree", {"summary": summary, "per_seed": per_seed})
    print(json.dumps(_to_json(summary), indent=1))


# ----------------------------------------------------------------------------- e2
def _select(mean: NDArray, lambdas: NDArray) -> tuple[int, float, bool]:
    i, j = np.unravel_index(np.argmin(mean), mean.shape)
    return C.DEGREES[int(i)], float(lambdas[j]), bool(j in (0, len(lambdas) - 1))


def _e2_seed(seed: int) -> dict:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    x, y, x_tr, x_te, y_tr, y_te = _data(seed)
    k = C.K_SELECT

    ols = kfold_mse_degree_sweep(x_tr, y_tr, C.DEGREES, OLS, k=k, seed=seed)
    ridge_folds = np.stack(
        [
            kfold_mse_degree_sweep(
                x_tr, y_tr, C.DEGREES, lambda lam=lam: _ridge(lam), k=k, seed=seed
            )["mse_folds"]
            for lam in C.RIDGE_LAMBDAS
        ],
        axis=1,
    )
    lasso_folds = np.stack(
        [
            kfold_mse_degree_sweep(
                x_tr, y_tr, C.DEGREES, lambda lam=lam: _lasso(lam), k=k, seed=seed
            )["mse_folds"]
            for lam in C.LASSO_LAMBDAS
        ],
        axis=1,
    )

    i = int(np.argmin(ols["mse_mean"]))
    sel: dict[str, dict[str, Any]] = {
        "ols": {"degree": C.DEGREES[i], "lambda": 0.0, "cv_folds": ols["mse_folds"][i]}
    }
    for name, folds, lambdas in (
        ("ridge", ridge_folds, C.RIDGE_LAMBDAS),
        ("lasso", lasso_folds, C.LASSO_LAMBDAS),
    ):
        d, lam, edge = _select(folds.mean(axis=2), lambdas)
        j = int(np.flatnonzero(lambdas == lam)[0])
        sel[name] = {
            "degree": d,
            "lambda": lam,
            "lambda_at_edge": edge,
            "cv_folds": folds[C.DEGREES.index(d), j],
        }

    factories = {
        "ols": lambda _lam: OLS(),
        "ridge": _ridge,
        "lasso": _lasso,
    }
    for name, s in sel.items():
        s["cv_mse"] = float(np.mean(s["cv_folds"]))
        model = factories[name](s["lambda"])
        p_te, p_grid = _fit_predict(s["degree"], model, x_tr, y_tr, x_te, C.X_GRID)
        s["test_mse"] = float(np.mean((y_te - p_te) ** 2))
        s["oracle_mse"] = _oracle_mse(p_grid)
        s["n_nonzero"] = int(np.sum(model.coef_[1:] != 0))
        s["coef"] = model.coef_

    out: dict[str, Any] = {"seed": seed, "selected": sel}
    if seed == C.SEED:
        out["ols_cv_mean"] = ols["mse_mean"]
        out["ridge_grid_mean"] = ridge_folds.mean(axis=2)
        out["lasso_grid_mean"] = lasso_folds.mean(axis=2)
        r10 = np.stack(
            [
                kfold_mse_degree_sweep(
                    x_tr, y_tr, C.DEGREES, lambda lam=lam: _ridge(lam), k=10, seed=seed
                )["mse_mean"]
                for lam in C.RIDGE_LAMBDAS
            ],
            axis=1,
        )
        out["ridge_grid_mean_cv10"] = r10
        d, lam, edge = _select(r10, C.RIDGE_LAMBDAS)
        out["ridge_cv10_selected"] = {
            "degree": d,
            "lambda": lam,
            "lambda_at_edge": edge,
            "cv_mse": float(r10.min()),
        }
    return out


def _paired(a: Sequence[float] | NDArray, b: Sequence[float] | NDArray) -> dict[str, Any]:
    a, b = np.asarray(a), np.asarray(b)
    return {
        "diff_mean_se": _mean_se(a - b),
        "rel_median": float(np.median(a / b - 1)),
        "fraction_better": float(np.mean(a < b)),
        "n_better": int(np.sum(a < b)),
    }


def run_e2(a: argparse.Namespace) -> None:
    t0 = time.time()
    per_seed = _map(_e2_seed, C.seeds(), a.workers)
    methods = list(per_seed[0]["selected"])
    summary: dict[str, Any] = {"runtime_s": time.time() - t0, "n_datasets": len(per_seed)}
    for m in methods:
        for metric in ("cv_mse", "test_mse", "oracle_mse"):
            v = [r["selected"][m][metric] for r in per_seed]
            summary[f"{m}.{metric}"] = {"mean_se": _mean_se(v), "median": float(np.median(v))}
        summary[f"{m}.degrees"] = [r["selected"][m]["degree"] for r in per_seed]
        summary[f"{m}.n_nonzero"] = [r["selected"][m]["n_nonzero"] for r in per_seed]
        if m != "ols":
            summary[f"{m}.lambdas"] = [r["selected"][m]["lambda"] for r in per_seed]
            summary[f"{m}.n_lambda_at_edge"] = sum(
                r["selected"][m]["lambda_at_edge"] for r in per_seed
            )
        for metric in ("cv_mse", "test_mse", "oracle_mse"):
            if m != "ols":
                summary[f"{m}-vs-ols.{metric}"] = _paired(
                    [r["selected"][m][metric] for r in per_seed],
                    [r["selected"]["ols"][metric] for r in per_seed],
                )
    ref = next(r for r in per_seed if r["seed"] == C.SEED)
    summary["ref_seed"] = {
        m: {k: v for k, v in s.items() if k not in ("cv_folds", "coef")}
        for m, s in ref["selected"].items()
    }
    for m in methods:
        if m != "ols":
            summary["ref_seed"][f"{m}-vs-ols.cv_folds"] = _mean_se(
                np.asarray(ref["selected"][m]["cv_folds"])
                - np.asarray(ref["selected"]["ols"]["cv_folds"])
            )
    summary["ref_seed"]["ridge_cv10"] = ref["ridge_cv10_selected"]
    _save("e2_model_selection", {"summary": summary, "per_seed": per_seed})
    print(json.dumps(_to_json(summary), indent=1))


# ----------------------------------------------------------------------------- e3
def _downsample(h: NDArray, n: int = 400) -> dict[str, list]:
    idx = np.unique(np.geomspace(1, len(h), n).astype(int)) - 1
    return {"iter": (idx + 1).tolist(), "value": np.asarray(h)[idx].tolist()}


def _e3_config(d: int, lam: float) -> dict[str, Any]:
    _, _, x_tr, _, y_tr, _ = _data(C.SEED)
    (X,) = design(x_tr, d)

    def F(theta: NDArray) -> float:
        return float(lasso_cost(X, y_tr, theta, lam, True))

    sk = _sk_lasso(lam).fit(X, y_tr)
    sk_theta = np.concatenate([[sk.intercept_], sk.coef_[1:]])
    F_star = F(sk_theta)
    L = float(np.linalg.eigvalsh(2 / len(y_tr) * X.T @ X)[-1])

    n_it = 50_000
    runs = {
        "plain": {"learning_rate": 1 / L, "optimizer": "plain"},
        "momentum": {"learning_rate": 1 / L, "optimizer": "momentum"},
        "adagrad": {"learning_rate": 0.05, "optimizer": "adagrad"},
        "rmsprop": {"learning_rate": 1e-3, "optimizer": "rmsprop"},
        "adam": {"learning_rate": C.LASSO_GD["learning_rate"], "optimizer": "adam"},
    }
    traces = {}
    for name, kw in runs.items():
        model = Lasso(lam=lam, max_iter=n_it, tol=0.0, fit_intercept_column=True, **kw)
        model.fit(X, y_tr)
        gap = (np.asarray(model.cost_history_) - F_star) / F_star
        traces[name] = _downsample(np.maximum(gap, 1e-16)) | {"learning_rate": kw["learning_rate"]}
    finals: dict[str, Any] = {}
    for name, iters in (("adam_default", C.LASSO_GD["max_iter"]), ("adam_long", n_it)):
        th = _lasso(lam, max_iter=iters, tol=0.0).fit(X, y_tr).coef_
        on_zero = th[1:][sk_theta[1:] == 0]
        finals[name] = {
            "iters": iters,
            "rel_cost_gap": (F(th) - F_star) / F_star,
            "n_exact_zero": int(np.sum(th[1:] == 0)),
            "max_abs_on_sklearn_zeros": float(np.max(np.abs(on_zero), initial=0.0)),
            "max_abs_diff_vs_sklearn": float(np.max(np.abs(th - sk_theta))),
            "coef": th,
        }
    finals["sklearn"] = {
        "n_exact_zero": int(np.sum(sk_theta[1:] == 0)),
        "n_iter": int(sk.n_iter_),
        "coef": sk_theta,
    }
    return {
        "degree": d,
        "lambda": lam,
        "F_star": F_star,
        "L": L,
        "kappa": float(L / max(np.linalg.eigvalsh(2 / len(y_tr) * X.T @ X)[0], 1e-300)),
        "traces": traces,
        "finals": finals,
    }


def run_e3(a: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    sel = load("e2_model_selection")["summary"]["ref_seed"]["lasso"]
    configs = {
        "selected": _e3_config(int(sel["degree"]), float(sel["lambda"])),
        "high_degree": _e3_config(C.SHOWCASE_DEGREE, 1e-4),
    }
    _save("e3_lasso", {"configs": configs})
    print(
        json.dumps(
            _to_json(
                {
                    c: {
                        k: {kk: vv for kk, vv in v.items() if kk != "coef"}
                        for k, v in cfg["finals"].items()
                    }
                    | {"kappa": cfg["kappa"]}
                    for c, cfg in configs.items()
                }
            ),
            indent=1,
        )
    )


# ----------------------------------------------------------------------------- e4
OPTIMIZERS = {
    "plain": {},
    "momentum": {"beta": 0.9},
    "adagrad": {},
    "rmsprop": {"rho": 0.9},
    "adam": {"beta1": 0.9, "beta2": 0.999},
}


def _e4_problem(d: int, lam: float) -> tuple[NDArray, NDArray, NDArray, dict]:
    _, _, x_tr, _, y_tr, _ = _data(C.SEED)
    (X,) = design(x_tr, d)
    theta_star = (OLS() if lam == 0 else _ridge(lam)).fit(X, y_tr).coef_
    pen = np.eye(X.shape[1])
    pen[0, 0] = 0
    eig = np.linalg.eigvalsh(2 / X.shape[0] * X.T @ X + 2 * lam * pen)
    return X, y_tr, theta_star, {"L": eig[-1], "mu": eig[0], "kappa": eig[-1] / eig[0]}


def _e4_task(args: tuple) -> dict:
    from fys_stk4155_p1.optimization.benchmark import iterations_to_tolerance

    d, lam, opt, lr = args
    X, y, theta_star, _ = _e4_problem(d, lam)
    r = iterations_to_tolerance(
        X,
        y,
        theta_star,
        opt,
        float(lr),
        lam=lam,
        fit_intercept_column=True,
        param_tol=C.GD_PARAM_TOL,
        max_iter=C.GD_MAX_ITER,
        **OPTIMIZERS[opt],
    )
    return {"degree": d, "lambda": lam, "optimizer": opt, "learning_rate": float(lr)} | r


def run_e4(a: argparse.Namespace) -> None:
    tasks = [
        (d, lam, o, lr)
        for d, lam in C.GD_BENCH_PROBLEMS
        for o in OPTIMIZERS
        for lr in C.GD_LEARNING_RATES
    ]
    t0 = time.time()
    runs = _map(_e4_task, tasks, a.workers)
    summary: dict[str, Any] = {"runtime_s": time.time() - t0}
    for d, lam in C.GD_BENCH_PROBLEMS:
        spec = _e4_problem(d, lam)[3]
        key = f"deg{d}_lam{lam:g}"
        summary[key] = {"degree": d, "lambda": lam, "hessian": spec | {"2/L": 2 / spec["L"]}}
        for o in OPTIMIZERS:
            rs = [
                r for r in runs if r["degree"] == d and r["lambda"] == lam and r["optimizer"] == o
            ]
            ok = [r for r in rs if r["iters_param"] is not None]
            finite = [r for r in rs if not r["diverged"]]
            best = (
                min(ok, key=lambda r: r["iters_param"])
                if ok
                else min(finite, key=lambda r: r["final_param_err"])
            )
            summary[key][o] = {
                "iters": best["iters_param"],
                "learning_rate": best["learning_rate"],
                "final_param_err": best["final_param_err"],
                "n_lr_converged": len(ok),
                "n_lr": len(rs),
            }
    _save("e4_optimizers", {"optimizers": OPTIMIZERS, "summary": summary, "runs": runs})
    print(json.dumps(_to_json(summary), indent=1))


# ----------------------------------------------------------------------------- e5
def run_e5(a: argparse.Namespace) -> None:
    from fys_stk4155_p1.regression.autodiff import autodiff_gradient, lasso_autodiff_gradient
    from fys_stk4155_p1.regression.cost import analytical_gradient, lasso_subgradient

    _, _, x_tr, _, y_tr, _ = _data(C.SEED)
    rng = np.random.default_rng(C.SEED)
    out: dict[str, Any] = {}
    for d in (5, 10, 15):
        (X,) = design(x_tr, d)
        diffs: dict[str, list[float]] = {"ols": [], "ridge": [], "lasso": []}
        for _ in range(100):
            th = rng.normal(size=X.shape[1])
            for name, fa, fb, lam in (
                ("ols", analytical_gradient, autodiff_gradient, 0.0),
                ("ridge", analytical_gradient, autodiff_gradient, 0.1),
                ("lasso", lasso_subgradient, lasso_autodiff_gradient, 0.1),
            ):
                ga = np.asarray(fa(X, y_tr, th, lam, True))
                gb = np.asarray(fb(X, y_tr, th, lam, True))
                diffs[name].append(float(np.max(np.abs(ga - gb)) / np.max(np.abs(ga))))
        out[f"deg{d}"] = {
            k: {"max_rel": max(v), "median_rel": float(np.median(v))} for k, v in diffs.items()
        }
    # derivative of |theta| at exactly 0: analytical (np.sign) vs JAX
    (X,) = design(x_tr, 5)
    th0 = np.zeros(X.shape[1])
    ga = np.asarray(lasso_subgradient(X, y_tr, th0, 1.0, True))
    gb = np.asarray(lasso_autodiff_gradient(X, y_tr, th0, 1.0, True))
    out["at_zero"] = {
        "analytical_minus_mse_grad": ga - np.asarray(analytical_gradient(X, y_tr, th0, 0.0, True)),
        "autodiff_minus_mse_grad": gb - np.asarray(analytical_gradient(X, y_tr, th0, 0.0, True)),
    }
    _save("e5_autodiff", {"summary": out})
    print(json.dumps(_to_json(out), indent=1))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("experiment", choices=["e1", "e2", "e3", "e4", "e5"])
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = p.parse_args()
    {"e1": run_e1, "e2": run_e2, "e3": run_e3, "e4": run_e4, "e5": run_e5}[a.experiment](a)


if __name__ == "__main__":
    main()
