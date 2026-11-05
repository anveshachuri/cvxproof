"""Finance: choose portfolio weights with the least risk that still meets a return target."""

from __future__ import annotations

import numpy as np

from .certificate import QuadraticProblem


def portfolio(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    min_return: float,
    max_weight: float = 1.0,
) -> QuadraticProblem:
    """Minimize 0.5 w'Cw so that weights sum to 1, stay in [0, max_weight], and meet the target.

    Inequality rows: return target, then lower bounds, then upper bounds. The return
    target's multiplier says how much variance one more unit of return costs.
    """
    mu = np.asarray(expected_returns, dtype=float)
    n = mu.size
    return QuadraticProblem(
        Q=np.asarray(covariance, dtype=float),
        c=np.zeros(n),
        A=np.ones((1, n)),
        b=np.ones(1),
        G=np.vstack([-mu, -np.eye(n), np.eye(n)]),
        h=np.concatenate([[-min_return], np.zeros(n), np.full(n, max_weight)]),
    )


def example(n_assets: int = 8, seed: int = 0) -> QuadraticProblem:
    """A portfolio on synthetic returns. The numbers are illustrative, not market data."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n_assets, 3)) * 0.04
    covariance = loadings @ loadings.T + np.diag(rng.uniform(0.01, 0.04, n_assets) ** 2)
    expected_returns = rng.uniform(0.02, 0.12, n_assets)
    return portfolio(expected_returns, covariance, min_return=float(np.quantile(expected_returns, 0.75)), max_weight=0.4)
