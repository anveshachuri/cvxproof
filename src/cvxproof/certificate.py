"""Check an answer to a convex quadratic program without trusting the solver.

Problem:   minimize  0.5 x'Qx + c'x + constant
           subject to  Ax = b  and  Gx <= h

Weak duality does the work. Any multipliers y and z >= 0 give a lower bound on the
optimal value, however they were produced, and any feasible x gives an upper bound.
When the two bounds meet, x is optimal. This module uses only NumPy, so a bug in a
solver or modeling language can't change its verdict.

The verdict is about the model as written, within numerical tolerance. It doesn't say
whether the model describes the real world well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class QuadraticProblem:
    """A convex quadratic program. Missing constraints can be left out."""

    Q: np.ndarray
    c: np.ndarray
    A: np.ndarray | None = None
    b: np.ndarray | None = None
    G: np.ndarray | None = None
    h: np.ndarray | None = None
    constant: float = 0.0

    def __post_init__(self) -> None:
        Q, c = np.asarray(self.Q, dtype=float), np.asarray(self.c, dtype=float)
        n = c.size
        A = np.zeros((0, n)) if self.A is None else np.asarray(self.A, dtype=float)
        b = np.zeros(0) if self.b is None else np.asarray(self.b, dtype=float)
        G = np.zeros((0, n)) if self.G is None else np.asarray(self.G, dtype=float)
        h = np.zeros(0) if self.h is None else np.asarray(self.h, dtype=float)
        if Q.shape != (n, n):
            raise ValueError(f"Q must have shape ({n}, {n}), got {Q.shape}")
        if A.ndim != 2 or A.shape != (b.size, n):
            raise ValueError(f"A must have shape ({b.size}, {n}), got {A.shape}")
        if G.ndim != 2 or G.shape != (h.size, n):
            raise ValueError(f"G must have shape ({h.size}, {n}), got {G.shape}")
        if not np.allclose(Q, Q.T):
            raise ValueError("Q must be symmetric")
        scale = max(1.0, float(np.abs(Q).max(initial=0.0)))
        if n and np.linalg.eigvalsh(Q).min() < -1e-9 * scale:
            raise ValueError("Q must be positive semidefinite, or the problem isn't convex")
        for name, value in (("Q", Q), ("c", c), ("A", A), ("b", b), ("G", G), ("h", h)):
            object.__setattr__(self, name, value)

    def objective(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self.Q @ x + self.c @ x + self.constant)


@dataclass(frozen=True)
class Solution:
    """A candidate answer. The multipliers are untrusted, like everything else here."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class Certificate:
    """The measurements behind a verdict. All residuals are relative."""

    passed: bool
    upper_bound: float
    lower_bound: float
    relative_gap: float
    equality_violation: float
    inequality_violation: float
    negative_multiplier: float
    stationarity: float
    complementarity: float
    tolerance: float
    reasons: list[str] = field(default_factory=list)


def _max_abs(values: np.ndarray) -> float:
    return float(np.abs(values).max(initial=0.0))


def lower_bound(problem: QuadraticProblem, y: np.ndarray, z: np.ndarray, tolerance: float) -> float:
    """Evaluate the dual function. Any y and z >= 0 give a valid bound on the optimum.

    Returns -inf when the Lagrangian is unbounded below, which means these multipliers
    prove nothing.
    """
    linear = problem.c + problem.A.T @ y + problem.G.T @ z
    v = -np.linalg.pinv(problem.Q, hermitian=True) @ linear
    if _max_abs(problem.Q @ v + linear) > tolerance * (1.0 + _max_abs(problem.c)):
        return -np.inf
    return float(
        0.5 * v @ problem.Q @ v + linear @ v - problem.b @ y - problem.h @ z + problem.constant
    )


def certify(problem: QuadraticProblem, solution: Solution, tolerance: float = 1e-6) -> Certificate:
    """Decide whether a candidate is feasible and provably optimal within tolerance."""
    x = np.asarray(solution.x, dtype=float)
    y = np.asarray(solution.y, dtype=float)
    z = np.asarray(solution.z, dtype=float)
    if x.shape != problem.c.shape or y.shape != problem.b.shape or z.shape != problem.h.shape:
        raise ValueError("The solution's shapes don't match the problem")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        raise ValueError("The solution contains non-finite values")

    upper = problem.objective(x)
    slack = problem.h - problem.G @ x
    # Negative multipliers don't give a valid bound, so the bound uses their positive part.
    z_valid = np.maximum(z, 0.0)
    lower = lower_bound(problem, y, z_valid, tolerance)

    equality = _max_abs(problem.A @ x - problem.b) / (1.0 + _max_abs(problem.b))
    inequality = float(np.maximum(-slack, 0.0).max(initial=0.0)) / (1.0 + _max_abs(problem.h))
    negative = float(np.maximum(-z, 0.0).max(initial=0.0)) / (1.0 + _max_abs(z))
    gradient = problem.Q @ x + problem.c + problem.A.T @ y + problem.G.T @ z
    stationarity = _max_abs(gradient) / (1.0 + _max_abs(problem.c))
    complementarity = float(np.abs(z_valid @ slack)) / (1.0 + abs(upper))
    gap = np.inf if not np.isfinite(lower) else (upper - lower) / (1.0 + abs(upper))

    reasons = []
    checks = (
        (equality, "equality constraints are violated"),
        (inequality, "inequality constraints are violated"),
        (negative, "an inequality multiplier is negative"),
        (stationarity, "the multipliers don't satisfy stationarity"),
        (complementarity, "complementary slackness fails"),
        (abs(gap), "the upper and lower bounds don't meet"),
    )
    for value, message in checks:
        # Written so that NaN fails: a check that can't be evaluated proves nothing.
        if not value <= tolerance:
            reasons.append(f"{message} ({value:.3e})")

    return Certificate(
        passed=not reasons,
        upper_bound=upper,
        lower_bound=lower,
        relative_gap=float(gap),
        equality_violation=equality,
        inequality_violation=inequality,
        negative_multiplier=negative,
        stationarity=stationarity,
        complementarity=complementarity,
        tolerance=tolerance,
        reasons=reasons,
    )
