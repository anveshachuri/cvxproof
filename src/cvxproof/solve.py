"""Solve problems with cvxpy. Nothing here is trusted: `certify` checks the output."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from .certificate import QuadraticProblem, Solution


def solve(problem: QuadraticProblem, solver: str = "CLARABEL") -> Solution:
    x = cp.Variable(problem.c.size)
    objective = problem.c @ x
    if np.any(problem.Q):
        objective += 0.5 * cp.quad_form(x, cp.psd_wrap(problem.Q))
    constraints = []
    if problem.b.size:
        constraints.append(problem.A @ x == problem.b)
    if problem.h.size:
        constraints.append(problem.G @ x <= problem.h)
    model = cp.Problem(cp.Minimize(objective), constraints)
    model.solve(solver=solver)
    if model.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise ValueError(f"The solver reports status '{model.status}'")
    y = constraints[0].dual_value if problem.b.size else np.zeros(0)
    z = constraints[-1].dual_value if problem.h.size else np.zeros(0)
    return Solution(x=np.asarray(x.value), y=np.atleast_1d(y), z=np.atleast_1d(z))
