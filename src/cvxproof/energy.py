"""Energy: meet electricity demand at least cost across a grid with limited lines (DC power flow)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .certificate import QuadraticProblem, Solution


@dataclass(frozen=True)
class Generator:
    bus: int
    max_output: float
    linear_cost: float
    quadratic_cost: float = 0.0


@dataclass(frozen=True)
class Line:
    from_bus: int
    to_bus: int
    susceptance: float
    limit: float


def dispatch(demand: np.ndarray, generators: list[Generator], lines: list[Line]) -> QuadraticProblem:
    """Choose generator output and bus angles. Bus 0 is the angle reference.

    Variables: generator outputs, then angles for buses 1 onward. Equality rows are
    the power balance at each bus, so their multipliers give electricity prices.
    """
    demand = np.asarray(demand, dtype=float)
    n_gen, n_bus = len(generators), demand.size
    n = n_gen + n_bus - 1

    def angle(bus: int) -> np.ndarray:
        row = np.zeros(n)
        if bus > 0:
            row[n_gen + bus - 1] = 1.0
        return row

    Q = np.zeros((n, n))
    c = np.zeros(n)
    A = np.zeros((n_bus, n))
    for i, gen in enumerate(generators):
        Q[i, i] = 2.0 * gen.quadratic_cost
        c[i] = gen.linear_cost
        A[gen.bus, i] = 1.0

    G_rows, h_rows = [], []
    for line in lines:
        flow = line.susceptance * (angle(line.from_bus) - angle(line.to_bus))
        A[line.from_bus] -= flow
        A[line.to_bus] += flow
        G_rows += [flow, -flow]
        h_rows += [line.limit, line.limit]
    for i, gen in enumerate(generators):
        row = np.zeros(n)
        row[i] = 1.0
        G_rows += [row, -row]
        h_rows += [gen.max_output, 0.0]
    return QuadraticProblem(Q=Q, c=c, A=A, b=demand, G=np.array(G_rows), h=np.array(h_rows))


def prices(solution: Solution) -> np.ndarray:
    """Electricity price at each bus: the cost of serving one more unit of demand there."""
    return -solution.y


def example() -> QuadraticProblem:
    """A three-bus grid with a congested line. The numbers are illustrative."""
    return dispatch(
        demand=np.array([0.0, 60.0, 90.0]),
        generators=[
            Generator(bus=0, max_output=200.0, linear_cost=10.0, quadratic_cost=0.02),
            Generator(bus=1, max_output=100.0, linear_cost=25.0, quadratic_cost=0.05),
            Generator(bus=2, max_output=80.0, linear_cost=40.0, quadratic_cost=0.1),
        ],
        lines=[Line(0, 1, 1.0, 70.0), Line(1, 2, 1.0, 50.0), Line(0, 2, 1.0, 60.0)],
    )
