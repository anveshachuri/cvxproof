"""Transportation: predict how drivers spread over a road network (user equilibrium).

Equilibrium flows minimize the convex Beckmann objective. Convexity gives a proof that
needs no solver: the objective's tangent at the current flows, minimized by sending
everyone along today's shortest paths, is a lower bound on the optimum.

The checks use only NumPy and SciPy's shortest paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


@dataclass(frozen=True)
class Network:
    """Links use BPR travel times: free_flow * (1 + alpha * (flow / capacity) ** power)."""

    n_nodes: int
    tail: np.ndarray
    head: np.ndarray
    free_flow: np.ndarray
    capacity: np.ndarray
    alpha: np.ndarray
    power: np.ndarray
    demand: np.ndarray

    def travel_time(self, flow: np.ndarray) -> np.ndarray:
        return self.free_flow * (1.0 + self.alpha * (flow / self.capacity) ** self.power)

    def objective(self, flow: np.ndarray) -> float:
        extra = self.alpha * self.capacity * (flow / self.capacity) ** (self.power + 1) / (self.power + 1)
        return float(np.sum(self.free_flow * (flow + extra)))

    def shortest_path_cost(self, time: np.ndarray) -> float:
        # Parallel links are summed by the sparse constructor, so keep the fastest of each.
        fastest: dict[tuple[int, int], float] = {}
        for a, b, t in zip(self.tail, self.head, time):
            fastest[(int(a), int(b))] = min(fastest.get((int(a), int(b)), np.inf), float(t))
        rows, cols = zip(*fastest)
        graph = csr_matrix((list(fastest.values()), (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        distances = shortest_path(graph, directed=True)
        # Pairs with no demand don't count, even if no path connects them.
        wanted = self.demand > 0
        return float(np.sum(self.demand[wanted] * distances[wanted]))


@dataclass(frozen=True)
class TrafficCertificate:
    passed: bool
    upper_bound: float
    lower_bound: float
    relative_gap: float
    conservation_violation: float
    negative_flow: float
    tolerance: float
    reasons: list[str] = field(default_factory=list)


def certify_flows(network: Network, origin_flows: np.ndarray, tolerance: float = 1e-4) -> TrafficCertificate:
    """Check flows given per origin, shape (n_origins = n_nodes, n_links)."""
    origin_flows = np.asarray(origin_flows, dtype=float)
    if not np.isfinite(origin_flows).all():
        raise ValueError("Flows contain non-finite values")
    violation = 0.0
    for origin in range(network.n_nodes):
        net_out = np.zeros(network.n_nodes)
        np.add.at(net_out, network.tail, origin_flows[origin])
        np.add.at(net_out, network.head, -origin_flows[origin])
        wanted = -network.demand[origin].copy()
        wanted[origin] += network.demand[origin].sum()
        violation = max(violation, float(np.abs(net_out - wanted).max()))
    scale = 1.0 + float(network.demand.sum())
    conservation = violation / scale
    negative = float(np.maximum(-origin_flows, 0.0).max()) / scale
    flow = np.maximum(origin_flows.sum(axis=0), 0.0)

    upper = network.objective(flow)
    time = network.travel_time(flow)
    # Tangent lower bound: F(x) + grad F(x)'(y - x) at the all-or-nothing flows y.
    lower = upper - (float(time @ flow) - network.shortest_path_cost(time))
    gap = (upper - lower) / (1.0 + abs(upper))

    reasons = []
    for value, message in (
        (conservation, "flows don't deliver the demand"),
        (negative, "a flow is negative"),
        (abs(gap), "the upper and lower bounds don't meet"),
    ):
        # Written so that NaN fails: a check that can't be evaluated proves nothing.
        if not value <= tolerance:
            reasons.append(f"{message} ({value:.3e})")
    return TrafficCertificate(
        passed=not reasons,
        upper_bound=upper,
        lower_bound=lower,
        relative_gap=gap,
        conservation_violation=conservation,
        negative_flow=negative,
        tolerance=tolerance,
        reasons=reasons,
    )


def solve_network(network: Network, solver: str = "CLARABEL") -> np.ndarray:
    """Solve for per-origin flows with cvxpy. `certify_flows` checks the output."""
    n_links = network.tail.size
    origins = [o for o in range(network.n_nodes) if network.demand[o].sum() > 0]
    flows = cp.Variable((len(origins), n_links), nonneg=True)
    incidence = np.zeros((network.n_nodes, n_links))
    incidence[network.tail, np.arange(n_links)] = 1.0
    incidence[network.head, np.arange(n_links)] -= 1.0
    constraints = []
    for row, origin in enumerate(origins):
        wanted = -network.demand[origin].copy()
        wanted[origin] += network.demand[origin].sum()
        constraints.append(incidence @ flows[row] == wanted)
    total = cp.sum(flows, axis=0)
    terms = []
    for i in range(n_links):
        ratio = total[i] / network.capacity[i]
        extra = network.alpha[i] * network.capacity[i] / (network.power[i] + 1) * cp.power(ratio, int(network.power[i]) + 1)
        terms.append(network.free_flow[i] * (total[i] + extra))
    cp.Problem(cp.Minimize(sum(terms)), constraints).solve(solver=solver)
    result = np.zeros((network.n_nodes, n_links))
    result[origins] = flows.value
    return result


def example() -> Network:
    """Two towns joined by a fast congestible road and a slower uncongested one."""
    return Network(
        n_nodes=2,
        tail=np.array([0, 0]),
        head=np.array([1, 1]),
        free_flow=np.array([1.0, 2.0]),
        capacity=np.array([1.0, 1.0]),
        alpha=np.array([1.0, 0.0]),
        power=np.array([1.0, 1.0]),
        demand=np.array([[0.0, 2.0], [0.0, 0.0]]),
    )
