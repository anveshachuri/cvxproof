"""Solve Sioux Falls with cvxproof and compare with the published equilibrium.

Data: https://github.com/bstabler/TransportationNetworks (SiouxFalls_net, _trips, _flow .tntp).
It isn't bundled here. Download the three files and pass their folder:

    python examples/sioux_falls.py path/to/folder
"""

import re
import sys
import time
from pathlib import Path

import numpy as np

from cvxproof.traffic import Network, certify_flows, solve_network


def read_network(folder: Path) -> tuple[Network, dict]:
    links = []
    for line in (folder / "SiouxFalls_net.tntp").read_text().splitlines():
        parts = line.strip().rstrip(";").split()
        if len(parts) >= 7 and parts[0].isdigit():
            links.append([float(p) for p in parts[:7]])
    links = np.array(links)
    n_nodes = int(links[:, :2].max())
    demand = np.zeros((n_nodes, n_nodes))
    origin = None
    for line in (folder / "SiouxFalls_trips.tntp").read_text().splitlines():
        header = re.match(r"\s*Origin\s+(\d+)", line)
        if header:
            origin = int(header.group(1)) - 1
        elif origin is not None:
            for destination, value in re.findall(r"(\d+)\s*:\s*([\d.]+)", line):
                demand[origin, int(destination) - 1] = float(value)
    published = {}
    for line in (folder / "SiouxFalls_flow.tntp").read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].isdigit():
            published[(int(parts[0]) - 1, int(parts[1]) - 1)] = float(parts[2])
    network = Network(
        n_nodes=n_nodes,
        tail=links[:, 0].astype(int) - 1,
        head=links[:, 1].astype(int) - 1,
        capacity=links[:, 2],
        free_flow=links[:, 4],
        alpha=links[:, 5],
        power=links[:, 6],
        demand=demand,
    )
    return network, published


def main(folder: str) -> None:
    network, published = read_network(Path(folder))
    start = time.time()
    origin_flows = solve_network(network)
    seconds = time.time() - start
    result = certify_flows(network, origin_flows)
    flow = origin_flows.sum(axis=0)
    reference = np.array([published[(a, b)] for a, b in zip(network.tail, network.head)])
    print(f"{network.tail.size} links, {network.n_nodes} nodes, {network.demand.sum():,.0f} trips, solved in {seconds:.1f} s")
    print(f"certificate passed: {result.passed} {result.reasons}")
    print(f"upper bound {result.upper_bound:,.3f}, lower bound {result.lower_bound:,.3f}, relative gap {result.relative_gap:.3e}")
    print(f"published objective {network.objective(reference):,.3f}")
    print(f"our objective minus published {result.upper_bound - network.objective(reference):.4f}")
    print(f"largest link flow difference {np.max(np.abs(flow - reference) / reference):.4%}")


if __name__ == "__main__":
    main(sys.argv[1])
