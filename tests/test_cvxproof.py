import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cvxproof import QuadraticProblem, Solution, certify, energy, finance, lower_bound, medicine, solve, traffic
from cvxproof.energy import Generator, Line

SOURCE = Path(__file__).parents[1] / "src" / "cvxproof"
EXAMPLES = {
    "finance": finance.example,
    "energy": energy.example,
    "medicine": medicine.example,
}


def imported_modules(path):
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def test_the_checker_cannot_depend_on_a_solver():
    assert imported_modules(SOURCE / "certificate.py") <= {"__future__", "dataclasses", "numpy"}


# Hand-worked answers: each is derived in the comment, with no solver involved.


def test_unconstrained_quadratic_by_hand():
    # Minimize (x - 3)^2 / 2 + 1: x = 3, value 1.
    problem = QuadraticProblem(Q=np.eye(1), c=np.array([-3.0]), constant=5.5)
    result = certify(problem, Solution(np.array([3.0]), np.zeros(0), np.zeros(0)))
    assert result.passed and result.upper_bound == pytest.approx(1.0) and result.lower_bound == pytest.approx(1.0)


def test_small_linear_program_by_hand():
    # Minimize x1 + 2 x2 with x1 + x2 = 1, x >= 0: x = (1, 0), value 1. Duals: y = -1, z = (0, 1).
    problem = QuadraticProblem(
        Q=np.zeros((2, 2)), c=np.array([1.0, 2.0]), A=np.ones((1, 2)), b=np.ones(1), G=-np.eye(2), h=np.zeros(2)
    )
    answer = Solution(np.array([1.0, 0.0]), np.array([-1.0]), np.array([0.0, 1.0]))
    result = certify(problem, answer)
    assert result.passed and result.upper_bound == pytest.approx(1.0) and result.lower_bound == pytest.approx(1.0)
    assert np.allclose(solve(problem).x, [1.0, 0.0], atol=1e-6)


def test_two_uncorrelated_assets_by_hand():
    # Variances 1 and 4, no return target: w = (4, 1) / 5, the inverse variance portfolio.
    problem = finance.portfolio(np.zeros(2), np.diag([1.0, 4.0]), min_return=-1.0)
    answer = solve(problem)
    assert np.allclose(answer.x, [0.8, 0.2], atol=1e-6)
    assert certify(problem, answer).passed


def test_return_target_prices_risk_by_hand():
    # Returns 0 and 1, variances 1 and 1, target 0.75: w = (0.25, 0.75). Stationarity with
    # budget multiplier y and target multiplier t gives w1 + y = 0 and w2 + y - t = 0, so t = 0.5.
    problem = finance.portfolio(np.array([0.0, 1.0]), np.eye(2), min_return=0.75)
    answer = solve(problem)
    assert np.allclose(answer.x, [0.25, 0.75], atol=1e-6)
    assert answer.z[0] == pytest.approx(0.5, abs=1e-6)
    assert certify(problem, answer).passed


def test_a_small_gap_does_not_promise_precise_weights():
    # Target 0.5 is met exactly by the equal weights, so the target is only just active. Near a
    # flat optimum the objective can be proven tighter than the weights are known. The weights
    # here are only checked to 1e-3 even though the gap is far smaller.
    problem = finance.portfolio(np.array([0.0, 1.0]), np.eye(2), min_return=0.5)
    answer = solve(problem)
    assert np.allclose(answer.x, [0.5, 0.5], atol=1e-3)
    assert certify(problem, answer).relative_gap < 1e-6


def test_congested_grid_prices_by_hand():
    # Cheap generator at bus 0 costs 10, expensive one at bus 1 costs 30, demand 100 at bus 1,
    # line limit 60. Dispatch: 60 and 40. Prices: 10 at bus 0 and 30 at bus 1.
    problem = energy.dispatch(
        demand=np.array([0.0, 100.0]),
        generators=[Generator(0, 500.0, 10.0), Generator(1, 500.0, 30.0)],
        lines=[Line(0, 1, 1.0, 60.0)],
    )
    answer = solve(problem)
    assert np.allclose(answer.x[:2], [60.0, 40.0], atol=1e-4)
    assert np.allclose(energy.prices(answer), [10.0, 30.0], atol=1e-4)
    assert certify(problem, answer).passed


def test_uncongested_grid_has_one_price_by_hand():
    # A line limit of 500 doesn't bind, so the cheap generator serves everything and the price is 10 everywhere.
    problem = energy.dispatch(
        demand=np.array([0.0, 100.0]),
        generators=[Generator(0, 500.0, 10.0), Generator(1, 500.0, 30.0)],
        lines=[Line(0, 1, 1.0, 500.0)],
    )
    answer = solve(problem)
    assert np.allclose(answer.x[:2], [100.0, 0.0], atol=1e-4)
    assert np.allclose(energy.prices(answer), [10.0, 10.0], atol=1e-4)


def test_one_beam_dose_plan_by_hand():
    # One beam gives the tumor 1 per unit and healthy tissue 0.5, prescription 60, limit 20.
    # The limit caps the beam at 40, so the tumor gets 40. Stationarity: (40 - 60) + 0.5 z = 0, so z = 40.
    problem = medicine.dose_plan(np.array([[1.0]]), np.array([[0.5]]), prescription=60.0, healthy_limit=20.0)
    answer = solve(problem)
    assert answer.x == pytest.approx([40.0], abs=1e-5)
    assert answer.z[0] == pytest.approx(40.0, abs=1e-5)
    result = certify(problem, answer)
    assert result.passed and result.upper_bound == pytest.approx(200.0, abs=1e-3)


def two_road_network(demand):
    return replace(traffic.example(), demand=np.array([[0.0, demand], [0.0, 0.0]]))


@pytest.mark.parametrize(("demand", "expected"), [(2.0, [1.0, 1.0]), (0.5, [0.5, 0.0])])
def test_two_road_equilibrium_by_hand(demand, expected):
    # Road A takes 1 + x and road B takes 2. Demand 2 splits so both take 2: x = (1, 1).
    # Demand 0.5 all fits on road A, which still only takes 1.5.
    network = two_road_network(demand)
    flows = traffic.solve_network(network)
    assert np.allclose(flows.sum(axis=0), expected, atol=1e-5)
    assert traffic.certify_flows(network, flows).passed


def test_traffic_exact_answer_has_zero_gap():
    network = two_road_network(2.0)
    exact = np.array([[1.0, 1.0], [0.0, 0.0]])
    result = traffic.certify_flows(network, exact, tolerance=1e-12)
    assert result.passed and result.upper_bound == pytest.approx(3.5) and result.relative_gap == pytest.approx(0.0)


# Every example must be solved and proven.


@pytest.mark.parametrize("name", EXAMPLES)
def test_examples_are_certified(name):
    problem = EXAMPLES[name]()
    result = certify(problem, solve(problem))
    assert result.passed, result.reasons
    assert result.lower_bound <= result.upper_bound + 1e-6


# The checker has to reject wrong answers, or passing means nothing.


@pytest.mark.parametrize("name", EXAMPLES)
def test_changed_answers_are_rejected(name):
    problem = EXAMPLES[name]()
    good = solve(problem)
    rng = np.random.default_rng(1)
    moved = Solution(good.x + 0.05 * (1 + np.abs(good.x)) * rng.normal(size=good.x.size), good.y, good.z)
    assert not certify(problem, moved).passed
    if good.y.size:
        assert not certify(problem, Solution(good.x, good.y + 1.0, good.z)).passed
    assert not certify(problem, Solution(good.x, good.y, good.z - 1.0)).passed
    assert not certify(problem, Solution(good.x, good.y, good.z + 1.0)).passed
    assert not certify(problem, Solution(good.x, 0 * good.y, 0 * good.z)).passed


def test_nan_never_passes():
    # A check that can't be evaluated proves nothing, so it has to fail.
    network = two_road_network(2.0)
    broken = replace(network, capacity=np.array([np.nan, 1.0]))
    result = traffic.certify_flows(broken, np.array([[1.0, 1.0], [0.0, 0.0]]))
    assert not result.passed


def test_feasible_but_suboptimal_answer_is_rejected_by_the_bounds():
    # Same linear program as above. x = (0, 1) is feasible but costs 2, not 1.
    problem = QuadraticProblem(
        Q=np.zeros((2, 2)), c=np.array([1.0, 2.0]), A=np.ones((1, 2)), b=np.ones(1), G=-np.eye(2), h=np.zeros(2)
    )
    result = certify(problem, Solution(np.array([0.0, 1.0]), np.array([-1.0]), np.array([0.0, 1.0])))
    assert not result.passed
    assert result.relative_gap == pytest.approx(1 / 3)  # (2 - 1) / (1 + 2)
    assert any("bounds" in reason for reason in result.reasons)


def test_infeasible_answer_that_looks_better_is_rejected():
    # x = (0, 0) has a lower objective but breaks the budget.
    problem = QuadraticProblem(
        Q=np.zeros((2, 2)), c=np.array([1.0, 2.0]), A=np.ones((1, 2)), b=np.ones(1), G=-np.eye(2), h=np.zeros(2)
    )
    result = certify(problem, Solution(np.zeros(2), np.array([-1.0]), np.array([0.0, 1.0])))
    assert not result.passed and any("equality" in reason for reason in result.reasons)


def test_multipliers_give_a_valid_bound_whatever_they_are():
    # Weak duality: any y and z >= 0 must bound the true optimum from below.
    problem = finance.example()
    optimum = certify(problem, solve(problem)).upper_bound
    rng = np.random.default_rng(2)
    for _ in range(200):
        y = rng.normal(size=problem.b.size)
        z = rng.exponential(size=problem.h.size)
        assert lower_bound(problem, y, z, 1e-6) <= optimum + 1e-9


def test_lower_bound_is_infinite_when_multipliers_prove_nothing():
    # A linear objective with the wrong multiplier is unbounded below.
    problem = QuadraticProblem(Q=np.zeros((1, 1)), c=np.array([1.0]), G=-np.eye(1), h=np.zeros(1))
    assert lower_bound(problem, np.zeros(0), np.array([0.0]), 1e-6) == -np.inf


@pytest.mark.parametrize(
    ("flows", "reason"),
    [
        ([[2.0, 0.0], [0.0, 0.0]], "bounds"),  # Everyone on road A: delivered, but not an equilibrium.
        ([[0.0, 2.0], [0.0, 0.0]], "bounds"),  # Everyone on road B.
        ([[0.5, 0.5], [0.0, 0.0]], "demand"),  # Half the drivers vanish.
        ([[2.5, -0.5], [0.0, 0.0]], "negative"),  # Negative flow.
    ],
)
def test_wrong_traffic_flows_are_rejected(flows, reason):
    result = traffic.certify_flows(two_road_network(2.0), np.array(flows))
    assert not result.passed and any(reason in r for r in result.reasons)


def test_invalid_problems_are_refused():
    with pytest.raises(ValueError, match="positive semidefinite"):
        QuadraticProblem(Q=-np.eye(2), c=np.zeros(2))
    with pytest.raises(ValueError, match="symmetric"):
        QuadraticProblem(Q=np.array([[1.0, 1.0], [0.0, 1.0]]), c=np.zeros(2))
    with pytest.raises(ValueError, match="shape"):
        QuadraticProblem(Q=np.eye(3), c=np.zeros(2))
    problem = finance.example()
    with pytest.raises(ValueError, match="non-finite"):
        good = solve(problem)
        certify(problem, Solution(good.x * np.nan, good.y, good.z))


# Each check has to report its own failure, even when another check would also catch the answer.


def test_inequality_violation_is_reported_when_objective_is_flat():
    # With a zero objective the bounds meet anyway, so only this check can catch x = 2 > 1.
    problem = QuadraticProblem(Q=np.zeros((1, 1)), c=np.zeros(1), G=np.eye(1), h=np.ones(1))
    result = certify(problem, Solution(np.array([2.0]), np.zeros(0), np.zeros(1)))
    assert not result.passed and any("inequality constraints" in r for r in result.reasons)


def test_negative_multiplier_is_reported_and_cannot_inflate_the_bound():
    # Minimize x^2 / 2 with x <= 10: optimum 0 at x = 0. With z = -1 the raw dual value would
    # be 9.5, above the true optimum, so the bound has to use the positive part of z.
    problem = QuadraticProblem(Q=np.eye(1), c=np.zeros(1), G=np.eye(1), h=np.array([10.0]))
    result = certify(problem, Solution(np.zeros(1), np.zeros(0), np.array([-1.0])))
    assert result.lower_bound == pytest.approx(0.0)
    assert not result.passed and any("negative" in r for r in result.reasons)


def test_stationarity_failure_is_reported():
    problem = finance.example()
    good = solve(problem)
    result = certify(problem, Solution(good.x, good.y + 1.0, good.z))
    assert any("stationarity" in r for r in result.reasons)


def test_complementarity_failure_is_reported():
    # Minimize x^2 / 2 with x <= 10: the constraint is slack at x = 0, so z = 1 breaks complementarity.
    problem = QuadraticProblem(Q=np.eye(1), c=np.zeros(1), G=np.eye(1), h=np.array([10.0]))
    result = certify(problem, Solution(np.zeros(1), np.zeros(0), np.array([1.0])))
    assert any("complementary" in r for r in result.reasons)


def test_nan_in_problem_data_never_passes():
    problem = QuadraticProblem(Q=np.eye(1), c=np.zeros(1), G=np.eye(1), h=np.array([np.nan]))
    assert not certify(problem, Solution(np.zeros(1), np.zeros(0), np.zeros(1))).passed
