# cvxproof

**A solver saying "optimal" isn't proof. These answers come with one.**

cvxproof is a collection of real convex optimization problems from finance, energy, transportation and medicine. Each answer is checked by code that shares nothing with the solver.

> Early prototype. The API will change.

## The idea

For a convex problem, optimality can be proven with two numbers:

- Any **feasible** answer gives an **upper bound** on the best possible value.
- **Weak duality** turns any valid multipliers into a **lower bound**, however they were produced.

If the bounds meet, the answer is optimal. So we can treat the solver as an untrusted source of candidates and check its work with a few lines of NumPy. A test makes sure `certificate.py` never imports a solver.

```python
from cvxproof import certify, energy, solve

problem = energy.example()
answer = solve(problem)
result = certify(problem, answer)

print(result.passed)                      # True
print(result.lower_bound, result.upper_bound)
print(energy.prices(answer))              # Electricity price at each bus
```

If an answer is wrong, the result says why:

```python
from cvxproof import QuadraticProblem, Solution, certify
import numpy as np

# Minimize x1 + 2 x2 with x1 + x2 = 1 and x >= 0. The answer below is feasible, but not optimal.
problem = QuadraticProblem(
    Q=np.zeros((2, 2)), c=np.array([1.0, 2.0]), A=np.ones((1, 2)), b=np.ones(1), G=-np.eye(2), h=np.zeros(2)
)
wrong = Solution(np.array([0.0, 1.0]), np.array([-1.0]), np.array([0.0, 1.0]))
print(certify(problem, wrong).reasons)
```

## The problems

| Field | Module | Decision | What the multipliers mean |
|---|---|---|---|
| Finance | `finance.portfolio` | Lowest-risk weights that meet a return target | The variance cost of more return |
| Energy | `energy.dispatch` | Cheapest generation across a grid with line limits | The electricity price at each bus |
| Transportation | `traffic` | How drivers spread over a road network | Not used: see below |
| Medicine | `medicine.dose_plan` | Beam intensities that reach a tumor dose within healthy tissue limits | What each tissue limit costs in coverage |

**`medicine` is a teaching model and is not for clinical use.** Its numbers are made up. The `finance` and `energy` examples also use illustrative numbers rather than real data.

Traffic equilibrium isn't a quadratic program, so it has its own check. Its objective is convex, so its tangent at the current flows is a lower bound on the optimum, and minimizing that tangent only takes shortest paths.

## Results so far

Each problem has at least one case with an answer worked out by hand in the tests.

On the Sioux Falls network (76 links, 24 nodes, 360,600 trips), using the [TransportationNetworks](https://github.com/bstabler/TransportationNetworks) data:

| Measure | Value |
|---|---:|
| Certificate | Passed |
| Proven bounds on the objective | 4,231,304.6 to 4,231,338.5 |
| Published objective | 4,231,335.3, inside those bounds |
| Largest link flow difference from the published answer | 0.0075% |
| Solve time on a laptop | 1.5 s |

To reproduce it, download `SiouxFalls_net.tntp`, `SiouxFalls_trips.tntp` and `SiouxFalls_flow.tntp` from that repository, which we don't bundle, and run:

```bash
python examples/sioux_falls.py path/to/folder
```

## What a passing certificate means

- **It's about the model as written.** It can't tell you whether the model suits the real world.
- **It holds within a numerical tolerance,** `1e-6` by default and `1e-4` for traffic. The bounds are computed in floating point, so this isn't a formal proof.
- **It proves the objective value, not the exact answer.** Near a flat optimum, the value can be proven far more tightly than the variables are known. In one test, weights are off by about 5e-5 while the gap is about 6e-9.
- **It needs correct problem data.** A wrong number in the model gives a correct proof of the wrong problem.

## Install and test

```bash
git clone https://github.com/anveshachuri/cvxproof.git
cd cvxproof
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # On Linux or macOS: .venv/bin/python
.venv/Scripts/python -m pytest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The best contribution is one new problem from a field you know.

## License

Apache-2.0
