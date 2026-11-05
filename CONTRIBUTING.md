# Contributing to cvxproof

Thanks for helping. The best contribution is one new problem from a field you know.

## Adding a problem

A new problem needs:

1. **A plain-English description** of the decision, in the module's docstring.
2. **A builder** that returns a `QuadraticProblem`, plus an `example()` with data that's clearly labeled as illustrative or properly sourced and licensed.
3. **At least one answer worked out by hand** in the tests, with the derivation in a comment. A test that compares the solver with itself proves nothing.
4. **What the multipliers mean,** if they mean something useful, like electricity prices.
5. **Its limits:** where the model stops matching reality. Anything medical or safety-related must say clearly that it isn't for real-world use.

If your problem doesn't fit a quadratic program, open an issue first. It needs its own checker, like `traffic.py`, and an explanation of why its lower bound is valid.

## Rules that protect the proof

- `certificate.py` must never import a solver or modeling language. A test enforces this.
- A check that can't be evaluated must fail. Watch out for `NaN`, which fails every comparison.
- New checks need a test showing they reject a wrong answer. Run new tests against the old code to confirm they fail.
- Don't loosen a tolerance to make a test pass. Find out why the bounds don't meet.

## What a good pull request has

- One change.
- Tests that pass with `python -m pytest`.
- Only claims you've checked. If you report results, include the command and data source, and say what you didn't test.
- Data you have the right to share, with its source and license named.

## Using AI tools

AI-assisted contributions are welcome under the same standards. Say in the PR that you used AI and which tool, run the tests yourself, read the whole change, and be ready to explain it. Please don't open issues or PRs in bulk.

## License

By contributing, you agree that your work is licensed under the [Apache-2.0 License](LICENSE).
