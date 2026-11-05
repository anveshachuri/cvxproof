"""Medicine: a teaching model of radiation dose planning.

NOT FOR CLINICAL USE. This shows the shape of the optimization only. Real treatment
planning uses measured patient anatomy, validated dose calculation and clinical review.
"""

from __future__ import annotations

import numpy as np

from .certificate import QuadraticProblem


def dose_plan(
    tumor_dose: np.ndarray,
    healthy_dose: np.ndarray,
    prescription: float,
    healthy_limit: float,
) -> QuadraticProblem:
    """Choose beam intensities so the tumor dose is close to the prescription.

    Minimize 0.5 ||tumor_dose w - prescription||^2 so that every healthy region stays
    under its limit and no beam intensity is negative. A healthy region's multiplier
    says how much tumor coverage its limit is costing.
    """
    tumor_dose = np.asarray(tumor_dose, dtype=float)
    healthy_dose = np.asarray(healthy_dose, dtype=float)
    target = np.full(tumor_dose.shape[0], float(prescription))
    n_beams = tumor_dose.shape[1]
    return QuadraticProblem(
        Q=tumor_dose.T @ tumor_dose,
        c=-tumor_dose.T @ target,
        G=np.vstack([healthy_dose, -np.eye(n_beams)]),
        h=np.concatenate([np.full(healthy_dose.shape[0], float(healthy_limit)), np.zeros(n_beams)]),
        constant=0.5 * float(target @ target),
    )


def example(seed: int = 0) -> QuadraticProblem:
    """Six beams, a small tumor and nearby healthy tissue. All numbers are made up."""
    rng = np.random.default_rng(seed)
    tumor_dose = rng.uniform(0.6, 1.0, size=(12, 6))
    healthy_dose = rng.uniform(0.05, 0.5, size=(20, 6))
    return dose_plan(tumor_dose, healthy_dose, prescription=60.0, healthy_limit=25.0)
