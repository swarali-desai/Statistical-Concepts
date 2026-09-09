"""CUPED: Controlled-experiment Using Pre-Experiment Data.

CUPED (Deng, Xu, Kohavi & Walker, WSDM 2013 — developed at Microsoft's
Experimentation Platform) reduces the variance of an experiment metric by
subtracting out the part of it that's predictable from a pre-experiment
covariate. Because the covariate is measured *before* randomization, it is
independent of treatment assignment by construction, so this adjustment is
unbiased -- it only removes noise, never signal. Lower variance means the
same experiment reaches significance with a smaller minimum detectable
effect, or the same MDE with fewer users/days.

    Y_cuped = Y - theta * (X - E[X]),   theta = Cov(Y, X) / Var(X)

Var(Y_cuped) = Var(Y) * (1 - corr(Y, X)^2)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.frequentist import TestResult, continuous_test


@dataclass
class CupedResult:
    theta: float
    correlation: float
    variance_before: float
    variance_after: float
    variance_reduction_pct: float
    test_result: TestResult


def compute_theta(metric: np.ndarray, covariate: np.ndarray) -> float:
    """theta = Cov(Y, X) / Var(X), estimated pooled across both arms.

    Pooling across arms (rather than estimating theta per-arm) is what keeps
    the estimator unbiased: theta must not depend on treatment assignment.
    """
    cov_matrix = np.cov(metric, covariate, ddof=1)
    covariance = cov_matrix[0, 1]
    variance_x = cov_matrix[1, 1]
    if variance_x == 0:
        return 0.0
    return covariance / variance_x


def apply_cuped(df: pd.DataFrame, metric_col: str, covariate_col: str) -> pd.Series:
    """Return the CUPED-adjusted metric as a new Series (same index as df)."""
    metric = df[metric_col].to_numpy(dtype=float)
    covariate = df[covariate_col].to_numpy(dtype=float)
    theta = compute_theta(metric, covariate)
    adjusted = metric - theta * (covariate - covariate.mean())
    return pd.Series(adjusted, index=df.index, name=f"{metric_col}_cuped")


def cuped_test(
    df: pd.DataFrame,
    metric_col: str,
    covariate_col: str,
    variant_col: str = "variant",
    control: str = "A",
    treatment: str = "B",
    alpha: float = 0.05,
) -> CupedResult:
    """Apply CUPED, then run the same Welch's t-test on the adjusted metric.

    theta and the correlation are estimated on the pooled sample so the
    adjustment can't leak treatment-arm information into the covariate
    relationship.
    """
    metric = df[metric_col].to_numpy(dtype=float)
    covariate = df[covariate_col].to_numpy(dtype=float)
    theta = compute_theta(metric, covariate)
    correlation = np.corrcoef(metric, covariate)[0, 1] if covariate.std() > 0 else 0.0

    adjusted_col = apply_cuped(df, metric_col, covariate_col)
    adjusted_df = df[[variant_col]].copy()
    adjusted_df[adjusted_col.name] = adjusted_col

    result = continuous_test(
        adjusted_df,
        metric_col=adjusted_col.name,
        variant_col=variant_col,
        control=control,
        treatment=treatment,
        alpha=alpha,
    )
    # Report the lift in the original metric's units/scale for interpretability
    # (CUPED shifts the *variance*, not the *mean difference*, which is
    # invariant to the adjustment up to sampling noise).
    result.metric = metric_col

    variance_before = float(np.var(metric, ddof=1))
    variance_after = float(np.var(adjusted_col.to_numpy(), ddof=1))
    variance_reduction_pct = (
        1 - variance_after / variance_before if variance_before > 0 else 0.0
    )

    return CupedResult(
        theta=float(theta),
        correlation=float(correlation),
        variance_before=variance_before,
        variance_after=variance_after,
        variance_reduction_pct=float(variance_reduction_pct),
        test_result=result,
    )
