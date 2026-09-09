"""Power/MDE calculations and randomization checks.

Implements:
- Two-proportion power analysis to size an experiment for a target minimum detectable effect (MDE).
- Simple AA/randomization checks to validate even allocation and covariate balance.
"""
from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass
class CovariateResult:
    """Container for covariate balance test results."""

    column: str
    pvalue: float
    passed: bool


def calculate_sample_size(
    baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation_treatment: float = 0.5,
) -> Dict[str, float]:
    """Compute required sample size for a two-proportion test.

    Steps:
    - Compute Cohen's h effect size between baseline and baseline+mde.
    - Solve for per-group sample size using a normal approximation power analysis.
    - Scale control/treatment sizes by the desired allocation ratio.
    - Return per-group counts and total.
    """
    if not 0 < allocation_treatment < 1:
        raise ValueError("allocation_treatment must be between 0 and 1 (exclusive).")

    p_control = baseline
    p_treatment = baseline + mde
    if not (0 <= p_control <= 1) or not (0 <= p_treatment <= 1):
        raise ValueError("Baseline and baseline+MDE must both be between 0 and 1.")

    effect_size = proportion_effectsize(p_control, p_treatment)

    # ratio = n_treatment / n_control
    ratio = allocation_treatment / (1 - allocation_treatment)
    power_analyzer = NormalIndPower()

    # Solve for control-group size (nobs1); treatment size scales by ratio.
    n_control = power_analyzer.solve_power(
        effect_size=effect_size, power=power, alpha=alpha, ratio=ratio
    )
    n_treatment = n_control * ratio

    # Round up to whole participants.
    n_control = ceil(n_control)
    n_treatment = ceil(n_treatment)

    return {
        "control": n_control,
        "treatment": n_treatment,
        "total": n_control + n_treatment,
        "effect_size": effect_size,
    }


def _chi2_allocation_pvalue(counts: pd.Series) -> float:
    """Chi-square goodness-of-fit p-value for even allocation across variants."""
    observed = counts.values
    expected = np.full_like(observed, fill_value=counts.sum() / len(observed), dtype=float)
    stat, pvalue = stats.chisquare(f_obs=observed, f_exp=expected)
    return float(pvalue)


def _covariate_pvalue(df: pd.DataFrame, covariate: str, variant_col: str) -> float:
    """Chi-square p-value for covariate vs variant contingency."""
    contingency = pd.crosstab(df[covariate], df[variant_col])
    chi2, pvalue, _, _ = stats.chi2_contingency(contingency)
    return float(pvalue)


def run_randomization_checks(
    df: pd.DataFrame,
    variant_col: str = "variant",
    covariate_cols: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Run AA/randomization checks for variant split and covariate balance.

    Returns a summary dict with:
    - variant_counts: observed counts by variant
    - variant_pvalue: chi-square p-value for even split
    - covariates: list of CovariateResult for each requested covariate
    - passed: True if all p-values > alpha
    """
    if variant_col not in df.columns:
        raise ValueError(f"variant_col '{variant_col}' not found in dataframe.")
    if covariate_cols:
        missing = [c for c in covariate_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Covariate columns not found: {missing}")

    variant_counts = df[variant_col].value_counts()
    variant_pvalue = _chi2_allocation_pvalue(variant_counts)

    covariate_results: List[CovariateResult] = []
    for cov in covariate_cols or []:
        pvalue = _covariate_pvalue(df, cov, variant_col)
        covariate_results.append(
            CovariateResult(column=cov, pvalue=pvalue, passed=pvalue > alpha)
        )

    passed = variant_pvalue > alpha and all(res.passed for res in covariate_results)

    return {
        "variant_counts": variant_counts.to_dict(),
        "variant_pvalue": variant_pvalue,
        "covariates": covariate_results,
        "passed": passed,
    }
