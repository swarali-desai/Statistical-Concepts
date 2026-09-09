"""Tests for power/MDE calculations and randomization checks."""

import pandas as pd

from src.stats.power import calculate_sample_size, run_randomization_checks


def test_calculate_sample_size_decreases_with_larger_mde():
    """Larger detectable lifts should require fewer total samples."""
    base_params = dict(baseline=0.10, alpha=0.05, power=0.8)

    small_lift = calculate_sample_size(mde=0.02, **base_params)
    big_lift = calculate_sample_size(mde=0.05, **base_params)

    # Symmetry check: balanced allocation yields roughly equal group sizes.
    assert abs(small_lift["control"] - small_lift["treatment"]) <= 1
    # Sanity check: needing to detect a larger lift should use fewer observations.
    assert small_lift["total"] > big_lift["total"]


def test_randomization_checks_pass_for_balanced_split():
    """A balanced dataset should pass allocation and covariate checks."""
    # A fully-crossed design (every variant x device_type combination
    # repeated equally often) is exactly balanced by construction, unlike
    # e.g. ["A", "B"] * n paired with ["mobile", "desktop"] * n, which is
    # perfectly collinear and would correctly fail a balance check.
    combos = [(v, d) for v in ("A", "B") for d in ("mobile", "desktop")] * 250
    df = pd.DataFrame(combos, columns=["variant", "device_type"])

    results = run_randomization_checks(df, covariate_cols=["device_type"], alpha=0.05)

    assert results["passed"] is True
    assert results["variant_pvalue"] > 0.05
    # Covariate p-value should also be non-significant for balanced data.
    device_result = next(res for res in results["covariates"] if res.column == "device_type")
    assert device_result.passed is True


def test_randomization_checks_flag_imbalance():
    """Skewed allocation should trigger a failed randomization check."""
    df = pd.DataFrame(
        {
            "variant": ["A"] * 900 + ["B"] * 100,
            "device_type": ["mobile"] * 700 + ["desktop"] * 300,
        }
    )

    results = run_randomization_checks(df, covariate_cols=["device_type"], alpha=0.05)

    assert results["passed"] is False
    assert results["variant_pvalue"] < 0.05
    device_result = next(res for res in results["covariates"] if res.column == "device_type")
    assert device_result.passed is False
