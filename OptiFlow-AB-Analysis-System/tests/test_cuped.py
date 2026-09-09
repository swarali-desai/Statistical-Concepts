"""Tests for src.analysis.cuped."""
import numpy as np
import pandas as pd

from src.analysis.cuped import compute_theta, cuped_test


def _correlated_df(n=10_000, seed=0, treatment_effect=0.5):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(50, 10, n)
    noise = rng.normal(0, 5, n)
    variant = np.where(rng.random(n) < 0.5, "A", "B")
    metric = covariate * 0.8 + noise + np.where(variant == "B", treatment_effect, 0.0)
    return pd.DataFrame({"variant": variant, "metric": metric, "covariate": covariate})


def test_theta_recovers_the_linear_relationship():
    df = _correlated_df(n=20_000)
    theta = compute_theta(df["metric"].to_numpy(), df["covariate"].to_numpy())
    assert abs(theta - 0.8) < 0.05


def test_cuped_reduces_variance_when_covariate_is_predictive():
    df = _correlated_df(n=20_000)
    result = cuped_test(df, metric_col="metric", covariate_col="covariate")
    assert result.variance_after < result.variance_before
    assert result.variance_reduction_pct > 0.3


def test_cuped_preserves_the_treatment_effect_estimate():
    df = _correlated_df(n=40_000, treatment_effect=0.5)
    result = cuped_test(df, metric_col="metric", covariate_col="covariate")
    assert abs(result.test_result.abs_lift - 0.5) < 0.15


def test_cuped_no_reduction_when_covariate_uninformative():
    rng = np.random.default_rng(0)
    n = 20_000
    variant = np.where(rng.random(n) < 0.5, "A", "B")
    metric = rng.normal(0, 1, n) + np.where(variant == "B", 0.1, 0.0)
    covariate = rng.normal(0, 1, n)  # independent of metric
    df = pd.DataFrame({"variant": variant, "metric": metric, "covariate": covariate})

    result = cuped_test(df, metric_col="metric", covariate_col="covariate")
    assert result.variance_reduction_pct < 0.05
