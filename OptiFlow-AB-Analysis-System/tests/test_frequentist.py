"""Tests for src.analysis.frequentist."""
import numpy as np
import pandas as pd

from src.analysis.frequentist import bootstrap_diff_ci, continuous_test, proportion_test


def _binary_df(n=5000, p_a=0.10, p_b=0.13, seed=1):
    rng = np.random.default_rng(seed)
    variant = np.where(rng.random(n) < 0.5, "A", "B")
    p = np.where(variant == "A", p_a, p_b)
    converted = rng.binomial(1, p)
    return pd.DataFrame({"variant": variant, "converted": converted})


def test_proportion_test_detects_a_real_lift():
    df = _binary_df(n=20_000, p_a=0.10, p_b=0.15)
    result = proportion_test(df)
    assert result.abs_lift > 0
    assert result.significant
    assert result.ci_low < result.abs_lift < result.ci_high


def test_proportion_test_null_case_is_usually_not_significant():
    df = _binary_df(n=2000, p_a=0.10, p_b=0.10, seed=3)
    result = proportion_test(df)
    assert not result.significant


def test_continuous_test_matches_known_mean_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=10.0, scale=2.0, size=5000)
    b = rng.normal(loc=10.5, scale=2.0, size=5000)
    df = pd.DataFrame(
        {"variant": ["A"] * len(a) + ["B"] * len(b), "revenue": np.concatenate([a, b])}
    )
    result = continuous_test(df, metric_col="revenue")
    assert abs(result.abs_lift - 0.5) < 0.1
    assert result.significant


def test_bootstrap_diff_ci_contains_true_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0.3, 1, 2000)
    out = bootstrap_diff_ci(a, b, n_boot=1000, seed=0)
    assert out["ci_low"] < 0.3 < out["ci_high"]
    assert out["significant"]
