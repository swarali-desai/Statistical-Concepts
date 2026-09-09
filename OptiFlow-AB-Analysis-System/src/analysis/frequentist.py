"""Frequentist two-sample tests for experiment metrics.

Covers the two metric shapes that come up in almost every product experiment:
binary (conversion, click-through) and continuous (revenue, session length).
Both return a common ``TestResult`` shape so downstream reporting/decision
code doesn't need to branch on metric type.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    """Common result shape for a two-arm significance test."""

    metric: str
    control_n: int
    treatment_n: int
    control_mean: float
    treatment_mean: float
    abs_lift: float
    rel_lift: float
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float
    alpha: float
    significant: bool

    def summary(self) -> str:
        direction = "up" if self.abs_lift >= 0 else "down"
        sig = "significant" if self.significant else "not significant"
        return (
            f"{self.metric}: {self.control_mean:.4f} (A) -> {self.treatment_mean:.4f} (B), "
            f"{direction} {abs(self.rel_lift):.2%} rel / {self.abs_lift:+.4f} abs "
            f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}] p={self.p_value:.4f} ({sig})"
        )


def _rel_lift(control_mean: float, abs_lift: float) -> float:
    if control_mean == 0:
        return float("nan")
    return abs_lift / control_mean


def proportion_test(
    df: pd.DataFrame,
    metric_col: str = "converted",
    variant_col: str = "variant",
    control: str = "A",
    treatment: str = "B",
    alpha: float = 0.05,
) -> TestResult:
    """Two-proportion z-test with a Wald confidence interval on the difference."""
    a = df.loc[df[variant_col] == control, metric_col]
    b = df.loc[df[variant_col] == treatment, metric_col]

    n_a, n_b = len(a), len(b)
    p_a, p_b = a.mean(), b.mean()
    abs_lift = p_b - p_a

    pooled_p = (a.sum() + b.sum()) / (n_a + n_b)
    se_pooled = np.sqrt(pooled_p * (1 - pooled_p) * (1 / n_a + 1 / n_b))
    z_stat = abs_lift / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = abs_lift - z_crit * se_unpooled
    ci_high = abs_lift + z_crit * se_unpooled

    return TestResult(
        metric=metric_col,
        control_n=n_a,
        treatment_n=n_b,
        control_mean=float(p_a),
        treatment_mean=float(p_b),
        abs_lift=float(abs_lift),
        rel_lift=_rel_lift(p_a, abs_lift),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        statistic=float(z_stat),
        p_value=float(p_value),
        alpha=alpha,
        significant=bool(p_value < alpha),
    )


def continuous_test(
    df: pd.DataFrame,
    metric_col: str = "revenue",
    variant_col: str = "variant",
    control: str = "A",
    treatment: str = "B",
    alpha: float = 0.05,
) -> TestResult:
    """Welch's t-test (unequal variance) for a continuous metric."""
    a = df.loc[df[variant_col] == control, metric_col].to_numpy()
    b = df.loc[df[variant_col] == treatment, metric_col].to_numpy()

    n_a, n_b = len(a), len(b)
    mean_a, mean_b = a.mean(), b.mean()
    abs_lift = mean_b - mean_a

    t_stat, p_value = stats.ttest_ind(b, a, equal_var=False)

    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(var_a / n_a + var_b / n_b)
    dof = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    )
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    ci_low = abs_lift - t_crit * se
    ci_high = abs_lift + t_crit * se

    return TestResult(
        metric=metric_col,
        control_n=n_a,
        treatment_n=n_b,
        control_mean=float(mean_a),
        treatment_mean=float(mean_b),
        abs_lift=float(abs_lift),
        rel_lift=_rel_lift(mean_a, abs_lift),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        statistic=float(t_stat),
        p_value=float(p_value),
        alpha=alpha,
        significant=bool(p_value < alpha),
    )


def bootstrap_diff_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    statistic=np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Non-parametric bootstrap CI for the difference in a statistic (B - A).

    Useful when a metric is too skewed (e.g. revenue-per-user with a mass at
    zero) for the CLT-based CI to be trusted, or when the statistic of
    interest isn't the mean (e.g. median, p90).
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(values_a), np.asarray(values_b)
    n_a, n_b = len(a), len(b)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        resample_a = a[rng.integers(0, n_a, n_a)]
        resample_b = b[rng.integers(0, n_b, n_b)]
        diffs[i] = statistic(resample_b) - statistic(resample_a)

    point_estimate = statistic(b) - statistic(a)
    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "point_estimate": float(point_estimate),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "boot_diffs": diffs,
        "significant": bool(ci_low > 0 or ci_high < 0),
    }
