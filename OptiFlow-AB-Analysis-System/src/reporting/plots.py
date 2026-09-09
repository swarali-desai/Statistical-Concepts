"""Plotting helpers for experiment readouts.

Each function returns a matplotlib ``Figure`` so callers (notebooks,
Streamlit, tests) can decide whether to show, save, or embed it.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_A = "#7f8c8d"
COLOR_B = "#2E86AB"
COLOR_BAD = "#C0392B"
COLOR_GOOD = "#27AE60"


def plot_posteriors(posterior_a: np.ndarray, posterior_b: np.ndarray, metric_name: str = "conversion rate"):
    """Overlaid posterior densities for control vs. treatment (Bayesian test)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(posterior_a, bins=80, density=True, alpha=0.55, color=COLOR_A, label="A (control)")
    ax.hist(posterior_b, bins=80, density=True, alpha=0.55, color=COLOR_B, label="B (treatment)")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("posterior density")
    ax.set_title(f"Posterior distributions: {metric_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_posterior_diff(posterior_diff: np.ndarray, metric_name: str = "B - A"):
    """Posterior of the difference (or relative lift), with the zero line marked."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(posterior_diff, bins=80, density=True, color=COLOR_B, alpha=0.75)
    ax.axvline(0, color=COLOR_BAD, linestyle="--", linewidth=1.5, label="no effect")
    prob_positive = float(np.mean(posterior_diff > 0))
    ax.set_title(f"Posterior of {metric_name} (P(B > A) = {prob_positive:.1%})")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cumulative_lift(
    df: pd.DataFrame,
    metric_col: str = "converted",
    variant_col: str = "variant",
    day_col: str = "day_index",
):
    """Cumulative per-arm metric mean over the course of the experiment."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for variant, color, label in [("A", COLOR_A, "A (control)"), ("B", COLOR_B, "B (treatment)")]:
        sub = df[df[variant_col] == variant].sort_values(day_col)
        running = sub[metric_col].expanding().mean()
        ax.plot(sub[day_col].values, running.values, color=color, label=label, linewidth=2)

    ax.set_xlabel("experiment day")
    ax.set_ylabel(f"cumulative mean({metric_col})")
    ax.set_title("Cumulative metric by day")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_daily_lift(df: pd.DataFrame, metric_col: str = "converted", variant_col: str = "variant", day_col: str = "day_index"):
    """Daily (non-cumulative) lift, to visualize the novelty effect decaying."""
    daily = (
        df.groupby([day_col, variant_col])[metric_col]
        .mean()
        .unstack(variant_col)
    )
    daily["abs_lift"] = daily["B"] - daily["A"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(daily.index, daily["abs_lift"], color=np.where(daily["abs_lift"] >= 0, COLOR_GOOD, COLOR_BAD))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("experiment day")
    ax.set_ylabel(f"daily lift (B - A) in {metric_col}")
    ax.set_title("Daily treatment effect (watch for novelty decay)")
    fig.tight_layout()
    return fig


def plot_sequential_boundary(sequential_df: pd.DataFrame):
    """Z-statistic trajectory vs. the O'Brien-Fleming boundary."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sequential_df["day"], sequential_df["z_stat"], color=COLOR_B, marker="o", label="observed z-statistic")
    ax.plot(sequential_df["day"], sequential_df["obf_boundary"], color=COLOR_BAD, linestyle="--", label="O'Brien-Fleming boundary")
    ax.plot(sequential_df["day"], -sequential_df["obf_boundary"], color=COLOR_BAD, linestyle="--")
    ax.fill_between(sequential_df["day"], sequential_df["obf_boundary"], -sequential_df["obf_boundary"], color=COLOR_A, alpha=0.08)
    ax.set_xlabel("experiment day (look)")
    ax.set_ylabel("z-statistic")
    ax.set_title("Sequential monitoring: z-trajectory vs. O'Brien-Fleming boundary")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_subgroup_lift(subgroup_table: pd.DataFrame, group_col: str):
    """Horizontal bar chart of per-subgroup lift with BH-significance highlighted."""
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(subgroup_table) + 1.5))
    colors = np.where(subgroup_table["significant_bh"], COLOR_GOOD, COLOR_A)
    ax.barh(subgroup_table[group_col].astype(str), subgroup_table["abs_lift"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("absolute lift (B - A)")
    ax.set_title(f"Treatment effect by {group_col} (green = significant after BH-FDR correction)")
    fig.tight_layout()
    return fig


def plot_variance_reduction(variance_before: float, variance_after: float):
    """Bar chart comparing metric variance before/after CUPED adjustment."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    reduction_pct = 1 - variance_after / variance_before if variance_before > 0 else 0
    ax.bar(["raw metric", "CUPED-adjusted"], [variance_before, variance_after], color=[COLOR_A, COLOR_B])
    ax.set_ylabel("variance")
    ax.set_title(f"CUPED variance reduction: {reduction_pct:.1%}")
    fig.tight_layout()
    return fig
