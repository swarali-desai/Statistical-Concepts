"""Sequential monitoring: why "peek and stop when significant" inflates
false positives, and how a group-sequential boundary fixes it.

If you check a fixed-alpha p-value every day and stop the first time it
crosses 0.05, the *actual* false-positive rate under the null is much higher
than 5% -- each look is another chance to false-positive, and the checks are
highly correlated but not perfectly so. This module implements:

1. ``obrien_fleming_boundary`` -- the classic O'Brien-Fleming nominal
   significance boundary z_(alpha/2) / sqrt(information_fraction), derived
   from the Brownian-motion approximation to the sequence of z-statistics
   (Jennison & Turnbull, "Group Sequential Methods", 2000). This is the
   textbook *approximation*; production systems (e.g. Microsoft's ExP
   platform) typically use the exact Lan-DeMets alpha-spending recursion,
   which requires numerical integration and is out of scope here -- the
   approximation is directionally identical (very conservative early,
   relaxing to ~alpha at the final look) and is enough to demonstrate why
   sequential correction matters.
2. ``sequential_monitoring`` -- replays an experiment day-by-day and reports
   the running z-statistic against that boundary.
3. ``simulate_peeking_inflation`` -- a Monte Carlo under the null (A/A) that
   measures the empirical false-positive rate of naive daily peeking vs. the
   O'Brien-Fleming-corrected rule, to make the risk concrete rather than
   theoretical.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def obrien_fleming_boundary(info_fraction: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Nominal O'Brien-Fleming z-boundary at a given information fraction (0, 1]."""
    info_fraction = np.clip(np.asarray(info_fraction, dtype=float), 1e-6, 1.0)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return z_alpha / np.sqrt(info_fraction)


def _two_proportion_z(n_a: int, conv_a: int, n_b: int, conv_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return 0.0
    p_a, p_b = conv_a / n_a, conv_b / n_b
    pooled = (conv_a + conv_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    return (p_b - p_a) / se if se > 0 else 0.0


def sequential_monitoring(
    df: pd.DataFrame,
    metric_col: str = "converted",
    variant_col: str = "variant",
    day_col: str = "day_index",
    control: str = "A",
    treatment: str = "B",
    alpha: float = 0.05,
    final_day: Optional[int] = None,
) -> pd.DataFrame:
    """Replay the experiment look-by-look (one look per day of traffic).

    Returns one row per day with the cumulative sample sizes, the naive
    fixed-alpha z-statistic, and the O'Brien-Fleming boundary for that day's
    information fraction, so you can plot "z-trajectory vs. boundary" -- the
    canonical sequential-testing chart.
    """
    final_day = final_day if final_day is not None else int(df[day_col].max())
    days = sorted(df[day_col].unique())

    rows = []
    for day in days:
        if day > final_day:
            break
        cum = df[df[day_col] <= day]
        a = cum[cum[variant_col] == control][metric_col]
        b = cum[cum[variant_col] == treatment][metric_col]
        n_a, n_b = len(a), len(b)
        z = _two_proportion_z(n_a, int(a.sum()), n_b, int(b.sum()))

        info_fraction = (n_a + n_b) / len(df[df[day_col] <= final_day])
        boundary = obrien_fleming_boundary(info_fraction, alpha=alpha)
        naive_p = 2 * (1 - stats.norm.cdf(abs(z)))

        rows.append(
            {
                "day": day,
                "n_control": n_a,
                "n_treatment": n_b,
                "info_fraction": info_fraction,
                "z_stat": z,
                "naive_p_value": naive_p,
                "naive_significant": naive_p < alpha,
                "obf_boundary": boundary,
                "obf_significant": abs(z) > boundary,
            }
        )

    return pd.DataFrame(rows)


@dataclass
class PeekingSimulationResult:
    n_simulations: int
    n_looks: int
    n_per_look_per_arm: int
    alpha: float
    naive_false_positive_rate: float
    obf_false_positive_rate: float


def simulate_peeking_inflation(
    n_simulations: int = 1000,
    n_looks: int = 10,
    n_per_look_per_arm: int = 200,
    alpha: float = 0.05,
    baseline: float = 0.10,
    seed: int = 42,
) -> PeekingSimulationResult:
    """Monte Carlo under the null (true A/A, no effect) comparing false-positive
    rates of naive repeated peeking vs. O'Brien-Fleming-corrected stopping.
    """
    rng = np.random.default_rng(seed)
    naive_false_positives = 0
    obf_false_positives = 0

    info_fractions = np.arange(1, n_looks + 1) / n_looks
    boundaries = obrien_fleming_boundary(info_fractions, alpha=alpha)

    for _ in range(n_simulations):
        conv_a, conv_b, n_a, n_b = 0, 0, 0, 0
        naive_stopped = False
        obf_stopped = False
        for look in range(n_looks):
            new_a = rng.binomial(1, baseline, n_per_look_per_arm)
            new_b = rng.binomial(1, baseline, n_per_look_per_arm)  # same rate: true null
            conv_a += new_a.sum()
            conv_b += new_b.sum()
            n_a += n_per_look_per_arm
            n_b += n_per_look_per_arm

            z = _two_proportion_z(n_a, conv_a, n_b, conv_b)
            p = 2 * (1 - stats.norm.cdf(abs(z)))

            if not naive_stopped and p < alpha:
                naive_stopped = True
            if not obf_stopped and abs(z) > boundaries[look]:
                obf_stopped = True

        naive_false_positives += int(naive_stopped)
        obf_false_positives += int(obf_stopped)

    return PeekingSimulationResult(
        n_simulations=n_simulations,
        n_looks=n_looks,
        n_per_look_per_arm=n_per_look_per_arm,
        alpha=alpha,
        naive_false_positive_rate=naive_false_positives / n_simulations,
        obf_false_positive_rate=obf_false_positives / n_simulations,
    )
