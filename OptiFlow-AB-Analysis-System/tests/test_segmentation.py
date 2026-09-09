"""Tests for src.analysis.segmentation."""
import numpy as np
import pandas as pd

from src.analysis.segmentation import interaction_effect_regression, subgroup_lift_table


def _heterogeneous_df(n=30_000, seed=0):
    rng = np.random.default_rng(seed)
    variant = np.where(rng.random(n) < 0.5, "A", "B")
    device = rng.choice(["mobile", "desktop"], size=n)

    base = np.where(device == "mobile", 0.10, 0.12)
    # Only mobile responds to treatment; desktop is flat.
    lift = np.where((variant == "B") & (device == "mobile"), 0.06, 0.0)
    p = np.clip(base + lift, 0, 1)
    converted = rng.binomial(1, p)

    return pd.DataFrame({"variant": variant, "device_type": device, "converted": converted})


def test_subgroup_lift_table_finds_the_responsive_segment():
    df = _heterogeneous_df()
    table = subgroup_lift_table(df, group_col="device_type")
    mobile_row = table[table["device_type"] == "mobile"].iloc[0]
    desktop_row = table[table["device_type"] == "desktop"].iloc[0]

    assert mobile_row["significant_bh"]
    assert mobile_row["abs_lift"] > desktop_row["abs_lift"]


def test_interaction_regression_detects_heterogeneity():
    df = _heterogeneous_df()
    interaction = interaction_effect_regression(df, group_col="device_type")
    assert (interaction["p_value"] < 0.05).any()
