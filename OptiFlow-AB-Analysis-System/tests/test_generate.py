"""Tests for the synthetic data generator."""
from datetime import datetime

from src.simulation.generate import _generate_rows

EXPECTED_COLUMNS = {
    "user_id", "timestamp", "day_index", "variant", "device_type",
    "user_segment", "pre_period_sessions", "pre_period_revenue",
    "pre_period_session_duration", "session_duration", "converted",
    "revenue", "page_load_ms",
}


def _make_df(n=20_000, seed=1, srm_bug=False):
    return _generate_rows(
        n=n, baseline=0.10, lift=0.02, seed=seed,
        start=datetime(2026, 1, 1), span_days=14,
        novelty_decay=0.2, novelty_boost=1.6, novelty_asymptote=0.7,
        guardrail_ms=15.0, srm_bug=srm_bug,
    )


def test_schema_has_expected_columns():
    df = _make_df(n=1000)
    assert EXPECTED_COLUMNS.issubset(df.columns)


def test_variants_are_roughly_balanced():
    df = _make_df(n=20_000)
    counts = df["variant"].value_counts(normalize=True)
    assert abs(counts["A"] - counts["B"]) < 0.02


def test_treatment_has_positive_effect_on_average():
    df = _make_df(n=20_000)
    rate_a = df.loc[df.variant == "A", "converted"].mean()
    rate_b = df.loc[df.variant == "B", "converted"].mean()
    assert rate_b > rate_a


def test_novelty_effect_decays_over_time():
    """The treatment's daily lift should be larger on day 0 than in the steady
    state, since novelty_boost > novelty_asymptote by construction."""
    df = _make_df(n=60_000)
    daily = df.groupby(["day_index", "variant"])["converted"].mean().unstack("variant")
    daily["lift"] = daily["B"] - daily["A"]
    early_lift = daily.loc[0:2, "lift"].mean()
    late_lift = daily.loc[11:13, "lift"].mean()
    assert early_lift > late_lift


def test_device_heterogeneity_mobile_benefits_more_than_desktop():
    df = _make_df(n=60_000)
    lift_by_device = (
        df.groupby(["device_type", "variant"])["converted"].mean().unstack("variant")
    )
    lift_by_device["lift"] = lift_by_device["B"] - lift_by_device["A"]
    assert lift_by_device.loc["mobile", "lift"] > lift_by_device.loc["desktop", "lift"]


def test_pre_period_covariate_is_unaffected_by_treatment():
    """CUPED requires the covariate to be balanced across arms -- it's
    measured before assignment, so treatment must have no effect on it."""
    df = _make_df(n=40_000)
    mean_a = df.loc[df.variant == "A", "pre_period_sessions"].mean()
    mean_b = df.loc[df.variant == "B", "pre_period_sessions"].mean()
    assert abs(mean_a - mean_b) < 0.1


def test_session_duration_correlates_with_its_pre_period_covariate():
    """This pairing is designed as the strong-correlation CUPED case (vs.
    the more modest, zero-inflated revenue/pre_period_revenue pairing)."""
    df = _make_df(n=20_000)
    corr = df["session_duration"].corr(df["pre_period_session_duration"])
    assert corr > 0.3


def test_srm_bug_breaks_the_allocation_ratio():
    balanced = _make_df(n=40_000, srm_bug=False)
    broken = _make_df(n=40_000, srm_bug=True)

    balanced_ratio = (balanced.variant == "B").mean()
    broken_ratio = (broken.variant == "B").mean()

    assert abs(balanced_ratio - 0.5) < 0.02
    assert broken_ratio < balanced_ratio - 0.01


def test_reproducible_with_same_seed():
    df1 = _make_df(n=500, seed=7)
    df2 = _make_df(n=500, seed=7)
    pd_testing_equal = (df1["converted"].to_numpy() == df2["converted"].to_numpy()).all()
    assert pd_testing_equal
