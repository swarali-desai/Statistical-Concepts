"""Tests for src.analysis.decision."""
import numpy as np
import pandas as pd

from src.analysis.decision import GuardrailCheck, make_ship_decision
from src.analysis.frequentist import continuous_test, proportion_test


def _balanced_experiment_df(n=20_000, seed=0, oec_lift=0.03, guardrail_regression=0.0):
    rng = np.random.default_rng(seed)
    variant = np.where(rng.random(n) < 0.5, "A", "B")
    device = rng.choice(["mobile", "desktop", "tablet"], size=n)

    p = np.where(variant == "A", 0.10, 0.10 + oec_lift)
    converted = rng.binomial(1, p)

    load_ms = rng.normal(600 + np.where(variant == "B", guardrail_regression, 0.0), 80)

    return pd.DataFrame(
        {"variant": variant, "device_type": device, "converted": converted, "page_load_ms": load_ms}
    )


def _decide(df, alpha=0.05):
    oec = proportion_test(df, metric_col="converted", alpha=alpha)
    load_result = continuous_test(df, metric_col="page_load_ms", alpha=alpha)
    guardrail = GuardrailCheck(result=load_result, direction="lower_is_better", max_acceptable_regression=10.0)
    return make_ship_decision(
        df, oec_result=oec, guardrails=[guardrail],
        srm_covariate_cols=["device_type"], alpha=alpha,
    )


def test_ships_when_oec_positive_and_guardrails_clean():
    df = _balanced_experiment_df(oec_lift=0.05, guardrail_regression=0.0)
    decision = _decide(df)
    assert decision.verdict == "SHIP"


def test_holds_when_oec_positive_but_guardrail_regressed():
    df = _balanced_experiment_df(oec_lift=0.05, guardrail_regression=40.0)
    decision = _decide(df)
    assert decision.verdict == "HOLD"


def test_does_not_ship_when_oec_flat():
    df = _balanced_experiment_df(n=2000, oec_lift=0.0, seed=5)
    decision = _decide(df)
    assert decision.verdict == "DO_NOT_SHIP"


def test_does_not_ship_when_srm_detected():
    df = _balanced_experiment_df(oec_lift=0.05)
    # Break the allocation ratio directly to simulate an SRM bug.
    b_idx = df[df.variant == "B"].sample(frac=0.4, random_state=0).index
    df = df.drop(index=b_idx)

    decision = _decide(df)
    assert decision.verdict == "DO_NOT_SHIP"
    assert decision.srm_passed is False
