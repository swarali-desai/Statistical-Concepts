"""Tests for src.analysis.bayesian."""
import numpy as np

from src.analysis.bayesian import bayesian_bootstrap_mean_diff, beta_binomial_ab_test


def test_beta_binomial_favors_the_better_arm():
    result = beta_binomial_ab_test(conversions_a=100, n_a=1000, conversions_b=140, n_b=1000)
    assert result.prob_b_beats_a > 0.95
    assert result.expected_loss_choose_b < result.expected_loss_choose_a


def test_beta_binomial_uncertain_when_arms_are_identical():
    result = beta_binomial_ab_test(conversions_a=100, n_a=1000, conversions_b=102, n_b=1000)
    assert 0.3 < result.prob_b_beats_a < 0.7


def test_bayesian_bootstrap_recovers_known_mean_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 800)
    b = rng.normal(11, 2, 800)
    result = bayesian_bootstrap_mean_diff(a, b, n_bootstrap=1000, seed=0)
    assert result.mean_diff_ci_low < 1.0 < result.mean_diff_ci_high
    assert result.prob_b_beats_a > 0.9
