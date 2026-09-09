"""Tests for src.analysis.sequential."""
import numpy as np

from src.analysis.sequential import obrien_fleming_boundary, simulate_peeking_inflation


def test_obrien_fleming_boundary_shrinks_to_alpha_boundary_at_full_information():
    from scipy import stats

    boundary_at_1 = obrien_fleming_boundary(1.0, alpha=0.05)
    assert abs(boundary_at_1 - stats.norm.ppf(0.975)) < 1e-9


def test_obrien_fleming_boundary_is_higher_for_earlier_looks():
    early = obrien_fleming_boundary(0.1, alpha=0.05)
    late = obrien_fleming_boundary(0.9, alpha=0.05)
    assert early > late


def test_naive_peeking_inflates_false_positive_rate_above_alpha():
    result = simulate_peeking_inflation(
        n_simulations=400, n_looks=8, n_per_look_per_arm=100, alpha=0.05, seed=1,
    )
    assert result.naive_false_positive_rate > result.alpha


def test_obf_correction_controls_false_positive_rate_near_alpha():
    result = simulate_peeking_inflation(
        n_simulations=400, n_looks=8, n_per_look_per_arm=100, alpha=0.05, seed=1,
    )
    # Should be much closer to the nominal alpha than the naive rate, and
    # comfortably below it (OBF is conservative by construction).
    assert result.obf_false_positive_rate < result.naive_false_positive_rate
    assert result.obf_false_positive_rate < 0.10
