"""Bayesian A/B testing.

Frequentist tests answer "how surprising would this data be if there were no
effect?". Product stakeholders usually want the answer to a different
question: "how likely is B better than A, and what do we lose if we pick
wrong?". Bayesian testing answers that directly and doesn't require
correcting for peeking (the posterior is always valid given the data
observed so far), which is why it pairs well with the sequential-monitoring
module for teams that want to check results daily.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BetaBinomialResult:
    prob_b_beats_a: float
    expected_loss_choose_b: float
    expected_loss_choose_a: float
    rel_lift_ci_low: float
    rel_lift_ci_high: float
    posterior_a: np.ndarray
    posterior_b: np.ndarray


def beta_binomial_ab_test(
    conversions_a: int,
    n_a: int,
    conversions_b: int,
    n_b: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 200_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BetaBinomialResult:
    """Beta-Binomial conjugate model for a binary metric (e.g. conversion).

    Uses Monte Carlo on the posteriors (rather than closed-form) so the same
    pattern generalizes to metrics without a conjugate prior.
    """
    rng = np.random.default_rng(seed)

    post_a = rng.beta(prior_alpha + conversions_a, prior_beta + n_a - conversions_a, n_samples)
    post_b = rng.beta(prior_alpha + conversions_b, prior_beta + n_b - conversions_b, n_samples)

    prob_b_beats_a = float(np.mean(post_b > post_a))

    # Expected loss: the average shortfall (in conversion-rate points) you'd
    # incur by picking that arm, given full posterior uncertainty. This is a
    # standard Bayesian A/B decision rule (Stucchio, "Bayesian A/B Testing
    # at VWO") that's more actionable than a point estimate for teams
    # deciding whether the evidence is strong enough to ship.
    loss_choose_b = np.maximum(post_a - post_b, 0)
    loss_choose_a = np.maximum(post_b - post_a, 0)

    rel_lift = (post_b - post_a) / post_a
    ci_low, ci_high = np.percentile(rel_lift, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return BetaBinomialResult(
        prob_b_beats_a=prob_b_beats_a,
        expected_loss_choose_b=float(loss_choose_b.mean()),
        expected_loss_choose_a=float(loss_choose_a.mean()),
        rel_lift_ci_low=float(ci_low),
        rel_lift_ci_high=float(ci_high),
        posterior_a=post_a,
        posterior_b=post_b,
    )


@dataclass
class BayesianBootstrapResult:
    prob_b_beats_a: float
    mean_diff_ci_low: float
    mean_diff_ci_high: float
    posterior_diff: np.ndarray


def bayesian_bootstrap_mean_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 4_000,
    alpha: float = 0.05,
    seed: int = 42,
    batch_size: int = 200,
) -> BayesianBootstrapResult:
    """Bayesian bootstrap (Rubin, 1981) posterior for the difference in means
    of a continuous metric (e.g. revenue-per-user), with no conjugate prior
    needed. Each draw reweights the observed sample with Dirichlet(1,...,1)
    weights, which approximates sampling from the Bayesian nonparametric
    posterior over the data-generating distribution.

    Dirichlet(1,...,1) weights are equivalent to n i.i.d. Exponential(1)
    draws normalized to sum to 1, which vectorizes far better than sampling
    a length-n Dirichlet per draw -- important here since n can be in the
    tens of thousands. Draws are processed in batches to bound memory.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    n_a, n_b = len(a), len(b)

    diffs = np.empty(n_bootstrap)
    done = 0
    while done < n_bootstrap:
        cur = min(batch_size, n_bootstrap - done)
        w_a = rng.standard_exponential((cur, n_a))
        w_a /= w_a.sum(axis=1, keepdims=True)
        w_b = rng.standard_exponential((cur, n_b))
        w_b /= w_b.sum(axis=1, keepdims=True)
        diffs[done:done + cur] = (w_b @ b) - (w_a @ a)
        done += cur

    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return BayesianBootstrapResult(
        prob_b_beats_a=float(np.mean(diffs > 0)),
        mean_diff_ci_low=float(ci_low),
        mean_diff_ci_high=float(ci_high),
        posterior_diff=diffs,
    )
