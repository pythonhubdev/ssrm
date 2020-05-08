import warnings
from functools import reduce

import numpy as np
from pytest import approx
from scipy.stats import multinomial

from .ssrm_test import (
    bayes_factor,
    log_posterior_predictive,
    multinomiallogpmf,
    posterior_probability,
    sequential_posteriors,
)


def test_accumulator():
    """
    Tests that the posterior probability computed sequentially
    via accumulation is equal to the posterior probability
    computed in a batch manner.
    """
    theta = np.array([1 / 3, 1 / 3, 1 / 3])
    alpha = np.array([1, 3, 2])
    sample_size = 40
    observations = multinomial.rvs(1, theta, size=sample_size)
    observations_sum = reduce(lambda x, y: x + y, observations)
    final_posterior = sequential_posteriors(observations, theta, alpha)[-1]
    final_bf = bayes_factor(final_posterior)
    post_prob = posterior_probability(final_bf)
    log_marginal_likelihood_M1 = log_posterior_predictive(observations_sum, alpha)
    log_marginal_likelihood_M0 = multinomial.logpmf(
        observations_sum, observations_sum.sum(), theta
    )
    log_odds = log_marginal_likelihood_M1 - log_marginal_likelihood_M0
    odds = np.exp(log_odds)
    assert post_prob == approx(odds / (1 + odds))


def test_overflow():
    posterior = {
        "log_marginal_likelihood_M1": -23.585991739528254,
        "log_marginal_likelihood_M0": -76546.65811250894,
        "posterior_M1": [88519.5, 12540.25, 13002.25],
        "posterior_M0": [0.75, 0.125, 0.125],
    }
    # Assert there are no warnings in code exec
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bf = bayes_factor(posterior)
        assert np.isinf(bf)
        assert posterior_probability(bf) == approx(1)
        assert len(w) == 0


def test_multinomiallogpmf():
    xs = [[10, 20, 30], [1, 1, 1], [0, 0, 1], [12, 30, 8]]
    ns = [sum(x) for x in xs]
    ps = [[0.2, 0.4, 0.4], [1 / 3, 1 / 3, 1 / 3], [0.3, 0.4, 0.3], [0, 0, 1]]
    for x, n, p in zip(xs, ns, ps):
        assert multinomial.logpmf(x, n, p) == approx(multinomiallogpmf(x, n, p))
