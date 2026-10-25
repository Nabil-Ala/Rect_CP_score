import math
import numpy as np
from scipy.special import logsumexp


from CP_scores import PCP_score
from cgmm_model import sample_conditional
from utils import _union_length_1d


def MC_volume_estimate(x0, model, M, score_choice, Quantile, refs_x0=None, *, rng):
    """PCP volume by seeded importance sampling from the conditional mixture."""
    if score_choice != "PCP":
        raise ValueError("The supported Monte Carlo score is PCP.")
    if M < 1:
        raise ValueError("M must be positive.")
    if np.isposinf(Quantile):
        return np.inf
    samples = sample_conditional(model, x0, M, rng=rng)[0]
    covered = PCP_score(np.asarray(x0), samples, refs_x0) <= Quantile
    if not np.any(covered):
        return 0.0
    mixture = model.condition(np.atleast_2d(x0))
    if isinstance(mixture, list):
        mixture = mixture[0]
    weights = np.asarray(mixture.weights_)
    if (not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0)):
        raise ValueError("Conditional GMM weights must be finite, nonnegative, and sum to one.")
    # Conditioning can underflow negligible component weights to zero.
    # log(0) = -inf correctly gives those components zero density contribution.
    with np.errstate(divide="ignore"):
        log_density = mixture.score_samples(samples[covered])
    if not np.isfinite(log_density).all():
        raise FloatingPointError("Conditional GMM produced non-finite log densities.")
    return float(np.exp(logsumexp(-log_density) - np.log(M)))

def Length_PCP_set_i(x0, i, q, refs_x0):
    x0 = np.atleast_2d(x0)
 
    if refs_x0.ndim == 3:
        refs_x0 = refs_x0[0]   # (L, d)

    centers = refs_x0[:, i]

    return _union_length_1d(centers, q)

def volume_linf(qhat, d):
    return (2 * qhat) ** d

def lp_unit_ball_volume(d, p):
    return (2 * math.gamma(1 + 1/p))**d / math.gamma(1 + d/p)

def volume_lp(qhat, d, p):
    return lp_unit_ball_volume(d, p) * (qhat ** d)

def unit_ball_volume(d):
    return math.pi**(d/2) / math.gamma(d/2 + 1)

def mahalanobis_volumes(model, X_test, qhat):
    _, cov = model.predict(X_test, return_cov=True)
    d = cov.shape[-1]
    chol = np.linalg.cholesky(cov)
    log_sqrt_det = np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
    log_unit_volume = (d / 2) * np.log(np.pi) - math.lgamma(d / 2 + 1)
    with np.errstate(divide="ignore"):
        log_volumes = log_unit_volume + d * np.log(qhat) + log_sqrt_det
    return np.exp(log_volumes)


def mahalanobis_rectangular_volumes(model, X_test, qhat):
    _, cov = model.predict(X_test, return_cov=True)
    variances = np.diagonal(cov, axis1=-2, axis2=-1)
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("Predictive variances must be finite and positive.")
    d = cov.shape[-1]
    with np.errstate(divide="ignore"):
        log_volumes = d * (np.log(2) + np.log(qhat)) + 0.5 * np.log(variances).sum(axis=-1)
    return np.exp(log_volumes)
