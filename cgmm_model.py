from cgmm.regressor import ConditionalGMMRegressor
import numpy as np
from sklearn.utils import check_random_state

def make_model(seed,n_components=2):
    return ConditionalGMMRegressor(
        n_components=n_components,
        covariance_type="full",
        max_iter=500,
        random_state=seed,
        reg_covar=1e-3,
        init_params="kmeans")


def make_sampling_rngs(seed):
    return tuple(
        np.random.RandomState(child.generate_state(1)[0])
        for child in np.random.SeedSequence(seed).spawn(3)
    )


def sample_conditional(model, X, n_samples, *, rng):
    """Sample p(y|X) with an explicit RNG; always return (n, n_samples, d).
    """
    rng = check_random_state(rng)
    mixtures = model.condition(np.atleast_2d(X))
    if not isinstance(mixtures, list):
        mixtures = [mixtures]
    samples = []
    for mixture in mixtures:
        mixture.random_state = rng
        samples.append(mixture.sample(n_samples)[0])
    return np.stack(samples, axis=0)
