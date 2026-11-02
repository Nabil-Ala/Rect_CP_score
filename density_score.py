"""Compare density and rectangularized-density conformal prediction on real datasets."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from cgmm_model import make_model, make_sampling_rngs
from data.data_loading import load_real_data
from main_real_data import COMPONENTS_BY_DATASET, real_specs
from utils import split_data, conformal_quantile, conditional_metrics


# Experiment settings
ALPHA = 0.10
SEEDS = range(20)
MAX_SAMPLES = 8000
N_COMPONENTS = 3
MC_SAMPLES = 300
GRID_SIZE = 512
MAX_ITER = 300
TOL = 1e-7
BATCH_SIZE = 128

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = BASE_DIR / "results" / "density_real_results.csv"
METHODS = ("Density", "Density_rect")
DATASETS = None  # None = all real datasets; or set {"dataset_name"}


# 1. The two nonconformity scores

def log_density(params, y):
    """Evaluate log p(y|x), allowing zero-weight mixture components."""
    # Conditioning can make negligible component weights exactly zero.
    # Their log weights are -inf, which correctly contributes zero density.
    with np.errstate(divide="ignore"):
        values = params["mix"].score_samples(y)
    if not np.isfinite(values).all():
        raise FloatingPointError("The CGMM returned nonfinite log densities.")
    return values


def density_score(params, y):
    """s(x,y) = -log p(y|x)."""
    return -float(log_density(params, y[None, :])[0])


def rectangular_score(params, y):
    """s_rect(x,y) = max_j {-log sup_{y_-j} p(y_j, y_-j|x)}."""
    return max(
        -float(profile_log_density(params, j, [y[j]])[0])
        for j in range(params["d"])
    )


def scores(params, y):
    return density_score(params, y), rectangular_score(params, y)


# 2. Numerical computation of the profile density

def prepare_mixture(mix):
    """Precompute the quantities needed for a conditional Gaussian mixture."""
    if (not np.isfinite(mix.weights_).all() or np.any(mix.weights_ < 0)
            or not np.isclose(mix.weights_.sum(), 1.0)):
        raise ValueError("Mixture weights must be finite, nonnegative, and sum to one.")
    active = mix.weights_ > 0
    weights = mix.weights_[active]
    means = mix.means_[active]
    cov = mix.covariances_[active]
    k, d = means.shape
    chol = np.linalg.cholesky(cov)
    log_det = 2 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)

    return {
        "mix": mix, "means": means, "cov": cov, "k": k, "d": d,
        "var": np.diagonal(cov, axis1=1, axis2=2),
        "precision": np.linalg.solve(cov, np.broadcast_to(np.eye(d), cov.shape)),
        "log_peaks": np.log(weights) - (d * np.log(2 * np.pi) + log_det) / 2,
        "unconverged": 0,
    }


def maximize_profiles(means, amplitudes, precision, params):
    """Maximize each Gaussian-mixture profile with multi-start modal EM."""
    k = means.shape[1]
    starts = [means]
    for a, b in combinations(range(k), 2):
        starts.append(((means[:, a] + means[:, b]) / 2)[:, None, :])
    weights = np.exp(amplitudes - logsumexp(amplitudes, axis=1)[:, None])
    starts.append(np.sum(weights[:, :, None] * means, axis=1)[:, None, :])

    starts = np.concatenate(starts, axis=1)
    n_values, n_starts, dim = starts.shape
    positions = starts.reshape(-1, dim).copy()
    value_index = np.repeat(np.arange(n_values), n_starts)
    best = np.full(len(positions), -np.inf)
    active = np.arange(len(positions))

    def objective(z, index):
        diff = z[:, None, :] - means[index]
        p_diff = np.einsum("kij,nkj->nki", precision, diff)
        terms = amplitudes[index] - 0.5 * np.sum(diff * p_diff, axis=2)
        return logsumexp(terms, axis=1), terms, p_diff

    for _ in range(MAX_ITER):
        index = value_index[active]
        old, terms, p_diff = objective(positions[active], index)
        responsibilities = np.exp(terms - old[:, None])
        curvature = np.einsum("nk,kij->nij", responsibilities, precision)
        gradient = -np.sum(responsibilities[:, :, None] * p_diff, axis=1)
        step = np.linalg.solve(curvature, gradient[..., None])[..., 0]
        proposed = positions[active] + step
        new, _, _ = objective(proposed, index)
        improved = new >= old
        positions[active[improved]] = proposed[improved]
        best[active] = np.maximum(best[active], np.maximum(old, new))
        active = active[(new - old) > TOL]
        if len(active) == 0:
            break

    params["unconverged"] += len(active)
    return best.reshape(n_values, n_starts).max(axis=1)


def profile_log_density(params, coordinate, values):
    """log sup_{y_-j} p(y_j=t, y_-j | x), evaluated at each t."""
    values = np.atleast_1d(values)
    means, cov = params["means"], params["cov"]
    diff = values[:, None] - means[None, :, coordinate]
    amplitudes = params["log_peaks"][None, :] - diff**2 / (2 * params["var"][None, :, coordinate])

    if params["d"] == 1:
        return logsumexp(amplitudes, axis=1)
    if params["k"] == 1:
        return amplitudes[:, 0]

    other = np.arange(params["d"]) != coordinate
    slope = cov[:, other, coordinate] / params["var"][:, coordinate, None]
    conditional_means = means[:, other][None, :, :] + diff[:, :, None] * slope[None, :, :]
    precision = params["precision"][:, other, :][:, :, other]

    return np.concatenate([
        maximize_profiles(conditional_means[i:i + BATCH_SIZE],
                          amplitudes[i:i + BATCH_SIZE], precision, params)
        for i in range(0, len(values), BATCH_SIZE)
    ])


# 3. Prediction-set volumes

def density_log_volume(params, q, rng):
    """Monte Carlo estimate of log Vol({y: -log p(y|x) <= q})."""
    if np.isinf(q):
        return float(q)
    mix = params["mix"]
    mix.random_state = rng
    samples = mix.sample(MC_SAMPLES)[0]
    log_p = log_density(params, samples)
    inside = -log_p <= q
    if not inside.any():
        return -np.inf
    return float(logsumexp(-log_p[inside]) - np.log(len(samples)))


def coordinate_length(params, j, q):
    """Grid estimate of the length of {t: -log sup_{y_-j} p(y|x) <= q}."""
    if np.isposinf(q):
        return np.inf
    if np.isneginf(q):
        return 0.0

    ratio = params["log_peaks"] + q + np.log(params["k"])
    active = ratio > 0
    radius = np.sqrt(2 * params["var"][active, j] * ratio[active])
    centers = params["means"][active, j]
    intervals = sorted(zip(centers - radius, centers + radius))

    merged = []
    for left, right in intervals:
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])

    length = 0.0
    for left, right in merged:
        centers = params["means"][:, j]
        centers = centers[(left <= centers) & (centers <= right)]
        grid = np.unique(np.r_[np.linspace(left, right, GRID_SIZE), centers])
        h = profile_log_density(params, j, grid) + q
        inside_left, inside_right = h[:-1] >= 0, h[1:] >= 0
        fraction = (inside_left & inside_right).astype(float)
        crossing = inside_left != inside_right
        fraction[crossing] = np.where(
            inside_left[crossing],
            h[:-1][crossing] / (h[:-1][crossing] - h[1:][crossing]),
            h[1:][crossing] / (h[1:][crossing] - h[:-1][crossing]),
        )
        length += float(np.dot(np.diff(grid), fraction))
    return length


def rectangular_log_volume(params, q):
    """log Vol(C_rect) = sum_j log(length of coordinate set j)."""
    lengths = [coordinate_length(params, j, q) for j in range(params["d"])]
    with np.errstate(divide="ignore"):
        return float(np.log(lengths).sum())


# 4. Real-data experiments

def conditionals(model, X):
    mixtures = model.condition(np.atleast_2d(X))
    return mixtures if isinstance(mixtures, list) else [mixtures]


def evaluate_dataset(X, Y, name, seed, n_components):
    X, Y = np.asarray(X)[:MAX_SAMPLES], np.asarray(Y)[:MAX_SAMPLES]
    X_train, X_cal, Y_train, Y_cal, X_test, Y_test = split_data(X, Y, seed)
    model = make_model(seed=seed, n_components=n_components)
    model.fit(X_train, Y_train)
    _, _, rng = make_sampling_rngs(seed)

    cal_scores = []
    unfinished = 0
    for mix, y in zip(conditionals(model, X_cal), Y_cal):
        params = prepare_mixture(mix)
        cal_scores.append(scores(params, y))
        unfinished += params["unconverged"]

    cal_scores = np.asarray(cal_scores)
    q = [conformal_quantile(cal_scores[:, j], ALPHA) for j in range(2)]

    test_scores, log_volumes = [], []
    for mix, y in zip(conditionals(model, X_test), Y_test):
        params = prepare_mixture(mix)
        test_scores.append(scores(params, y))
        log_volumes.append([
            density_log_volume(params, q[0], rng),
            rectangular_log_volume(params, q[1]),
        ])
        unfinished += params["unconverged"]

    test_scores, log_volumes = np.asarray(test_scores), np.asarray(log_volumes)
    rows = []
    for j, method in enumerate(METHODS):
        covered = test_scores[:, j] <= q[j]
        ert, wsc = conditional_metrics(X_test, covered, ALPHA)
        rows.append({
            "dataset_name": name, "n": len(X), "d": Y.shape[1],
            "method": method, "alpha": ALPHA, "seed": seed,
            "n_components": n_components, "coverage": float(covered.mean()),
            "ERT": ert, "WSC": wsc,
            "median_volume": float(np.median(log_volumes[:, j])),
            "quantile": q[j], "mc_samples": MC_SAMPLES, "grid_size": GRID_SIZE,
            "profile_max_iter": MAX_ITER, "profile_tol": TOL,
            "profile_unconverged_starts": unfinished if j == 1 else 0,
            "zero_volume_estimates": int(np.isneginf(log_volumes[:, j]).sum()),
        })
    return pd.DataFrame(rows)

from tqdm.auto import tqdm
def run_experiments():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    first = True

    for root, family, name in tqdm(real_specs):
        if DATASETS is not None and name not in DATASETS:
            continue
        root = Path(root)
        if not root.is_absolute():
            root = BASE_DIR / root
        X, Y = load_real_data(root, family, name)
        components = COMPONENTS_BY_DATASET.get(name.lower(), N_COMPONENTS)

        for seed in SEEDS:
            print(f"{family}_{name} | seed={seed}", flush=True)
            result = evaluate_dataset(X, Y, f"{family}_{name}", seed, components)
            result.to_csv(OUTPUT_CSV, mode="w" if first else "a",
                          header=first, index=False)
            first = False

    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_experiments()
