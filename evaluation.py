"""Shared split-conformal evaluation of the three methods and their rectangularization."""

import numpy as np
import pandas as pd

from cgmm_model import make_model, make_sampling_rngs, sample_conditional
from CP_scores import (
    L_infty_score, Lp_score, Mahalanobis_score, Mahalanobis_score_i,
    PCP_score, PCP_score_i,
)
from utils import split_data, conformal_quantile, conditional_metrics
from volume_est import (
    volume_linf, volume_lp, mahalanobis_volumes,
    mahalanobis_rectangular_volumes, MC_volume_estimate, Length_PCP_set_i,
)


def evaluate_cp(X, y, alpha, seed, n_components, dataset_name="dataset", *,
                max_samples=None):
    if max_samples is not None:
        X, y = X[:max_samples], y[:max_samples]
    X_train, X_cal, Y_train, Y_cal, X_test, Y_test = split_data(X, y, seed)
    d = Y_test.shape[1]
    model = make_model(seed=seed, n_components=n_components)
    model.fit(X_train, Y_train)

    rng_cal, rng_test, rng_volume = make_sampling_rngs(seed)
    refs_cal = sample_conditional(model, X_cal, 200, rng=rng_cal)
    refs_test = sample_conditional(model, X_test, 200, rng=rng_test)

    def scores(X_part, Y_part, refs):
        return {
            "L_infty": L_infty_score(model, X_part, Y_part),
            "Lp": Lp_score(model, X_part, Y_part, p=2),
            "Mahalanobis": Mahalanobis_score(model, X_part, Y_part),
            "Maha_rect": np.max([
                Mahalanobis_score_i(model, X_part, Y_part, j) for j in range(d)
            ], axis=0),
            "PCP": PCP_score(X_part, Y_part, refs),
            "PCP_rect": np.max([
                PCP_score_i(X_part, Y_part, j, refs) for j in range(d)
            ], axis=0),
        }

    quantiles = {
        method: conformal_quantile(values, alpha)
        for method, values in scores(X_cal, Y_cal, refs_cal).items()
    }
    test_scores = scores(X_test, Y_test, refs_test)
    volumes = {
        "L_infty": np.full(len(X_test), volume_linf(quantiles["L_infty"], d)),
        "Lp": np.full(len(X_test), volume_lp(quantiles["Lp"], d, p=2)),
        "Mahalanobis": mahalanobis_volumes(model, X_test, quantiles["Mahalanobis"]),
        "Maha_rect": mahalanobis_rectangular_volumes(model, X_test, quantiles["Maha_rect"]),
        "PCP": np.array([
            MC_volume_estimate(
                x, model, M=300, score_choice="PCP", Quantile=quantiles["PCP"],
                refs_x0=refs_test[i], rng=rng_volume,
            ) for i, x in enumerate(X_test)
        ]),
        "PCP_rect": np.array([
            np.prod([
                Length_PCP_set_i(x, j, quantiles["PCP_rect"], refs_test[i])
                for j in range(d)
            ]) for i, x in enumerate(X_test)
        ]),
    }

    rows = []
    for method, values in test_scores.items():
        covered = values <= quantiles[method]
        ert, wsc = conditional_metrics(X_test, covered, alpha)
        with np.errstate(divide="ignore"):
            median_log_volume = float(np.median(np.log(volumes[method])))
        rows.append({
            "dataset_name": dataset_name,
            "n": len(X),
            "d": d,
            "method": method,
            "alpha": alpha,
            "seed": seed,
            "n_components": n_components,
            "coverage": float(np.mean(covered)),
            "ERT": ert,
            "WSC": wsc,
            "median_volume": median_log_volume,
        })
    return pd.DataFrame(rows)
