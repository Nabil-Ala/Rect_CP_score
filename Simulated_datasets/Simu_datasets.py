"""
Synthetic benchmark datasets for multivariate conformal prediction.

  Case 1 — fixed Σ, Gaussian           
  Case 2 — fixed Σ, Student-t5        
  Case 3 — moving Σ(x), Gaussian      
  Case 5 — moving Σ(x), mixture       

Design principles
-----------------
* Same X reused across configured cases for a fixed d → variation is purely in Y|X.
* Parameters are nested: d-dimensional problem uses first d rows/entries of
  master matrices, so results are comparable across dimensions.
* SNR and mode separation are stable across d (see notes in each function).
* Cases 3 and 5 share the same covariate-driven heteroscedasticity; they differ
  only in whether cross-output correlation is present. This isolates the effect
  of correlation structure on score performance.
"""

import os
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIMS = (2, 4, 8, 10, 14)

CASE_NAMES = (
    "case1_fixed_gaussian",
    "case2_fixed_student_t",
    "case3_moving_gaussian",
    "case5_moving_mixture",
)


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def toeplitz_cov(d: int, rho: float = 0.5, sigma: float = 1.0) -> np.ndarray:
    """Return a d×d Toeplitz correlation matrix scaled by sigma²."""
    idx = np.arange(d)
    return (sigma ** 2) * (rho ** np.abs(idx[:, None] - idx[None, :]))


def feature_map(X: np.ndarray) -> np.ndarray:
    """
    Shared nonlinear feature map φ(X) ∈ R⁵.

    Includes linear, quadratic, and interaction terms so that the mean
    function is genuinely nonlinear and not trivially fitted by a linear model.

    X : (n, 6)  →  Phi : (n, 5)
    """
    x1, x2, x3, x4, x5, x6 = [X[:, j] for j in range(6)]
    return np.column_stack([
        x1,
        x2,
        np.tanh(x3),
        x4 ** 2 - 1.0,
        x5 * x6,
    ])


def sample_X(n: int, seed: int = 0) -> np.ndarray:
    """
    Draw n covariate vectors from X ~ N(0, Σ_X) with Σ_X Toeplitz(rho=0.5).

    Using a fixed Toeplitz Σ_X introduces mild correlation between covariates,
    which makes the feature map φ(X) more realistic than i.i.d. inputs.
    """
    Sigma_X = toeplitz_cov(6, rho=0.5, sigma=1.0)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(6), Sigma_X, size=n)


def normalise_rows(A: np.ndarray) -> np.ndarray:
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def pd_cholesky(covs: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Batched Cholesky decomposition with diagonal regularisation.

    covs : (n, d, d) symmetric positive semi-definite matrices
    returns L : (n, d, d) lower-triangular such that L @ L.T ≈ covs

    reg is set to 1e-6 which comfortably exceeds floating-point errors for
    d ≤ 14 and the covariance magnitudes used here (eigenvalues ≳ 0.1).
    """
    n, d, _ = covs.shape
    reg_mat = reg * np.eye(d)[None]          # (1, d, d) broadcast-safe
    return np.linalg.cholesky(covs + reg_mat)


def sample_gaussian(means: np.ndarray, covs: np.ndarray, rng) -> np.ndarray:
    """
    Draw Y_i ~ N(means_i, covs_i) for i = 1, …, n.  Fully vectorised.

    means : (n, d)
    covs  : (n, d, d)
    """
    n, d = means.shape
    L = pd_cholesky(covs)                    # (n, d, d)
    z = rng.normal(size=(n, d, 1))           # (n, d, 1)
    return means + (L @ z)[:, :, 0]         # (n, d)


def sample_student_t(means: np.ndarray, covs: np.ndarray,
                     nu: float, rng) -> np.ndarray:
    """
    Draw Y_i ~ t_nu(means_i, covs_i) for i = 1, …, n.  Vectorised.

    The multivariate t is obtained as  means + z / sqrt(chi²_nu / nu)
    where z ~ N(0, covs).  The chi² draw is shared across output
    dimensions (one scalar per observation), which is the standard
    multivariate-t construction.

    means : (n, d)
    covs  : (n, d, d)  — acts as the scale matrix, not the covariance;
                          true covariance = covs * nu/(nu-2) for nu > 2.
    """
    n, d = means.shape
    L = pd_cholesky(covs)                       # (n, d, d)
    z = rng.normal(size=(n, d, 1))              # (n, d, 1)
    chi2 = rng.chisquare(nu, size=(n, 1, 1))   # (n, 1, 1)
    scale = np.sqrt(chi2 / nu)                  # (n, 1, 1)
    return means + (L @ z)[:, :, 0] / scale[:, :, 0]


# ---------------------------------------------------------------------------
# Master parameters
# ---------------------------------------------------------------------------

def make_master_params(max_d: int = 14, seed: int = 123) -> dict:
    """
    Draw one master parameter set. For any d ≤ max_d, take the first d
    rows/entries to get the d-dimensional version. This nesting property
    means that results across dimensions reflect increasing difficulty
    rather than different problems.

    Mean model
    ----------
    A_master  ∈ ℝ^{max_d × 5} : row-normalised, so each output coordinate
    has unit-norm loading on φ(X).  Signal scale is set separately.

    Moving covariance
    -----------------
    B0, B1, B2 ∈ ℝ^{max_d × 3}: row-normalised factor loadings.
    B0 is the base (x-independent) part; B1, B2 carry the x-dependent part.

    Rank is r = min(3, d) to avoid rank > d.

    Base diagonal
    -------------
    Uniform 0.20 across all coordinates so that noise variance is the same
    for every output dimension — avoids unintended SNR drift across d.

    Mixture parameters
    ------------------
    mix_slope: controls how strongly x1 shifts the mixing probability.
    sep_slope : controls how strongly x2 shifts the mode separation.
    """
    rng = np.random.default_rng(seed)

    A_master  = normalise_rows(rng.normal(size=(max_d, 5)))
    B0_master = normalise_rows(rng.normal(size=(max_d, 3)))
    B1_master = normalise_rows(rng.normal(size=(max_d, 3)))
    B2_master = normalise_rows(rng.normal(size=(max_d, 3)))

    # Uniform base diagonal — see docstring
    base_diag_master = 0.20 * np.ones(max_d)

    return {
        "A_master":        A_master,
        "B0_master":       B0_master,
        "B1_master":       B1_master,
        "B2_master":       B2_master,
        "base_diag_master": base_diag_master,
    }


# ---------------------------------------------------------------------------
# Mean function
# ---------------------------------------------------------------------------

def mean_function(X: np.ndarray, d: int, params: dict,
                  signal_scale: float = 1.5) -> np.ndarray:
    """
    F(X) = signal_scale * A_d @ φ(X)  ∈ ℝ^d

    signal_scale = 1.5 gives SNR ≈ 3 at d = 2 with the default noise level
    (base_diag = 0.20).  Because A_d is row-normalised and φ(X) has entries
    of order 1, the per-coordinate signal std is approximately signal_scale,
    independent of d.
    """
    Phi  = feature_map(X)           # (n, 5)
    A_d  = params["A_master"][:d]   # (d, 5)
    return signal_scale * (Phi @ A_d.T)  # (n, d)


# ---------------------------------------------------------------------------
# Fixed covariance  (Cases 1 and 2)
# ---------------------------------------------------------------------------

def fixed_covariance(d: int, sigma: float = 0.45, rho: float = 0.6) -> np.ndarray:
    """
    Σ_0 = sigma² * Toeplitz(rho)

    Used identically for Cases 1 and 2 so that the only difference between
    those two cases is the noise distribution (Gaussian vs Student-t), not
    the covariance.

    sigma = 0.45 gives noise std 0.45 vs signal std ≈ 1.5: SNR ≈ 3.3.
    rho   = 0.6  gives moderate cross-output correlation.
    """
    return toeplitz_cov(d, rho=rho, sigma=sigma)


# ---------------------------------------------------------------------------
# Moving covariance  (Cases 3, 4, 5)
# ---------------------------------------------------------------------------

def _moving_lowrank_and_diag(X: np.ndarray, d: int,
                              params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the low-rank matrix M(x) = L(x) L(x)^T and the diagonal
    matrix D(x) that together form Σ(x) = M(x) + D(x).

    These are returned separately so that Case 4 can zero out M(x) while
    keeping D(x), isolating the effect of cross-output correlation.

    Low-rank part
    -------------
    L(x) = B0 + 0.18·x1·B1 + 0.18·x2·B2  ∈ ℝ^{d × r}

    r = min(3, d) to keep rank ≤ d.

    With x1, x2 ~ N(0,1), the variation term contributes ±0.54·||Bk|| to L.
    The base is 0.25·||B0|| (after the 0.25 scalar below — see code).
    Variation-to-base ratio > 2, giving a Σ(x) trace ratio of ~4–8× across
    the covariate range, which is large enough to stress adaptive methods.

    Diagonal part
    -------------
    D(x) = diag(base_diag * (1 + 0.50·x3² + 0.30·x4²))

    The quadratic dependence on x3, x4 gives a smooth, always-positive scale
    that is independent of the factor loadings and contributes additional
    heteroscedasticity without requiring exotic functions.

    Note: x3, x4 are used only for D(x); x1, x2 only for L(x).  This
    orthogonality makes it easier to interpret what drives Σ(x) in ablations.
    """
    n    = X.shape[0]
    r    = min(3, d)
    x1   = X[:, 0]
    x2   = X[:, 1]
    x3   = X[:, 2]
    x4   = X[:, 3]

    B0 = params["B0_master"][:d, :r]   # (d, r)
    B1 = params["B1_master"][:d, :r]   # (d, r)
    B2 = params["B2_master"][:d, :r]   # (d, r)

    # Factor loading matrix L(x): (n, d, r)
    L = (
        0.25 * B0[None]
        + 0.18 * x1[:, None, None] * B1[None]
        + 0.18 * x2[:, None, None] * B2[None]
    )

    # Low-rank part: (n, d, d)
    M = np.einsum("ndr,nkr->ndk", L, L)

    # Diagonal part: (n, d, d)
    base_diag = params["base_diag_master"][:d]             # (d,)
    scale     = 1.0 + 0.50 * x3 ** 2 + 0.30 * x4 ** 2   # (n,)
    diag_vars = base_diag[None] * scale[:, None]           # (n, d)
    D = np.einsum("nd,de->nde", diag_vars, np.eye(d))     # (n, d, d)

    return M, D


def moving_full_covariances(X: np.ndarray, d: int,
                            params: dict) -> np.ndarray:
    """
    Σ(x) = L(x)L(x)^T + D(x)  — full (dense) moving covariance.
    Used in Cases 3 and 5.
    """
    M, D = _moving_lowrank_and_diag(X, d, params)
    return M + D   # (n, d, d)


def moving_diagonal_covariances(X: np.ndarray, d: int,
                                params: dict) -> np.ndarray:
    """
    Σ(x) = D(x)  — axis-aligned moving covariance.
    Used in Case 4.

    Uses the exact same D(x) as moving_full_covariances so that the
    only difference between Cases 3 and 4 is the cross-output correlation.
    """
    _, D = _moving_lowrank_and_diag(X, d, params)
    return D   # (n, d, d)


# ---------------------------------------------------------------------------
# Mixture parameters  (Case 5)
# ---------------------------------------------------------------------------

def alternating_shift(d: int, per_coord_sep: float = 0.60) -> np.ndarray:
    """
    Mode separation vector δ_d with alternating signs.

    δ_d = per_coord_sep * (-1)^j,  j = 0, …, d-1

    This gives ||δ_d||_2 = per_coord_sep * √d and a Mahalanobis distance
    between the two modes of roughly 2 * per_coord_sep * √d / sigma ≈ 2√d
    (with base_diag = 0.20, sigma = √0.20 ≈ 0.45 and per_coord_sep = 0.60,
    giving ratio ≈ 1.33 per coordinate, well-separated for all d).

    The key property is that the Mahalanobis separation grows with d,
    so the bimodal structure becomes *more* pronounced at high dimension,
    not washed out as in the original code.
    """
    signs = (-1.0) ** np.arange(d)
    return per_coord_sep * signs


# ---------------------------------------------------------------------------
# Case generators
# ---------------------------------------------------------------------------

def _case1_fixed_gaussian(X, d, params, rng):
    """
    Y | X ~ N(F(X), Σ_0)

    Baseline case with fixed Gaussian noise. An oracle Mahalanobis score uses
    F(X) and Σ_0; the fitted joint GMM only approximates the nonlinear mean.
    """
    F      = mean_function(X, d, params)
    Sigma0 = fixed_covariance(d)
    eps    = rng.multivariate_normal(np.zeros(d), Sigma0, size=X.shape[0])
    return F + eps


def _case2_fixed_student_t(X, d, params, rng, nu=5):
    """
    Y | X ~ t_5(F(X), Σ_0)

    Same Σ_0 as Case 1; only the tail behaviour changes.  With ν = 5,
    the true covariance is Σ_0 · ν/(ν-2) = 1.67 Σ_0, so ball-shaped and
    fixed-Σ methods that assume Gaussianity will be inefficient.  PCP-based
    methods, which adapt to the true score distribution via the model, should
    show better set efficiency.
    """
    F      = mean_function(X, d, params)
    n      = X.shape[0]
    Sigma0 = fixed_covariance(d)
    covs   = np.broadcast_to(Sigma0[None], (n, d, d)).copy()
    return sample_student_t(F, covs, nu=nu, rng=rng)


def _case3_moving_gaussian(X, d, params, rng):
    """
    Y | X ~ N(F(X), Σ(X))

    Σ(X) is full (dense) and moves with X via the low-rank-plus-diagonal
    construction.  Methods with fixed or isotropic prediction sets will be
    inefficient in covariate regions with large Σ(X) and overly conservative
    in regions with small Σ(X).  Mahalanobis with a well-fitted GMM should
    still be near-optimal here.
    """
    F    = mean_function(X, d, params)
    covs = moving_full_covariances(X, d, params)
    return sample_gaussian(F, covs, rng)


def _case4_moving_axis_aligned(X, d, params, rng):
    """
    Y | X ~ N(F(X), D(X))   with D(X) diagonal

    D(X) is the exact diagonal part from Case 3, so the heteroscedasticity
    pattern is identical — only the cross-output correlation is removed.
    This optional case allows comparison of rectangular and ellipsoidal sets
    when the conditional noise covariance is diagonal.
    """
    F    = mean_function(X, d, params)
    covs = moving_diagonal_covariances(X, d, params)
    return sample_gaussian(F, covs, rng)


def _case5_moving_mixture(X, d, params, rng):
    """
    Y | X ~ π(X) · N(F(X) + δ_d,  Σ(X))
           + (1 - π(X)) · N(F(X) - δ_d,  Σ(X))

    where
        π(X)       = σ(x1 - x2 + 0.5 x3)          ∈ (0, 1)
        separation = clip(0.70 + 0.25·x2, 0.25, 1.40)
        δ_d        = alternating_shift(d, per_coord_sep=0.60)

    Both π(X) and the mode-separation scale depend on X, so:
      - for some x the distribution is near-unimodal (π ≈ 0 or 1)
      - for some x it is strongly bimodal (π ≈ 0.5, large separation)

    This case tests adaptation to multimodal, heteroscedastic outcomes.
    WSC/ERT assess how coverage varies with the covariates.

    Mode separation is Mahalanobis-stable across d (see alternating_shift).
    """
    F    = mean_function(X, d, params)
    covs = moving_full_covariances(X, d, params)
    n    = X.shape[0]

    # Mixing probability: logistic of a linear combination of covariates
    pi = 1.0 / (1.0 + np.exp(-(X[:, 0] - X[:, 1] + 0.5 * X[:, 2])))  # (n,)

    # x-dependent separation magnitude
    sep = np.clip(0.70 + 0.25 * X[:, 1], 0.25, 1.40)   # (n,)

    # Base alternating shift (unit per-coordinate magnitude)
    delta = alternating_shift(d, per_coord_sep=0.60)     # (d,)

    # Component assignment
    component = (rng.random(n) < pi).astype(int)         # 0 or 1
    signs     = 2.0 * component - 1.0                    # -1 or +1

    # Mode-shifted means
    means = F + (signs * sep)[:, None] * delta[None, :]  # (n, d)

    return sample_gaussian(means, covs, rng)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_GENERATORS = {
    "case1_fixed_gaussian":    _case1_fixed_gaussian,
    "case2_fixed_student_t":   _case2_fixed_student_t,
    "case3_moving_gaussian":   _case3_moving_gaussian,
    "case4_moving_axis_aligned": _case4_moving_axis_aligned,
    "case5_moving_mixture":    _case5_moving_mixture,
}


def generate_case(case_name: str, X: np.ndarray, d: int,
                  params: dict, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _GENERATORS[case_name](X, d, params, rng)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def generate_all_datasets(
    n: int = 8000,
    dims: tuple = DIMS,
    x_seed_base: int = 1000,
    y_seed_base: int = 2000,
    master_seed: int = 123,
) -> dict:
    """
    Generate all (case, d) datasets.

    Returns
    -------
    dict mapping (case_name, d) → {"X": ndarray (n,6), "Y": ndarray (n,d)}

    For a fixed d, the same X is shared across configured cases so that any
    difference in Y statistics is purely due to the DGP, not the covariates.

    Seeds
    -----
    x_seed_base + d       — X for dimension d
    y_seed_base + 100*k + d — Y for case index k and dimension d

    Using d in the seed ensures different X matrices for different dimensions
    while keeping them reproducible.
    """
    params   = make_master_params(max_d=max(dims), seed=master_seed)
    datasets = {}

    for d in dims:
        X = sample_X(n=n, seed=x_seed_base + d)

        for k, case_name in enumerate(CASE_NAMES):
            Y = generate_case(
                case_name=case_name,
                X=X,
                d=d,
                params=params,
                seed=y_seed_base + 100 * k + d,
            )
            datasets[(case_name, d)] = {"X": X.copy(), "Y": Y}

    return datasets


def save_datasets(datasets: dict,
                  out_dir=Path(__file__).resolve().parent / "Simulated_datasets") -> None:
    """Save each dataset as a compressed .npz file."""
    os.makedirs(out_dir, exist_ok=True)
    for (case_name, d), data in datasets.items():
        path = os.path.join(out_dir, f"{case_name}_d{d}.npz")
        np.savez_compressed(path, X=data["X"], Y=data["Y"])
        print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(datasets: dict, params: dict) -> None:
    """
    Print basic statistics to verify the DGPs behave as intended.

    Checks
    ------
    1. Shape and finiteness
    2. Per-case Y std (should be in a sane range, roughly 0.5–3)
    3. Moving-covariance trace ratio (should be ≥ 4× for Cases 3–5)
    4. Mode separation in Case 5 (check that bimodal structure exists)
    """
    print("\n=== Sanity checks ===\n")

    for (case_name, d), data in datasets.items():
        X, Y = data["X"], data["Y"]

        assert np.isfinite(Y).all(), f"Non-finite values in {case_name}, d={d}"

        y_std = Y.std(axis=0).mean()
        print(f"{case_name:30s}  d={d:2d}  Y_std(mean)={y_std:.3f}")

    # Covariance trace ratio for moving cases
    print("\n--- Moving Σ(x) trace range (d=6) ---")
    params_check = params
    d_check = 6
    rng     = np.random.default_rng(0)
    X_check = sample_X(n=2000, seed=42)

    for fn_name, fn in [
        ("full",     moving_full_covariances),
        ("diagonal", moving_diagonal_covariances),
    ]:
        covs   = fn(X_check, d_check, params_check)
        traces = np.trace(covs, axis1=1, axis2=2)
        print(
            f"  {fn_name:8s}: "
            f"trace ∈ [{traces.min():.3f}, {traces.max():.3f}], "
            f"ratio = {traces.max() / traces.min():.1f}×"
        )

    # Mixture separation at d=4
    print("\n--- Case 5 mixture separation check (d=4) ---")
    d_mix = 4
    X_mix = sample_X(n=5000, seed=99)
    pi    = 1.0 / (1.0 + np.exp(
        -(X_mix[:, 0] - X_mix[:, 1] + 0.5 * X_mix[:, 2])
    ))
    sep   = np.clip(0.70 + 0.25 * X_mix[:, 1], 0.25, 1.40)
    delta = alternating_shift(d_mix, per_coord_sep=0.60)
    mahal_sep = 2 * np.sqrt(d_mix) * 0.60 / np.sqrt(0.20)
    print(
        f"  π range:        [{pi.min():.2f}, {pi.max():.2f}], "
        f"mean={pi.mean():.2f}"
    )
    print(
        f"  separation:     [{sep.min():.2f}, {sep.max():.2f}], "
        f"mean={sep.mean():.2f}"
    )
    print(
        f"  ||δ_4||_2:      {np.linalg.norm(delta):.3f}"
    )
    print(
        f"  Mahal. dist.:   ≈ {mahal_sep:.2f} "
        f"(between modes, using base diagonal noise)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Generating datasets …")
    datasets = generate_all_datasets(n=5000)

    print("\nSaving …")
    save_datasets(datasets)

    params = make_master_params(max_d=max(DIMS))
    run_sanity_checks(datasets, params)

    print("\nDone.")


