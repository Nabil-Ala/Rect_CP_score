from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def split_data(X, y, seed):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    X_cal = scaler_X.transform(X_cal)
    y_cal = scaler_y.transform(y_cal)

    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test)   

    return X_train, X_cal, y_train, y_cal, X_test, y_test

def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a nonempty finite vector.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    scores = np.sort(scores)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else scores[k - 1]

def conditional_metrics(x_test, covered, alpha):
    from covmetrics import ERT, WSC
    from tabicl import TabICLClassifier

    x_test = np.asarray(x_test, dtype=float)
    covered = np.asarray(covered, dtype=int)

    unique_classes = np.unique(covered)

    if unique_classes.size < 2:
        ert_value = float("nan")
    else:
        ert_estimator = ERT(model_cls=TabICLClassifier)
        ert_value = float(ert_estimator.evaluate(x_test, covered, alpha, n_splits=5))

    try:
        wsc_value = float(WSC().evaluate(x_test, cover=covered, delta=alpha))
    except Exception:
        wsc_value = float("nan")

    return ert_value, wsc_value


def _union_length_1d(centers, radius):
    centers = np.sort(np.asarray(centers, dtype=float))
    if len(centers) == 0:
        return 0.0
    if radius < 0:
        return 0.0

    starts = centers - radius
    ends = centers + radius

    total = 0.0
    cur_a, cur_b = starts[0], ends[0]

    for a, b in zip(starts[1:], ends[1:]):
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += cur_b - cur_a
            cur_a, cur_b = a, b

    total += cur_b - cur_a
    return float(total)
