import numpy as np

""" Scores and their rectangularizations"""

def L_infty_score(model, X_cal, y_cal):
    y_cal_pred = model.predict(X_cal)
    eps = np.abs(y_cal - y_cal_pred)
    scores= np.max(eps, axis=1)
    return scores

def Lp_score(model, X, y, p):
    y_cal_pred = model.predict(X)
    eps = np.abs(y - y_cal_pred)
    scores= np.power(np.sum(eps**p, axis=1), 1/p)
    return scores

def Mahalanobis_score(model, X, y):
    y_cal_pred, cov = model.predict(X, return_cov=True)
    eps = y - y_cal_pred
    scores = np.array([np.sqrt(eps[i].T @ np.linalg.inv(cov[i]) @ eps[i]) for i in range(len(X))])
    return scores

def Mahalanobis_score_i(model, X, y, i):
    y_pred, cov = model.predict(X, return_cov=True)
    eps = y - y_pred
    scores = np.sqrt(eps[:, i]**2 / cov[:, i, i])
    return scores

def PCP_score(X, y,refs):
    if X.ndim == 1:
        X = X[None, :]        # (1, p)
    if y.ndim == 1:
        y = y[None, :]        # (1, d)

    scores=np.min(np.linalg.norm(refs - y[:, np.newaxis, :], axis=2), axis=1)
    return scores

def PCP_score_i( X, y, i, refs):
    if X.ndim == 1:
        X = X[None, :]        # (1, p)
    if y.ndim == 1:
        y = y[None, :]        # (1, d)

    if refs.ndim == 2:
        refs = refs[None, :, :]       # force shape (n, L, d)

    scores = np.min(np.abs(refs[:, :, i] - y[:, None, i]), axis=1)
    return scores
