
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


def _ei(mu, sigma, y_best, xi=0.03):
    sigma = np.maximum(sigma, 1e-10)
    imp = y_best - mu - xi
    z = imp / sigma
    out = imp * norm.cdf(z) + sigma * norm.pdf(z)
    out = np.where(sigma < 1e-11, 0.0, out)
    return np.maximum(out, 0.0)


def gp_ei_search(objective, dim, n_calls=35, n_random=None, n_candidates=700, seed=0):
    """Minimize objective(x) for x in [0,1]^dim."""
    rng = np.random.default_rng(seed)
    if n_random is None:
        n_random = max(5, min(10, n_calls // 4))

    X = rng.random((n_random, dim))
    y = np.asarray([objective(row) for row in X], dtype=np.float64)

    rest = max(0, int(n_calls) - n_random)

    for _ in range(rest):
        kern = Matern(length_scale=np.ones(dim), nu=2.5) + WhiteKernel(
            noise_level=1e-5
        )
        gpr = GaussianProcessRegressor(
            kernel=kern,
            normalize_y=True,
            random_state=seed,
            alpha=1e-12,
        )
        gpr.fit(X, y)
        y_best = float(np.min(y))

        cand = rng.random((n_candidates, dim))
        mu, sd = gpr.predict(cand, return_std=True)
        ei = _ei(mu, sd, y_best)
        idx = int(np.argmax(ei))
        if ei[idx] <= 1e-14:
            x_next = rng.random(dim)
        else:
            x_next = cand[idx]

        y_next = float(objective(x_next))
        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.concatenate([y, np.array([y_next])])

    ib = int(np.argmin(y))
    return X[ib].copy(), float(y[ib]), X, y


def to_int(u, lo, hi):
    return int(np.clip(round(lo + u * (hi - lo)), lo, hi))
