"""Hao--Chiang (2008) fuzzy-output support vector regression.

The published method models a crisp input and a symmetric triangular fuzzy
output (center, common_spread). The implementation solves the published dual
convex quadratic program and recovers the two intercepts from its KKT rules.
"""
from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import linprog


def linear_kernel(a, b):
    return np.asarray(a, float) @ np.asarray(b, float).T


def rbf_kernel(a, b, h):
    a, b = np.asarray(a, float), np.asarray(b, float)
    distance = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return np.exp(-distance / (float(h) ** 2))


def solve_dual(features, center, spread, H, K_tradeoff, P, kernel):
    """Solve Eqs. (14)/(22) of Hao and Chiang (2008)."""
    x = np.atleast_2d(np.asarray(features, float))
    y, e = np.asarray(center, float), np.asarray(spread, float)
    n = len(y)
    kx, ka = kernel(x, x), kernel(np.abs(x), np.abs(x))
    identity = np.eye(n)
    difference = np.hstack((identity, -identity))
    total = np.hstack((identity, identity))
    scale = (1-H) ** 2 / K_tradeoff
    quadratic = difference.T @ kx @ difference + scale * total.T @ ka @ total
    quadratic = (quadratic + quadratic.T) / 2
    linear = -np.r_[y + (1-H)*e, -y + (1-H)*e]
    equality = np.r_[np.ones(n), -np.ones(n)][None, :]
    matrix = sparse.vstack((sparse.csc_matrix(equality),
                            sparse.csc_matrix(np.ones((1, 2*n))),
                            sparse.eye(2*n, format="csc")), format="csc")
    lower = np.r_[0.0, -np.inf, np.zeros(2*n)]
    upper = np.r_[0.0, K_tradeoff/(1-H), np.full(2*n, P)]
    problem = osqp.OSQP()
    problem.setup(P=sparse.csc_matrix(quadratic), q=linear, A=matrix,
                  l=lower, u=upper, verbose=False, polishing=True,
                  eps_abs=1e-7, eps_rel=1e-7, max_iter=100000)
    result = problem.solve(raise_error=False)
    if result.info.status not in {"solved", "solved inaccurate"}:
        raise RuntimeError("Hao--Chiang QP: " + result.info.status)
    alpha1, alpha2 = result.x[:n], result.x[n:]
    return alpha1, alpha2, kx, ka, int(result.info.iter)


def _intercepts(alpha1, alpha2, center_no_bias, spread_no_bias,
                center, spread, H, K_tradeoff, P):
    tol = 1e-6
    first = np.flatnonzero((alpha1 > tol) & (alpha1 < P-tol))
    second = np.flatnonzero((alpha2 > tol) & (alpha2 < P-tol))
    candidates = []
    for i in first:
        for j in second:
            b = -0.5 * (center_no_bias[i] + center_no_bias[j]
                        + (1-H)*(spread_no_bias[i]-spread_no_bias[j])
                        - center[i]-center[j] - (1-H)*(spread[i]-spread[j]))
            d = -1/(2*(1-H)) * (center_no_bias[i]-center_no_bias[j]
                    + (1-H)*(spread_no_bias[i]+spread_no_bias[j])
                    - center[i]+center[j] - (1-H)*(spread[i]+spread[j]))
            if d >= -tol:
                candidates.append((b, max(0.0, d)))
    if candidates:
        return np.median(np.asarray(candidates), axis=0), "published_interior_kkt"

    # Exact convex intercept/slack subproblem for the boundary-alpha case.
    n = len(center)
    objective = np.r_[0.0, K_tradeoff, np.full(2*n, P)]
    lower_side = np.zeros((n, 2+2*n)); upper_side = np.zeros_like(lower_side)
    lower_side[:, 0] = -1; lower_side[:, 1] = -(1-H)
    lower_side[:, 2:2+n] = -np.eye(n)
    upper_side[:, 0] = 1; upper_side[:, 1] = -(1-H)
    upper_side[:, 2+n:] = -np.eye(n)
    rhs_lower = center_no_bias+(1-H)*spread_no_bias-center-(1-H)*spread
    rhs_upper = center-(1-H)*spread-center_no_bias+(1-H)*spread_no_bias
    result = linprog(objective, A_ub=np.vstack((lower_side, upper_side)),
                     b_ub=np.r_[rhs_lower, rhs_upper],
                     bounds=[(None, None), (0, None)]+[(0, None)]*(2*n),
                     method="highs")
    if not result.success:
        raise RuntimeError("Hao--Chiang intercept LP: " + result.message)
    return result.x[:2], "exact_primal_lp"


class HaoChiangFuzzySVR:
    """Published symmetric-fuzzy-output SVR; targets have (center, spread)."""
    def __init__(self, H=0.5, K_tradeoff=25.0, P=15.0, h=1.5):
        self.H, self.K_tradeoff = float(H), float(K_tradeoff)
        self.P, self.h = float(P), float(h)
        if not 0 <= self.H < 1 or min(self.K_tradeoff, self.P, self.h) <= 0:
            raise ValueError("require 0 <= H < 1 and positive K_tradeoff, P, h")

    def _kernel(self, a, b):
        return rbf_kernel(a, b, self.h)

    def fit(self, features, targets):
        x, target = np.asarray(features, float), np.asarray(targets, float)
        if target.ndim != 2 or target.shape[1] != 2 or np.any(target[:, 1] < 0):
            raise ValueError("Hao--Chiang targets must be (center, nonnegative common spread)")
        a1, a2, kx, ka, iterations = solve_dual(
            x, target[:, 0], target[:, 1], self.H, self.K_tradeoff,
            self.P, self._kernel)
        self.features_ = x.copy(); self.diff_ = a1-a2; self.total_ = a1+a2
        center_no = kx @ self.diff_
        spread_no = (1-self.H)/self.K_tradeoff * (ka @ self.total_)
        (self.center_intercept_, self.spread_intercept_), self.intercept_recovery_ = _intercepts(
            a1, a2, center_no, spread_no, target[:, 0], target[:, 1],
            self.H, self.K_tradeoff, self.P)
        self.iterations_ = iterations
        return self

    def predict(self, features):
        x = np.asarray(features, float)
        center = self._kernel(x, self.features_) @ self.diff_ + self.center_intercept_
        spread = ((1-self.H)/self.K_tradeoff
                  * (self._kernel(np.abs(x), np.abs(self.features_)) @ self.total_)
                  + self.spread_intercept_)
        return np.column_stack((center, np.maximum(spread, 0.0)))
