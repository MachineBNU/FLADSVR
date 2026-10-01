"""Anchored-LAD fuzzy twin support vector regression.

The lower and upper objectives are optimized separately by ADMM. Triangular
fuzzy numbers use the convention (center, left_spread, right_spread).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _rbf_kernel(x: np.ndarray, z: np.ndarray, h: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    x2 = np.sum(x * x, axis=1)[:, None]
    z2 = np.sum(z * z, axis=1)[None, :]
    return np.exp(-np.maximum(x2 + z2 - 2.0 * x @ z.T, 0.0) / (h * h))


def _piecewise_prox(v: np.ndarray, rho: float, tau: float, c: float,
                    epsilon: float, side: str) -> np.ndarray:
    """Evaluate the exact piecewise proximal map by interval minimization."""
    v = np.asarray(v, dtype=float)
    if side == "lower":
        breaks = (-epsilon, 0.0)
        intervals = ((-np.inf, -epsilon, -tau),
                     (-epsilon, 0.0, c - tau),
                     (0.0, np.inf, c + tau))

        def loss(x):
            return tau * np.abs(x) + c * np.maximum(0.0, x + epsilon)
    elif side == "upper":
        breaks = (0.0, epsilon)
        intervals = ((-np.inf, 0.0, -c - tau),
                     (0.0, epsilon, tau - c),
                     (epsilon, np.inf, tau))

        def loss(x):
            return tau * np.abs(x) + c * np.maximum(0.0, epsilon - x)
    else:
        raise ValueError("side must be 'lower' or 'upper'")

    candidates = [np.full_like(v, b) for b in breaks]
    for lo, hi, slope in intervals:
        x = v - slope / rho
        if np.isfinite(lo):
            x = np.maximum(x, lo)
        if np.isfinite(hi):
            x = np.minimum(x, hi)
        candidates.append(x)
    cand = np.stack(candidates, axis=0)
    values = loss(cand) + 0.5 * rho * (cand - v[None, :]) ** 2
    return np.take_along_axis(cand, np.argmin(values, axis=0)[None, :], axis=0)[0]


@dataclass(frozen=True)
class _SideFit:
    alpha: np.ndarray
    bias: float
    history: tuple[dict, ...]
    stopping_reason: str


class FLADTSVR:
    """Anchored-LAD twin model with distinct lower and upper objectives.

    ``c_lower`` and ``c_upper`` must exceed ``tau`` for genuine opposing
    pointwise roles. Spread projection is an explicit part of ``predict``.
    """

    def __init__(self, *, h: float = 1.0, lambda_reg: float = 1.0,
                 alpha_ridge: float = 1e-6, intercept_ridge: float = 1e-6,
                 tau: float = 1.0, c_lower: float = 2.0,
                 c_upper: float = 2.0, epsilon1: float = 0.1,
                 epsilon2: float = 0.1, rho: float = 1.0,
                 max_iter: int = 3000, abs_tol: float = 1e-6,
                 rel_tol: float = 1e-5):
        vals = (h, lambda_reg, alpha_ridge, intercept_ridge, tau,
                c_lower, c_upper, epsilon1, epsilon2, rho)
        if any(not np.isfinite(v) for v in vals):
            raise ValueError("all hyperparameters must be finite")
        if min(h, alpha_ridge, intercept_ridge, tau, epsilon1, epsilon2, rho) <= 0:
            raise ValueError("h, ridges, tau, epsilons, and rho must be positive")
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be nonnegative")
        if c_lower <= tau or c_upper <= tau:
            raise ValueError("genuine twin conditions require c_lower,c_upper > tau")
        if max_iter < 1 or abs_tol <= 0 or rel_tol <= 0:
            raise ValueError("invalid iteration/tolerance setting")
        self.h = float(h)
        self.lambda_reg = float(lambda_reg)
        self.alpha_ridge = float(alpha_ridge)
        self.intercept_ridge = float(intercept_ridge)
        self.tau = float(tau)
        self.c_lower = float(c_lower)
        self.c_upper = float(c_upper)
        self.epsilon1 = float(epsilon1)
        self.epsilon2 = float(epsilon2)
        self.rho = float(rho)
        self.max_iter = int(max_iter)
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)

    def _fit_side(self, y: np.ndarray, side: str) -> _SideFit:
        hmat = self._design_
        p = hmat.shape[1]
        rho = self.rho
        chol = np.linalg.cholesky(
            self._regularizer_ + rho * hmat.T @ hmat)
        theta = np.zeros(p)
        z = np.zeros_like(y)
        dual = np.zeros_like(y)
        epsilon = self.epsilon1 if side == "lower" else self.epsilon2
        c_side = self.c_lower if side == "lower" else self.c_upper
        history: list[dict] = []
        reason = "max_iter"
        for iteration in range(1, self.max_iter + 1):
            old_theta = theta.copy()
            old_z = z.copy()
            # ADMM parameter update.
            rhs = rho * hmat.T @ (y + z - dual)
            theta = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
            residual = hmat @ theta - y
            # Piecewise residual-proximal update.
            z = _piecewise_prox(residual + dual, rho, self.tau,
                                c_side, epsilon, side)
            primal_vec = residual - z
            # Scaled-dual update.
            dual = dual + primal_vec
            dual_vec = rho * hmat.T @ (z - old_z)
            primal = float(np.linalg.norm(primal_vec))
            dual_norm = float(np.linalg.norm(dual_vec))
            parameter_change = float(np.linalg.norm(theta - old_theta))
            regularizer = 0.5 * float(theta @ self._regularizer_ @ theta)
            if side == "lower":
                side_loss = self.tau * np.abs(residual) + c_side * np.maximum(0.0, residual + epsilon)
            else:
                side_loss = self.tau * np.abs(residual) + c_side * np.maximum(0.0, epsilon - residual)
            objective = regularizer + float(np.sum(side_loss))
            eps_primal = np.sqrt(len(y)) * self.abs_tol + self.rel_tol * max(
                np.linalg.norm(residual), np.linalg.norm(z))
            eps_dual = np.sqrt(p) * self.abs_tol + self.rel_tol * np.linalg.norm(
                rho * hmat.T @ dual)
            history.append({"iteration": iteration, "objective": objective,
                            "primal_residual": primal, "dual_residual": dual_norm,
                            "parameter_change": parameter_change,
                            "primal_tolerance": float(eps_primal),
                            "dual_tolerance": float(eps_dual), "rho": float(rho)})
            if primal <= eps_primal and dual_norm <= eps_dual:
                reason = "residual_tolerance"
                break
            # Residual balancing changes the ADMM path without changing the objective.
            if iteration % 25 == 0:
                new_rho = rho
                if primal > 10.0 * dual_norm:
                    new_rho = min(2.0 * rho, 64.0)
                elif dual_norm > 10.0 * primal:
                    new_rho = max(0.5 * rho, 1.0 / 64.0)
                if new_rho != rho:
                    dual *= rho / new_rho
                    rho = new_rho
                    chol = np.linalg.cholesky(
                        self._regularizer_ + rho * hmat.T @ hmat)
        return _SideFit(theta[:-1], float(theta[-1]), tuple(history), reason)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FLADTSVR":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or y.shape != (len(x), 3):
            raise ValueError("X must be (n,d) and y must be (n,3)")
        if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("finite data with at least two samples required")
        if np.any(y[:, 1:] < 0):
            raise ValueError("training spreads must be nonnegative")
        self.x_train_ = x.copy()
        kernel = _rbf_kernel(x, x, self.h)
        ones = np.ones((len(x), 1))
        self._design_ = np.hstack((kernel, ones))
        n = len(x)
        self._regularizer_ = np.zeros((n + 1, n + 1))
        self._regularizer_[:n, :n] = (
            self.lambda_reg * kernel + self.alpha_ridge * np.eye(n))
        self._regularizer_[-1, -1] = self.intercept_ridge
        self.lower_fits_ = tuple(self._fit_side(y[:, j], "lower") for j in range(3))
        self.upper_fits_ = tuple(self._fit_side(y[:, j], "upper") for j in range(3))
        self.history_ = {
            "lower": tuple(f.history for f in self.lower_fits_),
            "upper": tuple(f.history for f in self.upper_fits_),
        }
        self.stopping_reasons_ = {
            "lower": tuple(f.stopping_reason for f in self.lower_fits_),
            "upper": tuple(f.stopping_reason for f in self.upper_fits_),
        }
        return self

    def predict_sides(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "x_train_"):
            raise RuntimeError("fit must be called first")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.x_train_.shape[1]:
            raise ValueError("X has incompatible dimensions")
        kernel = _rbf_kernel(x, self.x_train_, self.h)
        lower = np.column_stack([kernel @ f.alpha + f.bias for f in self.lower_fits_])
        upper = np.column_stack([kernel @ f.alpha + f.bias for f in self.upper_fits_])
        return lower, upper

    def predict(self, x: np.ndarray) -> np.ndarray:
        lower, upper = self.predict_sides(x)
        prediction = 0.5 * (lower + upper)
        # Declared Euclidean projection onto the valid TFN spread cone.
        prediction[:, 1:] = np.maximum(prediction[:, 1:], 0.0)
        return prediction
