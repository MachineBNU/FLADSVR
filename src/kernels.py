import numpy as np

def rbf_kernel(x, y=None, h=1.0):
    x = np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("compatible two-dimensional feature arrays required")
    if h <= 0 or not np.isfinite(h):
        raise ValueError("h must be finite and positive")
    x2 = np.sum(x*x, axis=1)[:, None]
    y2 = np.sum(y*y, axis=1)[None, :]
    return np.exp(-np.maximum(x2+y2-2*x@y.T, 0.0)/(h*h))
