"""Canonical TFN metrics for (center,left_spread,right_spread)."""
import numpy as np

def validate_tfn(a):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.all(np.isfinite(a)):
        raise ValueError("finite (n,3) TFN array required")
    if np.any(a[:, 1:] < 0):
        raise ValueError("spreads must be nonnegative")
    return a

def _membership(x, t):
    m,l,r = t; lo,hi = m-l,m+r
    out = np.zeros_like(x, dtype=float)
    if l == 0: out[x == m] = 1.0
    else:
        q = (x >= lo) & (x <= m); out[q] = (x[q]-lo)/l
    if r == 0: out[x == m] = 1.0
    else:
        q = (x >= m) & (x <= hi); out[q] = np.maximum(out[q], (hi-x[q])/r)
    return out

def similarity(a, b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape!=(3,) or b.shape!=(3,) or np.any(a[1:]<0) or np.any(b[1:]<0):
        raise ValueError("two valid TFNs required")
    points=sorted(set([a[0]-a[1],a[0],a[0]+a[2],b[0]-b[1],b[0],b[0]+b[2]]))
    # Add every crossing of the two linear membership functions.
    crossings=[]
    for lo,hi in zip(points[:-1],points[1:]):
        va=_membership(np.array([lo,hi]),a); vb=_membership(np.array([lo,hi]),b)
        d0,d1=va[0]-vb[0],va[1]-vb[1]
        if d0*d1<0: crossings.append(lo+(hi-lo)*abs(d0)/(abs(d0)+abs(d1)))
    p=np.array(sorted(set(points+crossings)),float)
    ma=_membership(p,a); mb=_membership(p,b)
    inter=float(np.trapezoid(np.minimum(ma,mb),p)); union=float(np.trapezoid(np.maximum(ma,mb),p))
    return 1.0 if union==0 and np.array_equal(a,b) else (inter/union if union else 0.0)

def msm(y_true,y_pred):
    a,b=validate_tfn(y_true),validate_tfn(y_pred)
    if a.shape!=b.shape: raise ValueError("shape mismatch")
    return float(np.mean([similarity(x,y) for x,y in zip(a,b)]))

def rmse(y_true,y_pred):
    a,b=validate_tfn(y_true),validate_tfn(y_pred)
    if a.shape!=b.shape: raise ValueError("shape mismatch")
    return float(np.sqrt(np.mean(np.sum((a-b)**2,axis=1))))
