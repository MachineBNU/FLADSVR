"""Split--Bregman FLADSVR for triangular fuzzy responses."""
from __future__ import annotations
import numpy as np
from .kernels import rbf_kernel

def _shrink(x,t): return np.sign(x)*np.maximum(np.abs(x)-t,0.0)

class FLADSVR:
    def __init__(self, *, gamma=1.0, h=1.0, mu=1.0, max_iter=2000, abs_tol=1e-7, rel_tol=1e-5):
        if min(gamma,h,mu,abs_tol,rel_tol)<=0: raise ValueError("positive settings required")
        self.gamma=float(gamma); self.h=float(h); self.mu=float(mu)
        self.max_iter=int(max_iter); self.abs_tol=float(abs_tol); self.rel_tol=float(rel_tol)
    def _fit_component(self,K,y):
        n=len(y); e=np.ones(n)
        A=np.block([[K+np.eye(n)/self.gamma,e[:,None]],[e[None,:],np.zeros((1,1))]])
        d=np.zeros(n); c=np.zeros(n); alpha=np.zeros(n); history=[]; reason="max_iter"
        for iteration in range(1,self.max_iter+1):
            old_alpha=alpha.copy(); old_d=d.copy()
            # The scaled-dual formulation gives the right-hand side y + d - c.
            sol=np.linalg.solve(A,np.r_[y+d-c,0.0]); alpha,bias=sol[:-1],float(sol[-1])
            residual=K@alpha+bias-y
            d=_shrink(residual+c,1.0/self.mu); c=c+residual-d
            primal=float(np.linalg.norm(residual-d)); dual=float(self.mu*np.linalg.norm(d-old_d))
            ep=self.abs_tol+self.rel_tol*max(np.linalg.norm(residual),np.linalg.norm(d))
            ed=self.abs_tol+self.rel_tol*self.mu*np.linalg.norm(c)
            objective=float(self.mu/(2*self.gamma)*(alpha@K@alpha)+np.abs(residual).sum())
            history.append(dict(iteration=iteration,objective=objective,primal_residual=primal,
                dual_residual=dual,parameter_change=float(np.linalg.norm(alpha-old_alpha)),
                primal_tolerance=float(ep),dual_tolerance=float(ed)))
            if primal<=ep and dual<=ed: reason="residual_tolerance"; break
        return alpha,bias,tuple(history),reason
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float)
        if X.ndim!=2 or y.shape!=(len(X),3) or len(X)<2 or np.any(y[:,1:]<0):
            raise ValueError("X (n,d) and valid TFN y (n,3) required")
        self.X_train_=X.copy(); K=rbf_kernel(X,h=self.h); fits=[self._fit_component(K,y[:,j]) for j in range(3)]
        self.params_=tuple((q[0],q[1]) for q in fits); self.history_=tuple(q[2] for q in fits)
        self.stopping_reasons_=tuple(q[3] for q in fits); return self
    def predict_raw(self,X):
        K=rbf_kernel(np.asarray(X,float),self.X_train_,self.h)
        return np.column_stack([K@a+b for a,b in self.params_])
    def predict(self,X):
        out=self.predict_raw(X); out[:,1:]=np.maximum(out[:,1:],0.0); return out
