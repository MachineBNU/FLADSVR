"""Baseline regressors for triangular fuzzy responses."""
import numpy as np
from scipy.optimize import linprog
from sklearn.svm import SVR
from .kernels import rbf_kernel
from .hao_chiang import HaoChiangFuzzySVR

CLASSES={
 'FuzzyMean':'SIMPLE_REFERENCE',
 'FuzzyMedian':'SIMPLE_REFERENCE',
 'FuzzyNW':'NATIVE_FUZZY_OUTPUT',
 'HaoChiangFSVR':'PUBLISHED_SYMMETRIC_FUZZY_OUTPUT',
 'ComponentRidge':'COMPONENTWISE_ADAPTATION',
 'ComponentLAD':'COMPONENTWISE_ADAPTATION',
 'ComponentRBF_SVR':'COMPONENTWISE_ADAPTATION'}
GRIDS={
 'FuzzyMean':[{}],
 'FuzzyMedian':[{}],
 'FuzzyNW':[{'h':h} for h in (.25,.5,1.,2.)],
 'HaoChiangFSVR':[{'H':H,'K_tradeoff':K,'P':15.,'h':h}
                   for H in (.25,.5,.75) for K in (5.,25.) for h in (.5,1.5)],
 'ComponentRidge':[{'alpha':a} for a in (1e-3,1e-2,.1,1.,10.,100.)],
 'ComponentLAD':[{}],
 'ComponentRBF_SVR':[{'C':c,'h':h,'epsilon':.1} for c in (.5,5.) for h in (.5,1.5)]+
                    [{'C':c,'h':1.,'epsilon':e} for c in (.5,5.) for e in (.02,.25)]}

class _Constant:
 def __init__(self,kind): self.kind=kind
 def fit(self,x,y):
  self.value_=np.mean(y,0) if self.kind=='mean' else np.median(y,0); self.value_[1:]=np.maximum(self.value_[1:],0); return self
 def predict(self,x): return np.tile(self.value_,(len(x),1))

class FuzzyNW:
 """Native TFN Nadaraya-Watson: one nonnegative normalized weight vector."""
 def __init__(self,h): self.h=float(h)
 def fit(self,x,y): self.x_=np.asarray(x,float).copy(); self.y_=np.asarray(y,float).copy(); return self
 def predict(self,x):
  k=rbf_kernel(np.asarray(x,float),self.x_,self.h); den=k.sum(1,keepdims=True)
  # RBF is positive, but retain a nearest-neighbor fallback for underflow.
  bad=den[:,0]<1e-300; out=k@self.y_/np.maximum(den,1e-300)
  if np.any(bad):
   d=((np.asarray(x)[bad,None,:]-self.x_[None,:,:])**2).sum(2); out[bad]=self.y_[np.argmin(d,1)]
  return out

class ComponentRidge:
 def __init__(self,alpha): self.alpha=float(alpha)
 def fit(self,x,y):
  h=np.column_stack((np.asarray(x,float),np.ones(len(x)))); p=h.shape[1]
  reg=np.eye(p)*self.alpha; reg[-1,-1]=0.; self.coef_=np.linalg.solve(h.T@h+reg,h.T@np.asarray(y,float)); return self
 def predict(self,x):
  out=np.column_stack((np.asarray(x,float),np.ones(len(x))))@self.coef_; out[:,1:]=np.maximum(out[:,1:],0); return out

class ComponentLAD:
 def fit(self,x,y):
  h=np.column_stack((np.asarray(x,float),np.ones(len(x)))); n,p=h.shape; co=[]
  for j in range(3):
   c=np.r_[np.zeros(p),np.ones(2*n)]
   aeq=np.hstack((h,np.eye(n),-np.eye(n)))
   res=linprog(c,A_eq=aeq,b_eq=np.asarray(y,float)[:,j],bounds=[(None,None)]*p+[(0,None)]*(2*n),method='highs')
   if not res.success: raise RuntimeError(res.message)
   co.append(res.x[:p])
  self.coef_=np.column_stack(co); return self
 def predict(self,x):
  out=np.column_stack((np.asarray(x,float),np.ones(len(x))))@self.coef_; out[:,1:]=np.maximum(out[:,1:],0); return out

class ComponentRBFSVR:
 def __init__(self,C,h,epsilon): self.C=float(C); self.h=float(h); self.epsilon=float(epsilon)
 def fit(self,x,y):
  self.models_=[SVR(C=self.C,gamma=1/(self.h*self.h),epsilon=self.epsilon,kernel='rbf').fit(x,np.asarray(y)[:,j]) for j in range(3)]; return self
 def predict(self,x):
  out=np.column_stack([m.predict(x) for m in self.models_]); out[:,1:]=np.maximum(out[:,1:],0); return out

def make_model(name,params):
 if name=='FuzzyMean': return _Constant('mean')
 if name=='FuzzyMedian': return _Constant('median')
 if name=='FuzzyNW': return FuzzyNW(**params)
 if name=='HaoChiangFSVR': return HaoChiangFuzzySVR(**params)
 if name=='ComponentRidge': return ComponentRidge(**params)
 if name=='ComponentLAD': return ComponentLAD()
 if name=='ComponentRBF_SVR': return ComponentRBFSVR(**params)
 raise ValueError(name)
