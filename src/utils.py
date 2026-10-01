import numpy as np
class FeatureScaler:
 def fit(self,x):
  x=np.asarray(x,float); self.mean_=x.mean(0); self.scale_=x.std(0); self.scale_[self.scale_<1e-12]=1.; return self
 def transform(self,x): return (np.asarray(x,float)-self.mean_)/self.scale_
class TargetScale:
 """Positive scale only: preserves spread nonnegativity."""
 def fit(self,y):
  self.scale_=np.sqrt(np.mean(np.asarray(y,float)**2,axis=0)); self.scale_[self.scale_<1e-12]=1.; return self
 def transform(self,y): return np.asarray(y,float)/self.scale_
 def inverse(self,y): return np.asarray(y,float)*self.scale_
