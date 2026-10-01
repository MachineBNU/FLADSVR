from __future__ import annotations
import numpy as np
from .fladsvr import FLADSVR
from .fladtsvr import FLADTSVR
from .metrics import rmse, msm
from .utils import FeatureScaler, TargetScale
from .config import FLADSVR_GRID, FLADTSVR_GRID, SOLVER

def kfold_indices(n,k,seed,shuffle=True):
 idx=np.arange(n)
 if shuffle: idx=np.random.default_rng(seed).permutation(idx)
 parts=np.array_split(idx,k)
 for val in parts:
  train=np.setdiff1d(np.arange(n),val,assume_unique=True)
  yield train,val

def make_model(method,params):
 if method=='FLADSVR': return FLADSVR(**params,**SOLVER['fladsvr'])
 if method=='FLADTSVR': return FLADTSVR(**params,**SOLVER['fladtsvr'])
 raise ValueError(method)

def fit_predict(method,params,x_train,y_train,x_test):
 xs=FeatureScaler().fit(x_train); ys=TargetScale().fit(y_train)
 model=make_model(method,params).fit(xs.transform(x_train),ys.transform(y_train))
 pred=ys.inverse(model.predict(xs.transform(x_test)))
 pred[:,1:]=np.maximum(pred[:,1:],0.0)
 return pred,model,xs,ys

def tune(method,x,y,seed,inner_folds=3):
 grid=FLADSVR_GRID if method=='FLADSVR' else FLADTSVR_GRID
 splits=list(kfold_indices(len(x),min(inner_folds,len(x)),seed,True))
 records=[]
 for candidate_id,params in enumerate(grid):
  pred=np.empty_like(y); ok=True; error=''
  try:
   for tr,va in splits:
    pred[va]=fit_predict(method,params,x[tr],y[tr],x[va])[0]
   score_rmse=rmse(y,pred); score_msm=msm(y,pred)
  except Exception as exc:
   ok=False; error=f'{type(exc).__name__}: {exc}'; score_rmse=float('inf'); score_msm=-float('inf')
  records.append({'candidate_id':candidate_id,'params':params,'inner_rmse':score_rmse,'inner_msm':score_msm,'ok':ok,'error':error})
 valid=[r for r in records if r['ok']]
 if not valid: raise RuntimeError(f'all {method} candidates failed: {records}')
 best=min(valid,key=lambda r:(r['inner_rmse'],-r['inner_msm'],r['candidate_id']))
 return dict(best['params']),records
