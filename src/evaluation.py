from __future__ import annotations
import time, json
import numpy as np
from .datasets import load_example, PATHS
from .metrics import msm, rmse
from .tuning import kfold_indices,tune,fit_predict
from .config import SEED

def outer_splits(example,n):
 if example==3:
  for i in range(n): yield np.delete(np.arange(n),i),np.array([i])
 else:
  yield from kfold_indices(n,5,SEED+example,True)

def _convergence(model):
 reasons=[]
 if hasattr(model,'stopping_reasons_'):
  q=model.stopping_reasons_
  if isinstance(q,dict):
   for v in q.values(): reasons.extend(v)
  else: reasons.extend(q)
 return reasons

def evaluate_example(example,method):
 x,y=load_example(example); pred=np.empty_like(y); rows=[]; tuning_rows=[]; start=time.perf_counter()
 for fold,(tr,te) in enumerate(outer_splits(example,len(x))):
  params,search=tune(method,x[tr],y[tr],SEED+1000*example+100*fold+(0 if method=='FLADSVR' else 1),3)
  for q in search: tuning_rows.append({'example':example,'method':method,'outer_fold':fold,**q})
  yp,model,xs,ys=fit_predict(method,params,x[tr],y[tr],x[te]); pred[te]=yp
  reasons=_convergence(model); converged=all(r=='residual_tolerance' for r in reasons)
  lower_gap=epsilon1_effect=epsilon2_effect=float('nan')
  if method=='FLADTSVR':
   lower,upper=model.predict_sides(xs.transform(x[te])); lower_gap=float(np.linalg.norm(upper-lower))
   for key in ('epsilon1','epsilon2'):
    alt=dict(params); alt[key]=params[key]*1.5+0.01
    ap=fit_predict(method,alt,x[tr],y[tr],x[te])[0]
    if key=='epsilon1': epsilon1_effect=float(np.linalg.norm(ap-yp))
    else: epsilon2_effect=float(np.linalg.norm(ap-yp))
  for local,i in enumerate(te):
   rows.append({'example':example,'dataset_path':str(PATHS[example]),'sample_id':int(i),'outer_fold':fold,'method':method,
    'selected_hyperparameters':json.dumps(params,sort_keys=True),'predicted_center':yp[local,0],
    'predicted_left_spread':yp[local,1],'predicted_right_spread':yp[local,2],
    'true_center':y[i,0],'true_left_spread':y[i,1],'true_right_spread':y[i,2],
    'converged':converged,'stopping_reasons':'|'.join(reasons),
    'twin_side_gap':lower_gap,'epsilon1_prediction_effect':epsilon1_effect,'epsilon2_prediction_effect':epsilon2_effect})
 return rows,tuning_rows,{'example':example,'method':method,'n':len(y),'msm':msm(y,pred),'rmse':rmse(y,pred),
  'convergence_failures':sum(not r['converged'] for r in rows),'invalid_spreads':int(np.sum(pred[:,1:]<0)),
  'runtime_seconds':time.perf_counter()-start}
