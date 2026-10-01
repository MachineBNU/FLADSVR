SEED=20260820
FLADSVR_GRID=[
 {'gamma':g,'h':h,'mu':1.0} for g in (0.5,5.0,50.0) for h in (0.5,1.5)
]+[
 {'gamma':5.0,'h':1.0,'mu':m} for m in (0.25,4.0)
]
FLADTSVR_GRID=[
 {'h':h,'lambda_reg':lam,'epsilon1':e1,'epsilon2':e2,'tau':1.0,'c_lower':4.0,'c_upper':4.0}
 for h in (0.5,1.5) for lam in (0.05,0.5)
 for e1,e2 in ((0.05,0.20),(0.20,0.05),(0.10,0.10))
]
SOLVER={
 'fladsvr':{'max_iter':5000,'abs_tol':1e-6,'rel_tol':1e-4},
 'fladtsvr':{'rho':1.0,'alpha_ridge':1e-5,'intercept_ridge':1e-5,
             'max_iter':3000,'abs_tol':2e-6,'rel_tol':1e-4}}
