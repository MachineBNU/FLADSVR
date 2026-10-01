import inspect, unittest
import numpy as np
from src.fladsvr import FLADSVR
from src.fladtsvr import FLADTSVR
from src.hao_chiang import HaoChiangFuzzySVR, linear_kernel, solve_dual, _intercepts
from src.metrics import msm,rmse

def toy():
 x=np.linspace(-1.5,1.5,9)[:,None]
 y=np.column_stack((.4+.8*x[:,0],.25+.04*(x[:,0]+.3)**2,.35+.03*(x[:,0]-.2)**2))
 return x,y
def twin(**kw):
 p=dict(h=.8,lambda_reg=.05,alpha_ridge=1e-5,intercept_ridge=1e-5,tau=1.,c_lower=4.,c_upper=4.,epsilon1=.12,epsilon2=.18,rho=1.,max_iter=1800,abs_tol=2e-6,rel_tol=2e-5); p.update(kw); return FLADTSVR(**p)

class TestPipeline(unittest.TestCase):
 def test_metric_identities(self):
  _,y=toy(); self.assertEqual(msm(y,y),1.0); self.assertEqual(rmse(y,y),0.0)
 def test_fladsvr_deterministic_finite_valid(self):
  x,y=toy(); a=FLADSVR(gamma=5,h=.8,mu=1).fit(x,y); b=FLADSVR(gamma=5,h=.8,mu=1).fit(x,y)
  self.assertTrue(np.array_equal(a.predict(x),b.predict(x))); self.assertTrue(np.all(np.isfinite(a.predict(x)))); self.assertTrue(np.all(a.predict(x)[:,1:]>=0))
 def test_fladsvr_scaled_dual_rhs(self):
  s=inspect.getsource(FLADSVR._fit_component).replace(' ','')
  self.assertIn('np.r_[y+d-c,0.0]',s); self.assertEqual(s.count('np.r_['),1)
 def test_fladsvr_instrumentation(self):
  x,y=toy(); q=FLADSVR(gamma=5,h=.8,mu=1).fit(x,y)
  self.assertTrue(all(len(h)>0 for h in q.history_)); self.assertTrue(all(np.isfinite(h[-1]['objective']) for h in q.history_))
 def test_twin_finite_valid_distinct_and_scalar_bias(self):
  x,y=toy(); q=twin().fit(x,y); p=q.predict(x); lo,up=q.predict_sides(x)
  self.assertTrue(np.all(np.isfinite(p))); self.assertTrue(np.all(p[:,1:]>=0)); self.assertGreater(np.linalg.norm(up-lo),1e-3)
  self.assertTrue(all(np.isscalar(f.bias) for f in q.lower_fits_+q.upper_fits_))
 def test_epsilon1_operational_only_lower(self):
  x,y=toy(); a=twin(epsilon1=.05).fit(x,y); b=twin(epsilon1=.25).fit(x,y); al,au=a.predict_sides(x); bl,bu=b.predict_sides(x)
  self.assertGreater(np.linalg.norm(al-bl),1e-4); self.assertLess(np.linalg.norm(au-bu),1e-9)
 def test_epsilon2_operational_only_upper(self):
  x,y=toy(); a=twin(epsilon2=.05).fit(x,y); b=twin(epsilon2=.28).fit(x,y); al,au=a.predict_sides(x); bl,bu=b.predict_sides(x)
  self.assertLess(np.linalg.norm(al-bl),1e-9); self.assertGreater(np.linalg.norm(au-bu),1e-4)
 def test_twin_not_single(self):
  x,y=toy(); a=twin().fit(x,y).predict(x); b=FLADSVR(gamma=5,h=.8,mu=1).fit(x,y).predict(x)
  self.assertGreater(np.linalg.norm(a-b),1e-3)
 def test_hao_chiang_finite_symmetric_and_valid(self):
  x,y=toy(); symmetric=np.column_stack((y[:,0],(y[:,1]+y[:,2])/2))
  model=HaoChiangFuzzySVR(H=.5,K_tradeoff=25,P=15,h=.8).fit(x,symmetric)
  prediction=model.predict(x)
  self.assertEqual(prediction.shape,(len(x),2)); self.assertTrue(np.all(np.isfinite(prediction)))
  self.assertTrue(np.all(prediction[:,1]>=0))
 def test_hao_chiang_published_table_one(self):
  # Table I data from Hao and Chiang (2008), using their linear-kernel setting.
  x=np.arange(1.,9.)[:,None]/10
  center=np.array([2.25,2.875,2.5,4.25,4.,5.25,7.5,8.5])
  spread=np.array([.75,.875,1.,1.75,1.5,1.25,2.,1.5])
  a1,a2,kx,ka,_=solve_dual(x,center,spread,.5,25.,15.,linear_kernel)
  center_no=kx@(a1-a2); spread_no=.5/25.*(ka@(a1+a2))
  intercept,_=_intercepts(a1,a2,center_no,spread_no,center,spread,.5,25.,15.)
  self.assertTrue(np.allclose([a1@x[:,0]-a2@x[:,0],intercept[0],
                               .5/25.*((a1+a2)@x[:,0]),intercept[1]],
                              [8.,1.1525,.58,2.931],atol=2e-5))

if __name__=='__main__': unittest.main()
