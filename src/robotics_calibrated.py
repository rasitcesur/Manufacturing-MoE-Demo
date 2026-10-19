"""current learned gravity-aware residual conditioned on independent historic energy records."""
import numpy as np
from robotics_moe import RoboticsMoE51,prediction as parent_prediction
from robotics_moe import Prepared51

class PreparedCalibrated(Prepared51):
    def __init__(self,data,mean=None,std=None):
        super().__init__(data,mean,std)
        past=np.asarray(data['history_calibration'],np.float32)
        # Past energy ratios from six previous standardized diagnostic movements;
        # target trajectory energy is NEVER used in this calculation.
        logmachine=past[:,:6].mean(axis=1)
        if mean is None:
            self.mean['past']=past.mean(axis=0)
            self.std['past']=np.maximum(past.std(axis=0),.025)
        pn=np.clip((past-self.mean['past'])/self.std['past'],-7,7)
        self.norm['history']=np.concatenate([self.norm['history'],pn],axis=1).astype(np.float32)
        self.logmachine=logmachine.copy()
        self.nominal_energy_prior=self.prior[:,1].copy()
        self.prior[:,1]*=np.exp(logmachine)
        self.target=np.log(np.maximum(self.y,1e-6)/np.maximum(self.prior,1e-6)).astype(np.float32)

class RoboticsMoECalibrated(RoboticsMoE51):
    def __init__(self):super().__init__(history_dim=36)


def prediction(model,ds,batch=512,device=None,use_history=True):
    p=parent_prediction(model,ds,batch=batch,device=device,use_history=use_history)
    if not use_history:
        # TRUE no-history evaluation: remove the prior based on historic energy.
        p[:,1]*=np.exp(-ds.logmachine)
    return p
