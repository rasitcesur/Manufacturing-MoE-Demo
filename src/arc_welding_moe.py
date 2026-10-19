"""current: reused gravity-aware RoboticsMoE51 as actual robot-drive energy branch.
The process branch never predicts robot motor energy. All weights are random initialized.
"""
from __future__ import annotations
import numpy as np,torch
from torch import nn
from arc_welding_core import PreparedArc
from robotics_moe import RoboticsMoE51

class PreparedArc54(PreparedArc):
    def __init__(self,data,priors,gravity,past,center=None,scale=None,calibrate_drive=True):
        # Split train normalization into original energy and new signed-gravity/sensor channels.
        bc=center['base'] if center is not None else None
        bs=scale['base'] if scale is not None else None
        super().__init__(data,priors,center=bc,scale=bs)
        rawj=np.asarray(gravity['joint_gravity'],np.float32)
        rawg=np.asarray(gravity['global_gravity'],np.float32)
        rawh=np.asarray(data['history_power_W'],np.float32)
        measured=np.asarray(past['robot_history'],np.float32)
        self.calibration_scale=np.exp(np.clip(measured[:,0::4].mean(1),-.60,.60))
        if calibrate_drive:
            # Use physical priors and independent pre-target calibration measurements: BEFORE-target
            # movement meter ratios, rather than the target label, scale nominal drive.
            self.component_prior[:,0]*=self.calibration_scale
            self.prior[:,1]=self.component_prior.sum(1)
        nom= np.maximum(1.,rawh[:,6:7])
        # Historic sensor variations relative to the preceding idle measurement;
        # the six readings precede this welding program, never target telemetry.
        sensor=np.sign((rawh[:,6:12]-rawh[:,6:7])/nom)*np.log1p(np.abs((rawh[:,6:12]-rawh[:,6:7])/nom))
        if center is None:
            valid=rawj[self.mask>0]
            center={'base':self.center,'joint_gravity':valid.mean(0),'global_gravity':rawg.mean(0),'sensor':sensor.mean(0),'past':measured.mean(0)}
            scale={'base':self.scale,'joint_gravity':np.maximum(valid.std(0),.03),
                'global_gravity':np.maximum(rawg.std(0),.03),'sensor':np.maximum(sensor.std(0),.025),'past':np.maximum(measured.std(0),.025)}
        self.norm_center=center;self.norm_scale=scale
        nj=np.clip((rawj-center['joint_gravity'])/scale['joint_gravity'],-7,7)*self.mask[:,:,None]
        ng=np.clip((rawg-center['global_gravity'])/scale['global_gravity'],-7,7)
        nh=np.clip((sensor-center['sensor'])/scale['sensor'],-7,7)
        npast=np.clip((measured-center['past'])/scale['past'],-7,7)
        self.joint=np.concatenate([self.joint,nj],axis=2).astype(np.float32)
        self.global_x=np.concatenate([self.global_x,ng],axis=1).astype(np.float32)
        self.history=np.concatenate([self.history,nh,npast],axis=1).astype(np.float32)
        self.global_dim=self.global_x.shape[-1];self.history_dim=self.history.shape[-1]
        self.comp_target=np.log(np.maximum(self.c,1e-8)/np.maximum(self.component_prior,1e-8)).astype(np.float32)

class ArcWeldingMoE54(nn.Module):
    """GMAW-only process and sign-aware robot branches; no external weight sharing."""
    def __init__(self,global_dim,history_dim):
        super().__init__()
        from config import choice
        self.robot_energy_expert=RoboticsMoE51(joint_dim=25,global_dim=global_dim,history_dim=history_dim)
        m1,m2=choice((32,16),(48,24),(64,32))
        p1,p2=choice((56,28),(96,48),(144,72))
        e1,e2=choice((36,18),(64,32),(96,48))
        material=choice(24,48,72)
        self.process_mce=nn.Sequential(nn.Linear(history_dim,m1),nn.GELU(),nn.Linear(m1,m2),nn.GELU())
        self.process_expert=nn.Sequential(nn.Linear(global_dim+m2,p1),nn.GELU(),nn.Linear(p1,p2),nn.GELU(),nn.Linear(p2,6))
        self.external_axis_energy_expert=nn.Sequential(nn.Linear(global_dim+m2,e1),nn.GELU(),nn.Linear(e1,e2),nn.GELU(),nn.Linear(e2,1))
        self.material_addition_expert=nn.Sequential(nn.Linear(global_dim+m2,material),nn.GELU(),nn.Linear(material,1))
        nn.init.zeros_(self.robot_energy_expert.motion_energy_head[-1].weight)
        nn.init.zeros_(self.robot_energy_expert.motion_energy_head[-1].bias)
        for h in (self.process_expert,self.external_axis_energy_expert,self.material_addition_expert):
            nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)
    def forward(self,j,mask,g,h):
        robot_residual=self.robot_energy_expert(j,mask,g,h)[:,1:2]
        process=torch.cat([g,self.process_mce(h)],dim=1)
        arc_residual=self.process_expert(process)
        external_residual=self.external_axis_energy_expert(process)
        comp=torch.cat([robot_residual,torch.zeros_like(robot_residual),arc_residual,external_residual],dim=1)
        deposit=.945+.075*torch.tanh(self.material_addition_expert(process).flatten())
        return comp,deposit

def predict(model,ds,batch=256,zero_history=False):
    model.eval();device=next(model.parameters()).device;j,m,g,h=[t.to(device) for t in ds.tensors()]
    if zero_history: h=torch.zeros_like(h)
    parts=[];deps=[]
    with torch.no_grad():
        for s in range(0,len(j),batch):
            z,d=model(j[s:s+batch],m[s:s+batch],g[s:s+batch],h[s:s+batch])
            parts.append(z.detach().cpu().numpy());deps.append(d.detach().cpu().numpy())
    corr=np.clip(np.concatenate(parts),-1.,1.)
    components=ds.component_prior*np.exp(corr)
    if zero_history:components[:,0]/=ds.calibration_scale
    pp=ds.prior.copy();pp[:,1]=components.sum(1)
    pp[:,3]=pp[:,2]*np.concatenate(deps)
    return pp,components
