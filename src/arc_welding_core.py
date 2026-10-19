"""Arc Welding experts, extending Manufacturing-MoE's modular shared joint/MCE design.

Inputs: public topology/WPS and six + six *past* measured power readings; nominal
energy from recomputed physics. There is no access to generator-private parameters
or to target-component values at inference time.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class PreparedArc:
    def __init__(self, data, priors, center=None, scale=None):
        self.y=np.asarray(data['y'],np.float32)
        self.c=np.asarray(data['energy_components_Wh'],np.float32)
        self.prior=np.asarray(priors['prior'],np.float32)
        self.component_prior=np.asarray(priors['component_prior'],np.float32).copy()
        self.mask=np.asarray(data['mask'],np.float32)
        self.axes=np.asarray(data['n_axes'])
        self.topology=np.asarray(data['topology'])
        self.ids=np.asarray(data['robot_ids'])
        self.pids=np.asarray(data['program_ids'])
        self.industrial_scope=np.asarray(data['industrial_scope'],bool)
        j=np.asarray(data['joint'],np.float32)
        rawg=np.asarray(data['global_features'],np.float32)
        rawh=np.asarray(data['history_power_W'],np.float32)
        physics=np.asarray(priors['physics_extra'],np.float32)
        # Public, pre-weld calibration loads: intercept/slope from watts vs nominal arc V*I.
        loads=np.array([0,90,130,180,230,280],np.float32)
        va=loads*(17.5+.037*loads)
        slope=((rawh[:,1:6]-rawh[:,1:6].mean(1,keepdims=True))*(va[1:6]-va[1:6].mean())).sum(1)/np.sum((va[1:6]-va[1:6].mean())**2)
        # Calibrate welding PSU nameplate input using INDEPENDENT earlier
        # measurements at standard current/voltage loads; never use this job's energy.
        observed_to_nameplate=np.clip(rawg[:,21]*slope,.65,1.55)
        self.component_prior[:,2]*=observed_to_nameplate
        self.prior[:,1]=self.component_prior.sum(1)
        intercept=rawh[:,0]
        # Robot sensor probes describe different prior applied loads, not this job.
        drive_response=(rawh[:,7:12]-rawh[:,6:7])/np.maximum(rawh[:,6:7],1.)
        hist=np.concatenate([np.log1p(np.maximum(rawh,0)),slope[:,None],np.log1p(np.maximum(intercept,0))[:,None],drive_response],axis=1)
        # Positional axes/signs and branch flags retained, geometric scales compressed.
        g=rawg.copy()
        for k in [3,7,8,9,10,11,12,13,16,17,21,22,23,24]:
            g[:,k]=np.log1p(np.maximum(0,g[:,k]))
        extra=np.concatenate([np.log1p(np.maximum(physics,0)),np.log1p(np.maximum(self.component_prior,0)),
             np.log1p(np.maximum(self.prior,0)),
             self.component_prior/np.maximum(self.component_prior.sum(axis=1,keepdims=True),1e-6)],axis=1)
        full_g=np.concatenate([g,extra],axis=1)
        # Preserve zero-padding and shared per-joint architecture.
        if center is None:
            valid=j[self.mask>0]
            center={'joint':valid.mean(0),'global':full_g.mean(0),'history':hist.mean(0)}
            scale={'joint':np.maximum(valid.std(0),.025),'global':np.maximum(full_g.std(0),.025),'history':np.maximum(hist.std(0),.025)}
        self.center=center;self.scale=scale
        self.joint=(np.clip((j-center['joint'])/scale['joint'],-8,8)*self.mask[:,:,None]).astype(np.float32)
        self.global_x=np.clip((full_g-center['global'])/scale['global'],-8,8).astype(np.float32)
        self.history=np.clip((hist-center['history'])/scale['history'],-8,8).astype(np.float32)
        # Keep categorical branch flags intact.
        self.joint[:,:,9:12]=j[:,:,9:12]
        self.joint*=self.mask[:,:,None]
        self.comp_target=np.log(np.maximum(self.c,1e-8)/np.maximum(self.component_prior,1e-8)).astype(np.float32)
        self.transfer_target=(self.y[:,3]/np.maximum(self.y[:,2],1e-8)).astype(np.float32)
        self.global_dim=self.global_x.shape[-1];self.history_dim=self.history.shape[-1]
    def tensors(self):
        return tuple(torch.from_numpy(v) for v in (self.joint,self.mask,self.global_x,self.history))

class ArcWeldingMoE(nn.Module):
    def __init__(self,global_dim,history_dim,hidden=48):
        super().__init__()
        self.joint_motion_expert=nn.Sequential(nn.Linear(15,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.topology_coupling_expert=nn.Sequential(nn.Linear(3*hidden+global_dim,80),nn.GELU(),nn.Linear(80,48),nn.GELU())
        self.machine_characterization_expert=nn.Sequential(nn.Linear(history_dim,56),nn.GELU(),nn.Linear(56,32),nn.GELU())
        self.arc_power_expert=nn.Sequential(nn.Linear(48+32+global_dim,96),nn.GELU(),nn.Linear(96,64),nn.GELU(),nn.Linear(64,2))
        self.robot_drive_expert=nn.Sequential(nn.Linear(48+32+global_dim,80),nn.GELU(),nn.Linear(80,40),nn.GELU(),nn.Linear(40,2))
        self.auxiliary_energy_expert=nn.Sequential(nn.Linear(48+32+global_dim,80),nn.GELU(),nn.Linear(80,40),nn.GELU(),nn.Linear(40,4))
        self.material_addition_expert=nn.Sequential(nn.Linear(48+32+global_dim,48),nn.GELU(),nn.Linear(48,1))
        # Zero initial output weights -> meaningful nominal physics start.
        for m in (self.arc_power_expert,self.robot_drive_expert,self.auxiliary_energy_expert):
            nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias)
        nn.init.zeros_(self.material_addition_expert[-1].weight)
        nn.init.zeros_(self.material_addition_expert[-1].bias)
    def forward(self,j,mask,g,h):
        x=self.joint_motion_expert(j)*mask.unsqueeze(-1)
        n=mask.sum(1,keepdim=True).clamp(min=1)
        pooled=x.sum(1)/n
        xmax=x.masked_fill(mask.unsqueeze(-1)==0,-1e4).max(1).values
        sum_scale=x.sum(1)/torch.sqrt(n)
        topo=self.topology_coupling_expert(torch.cat([pooled,xmax,sum_scale,g],dim=1))
        machine=self.machine_characterization_expert(h)
        context=torch.cat([topo,machine,g],dim=1)
        robot=self.robot_drive_expert(context)
        arc=self.arc_power_expert(context)
        aux=self.auxiliary_energy_expert(context)
        comp=torch.cat([robot,arc,aux],dim=1)
        transfer=.945+.075*torch.tanh(self.material_addition_expert(context).flatten())
        return comp,transfer

def predict(model,ds,batch=256,zero_history=False):
    model.eval();j,m,g,h=ds.tensors(); outputs=[];ratios=[]
    if zero_history:h=torch.zeros_like(h)
    with torch.no_grad():
        for k in range(0,len(j),batch):
            z,dep=model(j[k:k+batch],m[k:k+batch],g[k:k+batch],h[k:k+batch])
            outputs.append(z.numpy());ratios.append(dep.numpy())
    z=np.clip(np.concatenate(outputs),-1.0,1.0)
    comps=ds.component_prior*np.exp(z)
    dep=np.concatenate(ratios)
    pred=ds.prior.copy()
    pred[:,1]=comps.sum(1)
    pred[:,3]=pred[:,2]*dep
    return pred,comps
