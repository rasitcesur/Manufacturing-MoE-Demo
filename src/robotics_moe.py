"""current sign-aware hierarchical gravity expert; learns robot design response.

Only robot expert is changed. Legacy 11 manufacturing operations untouched.
Features: 13 original joint + 10 signed gravity/topology per joint; 18 original
physical aggregate + 8 signed gravity global; 12 history probes.
"""
import numpy as np
import torch
from torch import nn
from robotics_core import Prepared as PreviousPrepared

class Prepared51(PreviousPrepared):
    def __init__(self,data,mean=None,std=None):
        super().__init__(data,mean,std)
        rawj=np.asarray(data['joint_gravity'],np.float32)
        rawg=np.asarray(data['global_gravity'],np.float32)
        if mean is None:
            active=rawj[data['mask']>0]
            self.mean['joint_gravity']=active.mean(axis=0)
            self.std['joint_gravity']=np.maximum(active.std(axis=0),.03)
            self.mean['global_gravity']=rawg.mean(axis=0)
            self.std['global_gravity']=np.maximum(rawg.std(axis=0),.03)
        nj=np.clip((rawj-self.mean['joint_gravity'])/self.std['joint_gravity'],-8,8)*data['mask'][:,:,None]
        ng=np.clip((rawg-self.mean['global_gravity'])/self.std['global_gravity'],-8,8)
        self.norm['joint']=np.concatenate([self.norm['joint'],nj],axis=2).astype(np.float32)
        self.norm['global_features']=np.concatenate([self.norm['global_features'],ng],axis=1).astype(np.float32)

class RoboticsMoE51(nn.Module):
    """Version-scaled sign-aware hierarchical robot model; 1-15 axes only."""
    def __init__(self,joint_dim=23,global_dim=26,history_dim=12,hidden=None):
        super().__init__()
        from config import choice
        hidden=choice(24,48,72) if hidden is None else hidden
        mc1,mc2=choice((24,16),(40,28),(64,40))
        co1,co2=choice((48,24),(72,40),(96,64))
        gr1,gr2=choice((24,16),(40,24),(64,40))
        he1,he2=choice((48,24),(96,48),(128,64))
        self.joint_expert=nn.Sequential(nn.Linear(joint_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.machine_expert=nn.Sequential(nn.Linear(history_dim,mc1),nn.GELU(),nn.Linear(mc1,mc2),nn.Tanh())
        self.coupling_expert=nn.Sequential(nn.Linear(hidden*6+global_dim,co1),nn.GELU(),nn.Linear(co1,co2),nn.GELU())
        self.gravity_expert=nn.Sequential(nn.Linear(8+mc2,gr1),nn.GELU(),nn.Linear(gr1,gr2),nn.GELU())
        self.motion_energy_head=nn.Sequential(nn.Linear(co2+mc2+gr2+global_dim,he1),nn.GELU(),nn.Linear(he1,he2),nn.GELU(),nn.Linear(he2,2))
    def forward(self,joint,mask,global_x,history):
        x=self.joint_expert(joint)*mask.unsqueeze(-1)
        count=mask.sum(dim=1,keepdim=True).clamp(min=1)
        pooled=x.sum(dim=1)/count
        mx=x.masked_fill(mask.unsqueeze(-1)==0,-1e4).max(dim=1).values
        arm1=(joint[:,:,10]>.5).float()*mask
        torso=(joint[:,:,11]>.5).float()*mask
        arm0=(1.-arm1)*(1.-torso)*mask
        def pool(w):return (x*w.unsqueeze(-1)).sum(dim=1)/w.sum(dim=1,keepdim=True).clamp(min=1)
        sum_scale=x.sum(dim=1)/count.sqrt()
        coupling=self.coupling_expert(torch.cat([pooled,mx,pool(arm0),pool(arm1),pool(torso),sum_scale,global_x],dim=1))
        machine=self.machine_expert(history)
        gravity=self.gravity_expert(torch.cat([global_x[:,-8:],machine],dim=1))
        return self.motion_energy_head(torch.cat([coupling,machine,gravity,global_x],dim=1))

def prediction(model,ds,batch=512,device=None,use_history=True):
    model.eval();device=device or next(model.parameters()).device;j,g,h,mask=ds.tensors(device)
    if not use_history:h=torch.zeros_like(h)
    parts=[]
    with torch.no_grad():
        for k in range(0,len(ds.y),batch):
            parts.append(model(j[k:k+batch],mask[k:k+batch],g[k:k+batch],h[k:k+batch]).cpu().numpy())
    ratio=np.exp(np.clip(np.concatenate(parts),-1.5,1.5))
    return ratio*ds.prior
