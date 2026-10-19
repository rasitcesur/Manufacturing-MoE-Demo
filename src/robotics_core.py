"""Modular, variable-DOF robot physics residual experts for Manufacturing-MoE.

Shared per-joint expert = weight sharing across arbitrary 1--15 joints.
Topology branch captured by arm_id, root relationships, torso flags.
Machine-characterization expert learns latent parameters from seven past probes,
not generator-private motor efficiency coefficients.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

class RoboticsMoE(nn.Module):
    def __init__(self,joint_dim=13,global_dim=18,history_dim=12,hidden=48):
        super().__init__()
        self.joint_expert=nn.Sequential(nn.Linear(joint_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.machine_expert=nn.Sequential(nn.Linear(history_dim,32),nn.GELU(),nn.Linear(32,24),nn.Tanh())
        self.coupling_expert=nn.Sequential(nn.Linear(hidden*6+global_dim,64),nn.GELU(),nn.Linear(64,32),nn.GELU())
        self.motion_energy_head=nn.Sequential(nn.Linear(32+24+global_dim,80),nn.GELU(),nn.Linear(80,48),nn.GELU(),nn.Linear(48,2))
    def forward(self,joint,mask,global_x,history):
        x=self.joint_expert(joint)*mask.unsqueeze(-1)
        pooled=x.sum(dim=1)/mask.sum(dim=1,keepdim=True).clamp(min=1)
        maxpool=x.masked_fill(mask.unsqueeze(-1)==0,-1e4).max(dim=1).values
        # Explicit arm / torso pooling avoids losing branch information in dual arms.
        # The two categorical joint channels are preserved as binary in Prepared.
        arm1=(joint[:,:,10]>.5).float()*mask
        torso=(joint[:,:,11]>.5).float()*mask
        arm0=(1.-arm1)*(1.-torso)*mask
        def pool(w):
            return (x*w.unsqueeze(-1)).sum(dim=1)/w.sum(dim=1,keepdim=True).clamp(min=1)
        sum_scale=x.sum(dim=1)/mask.sum(dim=1,keepdim=True).clamp(min=1).sqrt()
        coupling=self.coupling_expert(torch.cat([pooled,maxpool,pool(arm0),pool(arm1),pool(torso),sum_scale,global_x],dim=1))
        machine=self.machine_expert(history)
        z=self.motion_energy_head(torch.cat([coupling,machine,global_x],dim=1))
        # log predicted/nominal ratio; robust and inherently positive
        return z

class Prepared:
    def __init__(self,data,mean=None,std=None):
        self.raw=data
        # All extra features are computed from observable previous telemetry and
        # nominal physics, never from true target energy or generator-private gains.
        oldg=data['global_features']
        nominal_joules=np.expm1(np.clip(oldg[:,8:13],0,35)).astype(np.float32)
        fractions=nominal_joules/np.maximum(nominal_joules.sum(axis=1,keepdims=True),1e-6)
        oldh=data['history']
        probe_diffs=(oldh[:,1:6]-oldh[:,0:1])/np.maximum(oldh[:,0:1],1.)
        processed={'joint':data['joint'],
                   'global_features':np.concatenate([oldg,fractions],axis=1),
                   'history':np.concatenate([oldh,probe_diffs],axis=1)}
        if mean is None:
            real=data['joint'][data['mask']>0]
            self.mean={'joint':real.mean(axis=0),
                'global_features':processed['global_features'].mean(axis=0),
                'history':processed['history'].mean(axis=0)}
            self.std={'joint':np.maximum(real.std(axis=0),.02),
                'global_features':np.maximum(processed['global_features'].std(axis=0),.02),
                'history':np.maximum(processed['history'].std(axis=0),.02)}
        else:self.mean=mean;self.std=std
        self.norm={k:np.clip((processed[k]-self.mean[k])/self.std[k],-8,8).astype(np.float32)
                   for k in ['joint','global_features','history']}
        self.norm['joint'][:,:,10:12]=data['joint'][:,:,10:12]
        self.norm['joint']*=data['mask'][:,:,None]
        self.mask=data['mask'].astype(np.float32)
        self.prior=data['prior'].astype(np.float32)
        self.y=data['y'].astype(np.float32)
        self.target=np.log(np.maximum(self.y,1.e-6)/np.maximum(self.prior,1.e-6)).astype(np.float32)
        self.axes=data['n_axes'];self.topology=data['topology'];self.ids=data['robot_ids']
    def tensors(self,device='cpu'):
        return tuple(torch.from_numpy(self.norm[k]).to(device) for k in ('joint','global_features','history'))+(torch.from_numpy(self.mask).to(device),)

def prediction(model,ds,batch=512,device='cpu',use_history=True):
    model.eval();j,g,h,mask=ds.tensors(device)
    if not use_history:
        h=torch.zeros_like(h)
    output=[]
    with torch.no_grad():
        for i in range(0,len(ds.y),batch):
            r=model(j[i:i+batch],mask[i:i+batch],g[i:i+batch],h[i:i+batch])
            output.append(r.cpu().numpy())
    ratio=np.exp(np.clip(np.concatenate(output),-1.5,1.5))
    return ratio*ds.prior
