"""current separately trained process-specific welding MoEs.
Shared welding-cell robot physics and topology; NO trained weight sharing among
GMAW, RSW and TIG. MCE uses exclusively previous measured sensor data.
"""
import numpy as np
import torch
from torch import nn

class WeldingSet:
 def __init__(self,arrays,norm=None):
  self.y=np.asarray(arrays['y'],np.float32)
  self.c=np.asarray(arrays['components'],np.float32)
  self.prior=np.asarray(arrays['prior'],np.float32)
  self.comp_prior=np.asarray(arrays['component_prior'],np.float32)
  self.mask=np.asarray(arrays['mask'],np.float32)
  self.ids=np.asarray(arrays['robot_ids']);self.pids=np.asarray(arrays['program_ids'])
  self.topology=np.asarray(arrays['topology']);self.axes=np.asarray(arrays['n_axes'])
  joint=np.asarray(arrays['joint'],np.float32)
  g=np.asarray(arrays['global_features'],np.float32).copy()
  # log compression only for strictly non-negative continuous variables;
  # categorical flags and joint angles retain their signed coordinates.
  for col in (4,5,6,7,8,9,10,11,12,13,15,16,17,18,20,21,22,24,25,26,27,30,33,34,35,36,37):
   g[:,col]=np.log1p(np.maximum(g[:,col],0))
  h=np.asarray(arrays['history'],np.float32).copy()
  h[:,:12]=np.log1p(np.maximum(h[:,:12],0))
  cp=np.maximum(self.comp_prior,1e-9)
  x=np.concatenate([g,np.log1p(cp),cp/np.maximum(cp.sum(1,keepdims=True),1e-8),
     np.log1p(np.maximum(self.prior[:,1:2],0))],axis=1)
  if norm is None:
   active=joint[self.mask>0]
   norm={'jm':active.mean(0),'js':np.maximum(active.std(0),.04),
         'gm':x.mean(0),'gs':np.maximum(x.std(0),.04),
         'hm':h.mean(0),'hs':np.maximum(h.std(0),.04)}
  self.norm=norm
  self.joint=(np.clip((joint-norm['jm'])/norm['js'],-7,7)*self.mask[:,:,None]).astype(np.float32)
  self.global_x=np.clip((x-norm['gm'])/norm['gs'],-7,7).astype(np.float32)
  self.history=np.clip((h-norm['hm'])/norm['hs'],-7,7).astype(np.float32)
  self.y_logratio=np.log(np.maximum(self.c,1e-9)/np.maximum(self.comp_prior,1e-9)).astype(np.float32)
  # Only TIG with filler has material-addition target; RSW 0 is physically correct.
  self.material_ratio=(self.y[:,2]/np.maximum(self.prior[:,2],1e-8)).astype(np.float32)
 def tensors(self):return tuple(torch.from_numpy(x) for x in (self.joint,self.mask,self.global_x,self.history))

class WeldingMoE55(nn.Module):
 def __init__(self,global_dim,history_dim,process):
  super().__init__();self.process=process
  from config import choice
  jw=choice(24,48,72);tw=choice(16,32,48);mw=choice(16,32,48)
  th=choice(32,64,96);mh=choice(32,64,96)
  rw=choice(32,64,96);rr=choice(16,32,48)
  pw=choice(40,80,120);pp=choice(20,40,60);ew=choice(20,40,60)
  self.joint_dynamics=nn.Sequential(nn.Linear(15,jw),nn.GELU(),nn.Linear(jw,jw),nn.GELU())
  self.topology_expert=nn.Sequential(nn.Linear(jw*3+global_dim,th),nn.GELU(),nn.Linear(th,tw),nn.GELU())
  self.machine_characterization=nn.Sequential(nn.Linear(history_dim,mh),nn.GELU(),nn.Linear(mh,mw),nn.GELU())
  sd=tw+mw+global_dim
  self.robot_energy=nn.Sequential(nn.Linear(sd,rw),nn.GELU(),nn.Linear(rw,rr),nn.GELU(),nn.Linear(rr,2))
  self.process_energy=nn.Sequential(nn.Linear(sd,pw),nn.GELU(),nn.Linear(pw,pp),nn.GELU(),nn.Linear(pp,6))
  self.external_energy=nn.Sequential(nn.Linear(sd,ew),nn.GELU(),nn.Linear(ew,1))
  # RSW never deposits filler; remove its redundant material head in Small only.
  self.material_expert=None if (choice(True,False,False) and process=='rsw') else nn.Sequential(nn.Linear(sd,ew),nn.GELU(),nn.Linear(ew,1))
  for head in (self.robot_energy,self.process_energy,self.external_energy,self.material_expert):
   if head is not None:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
 def forward(self,j,m,g,h):
  z=self.joint_dynamics(j)*m.unsqueeze(-1);n=m.sum(1,keepdim=True).clamp_min(1)
  pooled=z.sum(1)/n;mx=z.masked_fill(m.unsqueeze(-1)==0,-1e4).max(1).values;ss=z.sum(1)/n.sqrt()
  topo=self.topology_expert(torch.cat([pooled,mx,ss,g],1));machine=self.machine_characterization(h)
  f=torch.cat([topo,machine,g],1)
  comp=torch.cat([self.robot_energy(f),self.process_energy(f),self.external_energy(f)],dim=1)
  transfer=torch.ones(len(j),dtype=j.dtype,device=j.device) if self.material_expert is None else .98+.08*torch.tanh(self.material_expert(f).flatten())
  return comp,transfer

def predict(model,ds,blend=1.,no_history=False,batch=512):
 model.eval();device=next(model.parameters()).device;j,m,g,h=[t.to(device) for t in ds.tensors()]
 if no_history:h=torch.zeros_like(h)
 comps=[];mr=[]
 with torch.no_grad():
  for k in range(0,len(j),batch):
   z,t=model(j[k:k+batch],m[k:k+batch],g[k:k+batch],h[k:k+batch]);comps.append(z.detach().cpu().numpy());mr.append(t.detach().cpu().numpy())
 residual=np.clip(np.concatenate(comps),-1.,1.)*blend
 predicted_components=ds.comp_prior*np.exp(residual)
 if no_history:
  # Remove automatic prior-history calibration when evaluating genuine cold start.
  pass
 out=ds.prior.copy();out[:,1]=predicted_components.sum(1)
 if model.process=='tig':out[:,2]=out[:,3]*np.maximum(.0,np.concatenate(mr))
 return out,predicted_components
