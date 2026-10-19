"""Signed gravity and downstream inertia features calculated from observable inputs."""
import numpy as np
from robot_physics import rnea_tree,quintic_trajectory,_rot

def potential_energy(c,q,payload=0.):
    n=c['n'];pa=c['parents'];A=np.asarray(c['axes'],float); O=np.asarray(c['offsets'],float);C=np.asarray(c['coms'],float);m=np.asarray(c['mass'],float).copy()
    ends=[i for i in range(n) if i not in pa]
    for e in ends:m[e]+=payload/len(ends)
    R=[];p=[];U=0.
    for i in range(n):
        Rp=R[pa[i]] if pa[i]>=0 else np.eye(3)
        pp=p[pa[i]] if pa[i]>=0 else np.zeros(3)
        pi=pp+Rp@O[i];Ri=Rp@_rot(A[i],q[i]);R.append(Ri);p.append(pi)
        U+=m[i]*9.81*(pi+Ri@C[i])[2]
    return U

def signed_features(c,mo,steps=11):
    n=c['n'];pa=np.asarray(c['parents'],np.int64);off=np.asarray(c['offsets'],float);axis=np.asarray(c['axes'],float)
    com=np.asarray(c['coms'],float);m=np.asarray(c['mass'],float).copy();I=np.asarray(c['inertia'],float).copy()
    payload=float(mo['payload_kg']);ends=[i for i in range(n) if i not in pa]
    for i in ends:m[i]+=payload/len(ends);I[i]+=payload*.012/len(ends)
    q0=np.asarray(mo['q_start'],float);q1=np.asarray(mo['q_end'],float);disp=np.abs(q1-q0);speed=mo['speed_factor']
    T=max(.18,1.875*disp.max()/(c['qmax']*speed),np.sqrt(5.78*disp.max()/(c['amax']*speed)))
    q,dq,_=quintic_trajectory(q0,q1,T,steps);z=np.zeros(n)
    gt=np.stack([rnea_tree(pa,off,axis,com,m,I,qi,z,z) for qi in q]);pg=gt*dq
    up=np.trapezoid(np.maximum(pg,0),dx=T/(steps-1),axis=0)
    down=np.trapezoid(np.maximum(-pg,0),dx=T/(steps-1),axis=0)
    net=np.trapezoid(pg,dx=T/(steps-1),axis=0)
    subtree=m.copy();depth=np.zeros(n)
    for i in range(n):
        if pa[i]>=0:depth[i]=depth[pa[i]]+1
    for i in range(n-1,-1,-1):
        if pa[i]>=0:subtree[pa[i]]+=subtree[i]
    j=np.zeros((15,10),np.float32)
    for i in range(n):
        j[i]=[np.log1p(up[i]),np.log1p(down[i]),np.sign(net[i])*np.log1p(abs(net[i])),
              np.sign(q1[i]-q0[i])*np.log1p(abs(q1[i]-q0[i])),
              np.log1p(np.sqrt(np.mean(gt[:,i]**2))),
              np.sign(gt[steps//2,i])*np.log1p(abs(gt[steps//2,i])),
              np.log1p(subtree[i]),subtree[i]/m.sum(),depth[i]/15.,float(up[i]+down[i]<1e-7)]
    G=np.asarray([np.log1p(up.sum()),np.log1p(down.sum()),np.sign(net.sum())*np.log1p(abs(net.sum())),
        float(up.sum()/(up.sum()+down.sum()+1e-8)),np.log1p(m.sum()),np.log1p(subtree.max()),
        float(np.mean(disp<1e-6)),float(np.mean(j[:n,9]))],np.float32)
    return j,G
