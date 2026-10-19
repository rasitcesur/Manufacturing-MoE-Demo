"""Variable topology 1--15 DOF robotics rigid-body inverse dynamics simulator.

Independent, reduced-order ideal revolute joint rigid-body RNEA-like Newton-Euler
backward recursion. NOT MuJoCo or Pinocchio; excludes contact, gearing backlash,
flexible joints, hard stops and servo controller transients.

Units: m, kg, seconds, rad, N, Nm, W and Wh. Topology supplied as rooted tree.
Joint axes are expressed in parent frame; inertia in child link frame.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit

MAX_JOINTS=15

@njit(cache=True)
def _cross(a,b):
    return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])

@njit(cache=True)
def _rot(axis,theta):
    c=math.cos(theta);s=math.sin(theta)
    x,y,z=axis[0],axis[1],axis[2]
    K=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    I=np.eye(3)
    return I+s*K+(1.-c)*(K@K)

@njit(cache=True)
def rnea_tree(parents, offsets, axes, coms, mass, inertia, q, dq, ddq):
    """Return inverse-dynamics joint torques for rooted revolute tree.
    All joint positions are at their preceding-link pivot, offset in parent coords.
    """
    n=len(q)
    r=np.zeros((n,3,3)); pivot=np.zeros((n,3));axis_w=np.zeros((n,3))
    omega=np.zeros((n,3));alpha=np.zeros((n,3));acc=np.zeros((n,3))
    force=np.zeros((n,3)); moment=np.zeros((n,3)); tau=np.zeros(n)
    gravity=np.array([0.,0.,-9.81])
    for i in range(n):
        p=parents[i]
        if p>=0:
            R0=r[p]; wp=omega[p];ap=alpha[p]; a0=acc[p];pivot0=pivot[p]
        else:
            R0=np.eye(3);wp=np.zeros(3);ap=np.zeros(3);a0=np.zeros(3);pivot0=np.zeros(3)
        arm=R0@offsets[i]
        pivot[i]=pivot0+arm
        axis_w[i]=R0@axes[i]
        r[i]=R0@_rot(axes[i],q[i])
        omega[i]=wp+axis_w[i]*dq[i]
        alpha[i]=ap+axis_w[i]*ddq[i]+_cross(wp,axis_w[i]*dq[i])
        acc[i]=a0+_cross(ap,arm)+_cross(wp,_cross(wp,arm))
        rc=r[i]@coms[i]
        ic=r[i]@np.diag(inertia[i])@r[i].T
        ci=acc[i]+_cross(alpha[i],rc)+_cross(omega[i],_cross(omega[i],rc))
        force[i]=mass[i]*(ci-gravity)
        moment[i]=ic@alpha[i]+_cross(omega[i],ic@omega[i])+_cross(rc,force[i])
    for i in range(n-1,-1,-1):
        tau[i]=np.dot(axis_w[i],moment[i])
        p=parents[i]
        if p>=0:
            force[p]+=force[i]
            moment[p]+=moment[i]+_cross(pivot[i]-pivot[p],force[i])
    return tau


def _unit(v):
    a=np.asarray(v,float);return a/np.linalg.norm(a)


def robot_configuration(n, topology, rng, geometry_shift=False):
    """Build physically plausible but non-OEM synthetic kinematic tree."""
    if not 1<=n<=15:raise ValueError('n must lie in 1..15')
    if topology=='serial': lengths=[n];torso=False
    elif topology=='dual':
        if n<2:raise ValueError('dual requires at least two joints')
        na=min(7,max(1,n//2));nb=n-na
        if nb>7:raise ValueError('dual arm supports no more than 14 joints')
        lengths=[na,nb];torso=False
    elif topology=='torso_dual':
        if n<3:raise ValueError('torso_dual needs torso and two arms')
        na=min(7,max(1,(n-1)//2));nb=n-1-na
        if nb>7:raise ValueError('torso arms <=7 each')
        lengths=[na,nb];torso=True
    else:raise ValueError(topology)
    pa=[];offs=[];axes=[];com=[];ms=[];ins=[];ids=[];armids=[]
    if torso:
        pa.append(-1);offs.append([0,0,0.45]);axes.append([0,0,1]);com.append([0.03,0,0]);ms.append(float(rng.uniform(20,45)));ins.append([1.6,1.6,2.2]);ids.append('torso');armids.append(2)
    for aid,L in enumerate(lengths):
        prev=0 if torso else -1
        for j in range(L):
            link=float(rng.uniform(.20,.50)*(1.20 if geometry_shift else 1))
            mass=float(rng.uniform(1.5,11.)*(1.25 if geometry_shift else 1))
            if j==0 and len(lengths)==2:
                off=[0,(-0.32 if aid==0 else +0.32),0.12 if torso else 0.50]
            elif j==0:off=[0,0,0.48]
            else:off=[last_length,0,0]
            axislist=[[0,0,1],[0,1,0],[1,0,0],[0,1,0],[1,0,0],[0,1,0],[0,0,1]]
            axis=axislist[j%7]
            pa.append(prev);offs.append(off);axes.append(axis)
            com.append([link*.50,0,0]);ms.append(mass)
            I=mass*link*link/12
            ins.append([max(.002,I*.12),max(.002,I),max(.002,I)])
            idx=len(pa)-1;ids.append(f'arm{aid}_joint{j+1}');armids.append(aid)
            prev=idx;last_length=link
    if len(pa)!=n:raise AssertionError('wrong joint count')
    return {'n':n,'topology':topology,'parents':pa,'offsets':offs,'axes':axes,'coms':com,'mass':ms,
            'inertia':ins,'joint_names':ids,'arm_id':armids,'payload_kg':0.,
            'qmax':float(rng.uniform(.8,1.8)),'amax':float(rng.uniform(2.,4.5)),
            'motor_nominal_eff':float(rng.uniform(.82,.93)),
            'electronic_power_W':float(rng.uniform(90.,240.)),
            'regen_capability':float(rng.uniform(.30,.75)),
            'private_motor_efficiency':float(rng.uniform(.79,1.08)),
            'private_driver_loss':float(rng.uniform(.75,1.28)),
            'private_friction_scale':float(rng.uniform(.78,1.27)),
            'private_settle_s':float(rng.uniform(.015,.145)),
            'private_regen_scale':float(rng.uniform(.75,1.12))}


def quintic_trajectory(q0,q1,duration,steps=22):
    s=np.linspace(0.,1.,steps)
    blend=10*s**3-15*s**4+6*s**5
    db=(30*s**2-60*s**3+30*s**4)/duration
    ddb=(60*s-180*s**2+120*s**3)/(duration**2)
    move=q1-q0
    return q0[None,:]+blend[:,None]*move[None,:], db[:,None]*move[None,:],ddb[:,None]*move[None,:]


def simulate(cfg, motion, steps=22, detailed=False):
    """Compute forward specified trajectory inverse dynamics + electrical model.

    Motion is an independent plan: q_start/q_end, speed_factor, dwell_s and payload.
    The private efficiency parameters are intentionally unavailable to training code.
    """
    n=cfg['n'];q0=np.asarray(motion['q_start'],float);q1=np.asarray(motion['q_end'],float)
    disp=np.abs(q1-q0);speed=float(motion['speed_factor']);
    maxv=cfg['qmax']*speed;maxa=cfg['amax']*speed
    move_t=max(.18, 1.875*disp.max()/maxv,math.sqrt(5.78*disp.max()/maxa))
    dwell=float(motion['dwell_s']);q,dq,ddq=quintic_trajectory(q0,q1,move_t,steps)
    pa=np.array(cfg['parents'],np.int64);offset=np.array(cfg['offsets'],float)
    axis=np.array(cfg['axes'],float);com=np.array(cfg['coms'],float);masses=np.array(cfg['mass'],float)
    inertia=np.array(cfg['inertia'],float)
    payload=float(motion['payload_kg'])
    # Payload mass and approximate isotropic inertia at active grippers; map to end links.
    ends=[i for i in range(n) if i not in pa]
    for idx in ends:
        masses[idx]+=payload/len(ends)
        inertia[idx]+=payload*.012/len(ends)
    torques=np.stack([rnea_tree(pa,offset,axis,com,masses,inertia,q[i],dq[i],ddq[i]) for i in range(steps)])
    fric=(.12+.015*masses[None,:])*np.tanh(dq*18.)+.08*dq
    tau_motor=torques+fric*cfg['private_friction_scale']
    mechanical=tau_motor*dq
    nominal_tau=torques+fric
    positive=np.maximum(mechanical,0.);negative=np.minimum(mechanical,0.)
    # Real efficiency and driver losses are latent, only observable through external probe.
    efficiency=float(np.clip(cfg['motor_nominal_eff']*cfg['private_motor_efficiency'],.60,.98))
    copper=.020*tau_motor**2*cfg['private_driver_loss']
    drive_idle=(2.0+.35*masses[None,:])*cfg['private_driver_loss']
    regen=cfg['regen_capability']*cfg['private_regen_scale']
    electric=positive/efficiency+negative*regen+copper+drive_idle
    # Shared DC bus; negative joint contribution can offset positive joint demand.
    bus=np.maximum(electric.sum(axis=1),0.)
    control=cfg['electronic_power_W']+16.*n
    power=bus+control
    energy=float(np.trapezoid(power,dx=move_t/(steps-1))/3600.)
    settle=cfg['private_settle_s']*(1.+.08*n/15.)
    settle_e=settle*(control+np.sum(drive_idle))/3600.
    dwell_e=dwell*(control+np.sum(drive_idle))/3600.
    true_energy=energy+settle_e+dwell_e
    true_time=move_t+settle+dwell
    # Reference nominal prediction, using only public nameplate physics.
    nominal_electric=np.maximum(nominal_tau*dq,0.)/cfg['motor_nominal_eff']+np.minimum(nominal_tau*dq,0.)*cfg['regen_capability']+.020*nominal_tau**2+2.+.35*masses[None,:]
    nominal_energy=(np.trapezoid(np.maximum(nominal_electric.sum(axis=1),0.)+control,dx=move_t/(steps-1))+dwell*(control+np.sum(2.+.35*masses)))/3600.
    mot_nom=nominal_tau*dq
    nom_pos=float(np.trapezoid(np.maximum(mot_nom,0).sum(axis=1),dx=move_t/(steps-1)))
    nom_regen=float(np.trapezoid(np.maximum(-mot_nom,0).sum(axis=1),dx=move_t/(steps-1)))
    nom_copper=float(np.trapezoid((.020*nominal_tau**2).sum(axis=1),dx=move_t/(steps-1)))
    nom_idle=float((move_t+dwell)*np.sum(2.+.35*masses))
    nom_aux=float((move_t+dwell)*control)
    out={'time_s':true_time,'energy_Wh':true_energy,'nominal_time_s':move_t+dwell,'nominal_energy_Wh':float(nominal_energy),
         'mechanical_positive_J':float(np.trapezoid(positive.sum(axis=1),dx=move_t/(steps-1))),
         'mechanical_regen_J':float(np.trapezoid(-negative.sum(axis=1),dx=move_t/(steps-1))),
         'torque_rms_Nm':np.sqrt((torques**2).mean(axis=0)).tolist(),
         'torque_max_Nm':np.max(np.abs(torques),axis=0).tolist(),
         'velocity_rms_rad_s':np.sqrt((dq**2).mean(axis=0)).tolist(),
         'mean_abs_velocity_rad_s':np.mean(np.abs(dq),axis=0).tolist(),
         'torque_work_J':np.trapezoid(np.abs(nominal_tau*dq),dx=move_t/(steps-1),axis=0).tolist(),
         'torque_rms_sq_Nm2':np.mean(nominal_tau**2,axis=0).tolist(),
         'material_removed_g':0.,
         'nominal_positive_J':nom_pos,'nominal_regen_J':nom_regen,'nominal_copper_J':nom_copper,
         'nominal_idle_J':nom_idle,'nominal_aux_J':nom_aux}
    if detailed:
        out['trace']={'time_s':np.linspace(0,move_t,steps).tolist(),'torque_Nm':torques.tolist(),
                      'joint_position_rad':q.tolist(),'joint_velocity_rad_s':dq.tolist(), 'bus_power_W':power.tolist()}
    return out


def sensor_probes(cfg,rng):
    """Six *past* power measurements (Watts per motor, normalized against size).
    Standstill, low/medium/high joint speed, loaded, braking/regen.
    These measurements exclude target program observations and private factors.
    """
    n=cfg['n'];eff=cfg['motor_nominal_eff']*cfg['private_motor_efficiency']
    friction=cfg['private_friction_scale'];loss=cfg['private_driver_loss']
    base=(cfg['electronic_power_W']+16*n)/n+(2+.35*np.mean(cfg['mass']))*loss
    probes=[base,base+7*friction+5*loss,base+18/eff+13*loss,base+35/eff+24*loss,
            base+52/eff+46*loss,base+19/eff+16*loss-8*cfg['regen_capability']*cfg['private_regen_scale']]
    noise=rng.normal(0,.006,6)
    return np.maximum(np.array(probes)*(1+noise),.001).tolist()
