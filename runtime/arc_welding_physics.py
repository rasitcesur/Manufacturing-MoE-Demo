"""Physics-informed GMAW/MIG-MAG robotic welding dataset teacher.

Uses the shared 1–15 axis robot inverse-dynamics routine in robot_physics.rnea_tree
inverse-dynamics simulator. One torch per welding segment; additional arms may
move as fixtures/supports. No weld metallurgical quality/penetration prediction.

Arc electric energy and thermal heat to workpiece are NOT additive. Gas is
volumetric consumption, not electrical energy. Synthetic and uncalibrated.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from robot_physics import rnea_tree, quintic_trajectory, _rot, robot_configuration

PRIVATE_SOURCE_FIELDS = ('private_source_efficiency','private_wire_drive_scale','private_cooling_scale',
                         'private_fume_scale','private_valve_scale')

DENSITY_G_MM3 = {'carbon_steel':.00785,'stainless':.0079,'aluminum':.00270}
WIRE_CHOICES = {'carbon_steel':('ER70S-6',.00785), 'stainless':('ER308LSi',.00790), 'aluminum':('ER4043',.00270)}

def machine_source(rng, ood=False):
    return {'source_nominal_efficiency':.86,'source_nameplate_W':15000.,
            'wire_feeder_nominal_W':55.,'cooling_nominal_W':210.,
            'fume_extraction_nominal_W':260.,'gas_valve_nominal_W':9.,
            'source_standby_nominal_W':68.,
            'private_source_efficiency':float(rng.uniform(.71,.78) if ood else rng.uniform(.80,.92)),
            'private_wire_drive_scale':float(rng.uniform(1.25,1.55) if ood else rng.uniform(.78,1.24)),
            'private_cooling_scale':float(rng.uniform(1.18,1.48) if ood else rng.uniform(.75,1.20)),
            'private_fume_scale':float(rng.uniform(1.18,1.48) if ood else rng.uniform(.75,1.21)),
            'private_valve_scale':float(rng.uniform(.85,1.2))}

def public_source(source):
    return {k:v for k,v in source.items() if k not in PRIVATE_SOURCE_FIELDS}

def fk_positions(cfg, q):
    """Return link pivot and link orientation, in world coordinates."""
    n=cfg['n']; rotations=[None]*n; pivots=[None]*n
    for i in range(n):
        p=cfg['parents'][i]
        R0=np.eye(3) if p<0 else rotations[p]
        base=np.zeros(3) if p<0 else pivots[p]
        pivots[i]=base+R0@np.asarray(cfg['offsets'][i],float)
        rotations[i]=R0@_rot(np.asarray(cfg['axes'][i],float),float(q[i]))
    return rotations,pivots

def torch_tips(cfg):
    pa=set(x for x in cfg['parents'] if x>=0)
    return [i for i in range(cfg['n']) if i not in pa]

def torch_trajectory(cfg, q_sequence, tip_idx):
    positions=[]
    for q in q_sequence:
        rotations,pivots=fk_positions(cfg,q)
        # tool center point is distal from last link, approximated from COM
        distal=2.*np.asarray(cfg['coms'][tip_idx],float)+np.array([.07,0,.025])
        positions.append(pivots[tip_idx]+rotations[tip_idx]@distal)
    return np.asarray(positions)

def duration_for_joint_motion(cfg, q0, q1, speed=0.8):
    d=float(np.max(np.abs(np.asarray(q1)-np.asarray(q0))))
    return max(.25,1.875*d/max(cfg['qmax']*speed,1e-5), math.sqrt(5.78*d/max(cfg['amax']*speed,1e-5)))

def timed_robot_motion(cfg, q0, q1, T, payload_kg=1.2, steps=20):
    """Use the shared joint-tree inverse-dynamics and drive-loss equations for the requested duration.

    Returns Wh split into robotic joint drive and shared controller; includes no
    seam process electric power. Joints are assumed position-controlled along
    smooth quintic interpolation. 20 samples resolve smooth joint dynamics, not
    high-frequency welding arc fluctuations.
    """
    assert T>0
    q0=np.asarray(q0,float);q1=np.asarray(q1,float);n=cfg['n']
    q,dq,ddq=quintic_trajectory(q0,q1,T,steps)
    pa=np.asarray(cfg['parents'],np.int64)
    offsets=np.asarray(cfg['offsets'],float);axes=np.asarray(cfg['axes'],float)
    coms=np.asarray(cfg['coms'],float);mass=np.asarray(cfg['mass'],float).copy()
    inertia=np.asarray(cfg['inertia'],float).copy()
    ends=torch_tips(cfg)
    for i in ends:
        mass[i]+=payload_kg/len(ends)
        inertia[i]+=payload_kg*.012/len(ends)
    tau=np.array([rnea_tree(pa,offsets,axes,coms,mass,inertia,q[k],dq[k],ddq[k])
                  for k in range(steps)])
    fric=(.12+.015*mass[None,:])*np.tanh(dq*18.)+.08*dq
    actual=tau+fric*cfg['private_friction_scale']
    mechanical=actual*dq
    efficiency=float(np.clip(cfg['motor_nominal_eff']*cfg['private_motor_efficiency'],.6,.98))
    copper=.020*actual**2*cfg['private_driver_loss']
    idle=(2.+.35*mass[None,:])*cfg['private_driver_loss']
    regen=cfg['regen_capability']*cfg['private_regen_scale']
    electric=np.maximum(mechanical,0)/efficiency+np.minimum(mechanical,0)*regen+copper+idle
    bus=np.maximum(electric.sum(axis=1),0)
    dt=T/(steps-1)
    joints_Wh=float(np.trapezoid(bus,dx=dt)/3600)
    controller_W=cfg['electronic_power_W']+16*n
    controller_Wh=float(controller_W*T/3600)
    # Nameplate-only references; does not use hidden motor factors.
    nom=tau+fric; nom_mech=nom*dq
    nom_el=np.maximum(nom_mech,0)/cfg['motor_nominal_eff']+np.minimum(nom_mech,0)*cfg['regen_capability']+.020*nom**2+2.+.35*mass[None,:]
    nom_joints=float(np.trapezoid(np.maximum(nom_el.sum(axis=1),0),dx=dt)/3600)
    return {'robot_joint_drives_Wh':joints_Wh,'robot_controller_Wh':controller_Wh,
            'nominal_robot_joint_drives_Wh':nom_joints,
            'nominal_robot_controller_Wh':controller_Wh,
            'mechanical_positive_J':float(np.trapezoid(np.maximum(mechanical,0).sum(axis=1),dx=dt)),
            'mechanical_regen_J':float(np.trapezoid(np.maximum(-mechanical,0).sum(axis=1),dx=dt))}

def welding_setting(rng, material, thickness_mm):
    # Paired parameter regimes, not an OEM WPS. Steel/SS/aluminum have separate
    # consumable density and approximate application bands.
    dia=float(rng.choice([.8,1.0,1.2,1.6],p=[.16,.36,.37,.11]))
    mode='short_circuit' if dia<=1.0 and thickness_mm<=5 else rng.choice(['short_circuit','spray'],p=[.25,.75])
    feed=float(rng.uniform(3.5,9.5)) # m/min
    amps=float(np.clip(32.+18.*feed*(dia**1.85)+(15. if mode=='spray' else 0.),75.,315.))
    volts=float(np.clip(16.+.040*amps+rng.normal(0,.7)+(1.0 if mode=='spray' else 0.),18.,33.))
    gas_mix='argon_co2' if material!='aluminum' else 'argon'
    flow=float(rng.uniform(11.,15.) if material=='aluminum' else rng.uniform(10.,16.))
    travel=float(rng.uniform(4.2,12.5)) # mm/sec
    thermal_eff=float(rng.uniform(.76,.84)) # 0.8 typical GMAW/TWI
    trans_eff=float(rng.uniform(.91,.98))
    return {'wire_diameter_mm':dia,'wire_feed_m_min':feed,'current_A':amps,'voltage_V':volts,
            'travel_speed_mm_s':travel,'gas_flow_L_min':flow,'gas_type':gas_mix,
            'arc_thermal_efficiency':thermal_eff,'transfer_efficiency':trans_eff,
            'transfer_mode':mode}

def historical_power_probes(cfg,source,rng):
    """Measurements from six BEFORE-target test loads. No target program observation.

    Watt probes include welding source loads and robot nameplate-like idle survey.
    Pair current with source observed grid W, allowing MCE to identify losses.
    """
    result=[]
    for amp in (0.,90.,130.,180.,230.,280.):
        volts=(0 if amp==0 else 17.5+.037*amp)
        src=((amp*volts)/source['private_source_efficiency'] if amp else 0)
        cooling=source['cooling_nominal_W']*source['private_cooling_scale']
        standby=source['source_standby_nominal_W']
        feeder=(source['wire_feeder_nominal_W']*.7*source['private_wire_drive_scale'] if amp else 0)
        p=src+cooling+standby+feeder
        result.append(float(max(0,p*(1+rng.normal(0,.005)))))
    return result

def _power_for_source(source,setting):
    weld_W=setting['voltage_V']*setting['current_A']/source['private_source_efficiency']
    feeder_W=source['wire_feeder_nominal_W']*source['private_wire_drive_scale']*(.5+setting['wire_feed_m_min']/11.)
    return weld_W,feeder_W



def external_trajectory(cfg,seg,base_tcp):
    """Transform robot TCP into the moving-workpiece reference frame.
    Rail translates robot base; positioner rotates workpiece, not robot links.
    """
    s=np.linspace(0,1,len(base_tcp));b=10*s**3-15*s**4+6*s**5
    m=seg.get('external_motion',{})
    rail=np.asarray(m.get('rail_m',[0.,0.]),float)
    angle=np.asarray(m.get('positioner_rad',[0.,0.]),float)
    tcp=np.array(base_tcp,float,copy=True)
    tcp[:,0]+=rail[0]+(rail[1]-rail[0])*b
    if np.max(np.abs(angle))>0:
        ph=angle[0]+(angle[1]-angle[0])*b
        # Transform torch positions into rotating part coordinates around fixture center.
        center=np.array(seg.get('positioner_center_m',[1.0,0.,.6]))
        rel=tcp-center[None,:];out=np.empty_like(rel)
        out[:,0]=np.cos(ph)*rel[:,0]+np.sin(ph)*rel[:,1]
        out[:,1]=-np.sin(ph)*rel[:,0]+np.cos(ph)*rel[:,1]
        out[:,2]=rel[:,2]
        tcp=out
    return tcp

def external_axis_energy(cfg,seg,T,steps):
    """Independent measured/nominal rail and positioner electric Wh.
    Power is from actuation of external mechanical loads, not a fictitious
    seventh or eighth robot arm joint. Includes torque-squared copper proxy.
    """
    m=seg.get('external_motion',{});cell=cfg.get('external_cell',{})
    frac=np.linspace(0,1,steps)
    bdot=(30*frac**2-60*frac**3+30*frac**4)/T
    bdd=(60*frac-180*frac**2+120*frac**3)/(T*T)
    total=0.;nom=0.
    hidden=cfg.get('private_external_eff',1.)
    loss=cfg.get('private_external_loss',1.)
    for kind,param in [('rail','rail_m'),('positioner','positioner_rad')]:
        if kind not in cell:continue
        dist=float(np.diff(m.get(param,[0.,0.]))[0]);vel=dist*bdot;acc=dist*bdd
        c=cell[kind];idle=float(c['idle_W']);eff=float(c['nominal_eff'])
        if kind=='rail':
            force=c['moving_mass_kg']*acc+c['friction_N']*np.tanh(vel*100)
            torque2=.0002*force**2
        else:
            force=c['inertia_kg_m2']*acc+c['friction_Nm']*np.tanh(vel*40)+c['gravity_Nm']*np.sin((float(m.get(param,[0,0])[0])+dist* (10*frac**3-15*frac**4+6*frac**5)))
            torque2=.012*force**2
        work=force*vel
        actual=np.maximum(work,0)/np.clip(eff*hidden,.55,.98)+np.minimum(work,0)*c['regen']+torque2*loss+idle*loss
        nominal=np.maximum(work,0)/eff+np.minimum(work,0)*c['regen']+torque2+idle
        total+=float(np.trapezoid(np.maximum(actual,0),dx=T/(steps-1))/3600)
        nom+=float(np.trapezoid(np.maximum(nominal,0),dx=T/(steps-1))/3600)
    return total,nom

def external_idle_W(cfg,nominal=False):
    return sum(float(c['idle_W'])*(1 if nominal else cfg.get('private_external_loss',1.)) for c in cfg.get('external_cell',{}).values())

def simulate_weld(cfg, source, program, steps=12, trace=False):
    """One robotic GMAW job, multiple seams. Segments own their arc state.

    `program`: {setting:..., segments:[{q0,q1,kind,tip_index,payload_kg,...}],
    material, shielding_preflow_s, shielding_postflow_s, gap_s}
    """
    setting=program['setting']; material=program['material']; rho=DENSITY_G_MM3[material]
    components={key:0. for key in ('robot_joint_drives_Wh','robot_controller_Wh',
      'welding_source_Wh','wire_feeder_Wh','cooling_Wh','fume_extraction_Wh',
      'gas_valve_Wh','source_standby_Wh','external_axes_Wh')}
    nominal_energy=0.; total_s=0.;arc_s=0.;torch_length_mm=0.
    wire_consumed_g=0.;wire_deposited_g=0.;wire_spatter_g=0.;arc_heat_kJ=0.
    gas_open_s=0.;gas_consumed_L=0.;mech_pos_J=0.;mech_regen_J=0.
    weld_source_W,wire_W=_power_for_source(source,setting)
    coolant_W=source['cooling_nominal_W']*source['private_cooling_scale']
    fume_W=source['fume_extraction_nominal_W']*source['private_fume_scale']
    valve_W=source['gas_valve_nominal_W']*source['private_valve_scale']
    standby_W=source['source_standby_nominal_W']
    wire_area_mm2=math.pi*setting['wire_diameter_mm']**2/4.
    segment_rows=[]
    for si,seg in enumerate(program['segments']):
        q0=np.asarray(seg['q0'],float);q1=np.asarray(seg['q1'],float)
        qtraj,_,_=quintic_trajectory(q0,q1,1,steps)
        tip=int(seg['tip_index'])
        path=0.;
        if seg['kind']=='arc':
            tipxyz=external_trajectory(cfg,seg,torch_trajectory(cfg,qtraj,tip))
            path=float(np.sum(np.linalg.norm(np.diff(tipxyz,axis=0),axis=1))*1000)
            if path<8:raise ValueError('weld segment path too short (<8 mm)')
            # Torch feed speed is physical tip path length / arc-on time.
            T=path/setting['travel_speed_mm_s']
        else:
            T=duration_for_joint_motion(cfg,q0,q1,seg.get('speed_factor',.72))
        ext=seg.get('external_motion',{})
        T=max(T,1.875*abs(np.diff(ext.get('rail_m',[0.,0.]))[0])/.18,1.875*abs(np.diff(ext.get('positioner_rad',[0.,0.]))[0])/.26)
        external_Wh,external_nom_Wh=external_axis_energy(cfg,seg,T,steps)
        components['external_axes_Wh']+=external_Wh
        nominal_energy+=external_nom_Wh
        robot=timed_robot_motion(cfg,q0,q1,T,seg.get('payload_kg',1.2),steps)
        for key in ('robot_joint_drives_Wh','robot_controller_Wh'):components[key]+=robot[key]
        nominal_energy+=robot['nominal_robot_joint_drives_Wh']+robot['nominal_robot_controller_Wh']
        mech_pos_J+=robot['mechanical_positive_J'];mech_regen_J+=robot['mechanical_regen_J']
        total_s+=T
        # Standby source on in all robot phases; active arc draws *additional* main circuit power.
        components['source_standby_Wh']+=standby_W*T/3600
        components['cooling_Wh']+=coolant_W*T/3600
        components['fume_extraction_Wh']+=fume_W*T/3600
        nominal_energy+=(standby_W+source['cooling_nominal_W']+source['fume_extraction_nominal_W'])*T/3600
        if seg['kind']=='arc':
            arc_s+=T; torch_length_mm+=path
            components['welding_source_Wh']+=weld_source_W*T/3600
            components['wire_feeder_Wh']+=wire_W*T/3600
            nominal_energy+=(setting['voltage_V']*setting['current_A']/source['source_nominal_efficiency']+source['wire_feeder_nominal_W']*(.5+setting['wire_feed_m_min']/11.))*T/3600
            feed_wire_mm=setting['wire_feed_m_min']*1000*T/60
            consumed=wire_area_mm2*feed_wire_mm*rho
            wire_consumed_g+=consumed
            wire_deposited_g+=consumed*setting['transfer_efficiency']
            wire_spatter_g+=consumed*(1-setting['transfer_efficiency'])
            arc_heat_kJ+=setting['arc_thermal_efficiency']*setting['voltage_V']*setting['current_A']*T/1000
        segment_rows.append({'segment_index':si,'kind':seg['kind'],'duration_s':T,'torch_path_mm':path,
                             'robot_energy_Wh':robot['robot_joint_drives_Wh']+robot['robot_controller_Wh']})
    # Shielding flows during each arc; pre/post once for entire job; gas valve electrical draw.
    pre=float(program['shielding_preflow_s']);post=float(program['shielding_postflow_s'])
    gas_open_s=arc_s+pre+post
    gas_consumed_L=gas_open_s*setting['gas_flow_L_min']/60
    # Pre/post-flows: no robot moves; controller, source standby, cooling and extraction remain on.
    idle_T=pre+post
    hold_joint_W=sum((2+.35*float(m))*cfg['private_driver_loss'] for m in cfg['mass'])
    hold_ctrl=cfg['electronic_power_W']+16*cfg['n']
    components['robot_joint_drives_Wh']+=hold_joint_W*idle_T/3600
    components['external_axes_Wh']+=external_idle_W(cfg)*idle_T/3600
    nominal_energy+=external_idle_W(cfg,nominal=True)*idle_T/3600
    components['robot_controller_Wh']+=hold_ctrl*idle_T/3600
    components['source_standby_Wh']+=standby_W*idle_T/3600
    components['cooling_Wh']+=coolant_W*idle_T/3600
    components['fume_extraction_Wh']+=fume_W*idle_T/3600
    components['gas_valve_Wh']+=valve_W*gas_open_s/3600
    nominal_energy+=(hold_ctrl+sum((2+.35*float(m)) for m in cfg['mass'])+standby_W+source['cooling_nominal_W']+source['fume_extraction_nominal_W'])*idle_T/3600
    nominal_energy+=source['gas_valve_nominal_W']*gas_open_s/3600
    total_s+=idle_T
    # Cooling tail after arc extinguishes, independently sampled and surfaced as an input.
    cool_tail=float(program['cooling_tail_s']);total_s+=cool_tail
    components['cooling_Wh']+=coolant_W*cool_tail/3600
    components['external_axes_Wh']+=external_idle_W(cfg)*cool_tail/3600
    nominal_energy+=external_idle_W(cfg,nominal=True)*cool_tail/3600
    components['fume_extraction_Wh']+=fume_W*cool_tail/3600
    components['robot_controller_Wh']+=(cfg['electronic_power_W']+16*cfg['n'])*cool_tail/3600
    components['source_standby_Wh']+=standby_W*cool_tail/3600
    nominal_energy+=(source['cooling_nominal_W']+source['fume_extraction_nominal_W']+cfg['electronic_power_W']+16*cfg['n']+standby_W)*cool_tail/3600
    total_energy=float(sum(components.values()))
    result={'time_s':float(total_s),'energy_Wh':total_energy,'nominal_energy_Wh':float(nominal_energy),
        'arc_on_s':float(arc_s),'arc_length_mm':float(torch_length_mm),'gas_on_s':float(gas_open_s),
        'gas_consumed_L':float(gas_consumed_L),'wire_consumed_g':float(wire_consumed_g),
        'wire_deposited_g':float(wire_deposited_g),'spatter_g':float(wire_spatter_g),
        'arc_energy_kJ':float(setting['current_A']*setting['voltage_V']*arc_s/1000),
        'heat_to_workpiece_kJ':float(arc_heat_kJ),'heat_input_kJ_mm':float(arc_heat_kJ/max(torch_length_mm,1e-9)),
        'material_removed_g':0.,'material_added_g':float(wire_deposited_g),
        'positive_mechanical_J':mech_pos_J,'regenerative_mechanical_J':mech_regen_J,
        'energy_components_Wh':components,
        'segment_summary':segment_rows}
    return result
