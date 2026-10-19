"""Manufacturing-MoE shared robot-cell inverse dynamics and energy accounting.
Not a qualified RSW/TIG WPS or thermal-metallurgical solver.
"""
from __future__ import annotations
import sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
from arc_welding_physics import timed_robot_motion,duration_for_joint_motion,external_axis_energy,external_idle_W,external_trajectory,quintic_trajectory,torch_trajectory

COMPONENTS=('robot_joint_drives_Wh','robot_controller_Wh','process_power_source_Wh','process_actuator_Wh','cooling_Wh','fume_extraction_Wh','gas_valve_Wh','source_standby_Wh','external_axes_Wh')

def machine_source(rng,process,ood=False):
    spot=process=='rsw'
    return dict(source_nominal_efficiency=.84 if spot else .87,
        source_nameplate_W=120000. if spot else 9000.,
        process_actuator_nominal_W=420. if spot else 34.,
        cooling_nominal_W=420. if spot else 170.,
        fume_extraction_nominal_W=90. if spot else 120.,
        gas_valve_nominal_W=0. if spot else 8.,
        source_standby_nominal_W=85. if spot else 64.,
        private_source_efficiency=float(rng.uniform(.67,.76) if ood else rng.uniform(.80,.91)),
        private_actuator_scale=float(rng.uniform(1.24,1.50) if ood else rng.uniform(.83,1.17)),
        private_cooling_scale=float(rng.uniform(1.19,1.46) if ood else rng.uniform(.80,1.20)),
        private_fume_scale=float(rng.uniform(1.16,1.43) if ood else rng.uniform(.83,1.17)),
        private_valve_scale=float(rng.uniform(.85,1.18)),
        private_contact_scale=float(rng.uniform(1.18,1.40) if ood else rng.uniform(.86,1.14)),
        private_transfer_eff=float(rng.uniform(.91,.97) if ood else rng.uniform(.965,.994)))

def public_source(src):return {k:v for k,v in src.items() if not k.startswith('private_')}

def current_setting(rng,process,material,thickness):
    if process=='rsw':
        amp=float(rng.uniform(5500,13000) if material!='aluminum' else rng.uniform(18000,27500))
        force=float(rng.uniform(1500,5600));res=float(rng.uniform(.000055,.00017))
        return dict(current_A=amp,electrode_force_N=force,contact_resistance_nominal_ohm=res,
         squeeze_s=float(rng.uniform(.12,.40)),pulse_s=float(rng.uniform(.08,.34)),
         hold_s=float(rng.uniform(.10,.34)),pulses=int(rng.integers(1,4)),
         inter_pulse_s=float(rng.uniform(.025,.075)),gas_flow_L_min=0.,filler_diameter_mm=0.,
         filler_feed_m_min=0.,travel_speed_mm_s=0.,ac_mode=0.,pulse_duty=float(rng.uniform(.55,.95)))
    ac=int(material=='aluminum' and rng.random()<.80)
    amp=float(rng.uniform(70,190));volts=float(rng.uniform(9,16))
    filler=int(rng.random()<.70);dia=float(rng.choice([1.0,1.2,1.6,2.0])) if filler else 0.
    feed=float(rng.uniform(.35,1.5)) if filler else 0.
    return dict(current_A=amp,voltage_V=volts,gas_flow_L_min=float(rng.uniform(8,17)),
        filler_diameter_mm=dia,filler_feed_m_min=feed,travel_speed_mm_s=float(rng.uniform(1.7,5.4)),
        ac_mode=ac,pulse_duty=float(rng.uniform(.55,.95)),squeeze_s=0.,pulse_s=0.,hold_s=0.,pulses=0,
        inter_pulse_s=0.,electrode_force_N=0.,contact_resistance_nominal_ohm=0.)

def probes(source,process,rng):
    # SIX prior independent load tests, not the predicted welding job.
    levels=np.array([0.,.25,.40,.55,.72,.90]);vals=[]
    for x in levels:
        if process=='rsw':
            I=12000*x
            # prior gauge: circuit equivalent resistance under a standardized electrode force
            nominal_power=(I*I*.000110)
            actual_power=nominal_power*source['private_contact_scale']/source['private_source_efficiency']
        else:
            nominal_power=(180*x)*(12.5)
            actual_power=nominal_power/source['private_source_efficiency']
        p=source['source_standby_nominal_W']+source['cooling_nominal_W']*source['private_cooling_scale']+actual_power
        vals.append(float(p*(1+rng.normal(0,.002))))
    return np.asarray(vals,np.float32)

def robot_segment(cfg,seg,T):
    r=timed_robot_motion(cfg,np.asarray(seg['q0']),np.asarray(seg['q1']),T,float(seg.get('payload_kg',1.7)),steps=12)
    e,nom=external_axis_energy(cfg,seg,T,12)
    return r,e,nom

def totals_common(cfg,source,program,process):
    comp={k:0. for k in COMPONENTS}
    nom={k:0. for k in COMPONENTS}
    source_w=source['source_standby_nominal_W'];cool=source['cooling_nominal_W'];fume=source['fume_extraction_nominal_W']
    ctrl_w=cfg['electronic_power_W']+16*cfg['n']
    total_t=active_t=0.;arc_heat=0.;deposit=0.;consumed=0.;gas=0.;spots=0
    spot_detail=[]
    for seg in program['segments']:
        q0=np.asarray(seg['q0']);q1=np.asarray(seg['q1'])
        motion_T=duration_for_joint_motion(cfg,q0,q1,.7)
        if seg['kind']=='travel' or process=='rsw':T=motion_T
        else:
            qseq,_,_=quintic_trajectory(q0,q1,1.,12)
            tcp=external_trajectory(cfg,seg,torch_trajectory(cfg,qseq,seg['tip_index']))
            path_mm=float(np.linalg.norm(np.diff(tcp,axis=0),axis=1).sum()*1000)
            T=max(motion_T,path_mm/max(program['setting']['travel_speed_mm_s'],.001))
        r,e,enom=robot_segment(cfg,seg,T)
        comp['robot_joint_drives_Wh']+=r['robot_joint_drives_Wh'];nom['robot_joint_drives_Wh']+=r['nominal_robot_joint_drives_Wh']
        comp['robot_controller_Wh']+=r['robot_controller_Wh'];nom['robot_controller_Wh']+=r['robot_controller_Wh']
        comp['external_axes_Wh']+=e;nom['external_axes_Wh']+=enom
        t=T
        if seg['kind'] in ('arc','spot'):
            if process=='rsw':
                s=program['setting'];nd=s['pulses'];tw=s['pulse_s'];wait=s['inter_pulse_s']*(nd-1)
                t_hold=s['squeeze_s']+nd*tw+wait+s['hold_s']
                t+=t_hold
                # dynamic resistance proxy: electrode force contact compression, interface heating and cooling.
                timegrid=np.linspace(0,1,12)
                resistance=s['contact_resistance_nominal_ohm']*(.7+.3*np.exp(-1.2*timegrid))*np.power(2800/s['electrode_force_N'],.18)
                resistance=np.clip(resistance,1e-6,.001)
                i_rms=s['current_A']*math.sqrt(s['pulse_duty'])
                Q_nom=i_rms**2*float(resistance.mean())*nd*tw
                Q_actual=Q_nom*source['private_contact_scale']
                comp['process_power_source_Wh']+=Q_actual/source['private_source_efficiency']/3600
                nom['process_power_source_Wh']+=Q_nom/source['source_nominal_efficiency']/3600
                heat=Q_actual*.66/1000;arc_heat+=heat
                spots+=1;active_t+=nd*tw
                nominal_act=source['process_actuator_nominal_W']*(.35+s['electrode_force_N']/6500)
                comp['process_actuator_Wh']+=nominal_act*source['private_actuator_scale']*t_hold/3600
                nom['process_actuator_Wh']+=nominal_act*t_hold/3600
                spot_detail.append(dict(heat_kJ=heat,electrode_force_N=s['electrode_force_N'],contact_resistance_ohm=float(resistance.mean())))
            else:
                s=program['setting'];active_t+=T
                arc_W=s['current_A']*s['voltage_V']*(s['pulse_duty']*(1.08 if s['ac_mode'] else 1.0)+(1-s['pulse_duty'])*.68)
                nom['process_power_source_Wh']+=arc_W/source['source_nominal_efficiency']*T/3600
                comp['process_power_source_Wh']+=arc_W/source['private_source_efficiency']*T/3600
                arc_heat+=arc_W*T*(.66 if s['ac_mode'] else .73)/1000
                if s['filler_diameter_mm']>0:
                    rho=.00270 if program['material']=='aluminum' else .0079
                    consumed_t=math.pi*s['filler_diameter_mm']**2/4*(s['filler_feed_m_min']*1000*T/60)*rho
                    consumed+=consumed_t;deposit+=consumed_t*source['private_transfer_eff']
                    act=source['process_actuator_nominal_W']*(.2+s['filler_feed_m_min'])
                    nom['process_actuator_Wh']+=act*T/3600
                    comp['process_actuator_Wh']+=act*source['private_actuator_scale']*T/3600
        # Source standby, controller/servo holding and auxiliaries across ALL phases.
        nom['source_standby_Wh']+=source_w*t/3600;comp['source_standby_Wh']+=source_w*t/3600
        nom['cooling_Wh']+=cool*t/3600;comp['cooling_Wh']+=cool*source['private_cooling_scale']*t/3600
        nom['fume_extraction_Wh']+=fume*t/3600;comp['fume_extraction_Wh']+=fume*source['private_fume_scale']*t/3600
        extra=t-T
        if extra>0:
            nom['robot_controller_Wh']+=ctrl_w*extra/3600;comp['robot_controller_Wh']+=ctrl_w*extra/3600
            hold=float(np.sum((2+.35*np.asarray(cfg['mass']))))
            nom['robot_joint_drives_Wh']+=hold*extra/3600
            comp['robot_joint_drives_Wh']+=hold*cfg['private_driver_loss']*extra/3600
            en=external_idle_W(cfg,nominal=False);en0=external_idle_W(cfg,nominal=True)
            comp['external_axes_Wh']+=en*extra/3600;nom['external_axes_Wh']+=en0*extra/3600
        total_t+=t
    s=program['setting']
    if process=='tig':
        gas_t=active_t+program['shielding_preflow_s']+program['shielding_postflow_s']
        gas=gas_t*s['gas_flow_L_min']/60
        nom['gas_valve_Wh']+=source['gas_valve_nominal_W']*gas_t/3600
        comp['gas_valve_Wh']+=source['gas_valve_nominal_W']*source['private_valve_scale']*gas_t/3600
        tail=program['shielding_preflow_s']+program['shielding_postflow_s']+program['cooling_tail_s']
    else:tail=program['cooling_tail_s']
    total_t+=tail
    for k,wat in [('robot_controller_Wh',ctrl_w),('source_standby_Wh',source_w),('cooling_Wh',cool),('fume_extraction_Wh',fume)]:
        nom[k]+=wat*tail/3600
        mult={'cooling_Wh':source['private_cooling_scale'],'fume_extraction_Wh':source['private_fume_scale']}.get(k,1.)
        comp[k]+=wat*mult*tail/3600
    hold=sum(2+.35*float(m) for m in cfg['mass'])
    comp['robot_joint_drives_Wh']+=hold*cfg['private_driver_loss']*tail/3600
    nom['robot_joint_drives_Wh']+=hold*tail/3600
    nom['external_axes_Wh']+=external_idle_W(cfg,nominal=True)*tail/3600
    comp['external_axes_Wh']+=external_idle_W(cfg,nominal=False)*tail/3600
    return dict(time_s=total_t,energy_Wh=sum(comp.values()),material_added_g=deposit,wire_consumed_g=consumed,
        gas_consumed_L=gas,spot_count=spots,heat_to_workpiece_kJ=arc_heat,active_time_s=active_t,
        components=comp,nominal_components=nom,spot_detail=spot_detail)
