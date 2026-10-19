"""Job-level energy features (10 independently supervised subsystems).

Inputs: complete G-code, machine profile, INFERENCE-PREDICTED command durations.
Outputs: one feature vector and one physically integrated baseline Wh per device/JOB.
No teacher duration or measured energy enters feature extraction. The physical
integration uses command chronology to obtain duty/regime exposure, but does
NOT make any neural energy prediction per G-code line.

All feature engineering is based on known G-code, machine metadata, estimated
motion time, and simulated (not observed) temperatures/duty. NOT calibrated.
"""
import json
import numpy as np
from physics import parse
from component_power import COMPONENTS,DEFAULT_POWER,simulate_components,MOTION

FEAT_NAMES={
 'electronics':['baseline_Wh','total_s','nominal_W','n_commands'],
 'stepper_hold':['baseline_Wh','total_s','hold_active_s_est','nominal_W','n_commands'],
 'motor_x':['baseline_Wh','motion_s','axis_mm_est','axis_mean_v','axis_rms_v','axis_peak_v','nominal_W','n_moves','total_s','accel_travel'],
 'motor_y':['baseline_Wh','motion_s','axis_mm_est','axis_mean_v','axis_rms_v','axis_peak_v','nominal_W','n_moves','total_s','accel_travel'],
 'motor_z':['baseline_Wh','motion_s','axis_mm_est','axis_mean_v','axis_rms_v','axis_peak_v','nominal_W','n_moves','total_s','accel_print'],
 'motor_e':['baseline_Wh','motion_s','axis_mm_est','axis_mean_v','axis_rms_v','axis_peak_v','nominal_W','n_extrusion_moves','total_s','accel_print'],
 'heater_nozzle':['baseline_Wh','total_s','nozzle_heating_s_est','target_temp_s','setpoint_mean_C','start_C','ambient_C','nominal_W','thermal_capacity','thermal_loss','fan_level_s','n_target_changes','extrusion_mm'],
 'heater_bed':['baseline_Wh','total_s','bed_heating_s_est','target_temp_s','setpoint_mean_C','start_C','ambient_C','nominal_W','thermal_capacity','thermal_loss','n_target_changes'],
 'fan_part':['baseline_Wh','total_s','fan_pwm_time_s','fan_active_s','nominal_W','fan_transition_n'],
 'fan_hotend':['baseline_Wh','total_s','hotend_on_time_s_est','nominal_W','nozzle_target_time_Cs','nozzle_start_C'],
}

# Feature magnitudes are log1p transformed, except unscaled temperatures may be
# zero or negative. Train-only scalers are used afterward.
def summary(rows,durations):
    t=np.asarray(durations,dtype=np.float64).reshape(-1)
    if len(t)!=len(rows) or len(rows)==0 or not np.all(np.isfinite(t)) or np.any(t<0):
        raise ValueError('Invalid sequence durations')
    profile=json.loads(rows[0]['initial_profile']) if isinstance(rows[0]['initial_profile'],str) else rows[0]['initial_profile']
    hw={**DEFAULT_POWER,**profile.get('power_hardware',{})}
    energy_line,power_line,vel=simulate_components(rows,t)
    base=np.asarray(energy_line,dtype=np.float64).sum(axis=0)
    assert base.shape==(len(COMPONENTS),)
    commands=[parse(r['gcode']) for r in rows]
    move=np.asarray([c in MOTION for c,_ in commands])
    nmove=int(move.sum());total=float(t.sum());move_s=float(t[move].sum())
    axis_len=np.sum(vel*t[:,None],axis=0)
    axis_mean=np.sum(vel*t[:,None],axis=0)/max(1.e-8,move_s)
    axis_rms=np.sqrt(np.sum(vel**2*t[:,None],axis=0)/max(1.e-8,move_s))
    axis_max=np.max(vel,axis=0)
    # G-code state-history (setpoint & PWM); no labels or sensor truth involved.
    noz_sp=float(profile.get('nozzle_start',profile.get('ambient',23)))
    bed_sp=float(profile.get('bed_start',profile.get('ambient',23)))
    fan=float(profile.get('fan_start',0.))
    fan_pwm_s=fan_s=noz_sp_time=bed_sp_time=0.
    n_noz=n_bed=n_fan=0
    noz_heating_s=bed_heating_s=hotend_on_s=0.
    for (c,a),dt,wh in zip(commands,t,energy_line):
        if c in ('M104','M109'):
            nxt=float(a.get('S',a.get('R',noz_sp)))
            n_noz+=int(abs(nxt-noz_sp)>1e-8);noz_sp=nxt
        if c in ('M140','M190'):
            nxt=float(a.get('S',a.get('R',bed_sp)))
            n_bed+=int(abs(nxt-bed_sp)>1e-8);bed_sp=nxt
        if c=='M106':
            nxt=max(0.,min(255.,a.get('S',255.)));n_fan+=int(abs(fan-nxt)>1e-8);fan=nxt
        if c=='M107':
            n_fan+=int(fan>0);fan=0.
        noz_sp_time+=noz_sp*dt
        bed_sp_time+=bed_sp*dt
        fan_pwm_s+=dt*fan/255.
        fan_s+=dt*(fan>0)
        # Estimated energy exposure, from reference physics, not measured label.
        noz_heating_s+=float(wh[COMPONENTS.index('heater_nozzle')])*3600/max(hw['nozzle_w'],1.e-7)
        bed_heating_s+=float(wh[COMPONENTS.index('heater_bed')])*3600/max(hw['bed_w'],1.e-7)
        hotend_on_s+=float(wh[COMPONENTS.index('fan_hotend')])*3600/max(hw['hotend_fan_w'],1.e-7)
    hold_s=base[1]*3600/max(hw['stepper_hold_w'],1.e-7)
    shared={}
    shared['electronics']=[base[0],total,hw['electronics_w'],len(t)]
    shared['stepper_hold']=[base[1],total,hold_s,hw['stepper_hold_w'],len(t)]
    for i,k in enumerate(['motor_x','motor_y','motor_z','motor_e']):
        kk=('x','y','z','e')[i]
        nominal=hw[f'{kk}_motor_w_at_'+('60' if i<2 else '5')]
        shared[k]=[base[2+i],move_s,axis_len[i],axis_mean[i],axis_rms[i],axis_max[i],nominal,
                   int((vel[:,i]>0).sum()) if i==3 else nmove,total,
                   float(profile.get('accel_travel' if i<2 else 'accel_print',500))]
    shared['heater_nozzle']=[base[6],total,noz_heating_s,noz_sp_time,
          noz_sp_time/max(total,1.e-8),float(profile['nozzle_start']),float(profile['ambient']),hw['nozzle_w'],
          hw['nozzle_cap_j_c'],hw['nozzle_loss_w_c'],fan_pwm_s,n_noz,axis_len[3]]
    shared['heater_bed']=[base[7],total,bed_heating_s,bed_sp_time,
          bed_sp_time/max(total,1.e-8),float(profile['bed_start']),float(profile['ambient']),hw['bed_w'],
          hw['bed_cap_j_c'],hw['bed_loss_w_c'],n_bed]
    shared['fan_part']=[base[8],total,fan_pwm_s,fan_s,hw['part_fan_w'],n_fan]
    shared['fan_hotend']=[base[9],total,hotend_on_s,hw['hotend_fan_w'],noz_sp_time,float(profile['nozzle_start'])]
    features={k:np.asarray(shared[k],dtype=np.float32) for k in COMPONENTS}
    for k in COMPONENTS:
        if len(features[k])!=len(FEAT_NAMES[k]):raise AssertionError(k)
        if not np.all(np.isfinite(features[k])):raise ValueError('Nonfinite feature '+k)
    return {'x':features,'base':base.astype('float32'),'time_s':total,
            'n_moves':nmove,
            'regimes':{'motion_time_s':move_s,'heater_nozzle_s_est':noz_heating_s,'heater_bed_s_est':bed_heating_s,
                'fan_pwm_equiv_s':fan_pwm_s,'axis_distance_mm_est':axis_len.tolist()}}


def fit_scalers(examples):
    return {k:(np.stack([e['summary']['x'][k] for e in examples]).mean(0).astype('float32'),
               np.maximum(np.stack([e['summary']['x'][k] for e in examples]).std(0),.001).astype('float32'))
            for k in COMPONENTS}

def normalized(x,scalers):
    return {k:np.clip((x[k]-scalers[k][0])/scalers[k][1],-8,8).astype('float32') for k in COMPONENTS}
