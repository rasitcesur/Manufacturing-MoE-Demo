import numpy as np
def feature(cfg,src,program):
    n=cfg['n'];j=np.zeros((15,15),np.float32)
    q0=np.asarray(program['segments'][0]['q0']);qf=np.asarray(program['segments'][-1]['q1'])
    for i in range(n):
        j[i]=[float(cfg['mass'][i])/15,float(cfg['coms'][i][0])*2,*map(float,cfg['inertia'][i]),
              *map(float,cfg['axes'][i]),float(cfg['parents'][i])/15,
              float(cfg['arm_id'][i]==0),float(cfg['arm_id'][i]==1),float(cfg['arm_id'][i]==2),
              float(q0[i]),float(qf[i]),float(abs(qf[i]-q0[i]))]
    s=program['setting'];ext=cfg.get('external_cell',{})
    rail=ext.get('rail',{});pos=ext.get('positioner',{})
    g=np.array([n/15,float(cfg['topology']=='dual'),float(program['material']=='aluminum'),float(program['material']=='stainless'),
      float(program['thickness_mm']),s['current_A'],s.get('voltage_V',0.),s['electrode_force_N'],s['contact_resistance_nominal_ohm'],
      s['squeeze_s'],s['pulse_s'],s['hold_s'],s['pulses'],s['inter_pulse_s'],s['pulse_duty'],
      s['filler_diameter_mm'],s['filler_feed_m_min'],s['gas_flow_L_min'],s['travel_speed_mm_s'],s['ac_mode'],
      program['shielding_preflow_s'],program['shielding_postflow_s'],program['cooling_tail_s'],
      src['source_nominal_efficiency'],src['source_nameplate_W'],src['process_actuator_nominal_W'],src['cooling_nominal_W'],src['fume_extraction_nominal_W'],
      cfg['motor_nominal_eff'],cfg['regen_capability'],float(np.sum(cfg['mass'])),float(bool(rail)),float(bool(pos)),
      rail.get('moving_mass_kg',0.),pos.get('inertia_kg_m2',0.),
      float(sum(seg['kind']!='travel' for seg in program['segments'])),
      float(sum(abs(seg['external_motion']['rail_m'][1]-seg['external_motion']['rail_m'][0]) for seg in program['segments'])),
      float(sum(abs(seg['external_motion']['positioner_rad'][1]-seg['external_motion']['positioner_rad'][0]) for seg in program['segments']))],np.float32)
    return j,g

def nominal(cfg,src):
    cc={k:v for k,v in cfg.items()}
    for k in ('private_friction_scale','private_motor_efficiency','private_driver_loss','private_regen_scale','private_external_eff','private_external_loss'):cc[k]=1.
    ss={**src,'private_source_efficiency':src['source_nominal_efficiency'],'private_actuator_scale':1.,'private_cooling_scale':1.,'private_fume_scale':1.,'private_valve_scale':1.,'private_contact_scale':1.,'private_transfer_eff':.98}
    return cc,ss

def calibrations(src,process,hist):
    # Measured historical watt probe ratio to public nameplate load; 0-load
    # source consumption is subtracted, avoiding target energy leakage.
    current=np.array([0.,.25,.40,.55,.72,.90]);
    expected=(12000*current)**2*.000110 if process=='rsw' else (180*current)*12.5
    # nominal is physical source conversion only; process electrode contact effects latent.
    ratio=np.median((hist[1:]-hist[0])/np.maximum(expected[1:]/src['source_nominal_efficiency'],1))
    return float(np.clip(ratio,.5,2.0))
