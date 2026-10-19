import numpy as np
def features(cfg,source,prog,result,history):
    n=cfg['n']
    jf=np.zeros((15,15),np.float32)
    parents=np.asarray(cfg['parents']);q0=np.asarray(prog['segments'][0]['q0'],float);qf=np.asarray(prog['segments'][-1]['q1'],float)
    arm=np.asarray(cfg['arm_id'])
    # Known nameplate geometry and planned trajectories only. No teacher labels.
    for i in range(n):
        jf[i]=[float(cfg['mass'][i])/15.,float(cfg['coms'][i][0])*2.,
           float(cfg['inertia'][i][0]),float(cfg['inertia'][i][1]),float(cfg['inertia'][i][2]),
           float(cfg['axes'][i][0]),float(cfg['axes'][i][1]),float(cfg['axes'][i][2]),
           float(parents[i])/15.,float(arm[i]==0),float(arm[i]==1),float(arm[i]==2),
           float(q0[i]),float(qf[i]),float(abs(qf[i]-q0[i]))]
    wf=prog['setting'];arc_count=sum(s['kind']=='arc' for s in prog['segments'])
    g=np.asarray([n/15.,float(cfg['topology']=='dual'),float(cfg['topology']=='torso_dual'),
        float(prog['thickness_mm']),float(prog['material']=='stainless'),float(prog['material']=='aluminum'),
        float(wf['wire_diameter_mm']),float(wf['wire_feed_m_min']),float(wf['current_A']),
        float(wf['voltage_V']),float(wf['travel_speed_mm_s']),float(wf['gas_flow_L_min']),
        float(source['cooling_nominal_W']),float(source['fume_extraction_nominal_W']),
        float(wf['transfer_mode']=='spray'),float(prog['shielding_preflow_s']),
        float(prog['shielding_postflow_s']),float(prog['cooling_tail_s']),float(arc_count),
        float(cfg['motor_nominal_eff']),float(cfg['regen_capability']),
        float(source['source_nominal_efficiency']),float(source['source_nameplate_W']),
        float(np.mean(cfg['mass'])),float(np.sum(cfg['mass']))],np.float32)
    return jf,g
