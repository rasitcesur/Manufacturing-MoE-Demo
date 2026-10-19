import numpy as np
JOINT_DIM=15
NF=13
def public_features(cfg,mo,sim):
    n=cfg['n'];j=np.zeros((JOINT_DIM,NF),np.float32);mask=np.zeros(JOINT_DIM,np.float32)
    qs=np.asarray(mo['q_start']);qe=np.asarray(mo['q_end']);lengths=np.asarray(cfg['coms'])[:,0]*2
    for i in range(n):
        # Known nominal physical features; NO true energy, actual duration, private efficiency.
        j[i]=[np.log1p(sim['torque_rms_Nm'][i]),np.log1p(sim['torque_max_Nm'][i]),
            sim['velocity_rms_rad_s'][i],sim['mean_abs_velocity_rad_s'][i],
            np.log1p(sim['torque_work_J'][i]),np.log1p(sim['torque_rms_sq_Nm2'][i]),
            np.log1p(cfg['mass'][i]),lengths[i], abs(qe[i]-qs[i]),
            cfg['parents'][i]/15.,float(cfg['arm_id'][i]==1),float(cfg['arm_id'][i]==2),
            float(cfg['axes'][i][2])]
    mask[:n]=1.
    g=np.asarray([n/15.,float(cfg['topology']=='dual'),float(cfg['topology']=='torso_dual'),
        np.log1p(mo['payload_kg']),mo['speed_factor'],mo['dwell_s'],
        np.log1p(sim['nominal_time_s']),np.log1p(sim['nominal_energy_Wh']),
        np.log1p(sim['nominal_positive_J']),np.log1p(sim['nominal_regen_J']),
        np.log1p(sim['nominal_copper_J']),np.log1p(sim['nominal_idle_J']),
        np.log1p(sim['nominal_aux_J'])],np.float32)
    return j,mask,g

