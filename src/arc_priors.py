"""Recompute *input-only* physics priors for GMAW from public robot and WPS profiles.
Private machine factors and target labels are never used as input. The simulator
supplies the nominal energy estimator, not training labels.
"""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from arc_welding_physics import simulate_weld

COMPONENTS=['robot_joint_drives_Wh','robot_controller_Wh','welding_source_Wh','wire_feeder_Wh','cooling_Wh','fume_extraction_Wh','gas_valve_Wh','source_standby_Wh','external_axes_Wh']

def calculate(split):
    base=ROOT/'data/gmaw'
    machines={d['robot_id']:d for d in json.load(open(base/f'{split}_machines.json'))}
    out=[];begin=time.time()
    with open(base/f'{split}_programs.jsonl') as f:
        for idx,line in enumerate(f):
            record=json.loads(line);m=machines[record['robot_id']]
            cfg=m['robot'].copy();source=m['welding_power_supply'].copy()
            for k in ('private_friction_scale','private_motor_efficiency','private_driver_loss','private_regen_scale','private_external_eff','private_external_loss'):cfg[k]=1.
            for k in ('private_source_efficiency','private_wire_drive_scale','private_cooling_scale','private_fume_scale','private_valve_scale'):
                source[k]=(source['source_nominal_efficiency'] if k=='private_source_efficiency' else 1.)
            program=record['program'].copy();program['setting']=dict(program['setting'],transfer_efficiency=.945,arc_thermal_efficiency=.8)
            r=simulate_weld(cfg,source,program,steps=12)
            arr=[r['time_s'],r['energy_Wh'],r['wire_consumed_g'],r['wire_deposited_g'],r['gas_consumed_L']]
            physics=[r['energy_components_Wh'][k] for k in COMPONENTS]
            extra=[r['arc_on_s'],r['arc_length_mm'],r['gas_on_s'],r['positive_mechanical_J'],r['regenerative_mechanical_J'],
                   len(program['segments']),r['nominal_energy_Wh']]
            out.append((arr,physics,extra,record['program_id']))
            if (idx+1)%250==0:print(split,idx+1,'elapsed_s',round(time.time()-begin,1),flush=True)
    with np.load(base/f'{split}.npz') as z:
        assert len(out)==len(z['y'])
        assert np.array_equal(np.array([v[3] for v in out]),z['program_ids'])
    prior=np.asarray([v[0] for v in out],np.float32)
    comp=np.asarray([v[1] for v in out],np.float32)
    extra=np.asarray([v[2] for v in out],np.float32)
    # Self-tests on deterministic labels (time/wire/gas independent of private factors).
    with np.load(base/f'{split}.npz') as z:
        true=z['y']
        for k in (0,2,4):
            mae=float(np.mean(abs(true[:,k]-prior[:,k])))
            assert np.allclose(true[:,k],prior[:,k],rtol=2e-5,atol=1e-4),(split,k,mae)
    assert np.allclose(prior[:,1],comp.sum(axis=1),rtol=2e-5,atol=1e-3)
    np.savez_compressed(base/f'{split}_input_priors.npz',prior=prior,component_prior=comp,physics_extra=extra)
    print('Completed',split,len(out),'s=',round(time.time()-begin,1),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','val','test','ood']);a=ap.parse_args()
    for split in ([a.split] if a.split else ['train','val','test','ood']):calculate(split)
