"""current inference for GMAW six/seven-axis arms with distinct rail/positioner actuators.
Prior robot/source/external-drive measurements are mandatory. No current target telemetry.
"""
import sys,json,argparse,importlib.util
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];SIM=ROOT/'runtime'
sys.path.insert(0,str(ROOT/'runtime'))
import arc_welding_physics
import arc_base_features as legacy
from arc_welding_physics import simulate_weld
from arc_gravity import gravity_context
from arc_welding_moe import PreparedArc54,ArcWeldingMoE54,predict
from arc_priors import COMPONENTS

def infer(bundle,checkpoint):
    machine=bundle['machine'];cfg=dict(machine['robot']);source=dict(machine['welding_power_supply'])
    obs=np.asarray(machine['historic_power_probes_W'],np.float32)
    if obs.shape!=(12,):raise ValueError('Twelve BEFORE-TARGET historical source/motor power probes required.')
    hist=bundle.get('previous_robot_motions',machine.get('previous_robot_motions'))
    if not isinstance(hist,list) or len(hist)!=6:raise ValueError('Six independent previous motion measurements required.')
    h=[]
    for m in hist:
        a=float(m['observed_robot_drive_Wh']);nom=float(m['nominal_robot_drive_Wh']);payload=float(m['payload_kg'])
        if a<=0 or nom<=0:raise ValueError('Previous measured/nominal energies must be positive')
        h.extend([np.log(a/nom),np.log1p(a),np.log1p(nom),np.log1p(payload)])
    ext_hist=machine.get('previous_external_drive_log_ratio')
    if not isinstance(ext_hist,list) or len(ext_hist)!=2:raise ValueError('External-axis previous idle power measurements required.')
    cfg.update({k:1. for k in ('private_friction_scale','private_motor_efficiency','private_driver_loss',
              'private_regen_scale','private_settle_s','private_external_eff','private_external_loss')})
    for k in ('private_source_efficiency','private_wire_drive_scale','private_cooling_scale','private_fume_scale','private_valve_scale'):
        source[k]=source['source_nominal_efficiency'] if k=='private_source_efficiency' else 1.
    prog=bundle['program'].get('program',bundle['program'])
    prog=dict(prog);prog['setting']=dict(prog['setting'],transfer_efficiency=.945,arc_thermal_efficiency=.8)
    nominal=simulate_weld(cfg,source,prog,steps=12)
    joint,g=legacy.features(cfg,source,prog,nominal,obs)
    cell=cfg.get('external_cell',{});rail=cell.get('rail',{});pos=cell.get('positioner',{})
    ext=np.array([float(bool(rail)),float(bool(pos)),float(cfg['topology']=='dual'),
      rail.get('moving_mass_kg',0.),rail.get('nominal_eff',0.),pos.get('inertia_kg_m2',0.),pos.get('nominal_eff',0.),
      sum(abs(np.diff(s.get('external_motion',{}).get('rail_m',[0,0]))[0]) for s in prog['segments']),
      sum(abs(np.diff(s.get('external_motion',{}).get('positioner_rad',[0,0]))[0]) for s in prog['segments']),
      *ext_hist],np.float32)
    g=np.concatenate([g,ext]);jgravity,ggravity=gravity_context(cfg,prog)
    prior=np.array([[nominal[k] for k in ('time_s','energy_Wh','wire_consumed_g','wire_deposited_g','gas_consumed_L')]],np.float32)
    comp=np.array([[nominal['energy_components_Wh'][k] for k in COMPONENTS]],np.float32)
    phy=np.array([[nominal[k] for k in ('arc_on_s','arc_length_mm','gas_on_s','positive_mechanical_J','regenerative_mechanical_J')]+[len(prog['segments']),nominal['nominal_energy_Wh']]],np.float32)
    data={'joint':joint[None], 'global_features':g[None], 'history_power_W':obs[None],
          'y':prior,'energy_components_Wh':comp,'mask':np.r_[np.ones(cfg['n']),np.zeros(15-cfg['n'])][None].astype(np.float32),
          'n_axes':np.array([cfg['n']]), 'topology':np.array([bundle['program'].get('topology','external')]),
          'robot_ids':np.array([machine.get('robot_id','external')]),'program_ids':np.array([bundle['program'].get('program_id','external')]),
          'industrial_scope':np.array([True])}
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if 'arc_welding' in ck:ck=ck['arc_welding']
    ds=PreparedArc54(data,{'prior':prior,'component_prior':comp,'physics_extra':phy},
      {'joint_gravity':jgravity[None],'global_gravity':ggravity[None]},
      {'robot_history':np.array(h,np.float32)[None,:]},center=ck['normalizer']['center'],scale=ck['normalizer']['scale'])
    model=ArcWeldingMoE54(**ck['model_config']);model.load_state_dict(ck['model'])
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    y,component=predict(model,ds)
    targets=('time_s','electricity_Wh','wire_consumed_g','wire_deposited_g','shielding_gas_L')
    return {'operation':'robotic_GMAW_welding','model_family':'gmaw','welding_arm_joints':cfg['n'],
       'external_axes':list(cell),'outputs':dict(zip(targets,[float(x) for x in y[0]])),
       'components_Wh':dict(zip(COMPONENTS,[float(x) for x in component[0]]))}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--checkpoint',default='reports/gmaw/arc_welding_model.pt');p.add_argument('--output');a=p.parse_args()
 r=infer(json.loads(Path(a.input).read_text()),a.checkpoint)
 if a.output:Path(a.output).write_text(json.dumps(r,indent=2))
 print(json.dumps(r,indent=2))
