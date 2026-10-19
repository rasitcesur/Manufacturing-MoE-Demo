"""Manufacturing-MoE current robotics: inference with observable pre-target history."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from robotics_calibrated import RoboticsMoECalibrated,PreparedCalibrated,prediction
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from robot_physics import simulate
from robot_features import public_features
from gravity_features import signed_features
PRIVATE_DEFAULTS={'private_motor_efficiency':1.,'private_driver_loss':1.,'private_friction_scale':1.,'private_settle_s':0.,'private_regen_scale':1.}

def robot_predict(checkpoint,robot,motion,history_probes=None,past_calibration=None):
    if history_probes is None:history_probes=robot.get('history_probes')
    if past_calibration is None:past_calibration=robot.get('past_calibration')
    if history_probes is None or past_calibration is None:
        raise ValueError('Calibrated robotics model requires seven previous power/settle observations and six independently measured calibration cycles (24 values).')
    h=np.asarray(history_probes,np.float32)
    hc=np.asarray(past_calibration,np.float32)
    if h.shape!=(7,) or hc.shape!=(24,) or not np.isfinite(h).all() or not np.isfinite(hc).all():
        raise ValueError('History observations must be finite and of length 7 and 24, respectively.')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    s=state.get('robotics_bundle',state)
    if s['model_type']!='RoboticsMoECalibrated':raise ValueError('Checkpoint does not contain calibrated robotics model.')
    model=RoboticsMoECalibrated();model.load_state_dict(s['robotics']);model.eval()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    cfg={**robot,**PRIVATE_DEFAULTS}
    out=simulate(cfg,motion,steps=22)
    joint,mask,global_x=public_features(cfg,motion,out)
    signed_joint,signed_global=signed_features(cfg,motion)
    d={'joint':joint[None],'mask':mask[None],'global_features':global_x[None],
       'joint_gravity':signed_joint[None],'global_gravity':signed_global[None],
       'history':h[None],'history_calibration':hc[None],
       'prior':np.array([[out['nominal_time_s'],out['nominal_energy_Wh']]],np.float32),
       'y':np.ones((1,2),np.float32),'n_axes':np.array([robot['n']]),
       'topology':np.array([robot['topology']]),'robot_ids':np.array([robot.get('robot_id','inference')])}
    scaler=s['train_normalizer'];ds=PreparedCalibrated(d,scaler['mean'],scaler['std'])
    pred=prediction(model,ds,device=device)[0]
    return {'model_family':'robotics','joint_count':robot['n'],'topology':robot['topology'],
       'predicted_cycle_time_s':float(pred[0]),'predicted_electricity_Wh':float(pred[1]),
       'material_removed_g':0.,'history_calibration_cycles':6,
       'signed_gravity_features':True,'simulator_nominal_energy_Wh':out['nominal_energy_Wh'],
       'not_a_real_machine_validated_model':True}

if __name__=='__main__':
    pa=argparse.ArgumentParser()
    pa.add_argument('--checkpoint',default='reports/v51/model.pt');pa.add_argument('--robot',required=True)
    pa.add_argument('--motion',required=True);pa.add_argument('--output',default='outputs/robotics_v51_prediction.json')
    a=pa.parse_args()
    r=json.loads(Path(a.robot).read_text());mo=json.loads(Path(a.motion).read_text())
    pred=robot_predict(a.checkpoint,r,mo)
    o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(pred,indent=2))
    print(json.dumps(pred,indent=2))
