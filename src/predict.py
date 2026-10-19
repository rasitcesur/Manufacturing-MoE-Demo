"""Inference for original Physical Transformer-MoE extended to 8 operations including surface grinding.
For hole operations, G17/Z and G19/X use identical canonical axial/lateral
features for the shared Hole Processing Expert. Machine orientation affects
hardware power prior only; old FDM/milling/turning parsers remain unchanged.
Research simulator only; CNC code is NEVER executed on machine.
"""
from __future__ import annotations
import argparse,csv,json,math,re
from pathlib import Path
import numpy as np,torch
from train import (TimeMoE,EnergyMoE,MODE,WORDS,COMMENTS,FEATURE_DIM,
      cnc_features,turning_features,drilling_features,compute_energy_case,energy_features,apply_energy_calibration,
      predict_time as _predict_time,machine_adjusted_case)
from machine_characterization import MachineCharacterizationExpert, SENSOR_NAMES, MACHINE_GROUPS
from hole_physics import simulate_program
from physics import state_features
from train import grinding_features,sawing_features,thermal_features

def predict(checkpoint,gcode,profile,mode,operation=None,device='cpu',machine_history=None):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if mode=='milling':mode='cnc_milling'
    if mode=='turning':mode='cnc_turning'
    if mode=='drilling':mode='cnc_drilling'
    if mode=='grinding':mode='cnc_grinding'
    if mode=='sawing':mode='cnc_sawing'
    if mode=='laser':mode='cnc_laser'
    if mode=='plasma':mode='cnc_plasma'
    if mode not in MODE:raise ValueError('Mode must be fdm, milling, turning, drilling, grinding, sawing, laser or plasma')
    profilej=Path(profile).read_text(encoding='utf-8');meta=json.loads(profilej)
    ops=('drilling','boring','reaming','tapping')
    if operation in ops:mode='cnc_drilling'
    if operation is None:operation=meta.get('operation','drilling' if mode=='cnc_drilling' else None)
    rows=[]
    for ln in Path(gcode).read_text(encoding='utf-8-sig').splitlines():
        line=COMMENTS.sub('',ln).strip().upper()
        if line and WORDS.search(line):
            rows.append({'program_id':'inference','step_index':str(len(rows)),'gcode':line,'initial_profile':profilej,
               'time_s':0.,'energy_Wh':0.,'material_g':0.})
    if not rows:raise ValueError('No G-code parsed')
    # The 180 original vertical drilling jobs use the original time parser
    # but the upgraded 12-component hardware-energy prior.
    legacy_hole=(operation=='drilling' and meta.get('firmware')=='analytic_drilling_surrogate_not_LinuxCNC')
    is_new_hole=(operation in ops and 'hidden_diameter_mm' in meta and not legacy_hole)
    if is_new_hole:
        codes,feat,prior,geom,energycase=simulate_program(rows,FEATURE_DIM)
        group=f'{operation}_'+('horizontal' if any(re.search(r'\bG19\b',r['gcode']) for r in rows) else 'vertical')
    elif mode=='fdm':
        codes,ff,prior=state_features(rows);feat=np.zeros((len(rows),FEATURE_DIM),np.float32)
        feat[:,:38]=ff;feat[:,38]=1.;geom=np.asarray(prior[:,2],float);group='fdm'
    elif mode=='cnc_milling':codes,feat,prior,geom=cnc_features(rows);group='cnc_milling'
    elif mode=='cnc_grinding':codes,feat,prior,geom=grinding_features(rows);group='grinding_surface'
    elif mode=='cnc_sawing':codes,feat,prior,geom=sawing_features(rows);group='sawing_band'
    elif mode in ('cnc_laser','cnc_plasma'):
        codes,feat,prior,geom=thermal_features(rows);group=mode.split('_')[-1]+'_sheet'
    elif mode=='cnc_turning':codes,feat,prior,geom=turning_features(rows);group='cnc_turning'
    elif mode=='cnc_drilling':codes,feat,prior,geom=drilling_features(rows);group='drilling_vertical_old'
    else:raise ValueError('Unknown mode')
    p={'id':'inference','mode':mode,'group':group,'rows':rows,'codes':np.asarray(codes,dtype=np.int64),
        'feat':np.asarray(feat,dtype=np.float32),'prior':np.asarray(prior,dtype=np.float32),
        'geometry_g':np.asarray(geom,dtype=float),'target':np.zeros((len(rows),3),np.float32)}
    if is_new_hole or legacy_hole:p['hole_new']=True
    if is_new_hole:p['hole_energy_physics']=energycase
    tm=TimeMoE().to(device);tm.load_state_dict(state['time']);tm.eval()
    em=EnergyMoE(state['components']).to(device);em.load_state_dict(state['energy']);em.eval()
    mean,std=state['time_scaler'];nt=_predict_time(tm,[p],mean,std,device)['inference']
    alpha=state['time_blends'].get(group,state['time_blends'].get('drilling_vertical_old',0.))
    pred_time=(1-alpha)*prior[:,0]+alpha*nt
    case=apply_energy_calibration(compute_energy_case(p,pred_time),group,
        state.get('milling_energy_calibration'),state.get('drilling_energy_calibration'))
    # Machine encoder receives watts observed BEFORE the target job; hidden gain ignored.
    if group in MACHINE_GROUPS and machine_history is not None:
        if isinstance(machine_history,(str,Path)):
            data=json.loads(Path(machine_history).read_text(encoding='utf-8'))
            observed=data['measurements_W']
        else:
            observed=machine_history['measurements_W'] if isinstance(machine_history,dict) else machine_history
        if 'machine_expert' not in state:raise ValueError('Checkpoint has no machine-history expert')
        mce=MachineCharacterizationExpert().to(device)
        mce.load_state_dict(state['machine_expert'])
        case=machine_adjusted_case(case,observed,mce,group)
    mn,sd=state['energy_scaler'];components=[]
    for name,b in zip(case['components'],case['base']):
        ix=state['components'].index((mode,name));ft=energy_features(case,name)
        xx=torch.from_numpy(np.clip((ft-mn)/sd,-8,8)).reshape(1,-1).to(device)
        with torch.inference_mode():learned=em(xx,torch.tensor([ix],device=device),
                   torch.tensor([MODE[mode]],device=device),torch.tensor([float(b)],device=device)).item()
        blend=state['energy_blends'].get(group,state['energy_blends'].get('drilling_vertical_old',0))
        components.append({'component':name,'energy_Wh':float((1-blend)*b+blend*learned)})
    total_time=float(np.sum(pred_time));total_energy=float(sum(x['energy_Wh'] for x in components));total_material=float(np.sum(geom))
    out={'operation':operation or mode,'mode':mode,'orientation':group.split('_')[-1] if (is_new_hole or legacy_hole) else None,
        'program_group':group,'time_s':total_time,'energy_Wh':total_energy,'material_g':total_material,
        'material_meaning':'filament' if mode=='fdm' else 'kerf_removed' if mode in ('cnc_laser','cnc_plasma') else 'chips_removed',
        'energy_components_Wh':components,'command_count':len(rows),
        'synthetic_only':True,'runtime_verified_on_LinuxCNC':False,
        'machine_characterization_used':bool(case.get('machine_history_used',False)),
        'machine_characterization_supported':group in MACHINE_GROUPS}
    return out,rows,pred_time

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--checkpoint',default='reports/final/model.pt')
    pa.add_argument('--gcode',required=True);pa.add_argument('--profile',required=True)
    pa.add_argument('--mode',required=True,choices=['fdm','milling','turning','drilling','grinding','sawing','cnc_sawing','cnc_milling','cnc_turning','cnc_grinding','laser','plasma','cnc_laser','cnc_plasma'])
    pa.add_argument('--operation',choices=['drilling','boring','reaming','tapping'])
    pa.add_argument('--machine-history',default=None,help='Prior machine power measurements JSON (six readings in W), not hidden machine gain')
    pa.add_argument('--output',default='outputs/prediction.json');pa.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    a=pa.parse_args();res,rows,pred=predict(a.checkpoint,a.gcode,a.profile,a.mode,a.operation,a.device,a.machine_history)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(res,ensure_ascii=False,indent=2))
    with out.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['step','gcode','pred_time_s']);w.writerows((i,r['gcode'],float(v)) for i,(r,v) in enumerate(zip(rows,pred)))
    print(json.dumps(res,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
