"""Standalone prediction from PUBLIC robot+power-supply description, target program
and PREVIOUS measured power history. No target label used during prediction.
Usage: python src/predict_welding.py --input examples/rsw_v55_input.json
"""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'runtime'),str(ROOT/'runtime')]
from process_features import feature,nominal,calibrations
from robot_cell import totals_common,COMPONENTS
from welding_specialists import WeldingSet,WeldingMoE55,predict

def calculate(item,checkpoint=None):
 process=item['process'];src=item['machine']['power_supply'];cfg=item['machine']['robot'];program=item['program']
 m=item['machine']
 hist=np.r_[m['previous_power_probes_W'],m['previous_robot_power_probes_W'],
            m['previous_motions'],m['prior_external_idle_log_ratios']].astype(np.float32)
 c,s=nominal(cfg,src)
 prior=totals_common(c,s,program,process)
 yc=np.array([prior[k] for k in ('time_s','energy_Wh','material_added_g','wire_consumed_g','gas_consumed_L','spot_count')],np.float32)
 cp=np.array([prior['components'][k] for k in COMPONENTS],np.float32)
 cr=float(np.exp(np.clip(np.asarray(m['previous_motions'])[::4].mean(),-.6,.6)))
 sc=calibrations(src,process,np.asarray(m['previous_power_probes_W']))
 cp[0]*=cr;cp[2]*=sc;yc[1]=float(cp.sum())
 j,g=feature(cfg,src,program)
 raw={'joint':j[None],'global_features':g[None],'history':hist[None],'mask':np.r_[np.ones(cfg['n']),np.zeros(15-cfg['n'])][None],
      'y':np.zeros((1,6),np.float32),'components':np.ones((1,9),np.float32),'prior':yc[None],'component_prior':cp[None],
      'robot_ids':np.array([item.get('robot_id','provided')]),'program_ids':np.array([item.get('program_id','provided')]),
      'topology':np.array([item.get('topology','unknown')]),'n_axes':np.array([cfg['n']])}
 checkpoint=checkpoint or ROOT/'reports'/('rsw' if process=='rsw' else 'tig')/'model.pt'
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
 d=WeldingSet(raw,norm=ck['normalizer'])
 net=WeldingMoE55(ck['global_dim'],ck['history_dim'],process)
 net.load_state_dict(ck['model_state'])
 net.to('cuda' if torch.cuda.is_available() else 'cpu')
 yp,comps=predict(net,d,blend=ck['blend'])
 return {'process':process,'predictions':dict(zip(('time_s','energy_Wh','material_added_g','wire_consumed_g','gas_consumed_L','spot_count'),map(float,yp[0]))),
         'components_Wh':dict(zip(COMPONENTS,map(float,comps[0]))),'source':'previous public machine histories and physics-informed nominal prior'}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--output');aa=a.parse_args()
 res=calculate(json.load(open(aa.input)))
 if aa.output:Path(aa.output).write_text(json.dumps(res,indent=2))
 else:print(json.dumps(res,indent=2))
