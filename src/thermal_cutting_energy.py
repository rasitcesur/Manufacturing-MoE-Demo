"""Hardware-level independent thermal cutter energy prior; labels are never read."""
from __future__ import annotations
import numpy as np,json
from thermal_cutting_physics import simulate,COMPONENTS,MAT

def components_by_row(rows,times):
    p=json.loads(rows[0]['initial_profile']);mode=p['mode'];ev=simulate(rows,teacher=False,times_override=times)
    mat=sum(e['material_removed_mm3'] for e in ev);dur=float(sum(times));motion=sum(e['motion_mm'] for e in ev)
    c=np.asarray([e['component_energy_Wh'] for e in ev],float)
    return {'components':list(COMPONENTS[mode]),'base':c.sum(axis=0),
     'motion_s':sum(e['time_s'] for e in ev if e['motion_mm']>0),
     'speed':motion/max(dur,.001),'rpm':p['laser_power_kW']*1000. if mode=='cnc_laser' else p['current_A'],
     'cut_vol':mat,'tool':sum(e['kerf_mm'] for e in ev)/max(1,len(ev)),
     'temp':0.,'density':MAT[p['material']][0]*1000.,
     'material':mat*MAT[p['material']][0], 'total_s':dur,'commands':len(rows),'mode':mode}
