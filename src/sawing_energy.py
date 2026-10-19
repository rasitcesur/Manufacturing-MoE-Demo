"""Independent inference physical energy prior for band sawing; never reads label columns."""
from __future__ import annotations
import json,numpy as np
from sawing_physics import COMPONENTS,DENSITY,simulate

def components_by_row(rows,times):
    ev=simulate(rows,teacher=False,times_override=times)
    power=np.asarray([e['component_energy_Wh'] for e in ev],dtype=float)
    profile=json.loads(rows[0]['initial_profile']);duration=float(np.sum(times))
    volume=sum(e['material_removed_mm3'] for e in ev);motion=sum(e['motion_mm'] for e in ev)
    return {'components':list(COMPONENTS),'base':power.sum(axis=0),
     'motion_s':sum(e['time_s'] for e in ev if e['motion_mm']>0),
     'speed':motion/max(duration,1e-8),'rpm':profile['band_speed_m_min'],
     'cut_vol':volume,'tool':profile['kerf_mm'],'temp':0.,
     'density':DENSITY[profile['material']]*1000.,
     'material':volume*DENSITY[profile['material']],
     'total_s':duration,'commands':len(rows),'mode':'cnc_sawing'}
