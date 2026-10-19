"""Surface-grinding job-level hardware energy, no measured labels in inference."""
from __future__ import annotations
import json, numpy as np
from grinding_physics import COMPONENTS,simulate,DENSITY

def components_by_row(rows,times):
    events=simulate(rows,teacher=False,times_override=times)
    arr=np.asarray([q['component_energy_Wh'] for q in events],float)
    p=json.loads(rows[0]['initial_profile']);dur=float(np.sum(times))
    vol=sum(q['material_removed_mm3'] for q in events)
    motion=sum(q['motion_mm'] for q in events)
    return {'components':list(COMPONENTS),'base':arr.sum(axis=0),
        'motion_s':sum(q['time_s'] for q in events if q['motion_mm']>0),
        'speed':motion/max(dur,1e-8),'rpm':p['wheel_rpm'], 'cut_vol':vol,
        'tool':p['wheel_diameter_mm'],'temp':0.,'density':DENSITY[p['material']]*1000,
        'material':vol*DENSITY[p['material']], 'total_s':dur,
        'commands':len(rows),'mode':'cnc_grinding'}
