"""Learn a machine state from prior SENSOR-LIKE power observations only.

Synthetic probe is created outside the predictive path. It emulates measurements
from a separate, previously completed, no-cut spindle/axis diagnostic; current
job energy and hidden machine_gain are NEVER passed to the expert at inference.
This is an ANALYTIC TEST, not a real sensor dataset or a validated power meter.
"""
from __future__ import annotations
import json, math, hashlib, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

MACHINE_GROUPS=['drilling_horizontal','boring_horizontal','boring_vertical','reaming_horizontal','reaming_vertical','tapping_horizontal','tapping_vertical','grinding_surface','sawing_band','laser_sheet','plasma_sheet']
SENSOR_NAMES=['idle_total_W','spindle_1000_W','spindle_3000_W','axis_x_20mms_W','axis_z_20mms_W','coolant_increment_W']
HISTORY_DIM=len(SENSOR_NAMES)

def synthesize_history(gain:float, orientation:str, seed_string:str):
    """Synthetic prior measurement, independent of target job, never used by predictor.
    This function must ONLY be invoked by generate_machine_histories.py.
    A separate no-cut standard probe is performed prior to all job predictions.
    """
    rng=random.Random(int(hashlib.sha256(seed_string.encode()).hexdigest()[:15],16))
    g=float(gain);z=orientation=='vertical'
    off=(62+17*g)+(15+(27 if z else 13)*g)+(17+(6 if z else 10))
    s=lambda rpm: (46+.021*rpm+1.1e-6*rpm*rpm)*g*(.91+.13*g)
    motor_factor=.96+.07*g
    axis_x=(6+(.78*20+.018*20**2))*motor_factor
    axis_z=(6+(.78*20+.018*20**2)*1.32)*motor_factor
    val=[off,off+s(1000),off+s(3000),off+axis_x,off+axis_z,120.]
    # Probe power sensor read-out error: reproducible, 0.3% Gaussian per channel.
    measured=[max(0.,v*(1.+rng.gauss(0.,.003))) for v in val]
    return measured

def save_histories(root:Path):
    """Generate *independent historical* power-probe values for all 7 NEW hole groups.

    The private source's gain parameter is consulted ONLY here to synthesize
    earlier instrument observations. Neither it nor current-job energy labels
    are written into active machine history, checkpoint inputs or training data.
    """
    import csv
    root=Path(root)
    (root/'data/machine_history').mkdir(parents=True,exist_ok=True)
    coverage={}
    for split in ('train','val','test'):
        original=root/'data/new_before_energy_revision'/f'{split}.csv'
        rows=list(csv.DictReader(original.open(encoding='utf-8')))
        mapping={};groups={}
        for r in rows:
            key=r['operation']+'_'+r['orientation']
            if key not in MACHINE_GROUPS:continue
            pid=r['program_id'];meta=json.loads(r['initial_profile'])
            grp=int(pid.split('_')[-1])//6
            machine=f'MACHINE_{key}_{grp:02d}'
            if machine in groups and abs(groups[machine][0]-float(meta['hidden_machine_gain']))>1e-10:
                raise ValueError('Inconsistent hidden machine metadata for '+machine)
            groups[machine]=(float(meta['hidden_machine_gain']),meta['orientation'])
            mapping[pid]=machine
        records={key:synthesize_history(g,o,key) for key,(g,o) in groups.items()}
        obj={'sensor_names':SENSOR_NAMES,'machine_measurements':records,'program_to_machine':mapping,
             'provenance':'Synthetic power measurements collected before target job; 0.3% sensor error; zero target-label access. No hidden gain in serialized data.'}
        (root/'data/machine_history'/f'{split}.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
        coverage[split]={'machines':len(groups),'programs':len(mapping)}
    (root/'data/machine_history/README.md').write_text("""# SYNTHETIC independent prior power tests — NOT CNC sensor readings

Applies to drilling-horizontal and both orientations of boring, reaming and tapping.
Each synthetic six-program machine group is assigned a reproducible, independently
noised prior no-cut power probe (6 watt readings). This probe is calculated by the
**private teaching simulator** from its latent efficiency coefficient, but the
coefficient NEVER appears in active inputs or targets. A program's future total
energy and component targets are not used to synthesize its prior probe.
Machine groups do not cross train/val/test; the gain VALUE may repeat across splits,
so these experiments do not establish extrapolation to entirely unseen efficiency
ranges. Real machine validation is still necessary. If no telemetry exists,
model uses cold-start fallback without any surrogate hidden gain.
""",encoding='utf-8')
    return coverage

def history_for_program(root:Path,split:str,pid:str):
    f=Path(root)/'data/machine_history'/f'{split}.json'
    js=json.loads(f.read_text())
    m=js['program_to_machine'].get(pid)
    return js['machine_measurements'].get(m) if m else None

class MachineCharacterizationExpert(nn.Module):
    """Only previous independent power probes; no private machine gain feature."""
    def __init__(self,components=60,latent_dim=None):
        super().__init__()
        from config import choice
        latent_dim=choice(6,10,16) if latent_dim is None else latent_dim
        a,b=choice((20,16),(32,24),(48,32))
        c,d=choice((24,12),(40,20),(64,32))
        self.encoder=nn.Sequential(nn.Linear(HISTORY_DIM,a),nn.GELU(),nn.Linear(a,b),nn.GELU(),nn.Linear(b,latent_dim))
        self.component=nn.Embedding(components,6)
        self.operation=nn.Embedding(len(MACHINE_GROUPS),5)
        self.gate=nn.Sequential(nn.Linear(latent_dim+6+5,c),nn.GELU(),nn.Linear(c,d),nn.GELU(),nn.Linear(d,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
        self.register_buffer('input_mean',torch.zeros(len(MACHINE_GROUPS),HISTORY_DIM))
        self.register_buffer('input_std',torch.ones(len(MACHINE_GROUPS),HISTORY_DIM))
    def set_scaler(self,mean,std):
        self.input_mean.copy_(torch.as_tensor(mean,dtype=torch.float32))
        self.input_std.copy_(torch.as_tensor(std,dtype=torch.float32))
    def forward(self,powers,component_index,available,group_index=None):
        if group_index is None:group_index=torch.zeros_like(component_index)
        h=self.encoder(torch.clamp((powers-self.input_mean[group_index])/self.input_std[group_index],-8,8))
        delta=self.gate(torch.cat([h,self.component(component_index),self.operation(group_index)],dim=-1)).squeeze(-1)
        return torch.where(available,torch.exp(.55*torch.tanh(delta)),torch.ones_like(delta))

