"""Manufacturing-MoE Physical Transformer-MoE: eleven operations, eight process families.
Shared Transformer, conditional time experts; conditional job-energy hardware experts.
All neural parameters initialized anew, no legacy weights loaded.
Labels: independent Marlin-style *approximation* (FDM), analytic stop-to-stop
*non-LinuxCNC* CNC surrogate. No physical machine measurements in this package.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, random, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from physics import state_features, CMAP
from job_features import summary as fdm_summary
from component_power import COMPONENTS as FDM_COMPONENTS
from drilling_physics import simulate as drilling_simulate
from hole_physics import simulate_program
from hole_energy_hardware import COMPONENTS as HOLE_COMPONENTS, components_by_row
from machine_characterization import MachineCharacterizationExpert, HISTORY_DIM, MACHINE_GROUPS
from grinding_physics import simulate as grinding_simulate, COMPONENTS as GRINDING_COMPONENTS, DENSITY as GRINDING_DENSITY
from grinding_energy import components_by_row as grinding_components_by_row
from sawing_physics import simulate as sawing_simulate, COMPONENTS as SAWING_COMPONENTS, DENSITY as SAWING_DENSITY
from sawing_energy import components_by_row as sawing_components_by_row
from thermal_cutting_physics import simulate as thermal_simulate, COMPONENTS as THERMAL_COMPONENTS, MAT as THERMAL_MAT
from thermal_cutting_energy import components_by_row as thermal_components_by_row
ROOT=Path(__file__).resolve().parents[1]
MODE={'fdm':0,'cnc_milling':1,'cnc_turning':2,'cnc_drilling':3,'cnc_grinding':4,'cnc_sawing':5,'cnc_laser':6,'cnc_plasma':7}
CNC_COMPONENTS=['electronics','axis_hold','axis_x','axis_y','axis_z','spindle_idle','cutting_load','coolant']
TURNING_COMPONENTS=['electronics','axis_hold','axis_x','axis_z','spindle_idle','cutting_load','coolant','hydraulic']
# Avoid double counting with FDM's components. Component identifiers are (mode, name).
MODES_COMPONENTS={'fdm':list(FDM_COMPONENTS),'cnc_milling':CNC_COMPONENTS,'cnc_turning':TURNING_COMPONENTS,'cnc_drilling':list(HOLE_COMPONENTS),'cnc_grinding':list(GRINDING_COMPONENTS),'cnc_sawing':list(SAWING_COMPONENTS),'cnc_laser':list(THERMAL_COMPONENTS['cnc_laser']),'cnc_plasma':list(THERMAL_COMPONENTS['cnc_plasma'])}
CODEMAP={**CMAP,'M3':40,'M5':41,'M8':42,'M9':43,'M2':44,'STATE':45,'G81':46,'G82':47,'G83':48}
WORDS=re.compile(r'([A-Z])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))',re.I)
COMMENTS=re.compile(r'\([^)]*\)|;[^\n]*')
MATERIALS={'AL6061':(.0027,1.4),'C45':(.00785,3.4),'POM':(.00141,.75)}
FEATURE_DIM=42 # 38 FDM native + 4 process flags

def read_groups(path,mode):
    groups=defaultdict(list)
    with Path(path).open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):groups[row['program_id']].append(row)
    ans=[]
    for k,rows in sorted(groups.items()):
        rows.sort(key=lambda r:int(r['step_index']))
        if mode=='fdm':
            codes,ff,pr=state_features(rows)
            feats=np.zeros((len(rows),FEATURE_DIM),np.float32);feats[:,:38]=ff;feats[:,38]=1.
            codes=np.asarray(codes,dtype=np.int64)
            geometry=np.asarray(pr[:,2],np.float64)
            meta={'material_kind':'filament_consumed'}
        elif mode=='cnc_milling':
            codes,feats,pr,geometry=cnc_features(rows)
            meta={'material_kind':'chips_removed'}
        elif mode=='cnc_turning':
            codes,feats,pr,geometry=turning_features(rows)
            meta={'material_kind':'chips_removed'}
        elif mode=='cnc_drilling':
            codes,feats,pr,geometry=drilling_features(rows)
            meta={'material_kind':'chips_removed'}
        elif mode=='cnc_grinding':
            codes,feats,pr,geometry=grinding_features(rows)
            meta={'material_kind':'chips_removed'}
        elif mode=='cnc_sawing':
            codes,feats,pr,geometry=sawing_features(rows)
            meta={'material_kind':'chips_removed'}
        elif mode in ('cnc_laser','cnc_plasma'):
            codes,feats,pr,geometry=thermal_features(rows)
            meta={'material_kind':'kerf_removed'}
        else:raise ValueError('Unknown mode: '+str(mode))
        y=np.array([[float(r['time_s']),float(r['energy_Wh']),float(r['material_g'])] for r in rows],dtype=np.float32)
        if len(feats)!=len(y):raise AssertionError(k)
        if not np.all(np.isfinite(feats)) or not np.all(np.isfinite(pr)):raise ValueError('Nonfinite input')
        ans.append({'id':k,'mode':mode,'rows':rows,'codes':codes,'feat':feats,'prior':pr.astype('float32'),
                    'geometry_g':geometry,'target':y,**meta})
    return ans

def cnc_features(rows):
    profile=json.loads(rows[0]['initial_profile']); feed=600.; spindle=0.; pos=np.zeros(3,np.float64)
    on=False;coolant=False;motion='G0';absolute=True;out_code=[];out_feat=[];out_prior=[];removed=[]
    d=float(profile['tool_diameter_mm']);rho=MATERIALS[profile['material']][0]
    for row in rows:
        gc=COMMENTS.sub('',row['gcode'].upper());pairs=[(k,float(v)) for k,v in WORDS.findall(gc)];w=dict(pairs)
        gvals=[v for k,v in pairs if k=='G'];mvals=[int(v) for k,v in pairs if k=='M']
        for g in gvals:
            if g in (0.,1.,2.,3.):motion='G'+str(int(g))
            elif g==90: absolute=True
            elif g==91: absolute=False
        if 'F' in w:feed=w['F']
        if 'S' in w:spindle=w['S']
        for m in mvals:
            if m in (3,4):on=True
            elif m==5:on=False
            elif m==8:coolant=True
            elif m==9:coolant=False
        move=any(k in w for k in 'XYZ')
        new=pos.copy()
        for i,k in enumerate('XYZ'):
            if k in w:new[i]=w[k] if absolute else new[i]+w[k]
        dx,dy,dz=(new-pos)
        length=float(np.linalg.norm(new-pos))
        if move and motion in ('G2','G3') and ('I' in w or 'J' in w):
            cx,cy=pos[0]+w.get('I',0.),pos[1]+w.get('J',0.)
            r0=math.hypot(pos[0]-cx,pos[1]-cy);r1=math.hypot(new[0]-cx,new[1]-cy)
            if r0>1e-7 and abs(r1-r0)<.02:
                a0=math.atan2(pos[1]-cy,pos[0]-cx);a1=math.atan2(new[1]-cy,new[0]-cx)
                sweep=(a1-a0)%(2*math.pi) if motion=='G3' else (a0-a1)%(2*math.pi)
                if sweep<1e-9:sweep=2*math.pi
                length=math.hypot(r0*sweep,dz)
        velocity=min(6000. if motion=='G0' else feed,6000. if motion=='G0' else 5000.)/60
        # Intentionally a nominal acceleration prior, NOT the original labeler's exact 600.
        accel=500.
        if move and length>0:
            switch=velocity*velocity/accel
            baseline=2*math.sqrt(length/accel) if length<switch else 2*velocity/accel+(length-switch)/velocity
        elif 'G4' in gc: baseline=max(0.,w.get('P',0.))
        else:baseline=0.
        vol=0.
        if move and motion=='G1' and on:
            if abs(dx)<1e-8 and abs(dy)<1e-8 and dz<0 and new[2]<0 and pos[2]>=0:
                vol=math.pi*(d/2)**2*(-new[2])
            elif abs(dz)<1e-8 and pos[2]<0 and math.hypot(dx,dy)>0:
                vol=d*math.hypot(dx,dy)*(-pos[2])
        mat=vol*rho
        code=motion if move else ('G4' if 'G4' in gc else ('M'+str(mvals[0]) if mvals else 'STATE'))
        rowf=np.zeros(FEATURE_DIM,np.float32)
        # Log-transform dynamic CNC quantities before train-only feature standardization.
        values=[math.log1p(max(length,0)),math.log1p(max(velocity,0)),math.log1p(max(spindle,0)),
                math.log1p(max(d,0)),math.log1p(float(profile['cut_depth_mm'])),
                math.log1p(abs(dx)),math.log1p(abs(dy)),math.log1p(abs(dz)),
                np.sign(dx),np.sign(dy),np.sign(dz),np.sign(new[2]),
                math.log1p(max(vol,0)),float(on),float(coolant),float(move),
                float(motion=='G0'),float(motion=='G1'),float(motion in ('G2','G3')),
                float(profile['material']=='C45'),float(profile['material']=='POM'),
                math.log1p(feed),math.log1p(max(baseline,0)),
                float('G4' in gc),float(len(rows)),
                math.log1p(abs(new[0])),math.log1p(abs(new[1])),
                float(profile['accel_mm_s2'])/1000.]
        rowf[:len(values)]=values;rowf[39]=1.;rowf[40]=rho*1000.;rowf[41]=math.log1p(float(profile['rapid_mm_min']))
        out_code.append(CODEMAP.get(code,CODEMAP['STATE']));out_feat.append(rowf)
        out_prior.append([baseline,0,mat]);removed.append(mat)
        pos=new
    return np.asarray(out_code,np.int64),np.asarray(out_feat,np.float32),np.asarray(out_prior,np.float32),np.asarray(removed,np.float64)

def turning_features(rows):
    """Parse limited LinuxCNC-compatible G7/G18/G94/G97 lathe dialect.

    X coordinates are *diameters*, so geometric motion uses delta-X/2.
    Material output is a cylinder-volume difference, NOT a learned prediction.
    Only simple G0/G1 turning passes are covered; canned cycles unsupported.
    """
    profile=json.loads(rows[0]['initial_profile'])
    rho=MATERIALS[profile['material']][0];feed=600.;rpm=0.;on=False;coolant=False
    dia=float(profile['stock_diameter_mm']);old_d=dia
    x=dia+4.;z=5.;motion='G0';absolute=True;diameter=True
    out_code=[];out_feat=[];out_prior=[];removed=[]
    for row in rows:
        gc=COMMENTS.sub('',row['gcode'].upper());pairs=[(k,float(v)) for k,v in WORDS.findall(gc)];w=dict(pairs)
        gvals=[v for k,v in pairs if k=='G'];mvals=[int(v) for k,v in pairs if k=='M']
        if any(g in (93,95,96) for g in gvals):raise ValueError('G93/G95/G96 not supported in initial turning model')
        for g in gvals:
            if g in (0.,1.):motion='G'+str(int(g))
            elif g in (2.,3.,76.,33.):raise ValueError('Turning arcs, threading or canned cycles not yet supported')
            elif g==90:absolute=True
            elif g==91:absolute=False
            elif g==7:diameter=True
            elif g==8:diameter=False
        if 'F' in w:feed=w['F']
        if 'S' in w:rpm=w['S']
        for m in mvals:
            if m in (3,4):on=True
            elif m==5:on=False
            elif m==8:coolant=True
            elif m==9:coolant=False
        nx=x if 'X' not in w else ((w['X'] if diameter else 2*w['X']) if absolute else x+(w['X'] if diameter else 2*w['X']))
        nz=z if 'Z' not in w else (w['Z'] if absolute else z+w['Z'])
        dx=(nx-x)/2.;dz=nz-z;dist=math.hypot(dx,dz)
        move=('X' in w or 'Z' in w)
        velocity=(float(profile['rapid_mm_min']) if motion=='G0' else feed)/60.
        accel=565. # intentionally differs from data generator's 650
        t=(2*math.sqrt(dist/accel) if dist<velocity*velocity/accel else 2*velocity/accel+(dist-velocity*velocity/accel)/velocity) if move and dist>0 else (max(0,w.get('P',0.)) if any(g==4 for g in gvals) else 0.)
        vol=0.
        if move and motion=='G1' and on and dz<0 and z<=1e-8 and nz<0 and abs(dx)<1e-8:
            # Label and prediction both track cumulative stock/profile geometry.
            vol=math.pi/4.*max(0,old_d**2-nx**2)*(-dz)
            if vol>0:old_d=nx
        vec=np.zeros(FEATURE_DIM,np.float32)
        values=[math.log1p(dist),math.log1p(velocity),math.log1p(max(rpm,0)),
            math.log1p(dia),math.log1p(float(profile['pass_depth_radial_mm'])),
            math.log1p(abs(dx)),0.,math.log1p(abs(dz)),np.sign(dx),0.,np.sign(dz),np.sign(nz),
            math.log1p(vol),float(on),float(coolant),float(move),float(motion=='G0'),float(motion=='G1'),0.,
            float(profile['material']=='C45'),float(profile['material']=='POM'),math.log1p(max(feed,0)),math.log1p(t),
            float(any(g==4 for g in gvals)),float(len(rows)),math.log1p(abs(nx)),math.log1p(abs(nz)),
            float(profile['accel_mm_s2'])/1000.]
        vec[:len(values)]=values;vec[40]=rho*1000.;vec[41]=math.log1p(float(profile['rapid_mm_min']))
        code=motion if move else ('G4' if any(g==4 for g in gvals) else ('M'+str(mvals[0]) if mvals else 'STATE'))
        out_code.append(CODEMAP.get(code,CODEMAP['STATE']));out_feat.append(vec)
        mass=vol*rho;removed.append(mass);out_prior.append([t,0.,mass]);x,z=nx,nz
    return np.asarray(out_code,np.int64),np.asarray(out_feat,np.float32),np.asarray(out_prior,np.float32),np.asarray(removed,np.float64)

def drilling_features(rows):
    """Drilling G-code features and independent approximate time prior.

    No target label or CSV-derived durations/energy enter this featurizer.
    Restricted to generator's explicit G81/G82/G83 and modal XY hole blocks.
    """
    profile=json.loads(rows[0]['initial_profile'])
    sim=drilling_simulate([r['gcode'] for r in rows],profile,accel=490.,rapid=6000.,clearance=.30)
    if len(sim)!=len(rows):raise ValueError('drilling program block mismatch')
    rho=MATERIALS[profile['material']][0];out_code=[];out_feat=[];prior=[];mat=[]
    for event in sim:
        code=event['command'];v=np.zeros(FEATURE_DIM,np.float32)
        length=event['motion_length_mm'];t=event['time_s']
        vol=event['material_removed_mm3'];d=profile['tool_diameter_mm']
        values=[math.log1p(length),math.log1p(length/max(t,1.e-5)),math.log1p(event['spindle_rpm']),
          math.log1p(d),math.log1p(float(profile['hole_depth_mm'])),
          math.log1p(abs(event['x_after_mm'])),math.log1p(abs(event['y_after_mm'])),
          math.log1p(abs(event['z_after_mm'])),
          float(event['command']=='G81'),float(event['command']=='G82'),float(event['command']=='G83'),
          float(event['command']=='G0'),math.log1p(vol),
          float(event['spindle_on']),float(event['coolant_on']),float(length>0),
          float(code=='G0'),float(code in ('G81','G82','G83')),float(code=='G83'),
          float(profile['material']=='C45'),float(profile['material']=='POM'),
          math.log1p(event['feed_mm_min']),math.log1p(max(t,0)),
          float(event['dwell_s']>0),float(len(rows)),
          math.log1p(event['peck_count']),math.log1p(float(profile['peck_q_mm'])),
          math.log1p(float(profile['hole_count'])),math.log1p(event['segments']),
          math.log1p(event['feed_motion_mm'])]
        v[:len(values)]=values
        v[37]=1.;v[40]=rho*1000.;v[41]=math.log1p(float(profile['rapid_mm_min']))
        out_code.append(CODEMAP.get(code,CODEMAP['STATE']));out_feat.append(v)
        mass=vol*rho;prior.append([t,0.,mass]);mat.append(mass)
    return (np.asarray(out_code,np.int64),np.asarray(out_feat,np.float32),
            np.asarray(prior,np.float32),np.asarray(mat,np.float64))

def read_new_hole_groups(path):
    groups=defaultdict(list)
    with Path(path).open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f): groups[r['program_id']].append(r)
    out=[]
    for pid,rows in sorted(groups.items()):
        rows.sort(key=lambda r:int(r['step_index']))
        codes,feats,pri,mass,energy_physics=simulate_program(rows,FEATURE_DIM)
        target=np.asarray([[float(r['time_s']),float(r['energy_Wh']),float(r['material_g'])] for r in rows],np.float32)
        if len(target)!=len(codes):raise AssertionError(pid)
        op=rows[0]['operation']; orient=rows[0]['orientation']
        out.append({'id':pid,'mode':'cnc_drilling','rows':rows,'codes':codes,
          'feat':feats,'prior':pri,'geometry_g':mass,'target':target,
          'material_kind':'chips_removed','hole_new':True,
          'hole_energy_physics':energy_physics,'group':op+'_'+orient})
    return out

def grinding_features(rows):
    """No energy/time label columns or hidden machine coefficient accessed here."""
    ev=grinding_simulate(rows,teacher=False)
    profile=json.loads(rows[0]['initial_profile'])
    codes=[];feat=[];pri=[];geom=[]
    for e in ev:
        v=np.zeros(FEATURE_DIM,np.float32);move=e['motion_mm'];vol=e['material_removed_mm3']
        values=[math.log1p(move),math.log1p(e['feed_mm_min']/60.),math.log1p(e['spindle_rpm']),
            math.log1p(e['wheel_diameter_mm']),math.log1p(e['wheel_width_mm']),
            math.log1p(abs(e['x_after_mm'])),math.log1p(abs(e['y_after_mm'])),math.log1p(abs(e['z_after_mm'])),
            float(e['command']=='G1'),float(e['contact']),math.log1p(vol),
            math.log1p(e['chip_thickness_equiv_mm']*1e5),math.log1p(e['time_s']),
            float(profile['material']=='C45'),float(profile['material']=='304SS'),
            float(profile['material']=='CAST_IRON'),float(profile['wheel_type']=='CBN'),
            math.log1p(profile['total_depth_mm']*1000.),math.log1p(profile['passes']),
            math.log1p(profile['tracks']),float(profile['sparkout']),
            math.log1p(profile['stroke_mm']),float(profile['accel_mm_s2'])/1000.,
            math.log1p(profile['table_feed_mm_min']),math.log1p(profile['wheel_rpm']),
            math.log1p(profile['wheel_width_mm']),math.log1p(len(rows))]
        v[:len(values)]=values;v[38]=.5;v[39]=.5
        v[40]=GRINDING_DENSITY[profile['material']]*1000.
        v[41]=math.log1p(profile['rapid_mm_min'])
        codes.append(CODEMAP.get(e['command'],CODEMAP['STATE']))
        feat.append(v);pri.append([e['time_s'],0.,e['material_g']]);geom.append(e['material_g'])
    return np.asarray(codes,np.int64),np.asarray(feat,np.float32),np.asarray(pri,np.float32),np.asarray(geom,float)

def sawing_features(rows):
    """Independent sawing feature extraction: physical prior only, never label values."""
    ev=sawing_simulate(rows,teacher=False)
    profile=json.loads(rows[0]['initial_profile'])
    codes=[];feat=[];prior=[];geometry=[]
    for e in ev:
        v=np.zeros(FEATURE_DIM,dtype=np.float32)
        vals=[math.log1p(e['motion_mm']),math.log1p(e['feed_mm_min']/60.),
          math.log1p(e['band_speed_m_min']),math.log1p(e['kerf_mm']*10.),
          math.log1p(e['height_mm']),math.log1p(abs(e['z_after_mm'])),
          math.log1p(e['mean_engaged_teeth']),math.log1p(e['feed_per_tooth_mm']*10000.),
          math.log1p(e['material_removed_mm3']),float(e['contact']),
          math.log1p(e['time_s']),float(e['command']=='G1'),
          float(profile['material']=='C45'),float(profile['material']=='304SS'),
          float(profile['material']=='CAST_IRON'),float(profile['blade_type']=='CARBIDE'),
          float(profile['stock_shape']=='round'),math.log1p(profile['width_mm']),
          math.log1p(profile['tpi']),math.log1p(profile['band_speed_m_min']),
          math.log1p(profile['feed_mm_min']),math.log1p(profile['segments']),
          math.log1p(profile['equivalent_blade_mass_kg']),
          float(profile['pause_midcut']),math.log1p(len(rows)),
          float(profile['accel_mm_s2'])/1000.,
          math.log1p(profile['rapid_mm_min']),float(e['blade_acceleration_event'])]
        v[:len(vals)]=vals
        v[38]=0.2;v[39]=.8;v[40]=SAWING_DENSITY[profile['material']]*1000.
        v[41]=math.log1p(profile['rapid_mm_min'])
        codes.append(CODEMAP.get(e['command'],CODEMAP['STATE']))
        feat.append(v);prior.append([e['time_s'],0.,e['material_g']]);geometry.append(e['material_g'])
    return (np.asarray(codes,np.int64),np.asarray(feat,np.float32),
      np.asarray(prior,np.float32),np.asarray(geometry,np.float64))

def thermal_features(rows):
    """Features from process settings, G-code, and independent analytic prior only.
    No private machine coefficient / ground-truth electrical labels used.
    """
    ev=thermal_simulate(rows,teacher=False)
    profile=json.loads(rows[0]['initial_profile']);mode=profile['mode']
    codes=[];ff=[];pri=[];material=[]
    for e in ev:
        v=np.zeros(FEATURE_DIM,dtype=np.float32)
        vals=[math.log1p(e['motion_mm']),math.log1p(e['feed_mm_min']/60.),
          math.log1p(profile['laser_power_kW']*1000 if mode=='cnc_laser' else profile['current_A']),
          math.log1p(profile['thickness_mm']),math.log1p(e['kerf_mm']*100),
          math.log1p(abs(e['x_after_mm'])),math.log1p(abs(e['y_after_mm'])),
          math.log1p(abs(e['z_after_mm'])),float(e['cutting']),float(e['piercing']),
          math.log1p(e['material_removed_mm3']),math.log1p(e['time_s']),
          float(e['source_on']),float(e['gas_on']),float(mode=='cnc_plasma'),
          float(profile['material']=='C45'),float(profile['material']=='304SS'),
          float(profile['material']=='AL6061'),math.log1p(profile['contours']),
          math.log1p(profile['cut_feed_mm_min']),math.log1p(profile['pierce_delay_s']),
          math.log1p(profile['arc_voltage_V']),math.log1p(profile['wall_plug_efficiency']*100),
          math.log1p(profile['power_supply_efficiency']*100),
          math.log1p(profile['assist_gas_power_W']),math.log1p(profile['compressor_power_W']),
          math.log1p(len(rows)),float(profile['accel_mm_s2'])/1000.]
        v[:len(vals)]=vals;v[38]=.14 if mode=='cnc_laser' else .24;v[39]=.86 if mode=='cnc_laser' else .76
        v[40]=THERMAL_MAT[profile['material']][0]*1000.;v[41]=math.log1p(profile['rapid_mm_min'])
        codes.append(CODEMAP.get(e['command'],CODEMAP['STATE']))
        ff.append(v);pri.append([e['time_s'],0.,e['material_g']]);material.append(e['material_g'])
    return np.asarray(codes,np.int64),np.asarray(ff,np.float32),np.asarray(pri,np.float32),np.asarray(material,np.float64)

class ItemDataset(Dataset):
    def __init__(self,programs,mean,std):self.data=programs;self.mean=mean;self.std=std
    def __len__(self):return len(self.data)
    def __getitem__(self,i):
        p=self.data[i];return {'id':p['id'],'mode':MODE[p['mode']],'codes':p['codes'],
            'features':np.clip((p['feat']-self.mean)/self.std,-6.,6.).astype('float32'),
            'prior':p['prior'][:,:1],'target':p['target'][:,:1]}

def collate(batch):
    n=len(batch);T=max(len(z['codes']) for z in batch);D=batch[0]['features'].shape[-1]
    codes=np.zeros((n,T),np.int64);ff=np.zeros((n,T,D),np.float32)
    pri=np.zeros((n,T,1),np.float32);yy=np.zeros((n,T,1),np.float32);mask=np.zeros((n,T),bool)
    for i,z in enumerate(batch):
        k=len(z['codes']);codes[i,:k]=z['codes'];ff[i,:k]=z['features'];pri[i,:k]=z['prior'];yy[i,:k]=z['target'];mask[i,:k]=True
    return { 'ids':[b['id'] for b in batch], 'mode':torch.tensor([b['mode'] for b in batch]),
            'codes':torch.from_numpy(codes),'feat':torch.from_numpy(ff),'prior':torch.from_numpy(pri),
            'target':torch.from_numpy(yy),'mask':torch.from_numpy(mask)}

class TimeMoE(nn.Module):
    def __init__(self,d=None):
        super().__init__()
        from config import choice
        d=choice(32,48,64) if d is None else d
        ff=choice(64,96,160)
        nl=choice(2,2,3)
        e1,e2=choice((24,16),(48,32),(72,48))
        self.code=nn.Embedding(64,d,padding_idx=0);self.mode=nn.Embedding(8,d)
        self.numeric=nn.Sequential(nn.Linear(FEATURE_DIM,d),nn.LayerNorm(d));self.prior=nn.Linear(1,d)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=ff,batch_first=True,activation='gelu',dropout=.05,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,nl,enable_nested_tensor=False)
        self.gate=nn.Linear(d,10)
        # 2 shared motion experts + eight mode-specific time experts.
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(d,e1),nn.GELU(),nn.Linear(e1,e2),nn.GELU(),nn.Linear(e2,1)) for _ in range(10)])
        for e in self.experts:nn.init.zeros_(e[-1].weight);nn.init.zeros_(e[-1].bias)
    def forward(self,codes,feat,prior,mask,mode):
        h=self.code(codes)+self.mode(mode)[:,None,:]+self.numeric(feat)+self.prior(torch.log1p(prior.clamp_min(0)))
        h=self.encoder(h,src_key_padding_mask=~mask)
        logits=self.gate(h)
        possible=torch.stack([torch.ones_like(mode,dtype=torch.bool),torch.ones_like(mode,dtype=torch.bool),mode==0,mode==1,mode==2,mode==3,mode==4,mode==5,mode==6,mode==7],-1)
        gates=torch.softmax(logits.masked_fill(~possible[:,None,:],-1.e6),-1)
        deltas=torch.cat([e(h) for e in self.experts],-1)
        delta=1.2*torch.tanh((gates*deltas).sum(-1))
        t=prior[...,0]*torch.exp(delta)
        t=torch.where(codes==CODEMAP['G4'],prior[...,0],t)
        return torch.where(mask,t,torch.zeros_like(t)),gates

def time_loss(p,b):
    y=b['target'][...,0];valid=b['mask'];positive=(y>1e-6)&valid
    # Event log loss weights short movements fairly; program totals penalized separately.
    lp=(torch.log1p(p)-torch.log1p(y)).square();rel=((p-y)/(y+.2)).abs()
    event=(lp*valid).sum()/valid.sum().clamp_min(1.) + .35*(rel*positive).sum()/positive.sum().clamp_min(1.)
    actual=y.sum(1);estimated=p.sum(1)
    prog=torch.abs(torch.log1p(estimated)-torch.log1p(actual)).mean()
    return event+1.5*prog

def predict_time(model,programs,mean,std,device,batch=16):
    loader=DataLoader(ItemDataset(programs,mean,std),batch_size=batch,shuffle=False,collate_fn=collate)
    model.eval();ans={}
    with torch.inference_mode():
        for z in loader:
            ids=z.pop('ids');inp={k:v.to(device) for k,v in z.items()};pred,_=model(inp['codes'],inp['feat'],inp['prior'],inp['mask'],inp['mode']);raw=pred.cpu().numpy()
            for j,name in enumerate(ids):ans[name]=raw[j,:int(z['mask'][j].sum())].copy()
    return ans

def compute_energy_case(p,pred_t):
    rows=p['rows'];mode=p['mode'];times=np.asarray(pred_t,dtype=np.float64)
    if p.get('hole_new'):
        # Component-wise physics: fixed cutting/rubbing work NEVER scales with time.
        # No measured labels, hidden wear state or hardness index enter the prior.
        _, raw = components_by_row(rows,times,teacher=False)
        return raw
    if mode=='cnc_grinding':
        return grinding_components_by_row(rows,times)
    if mode=='cnc_sawing':
        return sawing_components_by_row(rows,times)
    if mode in ('cnc_laser','cnc_plasma'):
        return thermal_components_by_row(rows,times)
    if mode=='fdm':
        sm=fdm_summary(rows,times);bas=np.asarray(sm['base'],np.float64)
        aux=sm['regimes'];speed=sum(float(x) for x in aux['axis_distance_mm_est'])/max(times.sum(),.001)
        raw={'components':list(FDM_COMPONENTS),'base':bas,'motion_s':float(aux['motion_time_s']),
             'speed':speed,'rpm':0.,'cut_vol':0.,'tool':0.,'temp':float(json.loads(rows[0]['initial_profile']).get('nozzle_target',200)),
             'density':1.25, 'material':p['target'][:,2].sum()}
    elif mode=='cnc_milling':
        profile=json.loads(rows[0]['initial_profile']);components=CNC_COMPONENTS
        bas=np.zeros(8,np.float64);rpm=0.;on=False;coolant=False
        mot_s=0.;dists=0.;cut_vol=0.; speed_v=0.;t_total=float(times.sum());
        # Reconstruct G-code motion and process state; no teacher outputs used.
        feed=600.;pos=np.zeros(3);m='G0';absolute=True
        for row,dt in zip(rows,times):
            gc=COMMENTS.sub('',row['gcode'].upper());pairs=[(k,float(v)) for k,v in WORDS.findall(gc)];w=dict(pairs)
            for g in [v for k,v in pairs if k=='G']:
                if g in (0.,1.,2.,3.):m='G'+str(int(g))
                if g==90:absolute=True
                if g==91:absolute=False
            if 'S' in w:rpm=w['S']
            if 'F' in w:feed=w['F']
            for z in [int(v) for k,v in pairs if k=='M']:
                if z in (3,4):on=True
                if z==5:on=False
                if z==8:coolant=True
                if z==9:coolant=False
            new=pos.copy()
            for i,k in enumerate('XYZ'):
                if k in w:new[i]=w[k] if absolute else new[i]+w[k]
            motion=any(k in w for k in 'XYZ');dx,dy,dz=new-pos
            dist=math.dist(pos,new)
            if motion and m in ('G2','G3') and ('I' in w or 'J' in w):
                cx,cy=pos[0]+w.get('I',0.),pos[1]+w.get('J',0.);r=math.hypot(pos[0]-cx,pos[1]-cy)
                a0=math.atan2(pos[1]-cy,pos[0]-cx);a1=math.atan2(new[1]-cy,new[0]-cx)
                sweep=(a1-a0)%(2*math.pi) if m=='G3' else (a0-a1)%(2*math.pi)
                if sweep<1e-9:sweep=2*math.pi
                dist=math.hypot(r*sweep,dz)
            vol=0.;d=float(profile['tool_diameter_mm'])
            if motion and m=='G1' and on:
                if abs(dx)<1e-8 and abs(dy)<1e-8 and dz<0 and new[2]<0 and pos[2]>=0:
                    vol=math.pi*(d/2)**2*(-new[2])
                elif abs(dz)<1e-8 and pos[2]<0 and math.hypot(dx,dy)>0:
                    vol=d*math.hypot(dx,dy)*(-pos[2])
            pos=new
            if dt<=0:continue
            # Deliberately approximate nominal physics; separate learned experts correct it.
            bas[0]+=36.*dt/3600.;bas[1]+=20.*dt/3600.;bas[7]+=(90. if coolant else 0.)*dt/3600.
            if on:bas[5]+=(37.+.010*rpm+1.15e-6*rpm*rpm)*dt/3600.
            if motion:
                mot_s+=dt;dists+=dist;vxyz=np.abs(np.asarray([dx,dy,dz]))/dt
                for axis in range(3):
                    v=vxyz[axis]
                    if v>1e-7:bas[2+axis]+=(.46+.20*v+.0022*v*v)*dt/3600.
                bas[6]+=MATERIALS[profile['material']][1]*vol/3600.*.92
            cut_vol+=vol
        raw={'components':list(components),'base':bas,'motion_s':mot_s,'speed':dists/max(mot_s,.001),
             'rpm':float(profile['spindle_nominal_rpm']),'cut_vol':cut_vol,'tool':float(profile['tool_diameter_mm']),
             'temp':0.,'density':MATERIALS[profile['material']][0]*1000,
             'material':cut_vol*MATERIALS[profile['material']][0]}
    elif mode=='cnc_turning':
        profile=json.loads(rows[0]['initial_profile']);components=TURNING_COMPONENTS
        bas=np.zeros(len(components),np.float64);rpm=0.;on=False;cool=False
        dia=float(profile['stock_diameter_mm']);x=dia+4.;z=5.;old_d=dia
        feed=600.;motion='G0';total_s=float(times.sum());mot_s=0.;distance=0.;cut_vol=0.
        for row,dt in zip(rows,times):
            gc=COMMENTS.sub('',row['gcode'].upper());pairs=[(k,float(v)) for k,v in WORDS.findall(gc)];w=dict(pairs)
            for g in [v for k,v in pairs if k=='G']:
                if g in (0.,1.):motion='G'+str(int(g))
            if 'S' in w:rpm=w['S']
            if 'F' in w:feed=w['F']
            for m in [int(v) for k,v in pairs if k=='M']:
                if m in (3,4):on=True
                elif m==5:on=False
                elif m==8:cool=True
                elif m==9:cool=False
            nx=w.get('X',x);nz=w.get('Z',z);dx=(nx-x)/2.;dz=nz-z
            move=('X' in w or 'Z' in w);dist=math.hypot(dx,dz)
            vol=0.
            if move and motion=='G1' and on and dz<0 and z<=1e-8 and nz<0 and abs(dx)<1e-8:
                vol=math.pi/4.*max(0,old_d**2-nx**2)*(-dz)
                if vol>0:old_d=nx
            if dt>0:
                bas[0]+=43.*dt/3600;bas[1]+=21.*dt/3600.
                if move:
                    mot_s+=dt;distance+=dist
                    vx=abs(dx)/dt;vz=abs(dz)/dt
                    if vx>1e-9:bas[2]+=(.5+.19*vx+.0018*vx*vx)*dt/3600.
                    if vz>1e-9:bas[3]+=(.66+.22*vz+.002* vz*vz)*dt/3600.
                if on:bas[4]+=(125+.025*rpm+9e-6*rpm*rpm)*dt/3600.
                bas[5]+=float(profile['cutting_specific_energy_J_mm3'])*vol*.91/3600.
                if cool:bas[6]+=76.*dt/3600.
                if on:bas[7]+=24.*dt/3600.
            cut_vol+=vol;x,z=nx,nz
        raw={'components':list(components),'base':bas,'motion_s':mot_s,'speed':distance/max(mot_s,1e-3),
          'rpm':float(profile['spindle_nominal_rpm']),'cut_vol':cut_vol,'tool':float(profile['pass_depth_radial_mm']),
          'temp':0.,'density':MATERIALS[profile['material']][0]*1000.,
          'material':cut_vol*MATERIALS[profile['material']][0]}
    elif mode=='cnc_drilling':
        profile=json.loads(rows[0]['initial_profile']);components=CNC_COMPONENTS
        # Separate approximate model, not teacher times or teacher power components.
        events=drilling_simulate([r['gcode'] for r in rows],profile,accel=490.,rapid=6000.,
            clearance=.30,base_power=38.,hold_power=18.,spindle_base=47.,
            spindle_linear=.0105,cutting_factor=.92,coolant_power=77.,axis_factor=.94)
        bas=np.zeros(len(components),np.float64)
        cut_vol=0.;distance=0.;mot_s=0.
        for e,t in zip(events,times):
            original=e['time_s'];ratio=t/max(original,1e-8) if original>0 else 0.
            for i,name in enumerate(components):
                energy=e['energy_components'][name]
                # Cut energy depends on volume, while other hardware energy follows time.
                bas[i]+=energy if name=='cutting_load' else energy*ratio
            cut_vol+=e['material_removed_mm3'];distance+=e['motion_length_mm'];
            if e['motion_length_mm']>0:mot_s+=float(t)
        raw={'components':list(components),'base':bas,'motion_s':mot_s,
          'speed':distance/max(mot_s,1.e-3),
          'rpm':float(profile['spindle_nominal_rpm']),'cut_vol':cut_vol,
          'tool':float(profile['tool_diameter_mm']),'temp':0.,
          'density':MATERIALS[profile['material']][0]*1000.,
          'material':cut_vol*MATERIALS[profile['material']][0]}
    else:raise ValueError(mode)
    raw['total_s']=float(times.sum());raw['commands']=len(rows);raw['mode']=mode
    return raw

class EnergyMoE(nn.Module):
    def __init__(self,components):
        super().__init__()
        from config import choice
        self.components=list(components);self.idx={k:i for i,k in enumerate(components)}
        emb=choice(8,12,16)
        b1,b2=choice((32,24),(48,32),(64,48))
        self.cembed=nn.Embedding(len(components),emb);self.membed=nn.Embedding(8,emb)
        self.body=nn.Sequential(nn.Linear(15+2*emb,b1),nn.GELU(),nn.Linear(b1,b2),nn.GELU())
        self.gate=nn.Linear(b2,9)
        shr=choice(16,24,32)
        self.shared=nn.Sequential(nn.Linear(b2,shr),nn.GELU(),nn.Linear(shr,1))
        widths=choice([(16,),(32,20),(16,),(36,24),(32,20),(32,20),(32,20),(32,20)],
                      [(24,),(48,32),(24,),(64,40),(56,36),(56,32),(56,32),(56,32)],
                      [(40,),(80,48),(40,),(96,64),(88,56),(88,48),(88,48),(88,48)])
        def head(widths):
            sizes=[b2,*widths,1];layers=[]
            for a,b in zip(sizes,sizes[1:]):layers.append(nn.Linear(a,b));
            # final activation must NOT be applied to output
            seq=[]
            for i,l in enumerate(layers):
                seq.append(l)
                if i+1<len(layers):seq.append(nn.GELU())
            return nn.Sequential(*seq)
        self.mode_experts=nn.ModuleList([head(w) for w in widths])
        for x in [self.shared]+list(self.mode_experts):nn.init.zeros_(x[-1].weight);nn.init.zeros_(x[-1].bias)
    def forward(self,x,component_idx,mode,b):
        h=self.body(torch.cat([x,self.cembed(component_idx),self.membed(mode)],-1))
        gate=self.gate(h)
        permit=torch.stack([torch.ones_like(mode,dtype=torch.bool),mode==0,mode==1,mode==2,mode==3,mode==4,mode==5,mode==6,mode==7],-1)
        w=torch.softmax(gate.masked_fill(~permit,-1.e6),-1)
        values=torch.cat([self.shared(h)]+[e(h) for e in self.mode_experts],-1)
        delta=.5*torch.tanh((w*values).sum(-1))
        return b*(1+delta)

def calibrate_milling_components(prepared, programs):
    """Fit positive component-level physical-prior gains on TRAIN programs only.

    Observations are synthetic per-component Wh labels already supplied in the
    training data. This calibrates the physical prior; it never reads validation
    or test labels, and is not a machine-power calibration.
    """
    a=prepared
    groups=np.asarray([p['mode'] for p in programs])
    gains={}
    for name in CNC_COMPONENTS:
        ix=np.asarray([groups[j]=='cnc_milling' and n==name
                       for j,n in zip(a['job'],a['names'])],dtype=bool)
        b=a['base'].numpy()[ix].astype('float64')
        y=a['target'].numpy()[ix].astype('float64')
        # Multiplicative least squares with a weak prior at gain=1; positives only.
        denom=np.dot(b,b)
        shrink=1e-5*denom
        gain=(np.dot(b,y)+shrink)/(denom+shrink) if denom>1e-15 else 1.
        gains[name]=float(np.clip(gain,0.5,1.5))
    return gains

def apply_milling_calibration(case, calibration):
    """Same component correction during training and production inference."""
    if not calibration or case['mode']!='cnc_milling':
        return case
    z=dict(case)
    z['base']=np.asarray(case['base'],dtype=np.float64)*np.asarray(
        [calibration.get(name,1.) for name in case['components']],dtype=np.float64)
    return z

def calibrate_drilling_components(prepared, programs):
    """Train-only, operation-group-specific physical calibration for drilling.

    Only observed component energies in TRAIN are used. The old vertical and
    horizontal drill groups use independent corrections; other operations
    remain unchanged. Regularization shrinks noisy coefficients towards 1.
    """
    gains = {}
    for group in ('drilling_horizontal', 'drilling_vertical_old'):
        ixprog = {j for j,p in enumerate(programs) if p['group'] == group}
        if not ixprog:
            continue
        gains[group] = {}
        for name in HOLE_COMPONENTS:
            ix = np.asarray([j in ixprog and n == name
                for j,n in zip(prepared['job'],prepared['names'])],dtype=bool)
            b = prepared['base'].numpy()[ix].astype('float64')
            y = prepared['target'].numpy()[ix].astype('float64')
            denom = float(np.dot(b,b))
            shrink = 1.e-4 * denom
            gain = (float(np.dot(b,y)) + shrink)/(denom + shrink) if denom>1.e-15 else 1.
            gains[group][name] = float(np.clip(gain,0.65,1.50))
    return gains


def apply_energy_calibration(case, group, milling_calibration=None, drilling_calibration=None):
    """Identical train/inference component scaling; never adjusts target labels."""
    out = apply_milling_calibration(case, milling_calibration)
    if out['mode'] != 'cnc_drilling' or not drilling_calibration or group not in drilling_calibration:
        return out
    gains = drilling_calibration[group]
    corrected = dict(out)
    corrected['base'] = np.asarray(out['base'],dtype='float64') * np.asarray(
        [gains.get(name,1.) for name in out['components']],dtype='float64')
    return corrected


def energy_features(z,name):
    b=float(z['base'][z['components'].index(name)])
    return np.asarray([math.log1p(max(b,0)),math.log1p(z['total_s']),math.log1p(z['motion_s']),
        math.log1p(max(z['speed'],0)),math.log1p(z['rpm']),math.log1p(z['cut_vol']),
        math.log1p(z['tool']),z['temp']/250.,z['density']/10.,
        math.log1p(z['commands']),z['motion_s']/max(z['total_s'],1e-8),
        float(name in ('electronics','stepper_hold','axis_hold')),
        float('motor' in name or 'axis_' in name),float('heater' in name),float('spindle' in name or name=='cutting_load')],np.float32)

def machine_adjusted_case(case, sensor_history=None, machine_expert=None, group=None):
    """Predict component corrections using ONLY prior power probe readings, no labels.
    Disabled exactly if no sensor history is passed or model is None.
    """
    if case['mode'] not in ('cnc_drilling','cnc_grinding','cnc_sawing','cnc_laser','cnc_plasma') or sensor_history is None or machine_expert is None:
        return case
    n=len(case['components'])
    hist=np.asarray(sensor_history,dtype=np.float32)
    if hist.shape!=(HISTORY_DIM,) or not np.isfinite(hist).all():
        raise ValueError('Machine history must be six finite prior power measurements in W')
    params=list(machine_expert.parameters());dev=params[0].device
    machine_expert.eval()
    with torch.inference_mode():
        factors=machine_expert(torch.from_numpy(np.tile(hist,(n,1))).to(dev),
             (torch.arange(n,dtype=torch.long,device=dev)+(12 if group=='grinding_surface' else 24 if group=='sawing_band' else 36 if group=='laser_sheet' else 48 if group=='plasma_sheet' else 0)),
             torch.ones(n,dtype=torch.bool,device=dev),
             torch.full((n,),MACHINE_GROUPS.index(group),dtype=torch.long,device=dev)).cpu().numpy()
    out=dict(case)
    out['base']=np.asarray(case['base'],dtype=np.float64)*factors
    out['machine_history_used']=True
    return out

def build_energy_arrays(programs,pred_time,component_names,normalizer=None,calibration=None,drilling_calibration=None,machine_expert=None,machine_history=None):
    rows=[];job_index=[];component_base=[];targets=[];features=[];indices=[];modes=[];names=[]
    for j,p in enumerate(programs):
        info=apply_energy_calibration(compute_energy_case(p,pred_time[p['id']]),p.get('group',p['mode']),calibration,drilling_calibration)
        if p.get('group') in MACHINE_GROUPS:
            info=machine_adjusted_case(info,(machine_history or {}).get(p['id']),machine_expert,p['group'])
        rows.append(info)
        for name in info['components']:
            feats=energy_features(info,name);bas=float(info['base'][info['components'].index(name)])
            target=sum(float(r[f'energy_{name}_Wh']) for r in p['rows'])
            features.append(feats);component_base.append(bas);targets.append(target)
            indices.append(component_names.index((p['mode'],name)));modes.append(MODE[p['mode']]);job_index.append(j);names.append(name)
    feature=np.asarray(features,np.float32);base=np.asarray(component_base,np.float32);y=np.asarray(targets,np.float32)
    if normalizer is None:
        mn=feature.mean(0);sd=np.maximum(feature.std(0),.05);normalizer=(mn,sd)
    mn,sd=normalizer;fn=np.clip((feature-mn)/sd,-8,8).astype('float32')
    return {'data':rows,'raw':feature,'feat':torch.from_numpy(fn),'base':torch.from_numpy(base),
            'target':torch.from_numpy(y),'ci':torch.tensor(indices),'mode':torch.tensor(modes),
            'job':np.asarray(job_index),'names':names,'normalizer':normalizer}

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);idx=y>1e-9;d=p-y
    return {'N':len(y),'positive_n':int(idx.sum()),'MSE':float(np.mean(d*d)),
        'RMSE':float(np.sqrt(np.mean(d*d))),'MAE':float(np.mean(np.abs(d))),
        'R2':float(1-np.sum(d*d)/np.sum((y-y.mean())**2)) if np.sum((y-y.mean())**2)>1e-12 else None,
        'MAPE_pct':float(100*np.mean(np.abs(d[idx]/y[idx]))) if idx.any() else None,
        'WAPE_pct':float(100*np.sum(np.abs(d[idx]))/np.sum(np.abs(y[idx]))) if idx.any() else None}

def sum_by_job(arr_values,job_idx,njobs):
    out=np.zeros(njobs,np.float64);np.add.at(out,job_idx,arr_values);return out

def energy_predict(model,a,dev):
    model.eval()
    with torch.inference_mode():return model(a['feat'].to(dev),a['ci'].to(dev),a['mode'].to(dev),a['base'].to(dev)).cpu().numpy()

def fit_machine_expert(raw_prepared,split_programs,machine_history,device,epochs=300):
    """Shared latent expert: train component and job Wh for 9 machine-characterization groups.

    Input: six pre-job measured power readings, component ID, process orientation.
    No gain parameter or target-job energy is supplied as a predictive feature.
    Validation selects checkpoint on the average of 7 group job-MAPEs.
    """
    sensors={}
    for split in ('train','val','test'):
        progs=split_programs[split];a=raw_prepared[split]
        hist=np.zeros((len(a['job']),HISTORY_DIM),np.float32)
        available=np.zeros(len(a['job']),bool)
        ci=np.zeros(len(a['job']),np.int64)
        gi=np.zeros(len(a['job']),np.int64)
        for k,(job,name) in enumerate(zip(a['job'],a['names'])):
            p=progs[job];group=p.get('group')
            if group not in MACHINE_GROUPS or p['id'] not in machine_history[split]:continue
            hist[k]=machine_history[split][p['id']];available[k]=True
            ci[k]=(list(THERMAL_COMPONENTS['cnc_laser']).index(name)+36 if group=='laser_sheet' else list(THERMAL_COMPONENTS['cnc_plasma']).index(name)+48 if group=='plasma_sheet' else list(SAWING_COMPONENTS).index(name)+24 if group=='sawing_band' else list(GRINDING_COMPONENTS).index(name)+12 if group=='grinding_surface' else list(HOLE_COMPONENTS).index(name));gi[k]=MACHINE_GROUPS.index(group)
        sensors[split]={'history':hist,'mask':available,'ci':ci,'gi':gi}
        sensors[split]['matched_programs']=len({progs[job]['id'] for job,valid in zip(a['job'],available) if valid})
    means=[];stds=[]
    for group in MACHINE_GROUPS:
        ix=sensors['train']['mask'] & (sensors['train']['gi']==MACHINE_GROUPS.index(group))
        vv=sensors['train']['history'][ix]
        means.append(vv.mean(axis=0));stds.append(np.maximum(vv.std(axis=0),1.0))
    mean=np.stack(means).astype('float32');std=np.stack(stds).astype('float32')
    expert=MachineCharacterizationExpert().to(device);expert.set_scaler(mean,std)
    opt=torch.optim.AdamW(expert.parameters(),lr=.004,weight_decay=.001)
    selected=float('inf');best_state=None;best_epoch=0;patience=55;wait=0;history=[]
    # All component/job targets remain in the loss. Only identical machine-probe
    # forward passes are memoized within an epoch (6 jobs / machine share probes).
    # This is algebraically identical to the full uncompressed loss, including
    # gradients, and avoids evaluating the same neural encoder 6x per machine.
    def tensors(split):
        a=raw_prepared[split];q=sensors[split];mask=q['mask']
        h=q['history'][mask];ci=q['ci'][mask];gi=q['gi'][mask]
        keys=np.column_stack([h,ci.astype('float32'),gi.astype('float32')])
        unique,inv=np.unique(keys,axis=0,return_inverse=True)
        return (torch.from_numpy(unique[:,:HISTORY_DIM].astype('float32')).to(device),
            torch.from_numpy(unique[:,HISTORY_DIM].astype('int64')).to(device),
            torch.from_numpy(unique[:,HISTORY_DIM+1].astype('int64')).to(device),
            torch.from_numpy(inv.astype('int64')).to(device),
            a['base'][mask].to(device),a['target'][mask].to(device),
            torch.from_numpy(a['job'][mask]).to(device))
    train_data=tensors('train');val_data=tensors('val')
    def predictions(data,split):
        h,ci,gi,inv,b,y,job=data
        factors=expert(h,ci,torch.ones(h.shape[0],dtype=torch.bool,device=device),gi)
        v=b*factors[inv]
        pj=torch.zeros(len(split_programs[split]),device=device).index_add(0,job,v)
        yj=torch.zeros(len(split_programs[split]),device=device).index_add(0,job,y)
        return v,y,pj,yj
    val_ids={g:torch.tensor([i for i,p in enumerate(split_programs['val']) if p['group']==g],dtype=torch.long,device=device) for g in MACHINE_GROUPS}
    train_ids={g:torch.tensor([i for i,p in enumerate(split_programs['train']) if p['group']==g],dtype=torch.long,device=device) for g in MACHINE_GROUPS}
    for ep in range(1,epochs+1):
        expert.train();opt.zero_grad(set_to_none=True)
        pred,y,pj,yj=predictions(train_data,'train')
        # Balance groups rather than allowing one operation with high Wh to dominate.
        group_losses=[]
        for group in MACHINE_GROUPS:
            ix=train_ids[group]
            if len(ix)==0:continue
            group_losses.append(torch.nn.functional.smooth_l1_loss(torch.log1p(pj[ix]),torch.log1p(yj[ix])))
        comp=torch.nn.functional.smooth_l1_loss(torch.log1p(pred*10),torch.log1p(y*10))
        loss=.30*comp+4.*sum(group_losses)/len(group_losses)
        loss.backward();torch.nn.utils.clip_grad_norm_(expert.parameters(),2.);opt.step()
        expert.eval()
        with torch.inference_mode():
            _pred,_y,pj,yj=predictions(val_data,'val')
            group_mapes={}
            for group in MACHINE_GROUPS:
                ix=val_ids[group]
                group_mapes[group]=metrics(yj[ix].cpu().numpy(),pj[ix].cpu().numpy())['MAPE_pct']
            sc=float(np.mean(list(group_mapes.values())))
        history.append({'epoch':ep,'train_loss':float(loss.detach()),'val_mean_group_MAPE_pct':sc})
        if sc<selected-1e-6:selected=sc;best_state=copy.deepcopy(expert.state_dict());best_epoch=ep;wait=0
        else:wait+=1
        if ep==1 or ep%40==0:print('MACHINE EXPERT',ep,'train_loss',round(float(loss.detach()),6),'val mean MAPE',round(sc,4),flush=True)
        if wait>=patience:break
    expert.load_state_dict(best_state);expert.eval()
    return expert,{'best_epoch':best_epoch,'val_mean_group_MAPE_pct':selected,'history':history,
         'sensor_names':list(__import__('machine_characterization').SENSOR_NAMES),
         'groups':MACHINE_GROUPS,'matched_programs':{s:q['matched_programs'] for s,q in sensors.items()},
         'training_target':'component and job energy, no private machine efficiency labels'}

def load_machine_history(root,split):
    all_history={}
    for filename in (f'{split}.json',f'grinding_{split}.json',f'sawing_{split}.json',f'laser_{split}.json',f'plasma_{split}.json'):
        d=json.loads((root/'data/machine_history'/filename).read_text())
        all_history.update({pid:d['machine_measurements'][key] for pid,key in d['program_to_machine'].items()})
    return all_history

def run(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(1)
    dev='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    if dev=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True); tstart=time.time()
    modes=['fdm','cnc_milling','cnc_turning','cnc_drilling','cnc_grinding','cnc_sawing','cnc_laser','cnc_plasma'];splits={}
    for s in ['train','val','test']:
        splits[s]=sum((read_groups(ROOT/'data'/({'fdm':'fdm','cnc_milling':'milling','cnc_turning':'turning'}[m])/(s+'.csv'),m) for m in modes if m not in ('cnc_drilling','cnc_grinding','cnc_sawing','cnc_laser','cnc_plasma')),[])
        splits[s] += read_new_hole_groups(ROOT/'data'/'new'/(s+'.csv'))
        grinds=read_groups(ROOT/'data'/'grinding'/(s+'.csv'),'cnc_grinding')
        for gp in grinds:gp['group']='grinding_surface'
        splits[s] += grinds
        saws=read_groups(ROOT/'data'/'sawing'/(s+'.csv'),'cnc_sawing')
        for saw in saws:saw['group']='sawing_band'
        splits[s] += saws
        for kind in ('laser','plasma'):
            progs=read_groups(ROOT/'data'/kind/(s+'.csv'),'cnc_'+kind)
            for pp in progs:pp['group']=kind+'_sheet'
            splits[s]+=progs
        # Original drilling time parser and Transformer features are retained exactly.
        # Only energy labels and energy hardware prior are upgraded to 12 components.
        for pp in read_groups(ROOT/'data'/'drilling_energy'/(s+'.csv'),'cnc_drilling'):
            pp['group']='drilling_vertical_old'
            pp['hole_new']=True
            splits[s].append(pp)
        for p in splits[s]:p.setdefault('group',p['mode'] if p['mode']!='cnc_drilling' else 'drilling_vertical_old')
        assert len({p['id'] for p in splits[s]})==len(splits[s])
    # Smoke-only deterministic per-operation subsample. Never enabled in full training.
    if args.smoke_max_per_group>0:
        for split_name in ('train','val','test'):
            groups_seen={};reduced=[]
            for pp in splits[split_name]:
                grp=pp['group'];cnt=groups_seen.get(grp,0)
                if cnt<args.smoke_max_per_group:
                    reduced.append(pp);groups_seen[grp]=cnt+1
            splits[split_name]=reduced
        print('SMOKE SUBSET (not publishable)',{s:len(splits[s]) for s in splits},flush=True)
    for a,b in [('train','val'),('train','test'),('val','test')]:assert {z['id'] for z in splits[a]}.isdisjoint({z['id'] for z in splits[b]})
    train=splits['train'];feat=np.concatenate([p['feat'] for p in train]);mean=feat.mean(0);std=np.maximum(feat.std(0),.05)
    groups_train=[p['group'] for p in splits['train']]
    counts={g:groups_train.count(g) for g in set(groups_train)}
    sampler=torch.utils.data.WeightedRandomSampler([1.0/counts[g] for g in groups_train],len(groups_train),replacement=True)
    loaders={s:DataLoader(ItemDataset(splits[s],mean,std),batch_size=args.batch,
             sampler=sampler if s=='train' else None,shuffle=False,collate_fn=collate) for s in splits}
    timemodel=TimeMoE().to(dev);optim=torch.optim.AdamW(timemodel.parameters(),lr=args.lr,weight_decay=.0001)
    best=float('inf');best_state=None;best_epoch=0;wait=0;history=[]
    for ep in range(1,args.epochs_time+1):
        timemodel.train();losses=[]
        for b in loaders['train']:
            b={k:v.to(dev) for k,v in b.items() if isinstance(v,torch.Tensor)}
            optim.zero_grad(set_to_none=True);pred,_=timemodel(b['codes'],b['feat'],b['prior'],b['mask'],b['mode']);loss=time_loss(pred,b)
            loss.backward();nn.utils.clip_grad_norm_(timemodel.parameters(),1.);optim.step();losses.append(float(loss.detach()))
        timemodel.eval();val_losses=[]
        with torch.inference_mode():
            for b in loaders['val']:
                b={k:v.to(dev) for k,v in b.items() if isinstance(v,torch.Tensor)}
                pred,_=timemodel(b['codes'],b['feat'],b['prior'],b['mask'],b['mode']);val_losses.append(float(time_loss(pred,b)))
        score=float(np.mean(val_losses));history.append({'epoch':ep,'train':float(np.mean(losses)),'val':score})
        if score<best-1e-7:best=score;best_state=copy.deepcopy(timemodel.state_dict());best_epoch=ep;wait=0
        else:wait+=1
        if ep==1 or ep%10==0:print(f'TIME epoch {ep}: train={np.mean(losses):.6f} val={score:.6f}',flush=True)
        if wait>=args.patience:break
    timemodel.load_state_dict(best_state)
    # Conservatively select time residual strength from validation separately for each process.
    time_neural={s:predict_time(timemodel,splits[s],mean,std,dev) for s in splits}
    time_final={};time_blends={}
    all_groups=sorted(set(p['group'] for p in splits['val']))
    for mode in all_groups:
        valprogs=[p for p in splits['val'] if p['group']==mode]
        def time_score(alpha):
            truth=np.array([p['target'][:,0].sum() for p in valprogs]);pred=np.array([((1-alpha)*p['prior'][:,0]+alpha*time_neural['val'][p['id']]).sum() for p in valprogs])
            return metrics(truth,pred)['MAPE_pct']
        alphas=[0.,.25,.5,.75,1.];chosen=min(alphas,key=time_score);time_blends[mode]=chosen
        print('TIME blend',mode,chosen,'val program MAPE',round(time_score(chosen),5),flush=True)
    for s in splits:
        time_final[s]={p['id']:((1-time_blends[p['group']])*p['prior'][:,0]+time_blends[p['group']]*time_neural[s][p['id']]) for p in splits[s]}
    components=[(m,n) for m in modes for n in MODES_COMPONENTS[m]]
    # Derive component-wise gains on TRAIN ONLY before fitting the energy MoE.
    uncalibrated_train=build_energy_arrays(splits['train'],time_final['train'],components)
    milling_calibration=calibrate_milling_components(uncalibrated_train,splits['train'])
    drilling_calibration=calibrate_drilling_components(uncalibrated_train,splits['train'])
    print('MILLING train-only prior calibration',milling_calibration,flush=True)
    print('DRILLING train-only prior calibration',drilling_calibration,flush=True)
    # Train a shared machine expert for 7 hole-operation groups ONLY from
    # previously observed power probes, not any gain parameter.
    # Raw component-energy labels provide the training loss, as for EnergyMoE.
    machine_history={s:load_machine_history(ROOT,s) for s in splits}
    raw_prepared={'train':build_energy_arrays(splits['train'],time_final['train'],components,
        calibration=milling_calibration,drilling_calibration=drilling_calibration)}
    raw_scaler=raw_prepared['train']['normalizer']
    for split_name in ('val','test'):
        raw_prepared[split_name]=build_energy_arrays(splits[split_name],time_final[split_name],components,
            raw_scaler,calibration=milling_calibration,drilling_calibration=drilling_calibration)
    machine_expert,machine_report=fit_machine_expert(raw_prepared,splits,machine_history,dev,args.epochs_machine)
    prepared={'train':build_energy_arrays(splits['train'],time_final['train'],components,
        calibration=milling_calibration,drilling_calibration=drilling_calibration,
        machine_expert=machine_expert,machine_history=machine_history['train'])}
    scaler=prepared['train']['normalizer']
    for s in ('val','test'):
        prepared[s]=build_energy_arrays(splits[s],time_final[s],components,scaler,
            calibration=milling_calibration,drilling_calibration=drilling_calibration,
            machine_expert=machine_expert,machine_history=machine_history[s])
    energymodel=EnergyMoE(components).to(dev);optimizer=torch.optim.AdamW(energymodel.parameters(),lr=args.lr_energy,weight_decay=.003)
    best=float('inf');best_state=None;best_e=0;wait=0;ehistory=[]
    for ep in range(1,args.epochs_energy+1):
        energymodel.train();a=prepared['train'];optimizer.zero_grad(set_to_none=True)
        pred=energymodel(a['feat'].to(dev),a['ci'].to(dev),a['mode'].to(dev),a['base'].to(dev))
        y=a['target'].to(dev);jobidx=torch.from_numpy(a['job']).to(dev)
        predjob=torch.zeros(len(splits['train']),device=dev).index_add(0,jobidx,pred)
        yjob=torch.zeros(len(splits['train']),device=dev).index_add(0,jobidx,y)
        # Component-relative loss + program relative loss, no individual-command Wh target.
        eligible=y>1.e-5
        lcomp=F.smooth_l1_loss(torch.log1p(pred[eligible]*10),torch.log1p(y[eligible]*10)) if eligible.any() else pred.mean()*0
        ljob=F.smooth_l1_loss(torch.log1p(predjob),torch.log1p(yjob))
        # Give each mode equal influence despite the ~8x difference in job Wh.
        mm=torch.tensor([MODE[q['mode']] for q in splits['train']],device=dev)
        mode_loss=[F.smooth_l1_loss(torch.log1p(predjob[mm==mi]),torch.log1p(yjob[mm==mi])) for mi in range(len(modes))]
        loss=.45*lcomp+2.0*sum(mode_loss)/len(mode_loss);loss.backward();nn.utils.clip_grad_norm_(energymodel.parameters(),1.);optimizer.step()
        energymodel.eval()
        with torch.inference_mode():
            b=prepared['val'];p=energymodel(b['feat'].to(dev),b['ci'].to(dev),b['mode'].to(dev),b['base'].to(dev))
            pv=sum_by_job(p.cpu().numpy(),b['job'],len(splits['val']))
            yr=np.array([q['target'][:,1].sum() for q in splits['val']]);modes_val=np.array([q['mode'] for q in splits['val']])
            # Equal-mode validation selection avoids ignoring CNC's smaller Wh scale.
            score=sum(metrics(yr[modes_val==m],pv[modes_val==m])['WAPE_pct'] for m in modes)/len(modes)
        ehistory.append({'epoch':ep,'train_loss':float(loss.detach()),'val_job_wape_pct':score})
        if score<best-1.e-8:best=score;best_state=copy.deepcopy(energymodel.state_dict());best_e=ep;wait=0
        else:wait+=1
        if ep==1 or ep%20==0:print(f'ENERGY epoch {ep}: train={loss.item():.6f} val_WAPE={score:.5f}%',flush=True)
        if wait>=args.patience_energy:break
    energymodel.load_state_dict(best_state)
    energy_neural={s:energy_predict(energymodel,prepared[s],dev) for s in splits}
    blends={}
    for mode in all_groups:
        v=prepared['val'];progs=splits['val'];ids=[i for i,p in enumerate(progs) if p['group']==mode]
        truth=np.array([progs[i]['target'][:,1].sum() for i in ids]);
        baseline=sum_by_job(v['base'].numpy(),v['job'],len(progs))[ids];neural=sum_by_job(energy_neural['val'],v['job'],len(progs))[ids]
        # The horizontal drilling teacher contains an unobserved, per-machine
        # gain. Constrain neural extrapolation to 12.5% of the energy prior;
        # this stability cap is fixed before validation and does not inspect test.
        # For drilling, user target is program-level MAPE, not aggregate WAPE.
        if mode == 'drilling_horizontal':
            weights = [0.,.125,.25,.5,.75,1.]
        elif mode == 'drilling_vertical_old':
            weights = [0.,.125,.25,.5,.75,1.]
        else:
            weights = [0.,.25,.5,.75,1.]
        criterion = 'MAPE_pct' if mode.startswith('drilling_') else 'WAPE_pct'
        choice = min(weights,key=lambda alpha:metrics(truth,baseline*(1-alpha)+neural*alpha)[criterion])
        blends[mode]=choice
        print('ENERGY blend',mode,choice,'val',criterion,metrics(truth,baseline*(1-choice)+neural*choice)[criterion],flush=True)
    results={}
    for s in splits:
        progs=splits[s];arr=prepared[s];byid={p['id']:i for i,p in enumerate(progs)};component_pred=arr['base'].numpy().copy();neur=energy_neural[s]
        for j,p in enumerate(progs):
            ix=arr['job']==j;alpha=blends[p['group']];component_pred[ix]=(1-alpha)*component_pred[ix]+alpha*neur[ix]
        jobenergy=sum_by_job(component_pred,arr['job'],len(progs));
        predictions=[]
        for j,p in enumerate(progs):
            predicted={'split':s,'mode':p['mode'],'operation_group':p['group'],'program_id':p['id'],'true_time_s':float(p['target'][:,0].sum()),
              'pred_time_s':float(np.sum(time_final[s][p['id']])),
              'true_energy_Wh':float(p['target'][:,1].sum()),'pred_energy_Wh':float(jobenergy[j]),
              'true_material_g':float(p['target'][:,2].sum()),'pred_material_g':float(p['geometry_g'].sum()),
              'material_definition':p['material_kind']}
            predictions.append(predicted)
        results[s]=predictions
        with (out/f'{s}_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    scores={}
    for s in ['val','test']:
        scores[s]={}
        for mode in modes:
            q=[p for p in results[s] if p['mode']==mode]
            scores[s][mode]={
                'program_count':len(q),
                'time_s':metrics([p['true_time_s'] for p in q],[p['pred_time_s'] for p in q]),
                'energy_Wh':metrics([p['true_energy_Wh'] for p in q],[p['pred_energy_Wh'] for p in q]),
                'material_g':metrics([p['true_material_g'] for p in q],[p['pred_material_g'] for p in q]),
                'material_type':q[0]['material_definition']}
    grouped_scores={}
    for split_name in ('val','test'):
        grouped_scores[split_name]={}
        for group in all_groups:
            qq=[p for p in results[split_name] if p['operation_group']==group]
            grouped_scores[split_name][group]={
              'program_count':len(qq),
              'time_s':metrics([z['true_time_s'] for z in qq],[z['pred_time_s'] for z in qq]),
              'energy_Wh':metrics([z['true_energy_Wh'] for z in qq],[z['pred_energy_Wh'] for z in qq]),
              'material_g':metrics([z['true_material_g'] for z in qq],[z['pred_material_g'] for z in qq])}
    stats={s:{m:{'programs':sum(p['mode']==m for p in splits[s]),'commands':sum(len(p['rows']) for p in splits[s] if p['mode']==m)} for m in modes} for s in splits}
    report={'model':'Manufacturing-MoE conventional-operation experts, trained from scratch','source':'from scratch; no pretrained checkpoints loaded; physics-informed priors and MCE training',
        'data_provenance':{'fdm':'approximate independent Marlin-style synthetic labels, NOT Marlin firmware / physical printer',
          'cnc_milling':'custom analytic single-block surrogate labels, NOT LinuxCNC simulator / real machine',
          'cnc_turning':'custom analytic turning surrogate labels with G7/G18 syntax; NOT LinuxCNC runtime / real machine',
          'cnc_drilling':'180 ORIGINAL vertical drilling programs: G-code/time/chips retained, energy re-labelled as 12 physical components; 1260 added hole programs likewise 12 components; NOT LinuxCNC runtime or real machine',
          'cnc_grinding':'564 new reciprocating surface-grinding programs; 12 independent hardware/process energy components; approximate analytic blockwise teacher, NOT LinuxCNC or measured data',
          'cnc_sawing':'564 metal band-saw programs; analytic synthetic teacher',
          'cnc_laser':'564 planar laser cutting programs; wall-plug efficiency/chiller/gas/optical source; synthetic analytical teacher, no real laser measurements',
          'cnc_plasma':'564 planar plasma cutting programs; arc supply/compressor/gas/torch; synthetic analytical teacher, no real plasma measurements'},
        'split':stats,'grouped_scores':grouped_scores,'best_epochs':{'time':best_epoch,'energy':best_e},
        'params':{'time':sum(p.numel() for p in timemodel.parameters()),'energy':sum(p.numel() for p in energymodel.parameters()),'machine':sum(p.numel() for p in machine_expert.parameters())},
        'chosen_blends_validation_only':{'time':time_blends,'energy':blends},'scores':scores,
        'milling_energy_train_only_component_gains':milling_calibration,
        'drilling_energy_train_only_component_gains_by_group':drilling_calibration,
        'machine_characterization':machine_report,
        'training_configuration_notes':['Extend MCE from 55 to 330 maximum epochs; EnergyMoE from 12 to 65; retain existing network architecture and physics','Preserve validation-only best epoch, early stopping, component gates, sensor scaling, and blend selection'],
        'drilling_blend_robustness':{'horizontal_max_neural_fraction':1.0,'selection_target':'validation MAPE','reason':'independent past telemetry replaces hidden gain'},
        'notes':['Machine characterization is fitted using prior SYNTHETIC power probes, NOT measured sensors; no gain input','Unseen machine GROUPS are split-disjoint, but test design and synthetic label formula have been previously inspected','Train-only normalization/calibration and validation-selected final blends; test set was inspected during iterative development and is not an untouched holdout',
           'Drilling blends chosen on validation MAPE; independent synthetic prior probe for all seven hole-operation orientations',
           'FDM material means filament consumption; milling/turning/drilling material means chips removed',
           'Geometric material output is deterministic; no neural material expert is trained',
           'No tool-wear model or hardness index is trained','No real-data adaptation validated','Synthetic teacher physics is correlated with inference physics; low synthetic errors do not establish physical accuracy'],
        'training_elapsed_s':time.time()-tstart,'time_history':history,'energy_history':ehistory}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    torch.save({'time':timemodel.cpu().state_dict(),'energy':energymodel.cpu().state_dict(),
      'time_scaler':(mean,std),'energy_scaler':scaler,'components':components,
      'milling_energy_calibration':milling_calibration,
      'drilling_energy_calibration':drilling_calibration,
      'machine_expert':machine_expert.cpu().state_dict(),
      'machine_sensor_names':machine_report['sensor_names'],
      'time_blends':time_blends,'energy_blends':blends,
      'config':{'FEATURE_DIM':FEATURE_DIM,'time_arch':'shared 2 Transformer layers 48x4, two shared and eight process experts',
        'energy_arch':'Eight process-conditioned hardware heads; shared 10-d machine expert for hole, grinding, sawing, laser and plasma; no wear heads'},
      'provenance':report['data_provenance'],'version':'manufacturing-normal-energy-machine-characterization-no-wear'},out/'model.pt')
    print('FINISHED',json.dumps({'epochs':report['best_epochs'],'scores':scores,'runtime_seconds':report['training_elapsed_s']},indent=2),flush=True)
    return report

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--output',default=str(ROOT/'reports'/'trained'))
    pa.add_argument('--epochs-time',type=int,default=65);pa.add_argument('--epochs-energy',type=int,default=120)
    pa.add_argument('--patience',type=int,default=15);pa.add_argument('--patience-energy',type=int,default=25)
    pa.add_argument('--epochs-machine',type=int,default=300)
    pa.add_argument('--batch',type=int,default=8);pa.add_argument('--lr',type=float,default=.0008);pa.add_argument('--lr-energy',type=float,default=.002)
    pa.add_argument('--smoke-max-per-group',type=int,default=0);pa.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');pa.add_argument('--seed',type=int,default=20261017)
    args=pa.parse_args();run(args)
