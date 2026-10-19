"""Canonical hole-cycle physics approximation, independent of training targets.
One Hole Processing Expert for G17/Z and G19/X; physical axis retained only for
machine-energy accounting. New synthetic metadata is NOT suitable for G-code-only
production deployments unless corresponding tool/stock data are available.
This is NOT a LinuxCNC runtime simulator.
"""
from __future__ import annotations
import math,re,json
import numpy as np
from hole_constants import MAT,TOOL,move_time
from hole_energy_hardware import components_by_row
RE=re.compile(r'([A-Z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))',re.I)
OPS=['drilling','boring','reaming','tapping']
CODE={'G81':46,'G82':47,'G83':48,'G84':49,'G85':50,'G89':51,'G80':52,'G0':1,'G4':5,'M3':40,'M5':41,'M8':42,'M9':43,'M2':44,'STATE':45}

def simulate_program(rows, feature_dim=42, accel_nominal=590, rapid_nominal=5700):
    profile=json.loads(rows[0]['initial_profile']);op=profile.get('operation',rows[0].get('operation','drilling'))
    if op not in OPS: raise ValueError(op)
    orient='horizontal' if any(re.search(r'\bG19\b',r['gcode']) for r in rows) else 'vertical'
    # Geometric stock/tool descriptors from simulated machine metadata. These are
    # optional input data (not tool-wear sensor feedback); no target is read here.
    diameter=float(profile.get('hidden_diameter_mm',0.0));pilot=float(profile.get('hidden_pilot_diameter_mm',0.0))
    depth=float(profile.get('hole_depth_mm',0.0));pitch=float(profile.get('pitch_mm',0.0))
    form=bool(profile.get('tap_form',False));material=profile.get('hidden_work_material','AL6061')
    tool=profile.get('hidden_tool_material','CARBIDE');density,se,_=MAT.get(material,MAT['AL6061'])
    tool_power=TOOL.get(tool,TOOL['CARBIDE'])[0]
    area=math.pi/4*max(0,diameter*diameter-pilot*pilot)
    if op=='tapping':
        area=0 if form else math.pi*diameter*pitch*.11
    area=max(0,area)
    # Energy teacher uses operation-specific friction/contact. No hidden wear or
    # hidden motor gain is supplied to this baseline; they remain unpredictable.
    contact_area=math.pi*diameter*pitch*.25 if op=='tapping' else math.pi/4*max(0,diameter**2-pilot**2)
    contact_geometry={'drilling':1.,'boring':.78,'reaming':1.42,'tapping':2.4 if form else 1.7}[op]
    feed=600.;rpm=0.;on=False;cool=False;axispos=5.;crosspos=(12.,12.);cycle=None;cycle_peck=4.5;cycle_dwell=0.3
    out_codes=[];out_feats=[];out_prior=[];out_mass=[];total_energy=0.
    running_total_distance=0.;running_cutvol=0.;running_motion=0.
    running_power_components=np.zeros(8,np.float64)
    ax='X' if orient=='horizontal' else 'Z';cross=('Y','Z') if orient=='horizontal' else ('X','Y')
    for ir,row in enumerate(rows):
        gc=row['gcode'].strip().upper(); pairs=[(k,float(v)) for k,v in RE.findall(gc)]; w=dict(pairs)
        gv=[int(v) for k,v in pairs if k=='G'];mv=[int(v) for k,v in pairs if k=='M']
        previous_cycle=cycle
        for g in gv:
            if g in (81,82,83,84,85,89):cycle='G'+str(g)
            if g==80:cycle=None
        if 'F' in w:feed=w['F']
        if cycle and any(g in (81,82,83,84,85,89) for g in gv):
            cycle_peck=float(w.get('Q',cycle_peck));cycle_dwell=float(w.get('P',cycle_dwell))
        if 'S' in w:rpm=w['S']
        for m in mv:
            if m in (3,4):on=True
            elif m==5:on=False
            elif m==8:cool=True
            elif m==9:cool=False
        code=('G'+str(gv[0])) if gv else ('M'+str(mv[0]) if mv else ('STATE'))
        dt=0.;volume=0.;axial_distance=0.;lateral_distance=0.;is_cycle=False;dwell=0.;rapid_distance=0.
        if 4 in gv:dt=max(0.,w.get('P',0.))
        if 0 in gv:
            new_axis=w.get(ax,axispos)
            new_cross=(w.get(cross[0],crosspos[0]),w.get(cross[1],crosspos[1]))
            axial_distance=abs(new_axis-axispos);lateral_distance=math.dist(crosspos,new_cross)
            rapid_distance=math.hypot(axial_distance,lateral_distance)
            dt=move_time(rapid_distance,rapid_nominal/60,accel_nominal)
            axispos=new_axis;crosspos=new_cross
        # A canned-cycle command OR a positional-only line while a cycle is modal.
        if cycle and (int(cycle[1:]) in gv or (not gv and not mv and any(k in w for k in cross))):
            nc=(w.get(cross[0],crosspos[0]),w.get(cross[1],crosspos[1]))
            lateral_distance=math.dist(crosspos,nc);crosspos=nc
            axial_distance=depth+2.
            speed=max(feed/60,0.01)
            dt=move_time(lateral_distance,rapid_nominal/60,accel_nominal)+move_time(axial_distance,speed,accel_nominal)
            dt+=move_time(axial_distance,speed if cycle in ('G85','G89','G84') else rapid_nominal/60,accel_nominal)
            if cycle=='G83':
                q=max(.1,cycle_peck)
                dt+=.55*(max(1,math.ceil(depth/q))-1)*depth/max(speed,.1)
            if cycle in ('G82','G84','G89'):
                dwell=cycle_dwell
                dt+=dwell
            if cycle=='G84': dt+=.27+.035*depth
            volume=area*depth;is_cycle=True;code=cycle
        energy=0.
        if dt>0:
            base_w=58.+18*(orient=='horizontal')+22.
            spindle_w=(48+rpm*.022+rpm*rpm*1e-6) if on else 0.
            aux_w=(82. if cool else 0.)
            motion_dist=lateral_distance+axial_distance if is_cycle else rapid_distance
            speed_motion=motion_dist/max(dt,1e-8)
            motion_w=(.4*speed_motion+.012*speed_motion**2) if motion_dist else 0.
            # Form tapping consumes contact work EVEN when zero chips are produced.
            vol_for_contact=contact_area*depth if (is_cycle and op=='tapping') else volume
            cutting_J=se*tool_power*vol_for_contact*contact_geometry if is_cycle else 0.
            power_comp=np.asarray([base_w*dt,0.,motion_w*dt*.5,motion_w*dt*.5,0.,spindle_w*dt,cutting_J,aux_w*dt],float)/3600.
            running_power_components+=power_comp
            energy=float(power_comp.sum())
            running_motion+=dt if (rapid_distance or is_cycle) else 0.
            running_total_distance+=motion_dist
        running_cutvol+=volume
        vec=np.zeros(feature_dim,np.float32)
        # Canonical axial/lateral motion ONLY. No raw X/Y/Z positions, signs,
        # physical orientation, or direction indicators leak into Hole Expert.
        fields=[math.log1p(axial_distance+lateral_distance),math.log1p(max(feed/60,0)),math.log1p(max(rpm,0)),
                math.log1p(max(diameter,0)),math.log1p(max(depth,0)),math.log1p(axial_distance),
                math.log1p(lateral_distance),math.log1p(abs(depth)),float(is_cycle),
                float(code in ('G81','G82','G83')),float(code in ('G84','G85','G89')),
                float(code=='G0'),math.log1p(max(volume,0)),float(on),float(cool),float(is_cycle),
                float(code=='G0'),float(code in ('G81','G82','G83')),float(code=='G83'),
                float(op=='boring'),float(op=='reaming'),math.log1p(max(feed,0)),math.log1p(dt),
                float(dwell>0),math.log1p(len(rows)),math.log1p(max(pitch,0)),
                float(form),math.log1p(max(pilot,0)),math.log1p(max(area,0)),math.log1p(max(contact_area,0))]
        vec[:len(fields)]=fields
        vec[37]=1.
        # These 2 are density and machine speed, which prior architecture uses.
        # They are not raw X/Z coordinates. For the hole canonical invariance
        # test, physical orientation affects only the hardware energy baseline.
        vec[40]=density*1000
        vec[41]=math.log1p(rapid_nominal)
        out_codes.append(CODE.get(code, CODE['STATE']))
        out_feats.append(vec)
        out_prior.append([dt,0.,volume*density]);out_mass.append(volume*density)
    # Corrected job-level, twelve-component electromechanical energy prior.
    # Here out_prior durations are G-code-only approximate time predictions.
    # The inference-stage prior is recomputed with the trained time prediction.
    _,raw=components_by_row(rows,np.asarray(out_prior)[:,0],teacher=False)
    return (np.asarray(out_codes,np.int64),np.asarray(out_feats,np.float32),np.asarray(out_prior,np.float32),
            np.asarray(out_mass,np.float64),raw)
