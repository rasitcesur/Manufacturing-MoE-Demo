"""Research-only analytic surrogate for continuous-band METAL SAWING (cut-off).

Supported: G21/G90/G94 G0/G1 G4 P M3/M5/M8/M9/M2, S in m/min,
F in mm/min, vertical Z feed, mm/seconds/Wh/g. Rectangular solid or round solid.
Not a controller, validated machine model, wear model, or measured electrical data.
Teacher-specific machine gain is PRIVATE and never included in prediction inputs.
Mechanism decomposition: tooth shear + edge ploughing + sliding/rubbing; the
sum represents mechanical blade load, not additional spindle idle electrical power.
"""
from __future__ import annotations
import math,re,json
import numpy as np
COMPONENTS=('electronics','hydraulic_hold','blade_drive_idle','blade_acceleration',
    'cutting_shear','ploughing','tooth_rubbing','feed_axis','band_guides',
    'coolant','chip_conveyor','control_aux')
DENSITY={'C45':.00785,'AL6061':.00270,'304SS':.00800,'CAST_IRON':.00720}
RESISTANCE={'C45':1.0,'AL6061':.49,'304SS':1.32,'CAST_IRON':.82}
WORD=re.compile(r'([A-Z])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))',re.I)
COMMENT=re.compile(r'\([^)]*\)|;[^\n]*')

def motion_time(mm,speed_mm_min,accel):
    if mm<=0:return 0.
    v=max(.01,speed_mm_min/60.);a=max(.01,accel)
    return 2*math.sqrt(mm/a) if mm<v*v/a else 2*v/a+(mm-v*v/a)/v

def cross_section_area(depth,profile):
    """Area removed by the blade plane from top down to depth (mm2)."""
    h=float(profile['height_mm']);d=max(0.,min(h,depth))
    if profile['stock_shape']=='rectangular':return d*float(profile['width_mm'])
    # Circle segment from top; stable analytic primitive for a cylinder bar.
    r=h/2.; yy=r-d
    return r*r*math.acos(max(-1.,min(1.,yy/r)))-yy*math.sqrt(max(0.,r*r-yy*yy))

def contact_width(depth,profile):
    if profile['stock_shape']=='rectangular':return float(profile['width_mm'])
    r=float(profile['height_mm'])/2.; yy=r-depth
    return 2*math.sqrt(max(0.,r*r-yy*yy))

def simulate(rows,teacher=False,machine_gain=None,times_override=None):
    if not rows:raise ValueError('Sawing needs G-code rows')
    profile=json.loads(rows[0]['initial_profile'])
    if teacher and machine_gain is None:raise ValueError('private teacher coefficient required')
    gain=float(machine_gain) if teacher else 1.
    density=DENSITY[profile['material']];res=RESISTANCE[profile['material']]
    kerf=float(profile['kerf_mm']);height=float(profile['height_mm']);blade_type=profile['blade_type']
    tooth_pitch_mm=25.4/float(profile['tpi'])
    feed=float(profile['feed_mm_min']);band_speed=0.
    z=3.;motion='G0';on=False;cool=False;absolute=True
    events=[];cumulative_area=0.;total_removed=0.
    for i,row in enumerate(rows):
        gc=COMMENT.sub('',row['gcode'].upper()).strip();pairs=[(a,float(b)) for a,b in WORD.findall(gc)]
        w=dict(pairs);gcodes=[int(v) for a,v in pairs if a=='G'];mcodes=[int(v) for a,v in pairs if a=='M']
        if any(g in (2,3,18,19,91,93,95) for g in gcodes):raise ValueError('Unsupported sawing G-code')
        for g in gcodes:
            if g==0:motion='G0'
            elif g==1:motion='G1'
            elif g==90:absolute=True
        if 'F' in w:feed=w['F']
        if 'S' in w:band_speed=w['S']
        was_on=on
        for m in mcodes:
            if m==3:on=True
            elif m==5:on=False
            elif m==8:cool=True
            elif m==9:cool=False
        nz=w.get('Z',z);move='Z' in w;dz=nz-z;dist=abs(dz) if move else 0.
        feed_actual=float(profile['rapid_mm_min']) if motion=='G0' else feed
        t=motion_time(dist,feed_actual,float(profile['accel_mm_s2'])*(1 if teacher else .85)) if move else 0.
        if 4 in gcodes:t=max(0.,w.get('P',0.))
        if 3 in mcodes and not was_on:t+=(1.15+.007*band_speed)*(1 if teacher else .93)
        if 8 in mcodes:t+=.24 if teacher else .22
        if teacher and move and motion=='G1':t*=1.+.011*math.tanh(dist/30.)+.006*(profile['material']=='304SS')
        if times_override is not None:t=float(times_override[i])
        t=max(0.,t)
        # Stock face is at Z=0; head moves in negative Z to -height.
        lo=max(0.,min(height,-z));hi=max(0.,min(height,-nz))
        cut=(bool(on and move and motion=='G1' and dz<0 and hi>lo))
        removed_area=max(0.,cross_section_area(hi,profile)-cross_section_area(lo,profile)) if cut else 0.
        vol=kerf*removed_area
        total_removed+=vol
        sweep=hi-lo if cut else 0.
        wavg=removed_area/sweep if sweep>1e-12 else 0.
        vfeed=sweep/max(t,1.e-8) # mm/s
        vblade=band_speed/60. # m/s (S specified m/min)
        fz=(vfeed*60)*tooth_pitch_mm/max(1.,band_speed*1000.) # mm / tooth
        engaged=wavg/max(tooth_pitch_mm,1.e-5)
        # Flexible empirical 'mechanistic' approximation. Constants are NOT
        # published/experimentally calibrated force coefficients.
        blade_factor=.84 if blade_type=='CARBIDE' else 1.
        u_shear=(1.7*res*blade_factor)*(1.+.015/(fz+.015))
        u_plough=(.70*res*blade_factor)/(1.+fz/.015)
        in_contact=bool(on and motion=='G1' and move and -nz>0 and z>=-height-.01 and nz<=0)
        # Friction/rubbing on engaged teeth even on near-zero-volume contact.
        rubbing_power=(8.+.42*engaged + .12*band_speed)*res*blade_factor if in_contact else 0.
        p=np.zeros(len(COMPONENTS),np.float64)
        if t>0:
            p[0]=(54.+13.*gain)*t/3600.
            p[1]=(92.+21.*gain)*t/3600.
            if on:p[2]=(195.+2.7*band_speed+.006*band_speed**2)*gain*t/3600.
            if 3 in mcodes and not was_on:
                # Approximate drive/band kinetic energy; not included in idle load.
                p[3]=(.5*float(profile['equivalent_blade_mass_kg'])*vblade*vblade)/(3600*.77)*gain
            p[4]=u_shear*vol/3600.
            p[5]=u_plough*vol/3600.
            p[6]=rubbing_power*t/3600.
            if move:
                axis_v=dist/max(t,.0001)
                p[7]=(11.+.8*axis_v+.012*axis_v*axis_v)*gain*t/3600.
            if on:p[8]=(31.+.18*band_speed)*t/3600.
            if cool:p[9]=(82.+12.*gain)*t/3600.
            if on:p[10]=64.*t/3600.
            p[11]=26.*t/3600.
        command='G4' if 4 in gcodes else (motion if move else 'M'+str(mcodes[0]) if mcodes else 'STATE')
        events.append({'command':command,'time_s':t,'energy_Wh':float(p.sum()),'component_energy_Wh':p.tolist(),
           'material_g':vol*density,'material_removed_mm3':vol,'cut_width_avg_mm':wavg,
           'mean_engaged_teeth':engaged,'feed_per_tooth_mm':fz,'band_speed_m_min':band_speed,
           'feed_mm_min':feed,'z_after_mm':nz,'motion_mm':dist,'contact':in_contact,
           'kerf_mm':kerf,'height_mm':height,'cut_depth_mm':sweep,'total_volume_mm3':total_removed,
           'blade_acceleration_event':bool(3 in mcodes and not was_on)})
        z=nz
    return events
