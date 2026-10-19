"""Independent blockwise research surrogate for reciprocating SURFACE grinding.

Supported G-code: G21/G17/G90/G94, G0/G1, G4 P, S/F, M3/M5/M8/M9/M2.
Simplified flat-wheel swept-band geometry; no arc interpolation, corner blending,
thermal/roughness/wheel wear, or real CNC driver. Units mm, s, W, Wh, g.
``teacher=True`` only for offline synthetic labels and PRIVATE coefficient passed as
an argument. The predictor always uses ``teacher=False`` and no private gain.
"""
from __future__ import annotations
import json,math,re
import numpy as np

COMPONENTS=('electronics','axis_hold','axis_x','axis_y','axis_z',
            'wheel_spindle_idle','wheel_acceleration','shearing','ploughing',
            'rubbing','coolant','filtration')
DENSITY={'C45':.00785,'AL6061':.00270,'304SS':.00800,'CAST_IRON':.00720}
SE={'C45':1.0,'AL6061':.52,'304SS':1.35,'CAST_IRON':.78}
WORD=re.compile(r'([A-Z])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))',re.I)
COMMENT=re.compile(r'\([^)]*\)|;[^\n]*')

def motion_time(mm,speed_mm_min,accel):
    if mm<=0:return 0.
    v=max(.01,speed_mm_min/60.);a=max(.01,accel)
    return 2*math.sqrt(mm/a) if mm<v*v/a else 2*v/a+(mm-v*v/a)/v

def simulate(rows,teacher=False,machine_gain=None,times_override=None):
    profile=json.loads(rows[0]['initial_profile'])
    d=float(profile['wheel_diameter_mm']);b=float(profile['wheel_width_mm'])
    wheel_type=profile['wheel_type'];material=profile['material'];rho=DENSITY[material]
    if teacher and machine_gain is None:raise ValueError('Synthetic teacher requires PRIVATE machine coefficient')
    gain=float(machine_gain) if teacher else 1.
    x=y=0.;z=5.;motion='G0';feed=600.;rpm=0.;on=False;cool=False;absolute=True
    prepared=[];lanedepth={};cutting_time=0.;total_volume=0.
    for i,row in enumerate(rows):
        gc=COMMENT.sub('',row['gcode'].upper()).strip();pairs=[(a,float(v)) for a,v in WORD.findall(gc)]
        words=dict(pairs);gg=[int(v) for a,v in pairs if a=='G'];mm=[int(v) for a,v in pairs if a=='M']
        if any(g in (2,3,18,19,91,93,95) for g in gg):raise ValueError('Unsupported grinding G-code')
        for g in gg:
            if g==0:motion='G0'
            elif g==1:motion='G1'
            elif g==90:absolute=True
        if 'S' in words:rpm=words['S']
        if 'F' in words:feed=words['F']
        was_on=on
        for m in mm:
            if m in (3,4):on=True
            if m==5:on=False
            if m==8:cool=True
            if m==9:cool=False
        nx=words.get('X',x);ny=words.get('Y',y);nz=words.get('Z',z)
        move=any(c in words for c in 'XYZ');dx=nx-x;dy=ny-y;dz=nz-z
        length=math.sqrt(dx*dx+dy*dy+dz*dz) if move else 0.
        speed=(float(profile['rapid_mm_min']) if motion=='G0' else feed)
        t=motion_time(length,speed, float(profile['accel_mm_s2'])*(1.0 if teacher else .86)) if move else 0.
        if 4 in gg:t=max(0.,words.get('P',0.))
        if 3 in mm and not was_on:t+= (1.35+.000037*rpm)*(1 if teacher else .90)
        if 8 in mm:t+=.35 if teacher else .33
        if teacher and length>0:t*=1.0+.016*math.tanh(length/80)+.008*(material=='304SS')
        if times_override is not None:t=float(times_override[i])
        t=max(0.,t)
        volume=0.;stockdepth=0.
        # Only X traverses with constant Y and negative surface Z remove stock.
        # Previously cut depth remembered per non-overlapping track, so a spark-out
        # pass at same depth removes no extra material.
        if move and motion=='G1' and on and abs(dy)<1e-7 and abs(dz)<1e-7 and abs(dx)>1 and nz<0:
            lane=round(ny/b)
            if abs(ny-lane*b)<1e-3:
                before=lanedepth.get(lane,0.)
                stockdepth=max(0.,-nz-before)
                volume=abs(dx)*b*stockdepth
                lanedepth[lane]=max(-nz,before)
        total_volume+=volume
        circumference_ms=math.pi*d*rpm/60000. # mm diameter -> m/s
        table_mm_s=abs(dx)/max(t,1.e-9)
        a_e=max(stockdepth,0.)
        heq=a_e*table_mm_s/max(circumference_ms*1000,1.)
        # Specific cutting energy model J/mm^3, controlled by chip thickness,
        # material and wheel abrasive. The phenomenon split avoids double count:
        # cutting + ploughing + rubbing are net mechanical wheel load only.
        f_wheel=.80 if wheel_type=='CBN' else 1.0
        u_shear=12.*SE[material]*f_wheel*(1.+.000065/(heq+.000065))
        u_plough=27.*SE[material]*f_wheel/(1.+heq/.000048)
        # Rubbing only on work contact, including spark-out; separate from chip work.
        in_contact=(move and motion=='G1' and on and abs(dy)<1.e-7 and abs(dz)<1.e-7 and nz<0 and abs(dx)>1)
        contact_length=math.sqrt(max(0.,min(-nz,.08))*d) if in_contact else 0.
        rubbing_power=(13.+.009*rpm + 1.4*contact_length*b)*SE[material]*f_wheel if in_contact else 0.
        powers=np.zeros(12,np.float64)
        if t>0:
            powers[0]=(80.+14.*gain)*t/3600.
            powers[1]=(34.+11.*gain)*t/3600.
            if move:
                for ix,dd,ff in [(2,abs(dx),1.),(3,abs(dy),1.08),(4,abs(dz),1.35)]:
                    if dd>0:powers[ix]+=(7.+ff*(.42*dd/max(t,.001)+.0012*(dd/max(t,.001))**2))*t/3600.
            if on:powers[5]=(350.+.024*rpm+1.6e-6*rpm*rpm)*gain*t/3600.
            if 3 in mm and not was_on:powers[6]+=(.009*rpm*rpm/1.e3)*gain/3600.
            # Work-derived energies in J converted to Wh, independent of duration.
            powers[7]=u_shear*volume/3600.
            powers[8]=u_plough*volume/3600.
            powers[9]=rubbing_power*t/3600.
            if cool:powers[10]=(160.+23.*gain)*t/3600.
            if on:powers[11]=95.*t/3600.
        # Teacher machine gain is NOT a global multiplier; it modulates selected
        # machine losses only. Cut energy is modified by wheel physics, not gain.
        prepared.append({'code':gc,'command':('G4' if 4 in gg else motion if move else ('M'+str(mm[0]) if mm else 'STATE')),
          'time_s':t,'material_removed_mm3':volume,'material_g':volume*rho,
          'energy_Wh':float(powers.sum()),'component_energy_Wh':powers.tolist(),
          'motion_mm':length,'feed_mm_min':feed,'spindle_rpm':rpm,'wheel_surface_m_s':circumference_ms,
          'depth_mm':-nz if nz<0 else 0.,'wheel_width_mm':b,'wheel_diameter_mm':d,
          'x_after_mm':nx,'y_after_mm':ny,'z_after_mm':nz,'contact':bool(in_contact),
          'chip_thickness_equiv_mm':heq,'removed_volume_cumulative_mm3':total_volume})
        x,y,z=nx,ny,nz
    return prepared
