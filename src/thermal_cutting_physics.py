"""Research analytical surrogates for sheet LASER / PLASMA thermal cutting.
NOT a heat-transfer CFD model, real CNC controller, or empirically calibrated model.
Shared planar CNC kinematics, independently modeled electrical hardware heads.
G20 and G91 are intentionally rejected; G2/G3 circular interpolation supports I/J.
No wear, dross/HAZ/thermal distortion, overlapping-kerf, nesting/yield or real sensor data.
"""
from __future__ import annotations
import math,json,re
import numpy as np
MAT={'C45':(.00785,1.0),'304SS':(.00800,1.21),'AL6061':(.00270,.80),'S235':(.00785,.91)}
COMPONENTS={
 'cnc_laser':('electronics','gantry_hold','axis_xy','z_focus','laser_standby','laser_emission_input','laser_pierce','chiller','assist_gas','exhaust_filter','servo_aux','control_aux'),
 'cnc_plasma':('electronics','gantry_hold','axis_xy','torch_height','power_supply_idle','arc_cutting_input','pilot_pierce_arc','air_compressor','gas_flow','fume_extraction','cooling','control_aux')}
WORD=re.compile(r'([A-Z])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))',re.I)
COMMENT=re.compile(r'\([^)]*\)|;[^\n]*')
def trapezoid(dist,v,a):
    if dist<=1.e-12:return 0.
    v=max(.05,v);a=max(1.,a)
    return 2*math.sqrt(dist/a) if dist<v*v/a else 2*v/a+(dist-v*v/a)/v

def path_arc(old,xy,w,motion):
    cx=old[0]+w.get('I',0.);cy=old[1]+w.get('J',0.)
    r=math.hypot(old[0]-cx,old[1]-cy)
    r2=math.hypot(xy[0]-cx,xy[1]-cy)
    if r<1e-7 or abs(r-r2)>max(.025,r*.001):raise ValueError('Invalid G2/G3 arc geometry')
    a0=math.atan2(old[1]-cy,old[0]-cx);a1=math.atan2(xy[1]-cy,xy[0]-cx)
    sweep=(a1-a0)%(2*math.pi) if motion=='G3' else (a0-a1)%(2*math.pi)
    if sweep<1.e-10:sweep=2*math.pi
    return r*sweep

def kerf(p,feed):
    t=float(p['thickness_mm']);res=MAT[p['material']][1]
    if p['mode']=='cnc_laser':
        kw=.095+.012*t+.023*res+.007*(float(p['laser_power_kW'])/4.)
        speed_factor=(float(p['cut_feed_mm_min'])/max(feed,1.))**.075
        return kw*speed_factor
    kw=.55+.022*t+.13*res+.0021*float(p['current_A'])
    speed_factor=(float(p['cut_feed_mm_min'])/max(feed,1.))**.055
    return kw*speed_factor

def simulate(rows,teacher=False,machine_gain=None,times_override=None):
    if not rows:raise ValueError('Empty program')
    p=json.loads(rows[0]['initial_profile']);mode=p['mode']
    if mode not in COMPONENTS:raise ValueError('laser/plasma mode only')
    if teacher and machine_gain is None:raise ValueError('Private teacher gain required')
    gain=float(machine_gain) if teacher else 1.
    den,res=MAT[p['material']];pos=np.array([0.,0.,float(p['safe_z_mm'])]);on=False;gas=False
    motion='G0';feed=float(p['cut_feed_mm_min']);absolute=True;events=[];prev_state=False
    for i,row in enumerate(rows):
        line=COMMENT.sub('',row['gcode'].upper()).strip()
        pairs=[(a,float(b)) for a,b in WORD.findall(line)];w=dict(pairs)
        gc=[int(v) for a,v in pairs if a=='G'];mc=[int(v) for a,v in pairs if a=='M']
        if any(z in (20,91,93,95) for z in gc):raise ValueError('Unsupported unit/mode: use G21 G90 G94')
        for g in gc:
            if g in (0,1,2,3):motion='G'+str(g)
            elif g==90:absolute=True
        if 'F' in w:feed=w['F']
        old=pos.copy();new=pos.copy()
        for k,j in [('X',0),('Y',1),('Z',2)]:
            if k in w:new[j]=w[k] if absolute else new[j]+w[k]
        ismove=any(k in w for k in 'XYZ')
        dx=float(new[0]-old[0]);dy=float(new[1]-old[1]);dz=float(new[2]-old[2])
        path=math.sqrt(dx*dx+dy*dy+dz*dz) if ismove else 0.
        if ismove and motion in ('G2','G3'):
            path=math.hypot(path_arc(old,new,w,motion),dz)
        # M3 starts cutting source; M5 stops; M8 and M9 switch gas collection state.
        was_on=on
        for m in mc:
            if m==3:on=True
            elif m==5:on=False
            elif m==8:gas=True
            elif m==9:gas=False
        rapid='G0'==motion
        v=(float(p['rapid_mm_min']) if rapid else max(1.,feed))/60.
        a=float(p['accel_mm_s2'])*(1. if teacher else .88)
        if ismove:t=trapezoid(path,v,a)
        else:t=0.
        if 4 in gc:t=max(0.,w.get('P',0.))
        if 3 in mc and not was_on:t+=float(p['ignition_delay_s'])*(1. if teacher else .93)
        if 8 in mc:t+=.17 if teacher else .15
        if teacher and ismove and not rapid:t*=1+.009*math.tanh(path/50.)
        if times_override is not None:t=float(times_override[i])
        t=max(0.,t)
        cutting=bool(on and ismove and motion in ('G1','G2','G3') and path>0 and abs(new[2]-float(p['cut_z_mm']))<.2)
        kwidth=kerf(p,feed) if cutting else 0.
        volume=path*kwidth*float(p['thickness_mm']) if cutting else 0.
        # Pierce hole is material removal and separate from cut path. Pierces use G4 after M3.
        piercing=bool(on and (3 in mc))
        pierce_vol=math.pi/4*(1.35*kerf(p,float(p['cut_feed_mm_min'])))**2*float(p['thickness_mm']) if piercing else 0.
        volume+=pierce_vol
        c=np.zeros(12,dtype=np.float64)
        if t>0:
            def add(k,watts):c[k]+=max(0.,watts)*t/3600.
            # Avoid double-counting mechanical cutting with electrical source output.
            add(0,68.+8.*gain);add(1,42.+10.*gain)
            if ismove:
                speed=path/max(t,1e-8)
                add(2,(40.+.65*speed+.007*speed**2)*gain)
                if abs(dz)>1e-8:add(3,(9.+1.2*abs(dz)/max(t,1e-8))*gain)
            if mode=='cnc_laser':
                # Nominal kW optical / wall-plug efficiency, no arbitrary mechanical shear term.
                opt=float(p['laser_power_kW'])*1000.
                eff=float(p['wall_plug_efficiency'])
                add(4,(230.+40.*gain)*(1 if on else .55))
                if on:
                    add(5,opt/max(eff,.10)*(1. if cutting else .30)*gain)
                if piercing:add(6,opt/max(eff,.10)*.55*gain)
                add(7,(370.+.22*opt+170.*gain) if on else 220.)
                if gas:add(8,(float(p['assist_gas_power_W'])+65.*gain))
                add(9,240. if gas else 95.)
                add(10,75.*gain);add(11,42.)
            else:
                arc=float(p['current_A'])*float(p['arc_voltage_V'])
                eff=float(p['power_supply_efficiency'])
                add(4,(110.+30.*gain)*(1. if on else .25))
                if on:add(5,arc/max(eff,.1)*(1. if cutting else .24)*gain)
                if piercing:add(6,arc/max(eff,.1)*.75*gain)
                if gas:add(7,(float(p['compressor_power_W'])+40.*gain))
                if gas:add(8,170.+.34*float(p['current_A']))
                add(9,360. if gas else 90.)
                add(10,190. if on else 90.)
                add(11,49.)
        event={'command':('G4' if 4 in gc else motion if ismove else f'M{mc[0]}' if mc else 'STATE'),
           'time_s':t,'energy_Wh':float(c.sum()),'component_energy_Wh':c.tolist(),
           'material_g':float(volume*den),'material_removed_mm3':volume,'kerf_mm':kwidth,
           'motion_mm':path,'cutting':cutting,'piercing':piercing,'feed_mm_min':feed,
           'x_after_mm':float(new[0]),'y_after_mm':float(new[1]),'z_after_mm':float(new[2]),
           'source_on':bool(on),'gas_on':bool(gas)}
        events.append(event);pos=new;prev_state=on
    return events
