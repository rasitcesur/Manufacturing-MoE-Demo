"""12-component analytical CNC hole-processing electric-energy surrogate (Wh).

NOT LinuxCNC's physical power model or electrical measurements. The teacher
has a machine-gain nuisance coefficient; inference prior uses a nominal gain.
No tool-wear label, hardness index, or wear state is consulted.

Energy bookkeeping distinguishes power*time from cutting/contact work (joules),
so changing predicted duration never rescales a fixed removed-volume energy.
"""
from __future__ import annotations
import json, math, re
import numpy as np
from hole_constants import MAT, TOOL

COMPONENTS=(
    'electronics','axis_hold','axis_x','axis_y','axis_z',
    'spindle_idle','spindle_transient','spindle_cutting',
    'feed_thrust','tool_rubbing','coolant','auxiliary'
)
RE=re.compile(r'([A-Z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))',re.I)


def components_by_row(rows, times, *, teacher=False):
    """Return Nx12 Wh; times must be independently predicted for inference.

    Ground truth energy labels may use source durations **only during relabeling**.
    Hidden machine gain is never used at inference, and hidden wear is never used.
    """
    profile=json.loads(rows[0]['initial_profile'])
    op=profile.get('operation',rows[0].get('operation','drilling'))
    orient=profile.get('orientation','vertical')
    axis='X' if orient=='horizontal' else 'Z'
    cross=('Y','Z') if axis=='X' else ('X','Y')
    depth=float(profile['hole_depth_mm']);diameter=float(profile['hidden_diameter_mm'])
    pilot=float(profile.get('hidden_pilot_diameter_mm',0.0));pitch=float(profile.get('pitch_mm',0.0))
    form=bool(profile.get('tap_form',False))
    material=profile.get('hidden_work_material','AL6061');tool=profile.get('hidden_tool_material','CARBIDE')
    _density,se,abras=MAT[material]
    tool_factor=TOOL[tool][0]
    gain=float(profile.get('hidden_machine_gain',1.0)) if teacher else 1.0
    # Cutting tap and forming tap: equal G-code may give different material volume.
    volume_area=max(0.,math.pi*(diameter**2-pilot**2)/4)
    if op=='tapping':volume_area=0.0 if form else math.pi*diameter*pitch*.11
    core_contact_area=(math.pi*diameter*pitch*.25 if op=='tapping' else volume_area)
    # Rubbing is independent of chip volume and nonzero for form tapping.
    wall_contact_mm2=math.pi*diameter*depth
    rubbing_rate={'drilling':.055,'boring':.27,'reaming':.85,'tapping':1.13}[op]
    if op=='tapping' and form:rubbing_rate*=1.38
    geometry_mult={'drilling':1.0,'boring':.78,'reaming':1.42,'tapping':2.0 if form else 1.65}[op]
    # Latent hardness values are intentionally NOT read here.
    feed=600.;rpm=0.;on=False;cool=False;pos_axis=5.;pos_cross=(12.,12.)
    active_cycle=None;power=[];total_vol=0.;total_dist=0.;total_motion=0.;last_m3=False
    for row,dt_in in zip(rows,times):
        dt=max(0.,float(dt_in));gc=row['gcode'].strip().upper();pairs=[(k,float(v)) for k,v in RE.findall(gc)]
        words=dict(pairs);gs=[int(v) for k,v in pairs if k=='G'];ms=[int(v) for k,v in pairs if k=='M']
        for g in gs:
            if g in (81,82,83,84,85,89):active_cycle=g
            elif g==80:active_cycle=None
        if 'F' in words:feed=words['F']
        if 'S' in words:rpm=words['S']
        start_spindle=3 in ms or 4 in ms
        for m in ms:
            if m in (3,4):on=True
            if m==5:on=False
            if m==8:cool=True
            if m==9:cool=False
        cycle=bool(active_cycle is not None and (active_cycle in gs or (not gs and not ms and any(k in words for k in cross))))
        rapid=0.;axial=0.;later=0.;vol=0.
        if 0 in gs:
            new_axis=words.get(axis,pos_axis)
            new_cross=(words.get(cross[0],pos_cross[0]), words.get(cross[1],pos_cross[1]))
            axial=abs(new_axis-pos_axis);later=math.dist(pos_cross,new_cross)
            rapid=math.hypot(axial,later)
            pos_axis=new_axis;pos_cross=new_cross
        if cycle:
            new_cross=(words.get(cross[0],pos_cross[0]),words.get(cross[1],pos_cross[1]))
            later=math.dist(pos_cross,new_cross);pos_cross=new_cross
            axial=2*(depth+2.)
            vol=volume_area*depth
        total_vol+=vol
        travel=rapid if rapid else axial+later
        total_dist+=travel
        if travel>0:total_motion+=dt
        E=np.zeros(12,dtype=np.float64)
        # Power*time terms, with tool axis orientation retained in machine physics.
        if dt>0:
            E[0]=(62.+17.*gain)*dt/3600.
            E[1]=(15. + (27. if axis=='Z' else 13.)*gain)*dt/3600.
            if travel>0:
                vel=travel/max(dt,1e-6)
                # Actuator power differs by axis; Z has gravity-dependent drive loss.
                cross_motion=later*2 if rapid==0 else later
                axis_motion=axial
                for letter,dd in ((axis,axis_motion),(cross[0],cross_motion*.5),(cross[1],cross_motion*.5)):
                    if dd<=0:continue
                    i={'X':2,'Y':3,'Z':4}[letter]
                    stiffness={'X':1.,'Y':1.08,'Z':1.32}[letter]
                    E[i]+=((6.+(.78*vel+.018*vel**2)*stiffness)*dd/max(travel,1e-6))*dt/3600.
            if on:E[5]=(46.+.021*rpm+1.1e-6*rpm*rpm)*gain*dt/3600.
            if cool:E[10]=(90.+30.*(op=='drilling'))*dt/3600.
            E[11]=(17.+(10. if orient=='horizontal' else 6.))*dt/3600.
        if start_spindle:
            omega=2*math.pi*rpm/60.
            E[6]=.5*.0040*omega*omega/(.78*3600.) * gain
        if cycle:
            # J/mm³ x mm³; no dt multiplier. Deliberate spindle/feed partition.
            # Calibrated only against the analytic research generator, NOT experiments.
            contact_v=(core_contact_area*depth if op=='tapping' else vol)
            total_contact_J=se*tool_factor*contact_v*geometry_mult
            thrust_share={'drilling':.095,'boring':.080,'reaming':.125,'tapping':.15}[op]
            E[7]=total_contact_J*(1-thrust_share)/(3600.*.91)
            E[8]=total_contact_J*thrust_share/(3600.*.79)
            # Rubbing can be large even for tiny reaming allowances and zero-chip form tap.
            E[9]=wall_contact_mm2*rubbing_rate*abras*tool_factor/3600.
            # Reversal/retiming tapping transition electric drive losses.
            if op=='tapping':E[6]+=.5*.0040*(2*math.pi*rpm/60.)**2/(.83*3600.)
        if teacher:
            # Nuisance machine-efficiency and fixture variation, never exposed to model.
            E[2:5]*=(.96+.07*gain)
            E[5:10]*=(.91+.13*gain)
        power.append(E)
    return np.asarray(power,dtype=np.float64),dict(components=list(COMPONENTS),
        base=np.asarray(power,dtype=np.float64).sum(0),motion_s=total_motion,
        speed=total_dist/max(total_motion,1e-8),rpm=rpm,cut_vol=total_vol,
        tool=diameter,temp=0.0,density=_density*1000,
        material=total_vol*_density,total_s=sum(times),commands=len(rows),
        mode='cnc_drilling',orientation_for_hardware=orient)
