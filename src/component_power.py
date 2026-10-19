"""Explicit, stateful *synthetic* power decomposition for Ender-3-like FDM systems.

NOT a calibrated Ender 3, nor the official Marlin Native Simulator. No implicit
'260 W always on' assumption. Heater state is integrated once per timeline.
All component powers expressed at AC input equivalence. Avoids double-counting
PSU losses; real devices require measured component/PSU calibration.

This module is used for synthetic-teacher label construction with teacher time,
and for inference priors with predicted/kinematic-prior time. Never look at
y_true while predicting.
"""
import json, math, hashlib
import numpy as np
from physics import parse, motion_info

COMPONENTS=(
 'electronics','stepper_hold','motor_x','motor_y','motor_z','motor_e',
 'heater_nozzle','heater_bed','fan_part','fan_hotend')
MOTION={'G0','G1','G2','G3'}

DEFAULT_POWER={
 'electronics_w':12.0,'stepper_hold_w':8.0,
 'x_motor_w_at_60':3.8,'y_motor_w_at_60':4.4,
 'z_motor_w_at_5':2.2,'e_motor_w_at_5':3.8,
 'nozzle_w':40.0,'bed_w':220.0,
 'part_fan_w':2.4,'hotend_fan_w':2.2,
 'nozzle_cap_j_c':25.0,'bed_cap_j_c':650.0,
 'nozzle_loss_w_c':0.13,'bed_loss_w_c':2.1,
 'psu_factor':1.0
}

def make_reference_hardware(profile_id):
    """Deterministic nominal hardware parameters: synthetic, not a measured device.
    Vary nominal ratings across jobs to exercise model generalization.
    Store this dictionary as known per-machine calibration metadata.
    """
    key=int(hashlib.sha256(str(profile_id).encode()).hexdigest()[:12],16)
    rng=np.random.default_rng(key)
    d=dict(DEFAULT_POWER)
    for name,lo,hi in [('electronics_w',.85,1.15),('stepper_hold_w',.8,1.2),
                       ('x_motor_w_at_60',.8,1.2),('y_motor_w_at_60',.8,1.2),
                       ('z_motor_w_at_5',.8,1.2),('e_motor_w_at_5',.8,1.2),
                       ('nozzle_w',.9,1.1),('bed_w',.9,1.1),
                       ('part_fan_w',.8,1.2),('hotend_fan_w',.8,1.2),
                       ('nozzle_cap_j_c',.9,1.1),('bed_cap_j_c',.9,1.1),
                       ('nozzle_loss_w_c',.8,1.2),('bed_loss_w_c',.8,1.2)]:
        d[name]*=float(rng.uniform(lo,hi))
    return d

class ComponentTimeline:
    def __init__(self,profile, hardware):
        self.ambient=float(profile['ambient']);self.t_nozzle=float(profile['nozzle_start']);self.t_bed=float(profile['bed_start'])
        self.target_nozzle=self.t_nozzle;self.target_bed=self.t_bed
        self.part_fan=float(profile.get('fan_start',0.));self.drivers_enabled=True
        self.p={**DEFAULT_POWER,**hardware}
        self.xyz=np.array([float(profile.get(k,0)) for k in ('start_x','start_y','start_z')],dtype=float)
        self.absolute_xyz=True;self.absolute_e=False;self.e_position=0.0;self.e_flow=1.0
        self.feed=1800.;self.speed_pct=1.0
    def change(self,cmd, args):
        if cmd in ('M104','M109'):self.target_nozzle=float(args.get('S',args.get('R',self.target_nozzle)))
        elif cmd in ('M140','M190'):self.target_bed=float(args.get('S',args.get('R',self.target_bed)))
        elif cmd=='M106':self.part_fan=max(0.,min(255.,float(args.get('S',255.))))
        elif cmd=='M107':self.part_fan=0.
        elif cmd=='M17':self.drivers_enabled=True
        elif cmd in ('M18','M84'):self.drivers_enabled=False
        elif cmd=='G90':self.absolute_xyz=True
        elif cmd=='G91':self.absolute_xyz=False
        elif cmd=='M82':self.absolute_e=True
        elif cmd=='M83':self.absolute_e=False
        elif cmd=='M220':self.speed_pct=max(.01,float(args.get('S',100))/100)
        elif cmd=='M221':self.e_flow=max(.01,float(args.get('S',100))/100)
        elif cmd=='G92':
            for i,k in enumerate('XYZ'):
                if k in args:self.xyz[i]=float(args[k])
            if 'E' in args:self.e_position=float(args['E'])
    def advance(self,cmd,args,dt):
        """Return Wh[10], time-averaged input-W[10], and observed motion state.
        Delta-E and velocity are calculated from G-code and provided duration.
        Re-traversing the same program with different dt yields different power
        because v=distance/dt; heater duty uses a common time-stepped clock.
        """
        dt=max(0.,float(dt))
        self.change(cmd,args)
        velocity=np.zeros(4,float)
        if cmd in MOTION:
            end,delta,path,feed,_,_=motion_info(cmd,args,self.xyz,self.feed,self.absolute_xyz)
            if 'F' in args:self.feed=float(args['F'])
            self.xyz=end
            if 'E' in args:
                new_e=float(args['E']) if self.absolute_e else self.e_position+float(args['E'])
                de=(new_e-self.e_position)*self.e_flow; self.e_position=new_e
            else:de=0.
            if dt>1e-8:
                velocity[:3]=np.abs(delta)/dt
                velocity[3]=abs(de)/dt
        result=np.zeros(len(COMPONENTS),float)
        if dt<=0.:return result,result,velocity
        # Motor power defined as grid-equivalent dynamic contributions.
        p=self.p
        move=cmd in MOTION and float(np.linalg.norm(velocity))>1e-6
        static=np.array([
            p['electronics_w'],
            p['stepper_hold_w'] if self.drivers_enabled else 0.,
            p['x_motor_w_at_60']*min(3.,(velocity[0]/60.)**.80) if move else 0.,
            p['y_motor_w_at_60']*min(3.,(velocity[1]/60.)**.80) if move else 0.,
            p['z_motor_w_at_5']*min(3.,(velocity[2]/5.)**.80) if move else 0.,
            p['e_motor_w_at_5']*min(3.,(velocity[3]/5.)**.75) if move else 0.,
            0.,0.,p['part_fan_w']*self.part_fan/255.,0.
        ],float)
        # At most 0.1 s integration to track thermostat duty and heat loss.
        n=max(1,int(math.ceil(dt/.1)));step=dt/n
        for _ in range(n):
            duty_n=float(self.target_nozzle>0 and self.t_nozzle < self.target_nozzle-.3)
            duty_b=float(self.target_bed>0 and self.t_bed < self.target_bed-.3)
            static[6]=p['nozzle_w']*duty_n
            static[7]=p['bed_w']*duty_b
            static[9]=p['hotend_fan_w'] if self.t_nozzle>45 or self.target_nozzle>0 else 0.
            result+=static*step/3600.
            h_n=p['nozzle_loss_w_c']*(1+.25*self.part_fan/255.)
            self.t_nozzle+=(p['nozzle_w']*duty_n-h_n*(self.t_nozzle-self.ambient))*step/p['nozzle_cap_j_c']
            self.t_bed+=(p['bed_w']*duty_b-p['bed_loss_w_c']*(self.t_bed-self.ambient))*step/p['bed_cap_j_c']
        result*=p['psu_factor']
        return result, result*3600/dt,velocity

def simulate_components(rows,durations):
    """Run a SINGLE shared clock for every component, returns (N,C) Wh, (N,C) W, (N,4) mm/s.
    Duration source must be explicit: teacher-only label generation OR inference-time prior/prediction.
    """
    if len(rows)!=len(durations):raise ValueError('Rows/time length mismatch')
    profile=json.loads(rows[0]['initial_profile']) if isinstance(rows[0]['initial_profile'],str) else rows[0]['initial_profile']
    hardware=profile.get('power_hardware',DEFAULT_POWER)
    state=ComponentTimeline(profile,hardware)
    energies=[];powers=[];speed=[]
    for r,d in zip(rows,durations):
        cmd,a=parse(r['gcode']);energy,power,vel=state.advance(cmd,a,d)
        energies.append(energy);powers.append(power);speed.append(vel)
    return np.array(energies,dtype='float32'),np.array(powers,dtype='float32'),np.array(speed,dtype='float32')
