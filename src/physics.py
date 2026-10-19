"""Independent stateful *approximate* physics features (no label access, not Marlin native).

Handles synchronous temperature waits, asynchronous setpoints, motion lengths,
lookahead angle hints, and energy = integrated predicted grid power. The C++
labeler is invoked only by dataset generation, never by this inference module.
"""
import math,re,json
import numpy as np

RE=re.compile(r'([A-Z])([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
CODE=['G0','G1','G2','G3','G4','M104','M109','M140','M190','M106','M107','M204','M205','G90','G91','M82','M83','G92','M220','M221','M400','OTHER','M105','M114','M115','M119','G21','M17','M18','M84']
CMAP={k:i+1 for i,k in enumerate(CODE)}

def parse(line):
    s=line.upper().split(';')[0].strip();t=s.split()
    if not t:return 'OTHER',{}
    c={'G00':'G0','G01':'G1','G02':'G2','G03':'G3'}.get(t[0],t[0]);a={k:float(v) for k,v in RE.findall(' '.join(t[1:]))}
    return (c if c in CMAP else 'OTHER'),a

def motion_info(code,a,pos,feed,absolute=True):
    old=np.array(pos[:3],dtype=float); end=old.copy()
    for j,k in enumerate('XYZ'):
        if k in a:end[j]=a[k] if absolute else end[j]+a[k]
    vec=end-old;dist=float(np.linalg.norm(vec));arc_r=0.;sweep=0.
    if code in ('G2','G3') and ('I' in a or 'J' in a):
        cx=old[0]+a.get('I',0.);cy=old[1]+a.get('J',0.)
        r=math.hypot(old[0]-cx,old[1]-cy);end_r=math.hypot(end[0]-cx,end[1]-cy)
        if r>1e-7 and abs(r-end_r)<max(.5,r*.05):
            b0=math.atan2(old[1]-cy,old[0]-cx);b1=math.atan2(end[1]-cy,end[0]-cx)
            if code=='G2':sweep=-((b0-b1)%(2*math.pi))
            else:sweep=(b1-b0)%(2*math.pi)
            if abs(sweep)<1e-8:sweep=(2*math.pi if code=='G3' else -2*math.pi)
            arc_r=r;dist=math.hypot(r*abs(sweep),vec[2])
    f=max(.1,a.get('F',feed)/60.) if 'F' in a else max(.1,feed/60.)
    return end,vec,dist,f,arc_r,abs(sweep)

def trapezoid(d,v,a,entry=0.,exit=0.):
    if d<=1.e-10:return 0.
    v=max(v,.01);a=max(a,1.);entry=min(max(entry,0),v);exit=min(max(exit,0),v)
    da=max(0,(v*v-entry*entry)/(2*a));dd=max(0,(v*v-exit*exit)/(2*a))
    if da+dd<=d:return (v-entry)/a+(d-da-dd)/v+(v-exit)/a
    vp=max(entry,exit,math.sqrt(max(0,a*d+(entry*entry+exit*exit)/2)))
    return max(0.,(vp-entry)/a+(vp-exit)/a)

class Thermal:
    def __init__(self,p):
        self.ambient=float(p['ambient']);self.nozzle=float(p['nozzle_start']);self.bed=float(p['bed_start'])
        self.nt=self.nozzle;self.bt=self.bed;self.fan=float(p['fan_start']);self.enabled=True
    def step(self,dt,moving):
        # Lumped coefficients match the labeled synthetic reference configuration.
        # These are NOT measured coefficients of a physical Ender 3.
        hn=.13*(1+.25*self.fan/255); hb=2.1
        un=1. if self.nt>0 and self.nozzle<self.nt-.3 else 0.
        ub=1. if self.bt>0 and self.bed<self.bt-.3 else 0.
        w=13.+(13. if self.enabled else 0.)+(7. if moving else 0.)+3*self.fan/255+40*un+220*ub
        self.nozzle+=(40*un-hn*(self.nozzle-self.ambient))*dt/25.0
        self.bed+=(220*ub-hb*(self.bed-self.ambient))*dt/650.
        return w*dt/3600.
    def advance(self,seconds,moving):
        if seconds<=0:return 0.
        steps=max(1,math.ceil(seconds/.1));dt=seconds/steps
        return sum(self.step(dt,moving) for _ in range(steps))
    def wait(self,which,target,allow_cooling=False):
        start=self.nozzle if which=='nozzle' else self.bed
        heating=target>start
        if not heating and not allow_cooling:return 0.,0.
        e=0.;t=0.
        while t<900.:
            current=self.nozzle if which=='nozzle' else self.bed
            if heating and current>=target-.8:break
            if not heating and current<=target+.8:break
            e+=self.step(.1,False);t+=.1
        return t,e

def planned_motion_times(rows, p, parsed):
    """Offline multi-block velocity planning: approximate junction and accel passes.

    This planner is independently implemented and does not query label targets.
    It is a deliberately simplified model (arcs are not firmware-segmented).
    """
    state=np.array([p['start_x'],p['start_y'],p['start_z']],float)
    feed=1800.;ap=float(p['accel_print']);at=float(p['accel_travel']);jerk=float(p['jerk']);
    absolute=True;speed_mult=1.;events=[]
    for idx,(c,a) in enumerate(parsed):
        if c=='G90':absolute=True
        if c=='G91':absolute=False
        if c=='G92':
            for ax,k in enumerate('XYZ'):
                if k in a:state[ax]=a[k]
        if c=='M204':
            ap=a.get('P',a.get('S',ap));at=a.get('T',a.get('S',at))
        if c=='M205':jerk=a.get('X',jerk)
        if c=='M220':speed_mult=a.get('S',100)/100
        if c in ('G0','G1','G2','G3'):
            new,vect,dist,v,radius,sweep=motion_info(c,a,state,feed,absolute)
            if 'F' in a:feed=a['F']
            v*=speed_mult;accel=at if c=='G0' else ap
            if radius>0:
                # For arc, represent local start/end tangents; full toolpath is curved.
                center=state[:2]+[a.get('I',0.),a.get('J',0.)]
                rs=state[:2]-center;re=new[:2]-center
                tangent_start=np.array([-rs[1],rs[0],0.]);tangent_end=np.array([-re[1],re[0],0.])
                if c=='G2':tangent_start*=-1;tangent_end*=-1
                vstart=tangent_start/(np.linalg.norm(tangent_start)+1.e-12)
                vend=tangent_end/(np.linalg.norm(tangent_end)+1.e-12)
            else:
                vend=vstart=vect/(np.linalg.norm(vect)+1.e-12)
            if dist>1e-8:events.append(dict(i=idx,d=dist,v=v,a=max(accel,1.),jerk=jerk,
                                    enter_dir=vstart,exit_dir=vend,arc=(radius>0)))
            state=new
    result={};block=[]
    def flush(b):
        if not b:return
        n=len(b);limits=np.zeros(n+1,dtype=float)
        for j in range(1,n):
            a=b[j-1];q=b[j]
            diff=np.abs(q['v']*q['enter_dir']-a['v']*a['exit_dir'])
            multiplier=min(1.,min(a['jerk'],q['jerk'])/(max(diff[0],diff[1],.000001)))
            limits[j]=max(0.,min(a['v'],q['v'])*multiplier)
        vel=limits.copy()
        for j in range(n-1,-1,-1):
            ev=b[j];vel[j]=min(vel[j],math.sqrt(max(0,vel[j+1]**2+2*ev['a']*ev['d'])))
        for j in range(n):
            ev=b[j];vel[j+1]=min(vel[j+1],math.sqrt(max(0,vel[j]**2+2*ev['a']*ev['d'])))
        for j,ev in enumerate(b):
            result[ev['i']]=(trapezoid(ev['d'],ev['v'],ev['a'],vel[j],vel[j+1]),vel[j],vel[j+1])
    last=-1
    for ev in events:
        if last>=0 and any(parsed[j][0] in ('G4','M109','M190','M400') for j in range(last+1,ev['i'])):
            flush(block);block=[]
        block.append(ev);last=ev['i']
    flush(block)
    return result

def state_features(program_rows):
    """Return (codes[T], raw_features[T,F], prior[T,3]) for ONE whole G-code program.
    No labels are touched. Offline lookahead uses at most next 2 commands.
    """
    rows=sorted(program_rows,key=lambda r:int(r['step_index']))
    p=json.loads(rows[0]['initial_profile']);thermal=Thermal(p)
    pos=np.array([p['start_x'],p['start_y'],p['start_z']],float)
    feed=1800.;acc_print=float(p['accel_print']);acc_travel=float(p['accel_travel']);jerk=float(p['jerk']);flow=1.;speed_mult=1.
    absolute=True;extruder_abs=True;e_pos=0.;ret_debt=0.;prev_vec=np.zeros(3);prev_feed=0.
    parsed=[parse(r['gcode']) for r in rows]
    plans=planned_motion_times(rows,p,parsed)
    codes=[];feats=[];priors=[]
    for i,(r,(c,a)) in enumerate(zip(rows,parsed)):
        prev=parsed[max(0,i-1)][0] if i else 'OTHER'
        nextc=parsed[min(i+1,len(rows)-1)][0] if i+1<len(rows) else 'OTHER'
        newpos,vec,length,v,radius,sweep=motion_info(c,a,pos,feed,absolute)
        if 'F' in a and c in ('G0','G1','G2','G3'):feed=a['F']
        v*=speed_mult
        # Estimate junction information from current/next segment.
        nextdir=np.zeros(3);nextfeed=0.;nextdist=0.
        if i+1<len(rows):
            nc,na=parsed[i+1]
            if nc in ('G0','G1','G2','G3'):
                _,nextdir,nextdist,nextfeed,_,_=motion_info(nc,na,newpos,feed,absolute)
        cos_next=0.;cos_prev=0.
        if np.linalg.norm(vec)>1e-8 and np.linalg.norm(nextdir)>1e-8:
            cos_next=float(np.dot(vec,nextdir)/(np.linalg.norm(vec)*np.linalg.norm(nextdir)))
        if np.linalg.norm(vec)>1e-8 and np.linalg.norm(prev_vec)>1e-8:
            cos_prev=float(np.dot(vec,prev_vec)/(np.linalg.norm(vec)*np.linalg.norm(prev_vec)))
        de=0.
        if c in ('G0','G1','G2','G3') and 'E' in a:
            e_new=(a['E'] if extruder_abs else e_pos+a['E'])
            de=(e_new-e_pos)*flow; e_pos=e_new
        accel=acc_travel if c=='G0' else acc_print
        # Approximate lookahead junction: geometry-dependent exit/entry speed.
        est_exit=0.;est_entry=0.
        if c in ('G0','G1','G2','G3'):
            if nextdist>1e-8:
                est_exit=min(v,nextfeed,jerk/(max(.1,math.sqrt(max(0,2-2*cos_next)))))
            if prev_feed>0 and np.linalg.norm(prev_vec)>1e-8:
                est_entry=min(v,prev_feed,jerk/(max(.1,math.sqrt(max(0,2-2*cos_prev)))))
        tt=0.;ee=0.;mm=0.
        if c in ('G0','G1','G2','G3'):
            # Arc motion's segmentation can shift time vs lumped arc estimate.
            plan=plans.get(i)
            if plan is not None:
                tt,est_entry,est_exit=plan
            else:
                tt=trapezoid(length,v,accel,est_entry,est_exit)
            ee=thermal.advance(tt,True)
            if de<0:ret_debt+=-de
            elif de>0:
                unretract=min(de,ret_debt);ret_debt-=unretract
                if length>1e-9:
                    mm=(de-unretract)*math.pi*(1.75/2)**2*1.24/1000.
            pos=newpos;prev_vec=vec.copy();prev_feed=v
        elif c=='G4':
            tt=max(0.,a.get('P',0)/1000+a.get('S',0));ee=thermal.advance(tt,False)
            prev_feed=0.
        elif c in ('M104','M109'):
            thermal.nt=a.get('S',a.get('R',thermal.nt))
            if c=='M109':tt,ee=thermal.wait('nozzle',thermal.nt,'R' in a)
            prev_feed=0.
        elif c in ('M140','M190'):
            thermal.bt=a.get('S',a.get('R',thermal.bt))
            if c=='M190':tt,ee=thermal.wait('bed',thermal.bt,'R' in a)
            prev_feed=0.
        elif c=='M106':thermal.fan=a.get('S',255.)
        elif c=='M107':thermal.fan=0.
        elif c=='M204':
            acc_print=a.get('P',a.get('S',acc_print));acc_travel=a.get('T',a.get('S',acc_travel))
        elif c=='M205':jerk=a.get('X',jerk)
        elif c=='M220':speed_mult=a.get('S',100)/100
        elif c=='M221':flow=a.get('S',100)/100
        elif c=='G90':absolute=True
        elif c=='G91':absolute=False
        elif c=='M82':extruder_abs=True
        elif c=='M83':extruder_abs=False
        elif c=='G92':
            for ax,k in enumerate('XYZ'):
                if k in a:pos[ax]=a[k]
            if 'E' in a:e_pos=a['E']
        # A compact physically meaningful feature vector, including state.
        f=[float(i)/max(1,len(rows)-1),float(CMAP.get(prev,CMAP['OTHER'])),float(CMAP.get(nextc,CMAP['OTHER'])),
           math.log1p(length),math.log1p(v),math.log1p(accel),math.log1p(jerk),
           cos_next,cos_prev,math.log1p(nextdist),math.log1p(nextfeed),
           math.log1p(est_entry),math.log1p(est_exit),math.log1p(radius),sweep,
           de,float(max(0.,de)),float(ret_debt),float(thermal.fan)/255,
           (thermal.nozzle-thermal.ambient)/200,(thermal.bed-thermal.ambient)/100,
           (thermal.nt-thermal.nozzle)/200,(thermal.bt-thermal.bed)/100,
           thermal.ambient/30,pos[0]/235,pos[1]/235,pos[2]/250,
           acc_print/1000,acc_travel/1000,speed_mult,flow,
           float(c=='G0'),float(c=='G1'),float(c in ('G2','G3')),
           float(c in ('M109','M190')),float(c=='G4'),
           math.log1p(max(tt,0.)),math.log1p(max(ee,0.)*3600)]
        codes.append(CMAP.get(c,CMAP['OTHER']));feats.append(f);priors.append([max(0.,tt),max(0.,ee),max(0.,mm)])
    return np.array(codes,np.int64),np.array(feats,np.float32),np.array(priors,np.float32)
