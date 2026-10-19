"""Gravity/subtree representation for GMAW motion; uses public config and target commanded q ONLY.
No private simulator coefficients, true target energies or target telemetry are consulted.
"""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from arc_base_physics import fk_positions,torch_tips

def gravity_context(cfg,program):
    n=cfg['n'];pa=np.asarray(cfg['parents'],int);mass=np.asarray(cfg['mass'],float).copy()
    edges=torch_tips(cfg);sub=np.zeros(n)
    pm=program['segments'][0].get('payload_kg',1.)
    for i in edges: mass[i]+=pm/max(1,len(edges))
    sub[:]=mass
    for i in range(n-1,-1,-1):
        if pa[i]>=0:sub[pa[i]]+=sub[i]
    depth=np.zeros(n)
    for i in range(n):
        if pa[i]>=0:depth[i]=depth[pa[i]]+1
    joint_up=np.zeros(n);joint_down=np.zeros(n);net=np.zeros(n);delta=np.zeros(n);max_tau=np.zeros(n)
    ancestors=[]
    for k in range(n):
        an=[];r=k
        while r>=0:an.append(r);r=pa[r]
        ancestors.append(an)
    for seg in program['segments']:
        q0=np.asarray(seg['q0'],float);q1=np.asarray(seg['q1'],float);d=q1-q0
        # Midpoint approximation to signed conservative potential work. Orientation determines its sign.
        q=.5*(q0+q1);rotation,pivot=fk_positions(cfg,q)
        axes=[]
        for i in range(n):
            rp=np.eye(3) if pa[i]<0 else rotation[pa[i]]
            axes.append(rp@np.asarray(cfg['axes'][i],float))
        cm=[pivot[i]+rotation[i]@np.asarray(cfg['coms'][i],float) for i in range(n)]
        tau=np.zeros(n)
        for k in range(n):
            for i in ancestors[k]:
                tau[i]+=mass[k]*9.81*np.cross(axes[i],cm[k]-pivot[i])[2]
        work=tau*d
        joint_up+=np.maximum(work,0);joint_down+=np.maximum(-work,0);net+=work
        delta+=d;max_tau=np.maximum(max_tau,np.abs(tau))
    j=np.zeros((15,10),np.float32)
    for i in range(n):
        j[i]=[np.log1p(joint_up[i]),np.log1p(joint_down[i]),np.sign(net[i])*np.log1p(abs(net[i])),
            np.sign(delta[i])*np.log1p(abs(delta[i])),np.log1p(max_tau[i]),
            np.sign(net[i])*np.log1p(max_tau[i]),np.log1p(sub[i]),sub[i]/max(mass.sum(),.01),
            depth[i]/15.,float(joint_up[i]+joint_down[i]<1e-8)]
    allup=joint_up.sum();alldown=joint_down.sum()
    g=np.array([np.log1p(allup),np.log1p(alldown),np.sign(net.sum())*np.log1p(abs(net.sum())),
         allup/(allup+alldown+1e-8),np.log1p(mass.sum()),np.log1p(sub.max()),
         float(np.mean(np.abs(delta)<1e-8)),float(np.mean(j[:n,9]))],np.float32)
    return j,g

def generate(split):
    base=ROOT/'data/arc_welding'
    machines={m['robot_id']:m['robot'] for m in json.loads((base/f'{split}_machines.json').read_text())}
    js=[];gs=[]
    for line in (base/f'{split}_programs.jsonl').open():
        record=json.loads(line)
        j,g=gravity_context(machines[record['robot_id']],record['program'])
        js.append(j);gs.append(g)
    assert len(js)==len(np.load(base/f'{split}.npz')['y'])
    np.savez_compressed(base/f'{split}_gravity_v53.npz',joint_gravity=np.array(js,np.float32),global_gravity=np.array(gs,np.float32))
    print(split,len(js),'signed gravity complete',flush=True)
if __name__=='__main__':
    for split in ('train','val','test','ood'):generate(split)
