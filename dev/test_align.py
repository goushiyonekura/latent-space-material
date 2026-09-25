import sys, time, numpy as np
sys.path.insert(0,'.')
from scoremap import mxl, align
S='materials/scores-diffusion/'; A='inputs/diffusion-003/'
def run(sec, stems, repeats=(), ref=None):
    scores=[mxl.parse(S+n+'.musicxml') for n in stems]
    alns=[]
    for sc,n in zip(scores,stems):
        t0=time.time(); al=align.align([sc],[A+n+'.wav'],ref_tempo=ref,repeats=repeats); alns.append(al)
        print(f"  {n[:26]:26s} cost {al.cost:.4f} passes {al.seg.max()+1} ({time.time()-t0:.1f}s)")
    t0=time.time(); J=align.align(scores,[A+n+'.wav' for n in stems],ref_tempo=ref,repeats=repeats)
    print(f"  JOINT cost {J.cost:.4f} passes {J.seg.max()+1} ({time.time()-t0:.1f}s)")
    grid=np.arange(0.5, J.audio_dur-0.5, 0.25)
    qj=np.array([J.q_at(t)[0] for t in grid])
    for n,al in zip(stems,alns):
        q=np.array([al.q_at(t)[0] for t in grid]); d=np.abs(q-qj)
        print(f"   {n[:20]:20s} vs joint: median {np.median(d):.3f} q, 90% {np.percentile(d,90):.3f} q, max {d.max():.2f} q")
    return J, alns
if __name__=='__main__':
    sec=sys.argv[1]
    stems={'papillon_1':['papillon_1_00.00.000_01.29.318','papillon_viola_1_00.00.000_01.29.318','papillon_violin_1_00.00.000_01.29.318'],
           'prism_1':['prism_1_00.00.000_01.28.904','prism_cello_1_00.00.000_01.28.904','prism_viola_1_00.00.000_01.28.904']}[sec]
    J,_=run(sec, stems)
    # tempo profile of the joint path: quarters per minute over 5-second windows
    for t in np.arange(0, J.audio_dur, 5.0):
        q0=J.q_at(t)[0]; q1=J.q_at(min(t+5, J.audio_dur))[0]
        print(f"   {t:5.1f}s  q {q0:6.2f} -> {q1:6.2f}  tempo {60*(q1-q0)/5:5.1f}")
