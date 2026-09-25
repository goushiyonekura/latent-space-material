"""Rough monophonic transcription of a stem: onsets (spectral flux peaks) + f0 by harmonic summation."""
import sys, numpy as np
sys.path.insert(0,'.')
from latent_space.audio_io import read_wav
NAMES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def nm(m): m=int(round(m)); return NAMES[m%12]+str(m//12-1)
def transcribe(path, t_max=30.0, hop=512, nfft=8192):
    x,info=read_wav(path); fs=info.sample_rate; x=x.astype(np.float64).mean(axis=1)[:int(t_max*fs)]
    pad=np.concatenate([np.zeros(nfft//2),x,np.zeros(nfft//2)])
    fr=np.lib.stride_tricks.sliding_window_view(pad,nfft)[::hop]
    S=np.abs(np.fft.rfft(fr*np.hanning(nfft),axis=1)); f=np.fft.rfftfreq(nfft,1/fs)
    L=np.log1p(1000*S/S.max())
    flux=np.maximum(0,np.diff(L,axis=0,prepend=L[:1])).sum(axis=1)
    db=20*np.log10(np.sqrt((fr**2).mean(axis=1))+1e-9)
    # f0 by harmonic summation over candidate midi 36..96
    cands=np.arange(36,97,0.25); f0s=440*2**((cands-69)/12)
    idx=lambda fq: np.clip(np.round(fq/(fs/nfft)).astype(int),0,S.shape[1]-1)
    H=np.zeros((len(fr),len(cands)))
    for h in range(1,6):
        H+=np.log1p(1000*S[:,idx(f0s*h)]/S.max())*(0.85**(h-1))
    best=cands[np.argmax(H,axis=1)]
    # onsets: local maxima of flux above adaptive threshold
    thr=np.median(flux)+1.5*np.std(flux)
    on=[i for i in range(1,len(flux)-1) if flux[i]>thr and flux[i]>=flux[i-1] and flux[i]>=flux[i+1]]
    t=np.arange(len(fr))*hop/fs
    out=[]
    for k,i in enumerate(on):
        j=on[k+1] if k+1<len(on) else len(fr)
        seg=slice(i+2, max(i+3, min(j, i+12)))
        m=np.median(best[seg])
        out.append((t[i], m, db[i]))
    return out
if __name__=='__main__':
    for t,m,d in transcribe(sys.argv[1], float(sys.argv[2]) if len(sys.argv)>2 else 30):
        print(f"{t:7.3f}s  {nm(m):4s} ({m:5.1f})  {d:6.1f} dB")
