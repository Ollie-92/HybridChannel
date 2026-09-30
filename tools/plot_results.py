"""Plot PDP/CFR and export metrics from trusted simulation outputs."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--trusted-local-output',action='store_true',required=True)
    args=p.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out=args.run/'analysis'
    out.mkdir(exist_ok=False)
    with np.load(args.run/'channel.npz',allow_pickle=True) as data:
        rows=[]; cfrs={}
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        offsets=np.linspace(-10e6,10e6,129)
        for i,step in enumerate(data['valid_steps']):
            h=np.asarray(data['H_by_step'][i]); tau=np.asarray(data['tau_by_step'][i])
            power=np.abs(h[0,0])**2
            active=power>0
            t=tau[active]; pw=power[active].astype(float)
            total=float(pw.sum()); mean=float(np.sum(t*pw)/total) if total else 0.0
            ds=float(np.sqrt(np.sum(pw*(t-mean)**2)/total)) if total else 0.0
            rows.append({'step':int(step),'time_s':float(data['times_s'][step]),
                         'active_rays':int(active.sum()),'power_sum_linear':total,
                         'rms_delay_spread_s':ds})
            cfr=np.einsum('usnm,nmf->usf',h,np.exp(-2j*np.pi*tau[:,:,None]*offsets))
            cfrs[f'cfr_step_{step:04d}']=cfr
            if i==0 and len(t):
                excess=(t-t.min())*1e9
                axes[0].stem(excess,10*np.log10(pw/pw.max()),basefmt=' ')
                axes[1].plot(offsets/1e6,20*np.log10(np.maximum(np.abs(cfr[0,0]),1e-30)))
        if not rows:raise ValueError('No valid channel steps to analyze')
        cfrs['frequency_offsets_hz']=offsets
        np.savez_compressed(out/'cfr.npz',**cfrs)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    axes[0].set(xlabel='Excess delay (ns)',ylabel='Relative ray power (dB)',title='First valid step: ray PDP')
    axes[1].set(xlabel='Frequency offset (MHz)',ylabel='Magnitude (dB)',title='CIR Fourier sum, RX0/TX0')
    for ax in axes:ax.grid(alpha=0.25)
    fig.savefig(out/'pdp_cfr.png',dpi=160)
    plt.close(fig)
    (out/'analysis.json').write_text(json.dumps({'new_analysis':True,'pdp':'incoherent ray powers',
        'cfr':'sum h_nm exp(-j 2*pi*frequency_offset*tau_nm)',
        'delay_reference':'as stored: excess in Hybrid; absolute in consistent RC',
        'scope':'No claim of exact 3GPP/OpenNTN equivalence'},indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
