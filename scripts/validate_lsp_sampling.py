"""Diagnostic sampling from the preserved local table (B10/B11/B20).

This checks the implemented sampler and reports its moments. It is not an
independent OpenNTN benchmark or proof of standard conformance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from hybridchannel.lsp_calculator import get_scenario_params,sample_lsps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--samples',type=int,default=10000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--elevation-deg',type=float,default=40.)
    args=p.parse_args()
    table=get_scenario_params('NTN-DenseUrban-LOS',fc_GHz=28.,elev_deg=args.elevation_deg)
    rng=np.random.default_rng(args.seed)
    samples=[sample_lsps(table,rng) for _ in range(args.samples)]
    keys=['K_R_dB','DS','ASD','ASA','ZSD','ZSA']
    values=np.array([[row[k] if k=='K_R_dB' else np.log10(row[k]) for k in keys] for row in samples])
    report={'experiment_id':'E04-lsp-sampling','samples':args.samples,'seed':args.seed,
            'elevation_deg':args.elevation_deg,'columns':keys,'domain':'K in dB; other columns log10',
            'mean':values.mean(axis=0).tolist(),'std':values.std(axis=0,ddof=1).tolist(),
            'covariance':np.cov(values,rowvar=False).tolist(),
            'input_correlation_eigenvalues':np.linalg.eigvalsh(table['C']).tolist(),
            'note':'Original diagonal stabilization and angle clipping retained; empirical moments are not a conformance pass/fail test.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(args.output)


if __name__=='__main__':main()
