"""Compare pristine-copy and packaged behavior on the same actual RT solution.

Use a PRIVATE copied reference directory and a trusted project run. A separate
seeded RT rerun is compared to the earlier saved channel, but same-input parity
is the primary extraction check because RT discovery can be nondeterministic.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import numpy as np
from hybridchannel import trajectory as new_driver
from hybridchannel.hybrid_channel import generate_full_channel as new_generate
from hybridchannel.consistent_random_clusters import freeze_random_cluster_state as new_freeze
from hybridchannel.consistent_random_clusters import evaluate_random_clusters as new_evaluate
from compare_baseline import compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.reference_dir.resolve()))
    old_driver=importlib.import_module('run_satellite_trajectory_channel')
    old_core=importlib.import_module('hybrid_channel')
    old_rc=importlib.import_module('consistent_random_clusters')
    config=json.loads((args.run/'config.json').read_text());cli=config['cli']
    with np.load(args.run/'channel.npz',allow_pickle=True) as z:
        pos=z['sat_positions_m'][0];vel=z['sat_velocities_mps'][0]
        rx=z['rx_position_m'];t=float(z['times_s'][0])
        earlier_h=np.asarray(z['H_by_step'][0]);earlier_tau=np.asarray(z['tau_by_step'][0])
    new_driver.set_mitsuba_variant()
    scene=new_driver.build_scene(config['resolved_scene'],cli['fc-ghz']*1e9,cli['tx-power-dbm'],rx)
    scene.transmitters['Sat'].position=new_driver.as_float_list(pos.astype(np.float32))
    solver=new_driver.PathSolver()
    los=solver(scene=scene,max_depth=0,los=True,specular_reflection=False,
        diffuse_reflection=False,refraction=False,synthetic_array=True,
        samples_per_src=cli['samples-per-src'],max_num_paths_per_src=1,seed=cli['seed'])
    paths=solver(scene=scene,max_depth=cli['max-depth'],los=True,specular_reflection=True,
        diffuse_reflection=True,refraction=True,synthetic_array=True,
        samples_per_src=cli['samples-per-src'],max_num_paths_per_src=cli['max-num-paths'],seed=cli['seed']+10000)
    a=old_driver.extract_paths_from_rt(paths,los)
    b=new_driver.extract_paths_from_rt(paths,los)
    counts={'array_elements':0,'scalars':0}
    compare(a,b,'extraction',counts)
    old_input=old_driver.rt_paths_to_hybrid_input(a,cli['rt-pre-filter-db'])
    new_input=new_driver.rt_paths_to_hybrid_input(b,cli['rt-pre-filter-db'])
    compare(old_input,new_input,'filtering',counts)
    geom=new_driver.make_array_geom(cli['fc-ghz'],cli['num-ant'])
    params=dict(scenario_name='NTN-DenseUrban-LOS',rx_array_geom=geom,tx_array_geom=geom,
                elev_deg=new_driver.elevation_deg(pos,rx),fc_GHz=cli['fc-ghz'],velocity_vec=vel,t=t,
                config={'is_LOS_first_cluster':True},tx_rx_distance_3D=float(np.linalg.norm(pos-rx)))
    old=old_core.generate_full_channel(old_input,**copy.deepcopy(params),rng=np.random.default_rng(cli['seed']))
    new=new_generate(new_input,**copy.deepcopy(params),rng=np.random.default_rng(cli['seed']))
    compare(old,new,'real_city_hybrid',counts)
    state_params=dict(scenario_name='NTN-DenseUrban-LOS',p_ut=rx,p_sat0=pos,fc_GHz=cli['fc-ghz'],
        elev_deg=params['elev_deg'],seed=cli['seed'],config={'is_LOS_first_cluster':True})
    so=old_rc.freeze_random_cluster_state(old_input,**copy.deepcopy(state_params))
    sn=new_freeze(new_input,**copy.deepcopy(state_params));compare(so,sn,'real_city_frozen_state',counts)
    for dt in (0.,.1,.2):
        kw=dict(p_sat=pos+vel*dt,rt_paths_current=new_input,rx_array_geom=geom,tx_array_geom=geom,t=dt)
        compare(old_rc.evaluate_random_clusters(so,**kw),new_evaluate(sn,**kw),f'real_city_rc/{dt}',counts)
    nh=new['channel']['H'];nt=new['channel']['tau']
    same_shape=nh.shape==earlier_h.shape and nt.shape==earlier_tau.shape
    repeated_exact=same_shape and np.array_equal(nh,earlier_h) and np.array_equal(nt,earlier_tau)
    report={'passed_same_input_parity':True,'rt_samples_per_source':cli['samples-per-src'],
        'raw_rt_paths':len(a['delays']),'filtered_rt_paths':len(new_input),'H_shape':list(nh.shape),
        'same_input_tolerance':{'rtol':0,'atol':0},'same_input_counts':counts,
        'independent_seeded_rt_rerun':{'same_shape':same_shape,'exactly_equal':bool(repeated_exact)},
        'scope':'one actual city RT position; all intermediate Hybrid values and three fixed-state RC evaluations compared'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__':main()
