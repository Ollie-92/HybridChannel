"""Compare every numeric intermediate against a pristine PRIVATE source copy.

Never point --reference-dir at the original projects. This is a developer
validation command, not a runtime dependency of the standalone package.
"""
import argparse
import ast
import copy
import importlib
import json
from pathlib import Path
import sys
import numpy as np
from hybridchannel import hybrid_channel as extracted
from hybridchannel import consistent_random_clusters as extracted_rc


def rt_fixture():
    """Synthetic scalar RT input; not an empirical or conference result."""
    return [{'tau_rt':0.003+float(d)*1e-9,'power_rt':float(p)*1e-16,
             'aoa':float(a),'aod':float(a+np.pi),'zoa':0.8+i*0.1,'zod':2.3-i*0.05}
            for i,(d,p,a) in enumerate(zip([0,9,24,61,110],[1,.2,.09,.03,.005],
                                          [0.1,0.4,-0.6,1.0,-1.2]))]


def compare(a,b,path,counts):
    if isinstance(a,dict):
        assert set(a)==set(b),path+' keys'
        for key in a:compare(a[key],b[key],path+'/'+str(key),counts)
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path+' length'
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),counts)
    elif isinstance(a,np.ndarray):
        assert a.dtype==b.dtype and a.shape==b.shape,path+' dtype/shape'
        np.testing.assert_array_equal(a,b,err_msg=path)
        counts['array_elements']+=a.size
    elif isinstance(a,(float,int,complex,np.number)):
        assert a==b or (np.isnan(a) and np.isnan(b)),path
        counts['scalars']+=1
    else:assert a==b,path


class Normalize(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        node.level=0
        return node
    def visit_FunctionDef(self,node):
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
            node.body.pop(0)
        return self.generic_visit(node)
    visit_ClassDef=visit_FunctionDef


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    ref=args.reference_dir.resolve()
    sys.path.insert(0,str(ref))
    original=importlib.import_module('hybrid_channel')
    original_rc=importlib.import_module('consistent_random_clusters')
    tests=[];counts={'array_elements':0,'scalars':0}
    rx=np.array([[0,0,0],[.005,0,0]],dtype=float)
    tx=np.array([[0,0,0],[.005,0,0],[.01,0,0]],dtype=float)
    for seed in (7,42,123,2026):
        for m in (1,3,20):
            kwargs=dict(rt_paths=rt_fixture(),scenario_name='NTN-DenseUrban-LOS',
                rx_array_geom=rx,tx_array_geom=tx,fc_GHz=28.0,elev_deg=40.0,
                velocity_vec=np.array([0.,7560.,0.]),t=.037,
                config={'M':m,'is_LOS_first_cluster':True},tx_rx_distance_3D=900000.)
            a=original.generate_full_channel(**copy.deepcopy(kwargs),rng=np.random.default_rng(seed))
            b=extracted.generate_full_channel(**copy.deepcopy(kwargs),rng=np.random.default_rng(seed))
            compare(a,b,f'hybrid/{seed}/M{m}',counts)
            tests.append({'kind':'hybrid','seed':seed,'M':m,'H_shape':list(b['channel']['H'].shape),'passed':True})
        params=dict(rt_paths_t0=rt_fixture(),scenario_name='NTN-DenseUrban-LOS',
                    p_ut=np.array([0.,0.,2.]),p_sat0=np.array([715050.,0.,600000.]),
                    fc_GHz=28.0,elev_deg=40.0,seed=seed)
        sa=original_rc.freeze_random_cluster_state(**copy.deepcopy(params))
        sb=extracted_rc.freeze_random_cluster_state(**copy.deepcopy(params))
        compare(sa,sb,f'state/{seed}',counts)
        for t in (0.,.01,.2):
            kw=dict(p_sat=params['p_sat0']+np.array([0.,7560.,0.])*t,
                    rt_paths_current=rt_fixture(),rx_array_geom=rx,tx_array_geom=tx,t=t)
            a=original_rc.evaluate_random_clusters(sa,**kw)
            b=extracted_rc.evaluate_random_clusters(sb,**kw)
            compare(a,b,f'rc/{seed}/{t}',counts)
        tests.append({'kind':'consistent_rc','seed':seed,'steps':3,'passed':True})
    structural=[]
    for name in ('hybrid_channel','consistent_random_clusters','cluster_splitter','lsp_calculator'):
        new_path=Path(importlib.import_module('hybridchannel.'+name).__file__)
        old_ast=ast.parse((ref/(name+'.py')).read_text())
        new_ast=ast.parse(new_path.read_text())
        olds={n.name:n for n in old_ast.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for node in new_ast.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                old=Normalize().visit(copy.deepcopy(olds[node.name]))
                new=Normalize().visit(copy.deepcopy(node))
                assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False),(name,node.name)
                structural.append(name+'.'+node.name)
    report={'passed':True,'comparison':'exact equality (rtol=0, atol=0), same clean environment',
            'synthetic_fixture':True,'does_not_certify_3gpp':True,'counts':counts,
            'cases':tests,'unchanged_retained_function_bodies':structural}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'passed':True,'cases':len(tests),'unchanged_functions':len(structural),**counts}))


if __name__=='__main__':main()
