from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.benchmark_v1_unified.identity import sha256_file
from src.benchmark_v1_unified.reporting import write_json_atomic
from src.benchmark_v1_unified.simulation import evaluate_s4_change,score_truth
from src.benchmark_v1_radiometric_r1.synthetic import generate_case

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixture-only',action='store_true',required=True);ap.add_argument('--output-root',default='data/output/benchmark_v1_unified/synthetic_smoke');args=ap.parse_args()
    out=ROOT/args.output_root
    if out.exists():raise SystemExit(f'refusing to overwrite smoke output: {out}')
    out.mkdir(parents=True)
    rng=np.random.default_rng(20261013);base=(np.sin(np.linspace(0,5,48))[None,:]+np.cos(np.linspace(0,4,32))[:,None]+rng.normal(0,.01,(32,48))).astype(np.float64);valid=np.ones(base.shape,bool)
    cases={};arrays={}
    configs={'S1':{'case':'S1','gain':1.2,'offset':-0.3},'S2':{'case':'S2','gain_base':1.0,'gain_gradient_x':.2,'offset_base':.1,'offset_gradient_y':.05},'S3':{'case':'S3','gains':[1.1,.9,1.2,1.0],'offsets':[.2,-.1,.3,0]},'S4':{'case':'S4','gain':1.15,'offset':.1,'change_region':[10,18,20,30],'change_amplitude':.6}}
    for case,cfg in configs.items():
        source=np.stack([base,base*1.7+2,base*.4-3,base*3+10]) if case=='S3' else base
        result=generate_case(source,valid,cfg,seed=20261013)
        arrays[f'{case}_clean']=result.clean_target;arrays[f'{case}_observed']=result.observed;arrays[f'{case}_valid']=result.valid_mask;arrays[f'{case}_change_mask']=result.change_mask
        cases[case]={'seed':20261013,'generator':'src/benchmark_v1_radiometric_r1/synthetic.py::generate_case','config':cfg,'status':'SYNTHETIC_FIXTURE_ONLY','truth_identity_check':score_truth(result.clean_target,result.clean_target,result.valid_mask)}
    s4=generate_case(base,valid,configs['S4'],seed=20261013)
    s4_after=s4.clean_target
    # Exact truth identity preserves designed change, while testing the S4 scorer.
    cases['S4']['change_score']=evaluate_s4_change(base,s4_after,base,s4_after,s4.change_mask,valid)
    npz=out/'s1_s4_fixture.npz';np.savez_compressed(npz,**arrays)
    report={'protocol_id':'DZ01V_UNIFIED_A_B_V1_20261013','status':'PASS','scope':'tiny synthetic fixture only; no real scenes and no radiometric algorithms','cases':cases,'fixture_npz':str(npz.relative_to(ROOT)),'fixture_sha256':sha256_file(npz)}
    write_json_atomic(out/'synthetic_smoke.json',report);print(json.dumps({'status':'PASS','cases':list(cases),'fixture':str(npz),'sha256':report['fixture_sha256']},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
