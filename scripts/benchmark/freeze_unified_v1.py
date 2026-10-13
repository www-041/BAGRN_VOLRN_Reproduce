from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import hashlib
import subprocess
import numpy as np
import rasterio
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.benchmark_v1_unified.config import load_benchmark_config
from src.benchmark_v1_unified.audit import audit_input_sources
from src.benchmark_v1_unified.geometry import load_grid,load_world_transforms
from src.benchmark_v1_unified.aligned import prepare_aligned_input
from src.benchmark_v1_unified.windows import make_fixed_windows,WindowConfig
from src.benchmark_v1_unified.identity import sha256_file
from src.benchmark_v1_unified.freeze import write_lock,verify_lock
from src.benchmark_v1_unified.reporting import write_json_atomic

def _csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def _write_csv(path,rows,fields):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def _digest_ordered(paths):
    h=hashlib.sha256()
    for p in paths:h.update(Path(p).name.encode('utf-8'));h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/benchmark_v1_unified.yaml');ap.add_argument('--output-root',default='data/output/benchmark_v1_unified/freeze_candidate_20261013');ap.add_argument('--skip-aligned-inputs',action='store_true');args=ap.parse_args()
    cfg=load_benchmark_config(ROOT/args.config);v=cfg.values;out=ROOT/args.output_root
    if out.exists():raise SystemExit(f'refusing to overwrite existing candidate directory: {out}')
    out.mkdir(parents=True);input_dir=out/'aligned_inputs';input_dir.mkdir()
    identity=audit_input_sources(Path(v['paths']['source_root']),ROOT/v['paths']['source_inventory'])
    if identity['status']!='PASS':
        write_json_atomic(out/'freeze_readiness.json',{'status':'NOT_FROZEN','reason':'BLOCKED_INPUT_IDENTITY','identity':identity});return 2
    grid=load_grid(ROOT/v['paths']['aligned_grid_manifest']);transforms=load_world_transforms(ROOT/v['paths']['source_transform_manifest'])
    rows=_csv(ROOT/v['paths']['source_inventory']);old_masks=ROOT/v['paths']['task15_input_masks'];mask_audits=[];aligned=[]
    for row in rows:
        i=int(row['scene_index']);source=Path(v['paths']['source_root'])/row['scene_id'][row['scene_id'].find('202'):row['scene_id'].find('202')+14]/row['scene_id']/f"{row['scene_id']}_B9.TIF"
        if not source.is_file():
            # Folder key is the acquisition timestamp embedded in the product ID.
            import re
            stamp=re.search(r'(20\d{12})',row['scene_id']).group(1);source=Path(v['paths']['source_root'])/stamp/row['scene_id']/f"{row['scene_id']}_B9.TIF"
        refmask=old_masks/f'scene_{i:03d}.tif'
        if args.skip_aligned_inputs:
            aligned.append({'scene_index':i,'source_path':str(source),'source_sha256':identity['scenes'][i]['current_sha256'],'aligned_path':None,'mask_path':str(refmask),'mask_source':'read_only_task15_stage06','status':'REUSED_VERIFIED_MASK_ONLY'})
            continue
        artifact=prepare_aligned_input(source,i,grid,transforms[i],input_dir,verify_mask_path=refmask)
        aligned.append({**artifact,'source_path':str(source),'source_sha256':identity['scenes'][i]['current_sha256']})
        mask_audits.append({'scene_index':i,'status':'PASS','mask_verified_against_task15':True})
    times={r['pair_id']:r for r in _csv(ROOT/v['paths']['temporal_inventory'])}
    masks_dir=input_dir if not args.skip_aligned_inputs else old_masks
    pair_rows=[]
    with rasterio.open(masks_dir/'scene_000_input_mask.tif' if not args.skip_aligned_inputs else masks_dir/'scene_000.tif') as base:
        windows=list(base.block_windows(1));grid_fingerprint={'crs':base.crs.to_string(),'transform':tuple(base.transform)[:6],'width':base.width,'height':base.height}
    mask_paths=[(masks_dir/f'scene_{i:03d}_input_mask.tif' if not args.skip_aligned_inputs else masks_dir/f'scene_{i:03d}.tif') for i in range(13)]
    # Count all pair intersections in one streaming pass, bounded to one raster block per scene.
    from contextlib import ExitStack
    from itertools import combinations
    counts={(i,j):0 for i,j in combinations(range(13),2)}
    with ExitStack() as stack:
        ds=[stack.enter_context(rasterio.open(p)) for p in mask_paths]
        for _,win in ds[0].block_windows(1):
            chunk=[s.read(1,window=win)>0 for s in ds]
            for i,j in counts:counts[i,j]+=int(np.count_nonzero(chunk[i]&chunk[j]))
    for i,j in counts:
        pid=f'{i:02d}_{j:02d}';old=times.get(pid,{});n=counts[i,j]
        pair_rows.append({'pair_id':pid,'scene_i':rows[i]['scene_id'],'scene_j':rows[j]['scene_id'],'time_stratum':old.get('time_bin','UNKNOWN'),'days_apart':old.get('days_apart',''),'quality_i':old.get('quality_i','UNKNOWN'),'quality_j':old.get('quality_j','UNKNOWN'),'native_valid_overlap_pixels':old.get('valid_overlap_pixels',''),'aligned_valid_overlap_pixels':n,'status':'OVERLAP' if n else 'NO_VALID_OVERLAP','inventory_definition':'Task15 canonical grid input masks, bilinear intensity warp; exact mask intersection'})
    _write_csv(out/'pair_overlap_inventory.csv',pair_rows,list(pair_rows[0]))
    windows_out=[]
    if not args.skip_aligned_inputs:
        for p in pair_rows:
            if p['status']!='OVERLAP':continue
            i,j=map(int,p['pair_id'].split('_'))
            with rasterio.open(mask_paths[i]) as a,rasterio.open(mask_paths[j]) as b:
                support=(a.read(1)>0)&(b.read(1)>0)
            refs=make_fixed_windows(p['pair_id'],support,WindowConfig(),time_stratum=p['time_stratum'])
            for ref in refs:windows_out.append({**ref.__dict__,'time_stratum':p['time_stratum'],'quality_i':p['quality_i'],'quality_j':p['quality_j']})
    _write_csv(out/'fixed_windows.csv',windows_out,list(windows_out[0]) if windows_out else ['window_id','pair_id','row_off','col_off','height','width','valid_count','support_sha256','stratum','time_stratum','quality_i','quality_j'])
    metric_spec={'protocol_id':cfg.protocol_id,'band':'B9','unit':'UNKNOWN','support':'frozen_input_mask_intersection','primary_eligibility':'full frozen input support required','metrics':{'MAE_V1':'mean(abs(x-y))','RMSE_V1':'sqrt(mean((x-y)^2))','BIAS_V1':'mean(x-y)','P95AE_V1':'linear_quantile(abs(x-y),0.95)','MAMD_V1':'abs(mean(x)-mean(y))','MSDD_V1':'abs(population_std_ddof0(x)-population_std_ddof0(y))','RDD_W1_EXACT_V1':'exact empirical 1D Wasserstein-1'},'aggregation':{'primary':'pair_macro','secondary':['pixel_weighted','pair_median','pair_p95']},'paper_metrics':{'CD':'FORMULA_UNVERIFIED','GL':'FORMULA_UNVERIFIED','RDOA':'NOT_COMPUTED','Ave':'NOT_COMPUTED'},'structure':{'direction':'SOBEL_DIRECTION_LOSS_V1; not paper GL','gradient_kernel':'3x3 Sobel / 8','edge_threshold':'max(1e-12, input-gradient P50)'},'windows':{'size_px':[128,128],'stride_px':128,'min_support_fraction':0.95,'cap_per_pair':32,'seed':20261013}}
    write_json_atomic(out/'metric_spec.json',metric_spec)
    code_paths=sorted([*ROOT.glob('src/benchmark_v1_unified/**/*.py'),*ROOT.glob('scripts/benchmark/**/*.py'),*ROOT.glob('tests/benchmark_v1_unified/**/*.py'),ROOT/'docs/benchmark/v1/unified_v1/plan.md',ROOT/'docs/benchmark/v1/unified_v1/source_formula_crosswalk.md',ROOT/'docs/benchmark/v1/unified_v1/benchmark_protocol_draft.md'])
    code_entries=[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha256_file(p)} for p in code_paths if p.is_file()]
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    branch=subprocess.run(['git','branch','--show-current'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    code_manifest={'protocol_id':cfg.protocol_id,'git_branch':branch,'git_head':head,'code_changes_committed':False,'files':code_entries,'file_count':len(code_entries)}
    write_json_atomic(out/'code_manifest.json',code_manifest)
    counts_summary={'total_pairs':len(pair_rows),'positive_overlap_pairs':sum(p['status']=='OVERLAP' for p in pair_rows),'no_valid_overlap_pairs':sum(p['status']=='NO_VALID_OVERLAP' for p in pair_rows),'native_inventory_positive_expected':48,'native_inventory_zero_expected':30,'aligned_00_01':next(p['aligned_valid_overlap_pixels'] for p in pair_rows if p['pair_id']=='00_01'),'native_00_01':next(p['native_valid_overlap_pixels'] for p in pair_rows if p['pair_id']=='00_01'),'aligned_00_10':next(p['aligned_valid_overlap_pixels'] for p in pair_rows if p['pair_id']=='00_10'),'native_00_10':next(p['native_valid_overlap_pixels'] for p in pair_rows if p['pair_id']=='00_10')}
    passed=counts_summary['positive_overlap_pairs']==48 and counts_summary['no_valid_overlap_pairs']==30 and (not args.skip_aligned_inputs and len(mask_audits)==13)
    write_json_atomic(out/'source_identity.json',identity)
    mask_manifest={'grid':grid_fingerprint,'masks':[{'scene_index':i,'path':str(mask_paths[i]),'sha256':sha256_file(mask_paths[i])} for i in range(13)],'definition':'explicit Task15-equivalent source nodata and finite bilinear-reprojection mask'}
    write_json_atomic(out/'mask_manifest.json',mask_manifest)
    input_freeze_sha=_digest_ordered([out/'source_identity.json',out/'mask_manifest.json',out/'pair_overlap_inventory.csv',out/'fixed_windows.csv'])
    geometry_freeze_sha=_digest_ordered([ROOT/v['paths']['aligned_grid_manifest'],ROOT/v['paths']['source_transform_manifest']])
    smoke_path=ROOT/'data/output/benchmark_v1_unified/synthetic_smoke/synthetic_smoke.json'
    smoke=json.loads(smoke_path.read_text(encoding='utf-8'))
    synthetic_npz=ROOT/smoke['fixture_npz']
    generator_path=ROOT/'src/benchmark_v1_radiometric_r1/synthetic.py'
    synthetic_manifest={'status':'SYNTHETIC_FIXTURE_ONLY','generator_path':str(generator_path),'generator_sha256':sha256_file(generator_path),'npz_path':str(synthetic_npz),'npz_sha256':sha256_file(synthetic_npz),'seed':20261013,'cases':['S1','S2','S3','S4'],'unit':'ARBITRARY_SYNTHETIC_UNIT','no_method_runs':True}
    write_json_atomic(out/'synthetic_truth_manifest.json',synthetic_manifest)
    manifest={'protocol_id':cfg.protocol_id,'status':'FROZEN' if passed else 'NOT_FROZEN','data_scope':'13-scene DZ01V B9; radiometric physical unit UNKNOWN','identity_status':identity['status'],'config_sha256':sha256_file(ROOT/args.config),'code_manifest_sha256':sha256_file(out/'code_manifest.json'),'git_head':head,'git_branch':branch,'source_manifest_sha256':sha256_file(ROOT/v['paths']['source_inventory']),'source_inventory_sha256':sha256_file(ROOT/v['paths']['source_inventory']),'temporal_inventory_sha256':sha256_file(ROOT/v['paths']['temporal_inventory']),'grid':grid_fingerprint,'geometry_transform_sha256':sha256_file(ROOT/v['paths']['source_transform_manifest']),'canonical_grid_sha256':sha256_file(ROOT/v['paths']['aligned_grid_manifest']),'input_freeze_sha256':input_freeze_sha,'geometry_freeze_sha256':geometry_freeze_sha,'mask_manifest_sha256':sha256_file(out/'mask_manifest.json'),'method_independent_metric_spec_sha256':sha256_file(out/'metric_spec.json'),'synthetic_truth_manifest_sha256':sha256_file(out/'synthetic_truth_manifest.json'),'intensity_resampling':'bilinear (verified Task15 implementation)','mask_resampling':'nearest for masks; validity is finite output after declared source nodata handling','scene_count':13,'source_hash_match_count':identity['verified_count'],'pair_counts':counts_summary,'aligned_input_artifacts':aligned,'window_count':len(windows_out),'freeze_claim':'input/geometry/support protocol only; no methods evaluated'}
    write_json_atomic(out/'frozen_manifest.json',manifest)
    files=[ROOT/args.config,ROOT/v['paths']['source_inventory'],ROOT/v['paths']['temporal_inventory'],ROOT/v['paths']['aligned_grid_manifest'],ROOT/v['paths']['source_transform_manifest'],generator_path,synthetic_npz,out/'code_manifest.json',out/'source_identity.json',out/'mask_manifest.json',out/'metric_spec.json',out/'synthetic_truth_manifest.json',out/'pair_overlap_inventory.csv',out/'fixed_windows.csv',out/'frozen_manifest.json']
    files.extend(Path(a['path']) for a in identity['scenes'])
    if not args.skip_aligned_inputs:
        for a in aligned:files.extend([Path(a['aligned_path']),Path(a['mask_path'])])
    else: files.extend(mask_paths)
    try:lock=write_lock(out,files)
    except Exception as exc:
        readiness={'status':'NOT_FROZEN','reasons':[f'LOCK_CREATION_FAILED:{exc}'],'pair_counts':counts_summary}
    else:
        lock_verification=verify_lock(out)
        ready=passed and lock_verification.valid
        readiness={'status':'FROZEN' if ready else 'NOT_FROZEN','reasons':[] if ready else (['OVERLAP_COUNTS_OR_MASK_AUDIT_MISMATCH'] if not passed else list(lock_verification.errors)),'pair_counts':counts_summary,'window_count':len(windows_out),'lock_path':str(lock),'lock_verification':{'valid':lock_verification.valid,'errors':list(lock_verification.errors)},'locked_file_count':len(files)}
    write_json_atomic(out/'freeze_readiness.json',readiness)
    print(json.dumps({'status':readiness['status'],'pairs':counts_summary,'windows':len(windows_out),'output':str(out)},ensure_ascii=False));return 0 if readiness['status']=='FROZEN' else 2
if __name__=='__main__':raise SystemExit(main())
