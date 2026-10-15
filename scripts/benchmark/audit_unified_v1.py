from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import rasterio
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.benchmark_v1_unified.config import load_benchmark_config
from src.benchmark_v1_unified.audit import audit_input_sources
from src.benchmark_v1_unified.geometry import load_grid,load_world_transforms
from src.benchmark_v1_unified.identity import sha256_file
from src.benchmark_v1_unified.reporting import write_json_atomic

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/benchmark_v1_unified.yaml');ap.add_argument('--output',default='data/output/benchmark_v1_unified/audit');args=ap.parse_args()
    cfg=load_benchmark_config(ROOT/args.config);v=cfg.values
    identity=audit_input_sources(Path(v['paths']['source_root']),ROOT/v['paths']['source_inventory'])
    grid=load_grid(ROOT/v['paths']['aligned_grid_manifest']);transforms=load_world_transforms(ROOT/v['paths']['source_transform_manifest'])
    task15_root=ROOT/v['paths']['task15_root'];maskdir=ROOT/v['paths']['task15_input_masks']
    mask_files=sorted(maskdir.glob('scene_*.tif'));mask_rows=[]
    for path in mask_files:
        with rasterio.open(path) as ds:
            exact=ds.crs.to_string()==grid.crs and tuple(ds.transform)[:6]==grid.affine and (ds.width,ds.height)==(grid.width,grid.height) and ds.dtypes[0]=='uint8'
            mask_rows.append({'path':str(path),'shape':[ds.height,ds.width],'crs':ds.crs.to_string() if ds.crs else None,'affine':tuple(ds.transform)[:6],'dtype':ds.dtypes[0],'nodata':ds.nodata,'grid_match':exact})
    geometry_pass=len(transforms)==13 and len(mask_rows)==13 and all(r['grid_match'] for r in mask_rows)
    report={'protocol_id':cfg.protocol_id,'status':'PASS' if identity['status']=='PASS' and geometry_pass else 'BLOCKED_INPUT_OR_GEOMETRY','identity':identity,'geometry':{'status':'PASS' if geometry_pass else 'BLOCKED_GEOMETRY_PROVENANCE','grid':{'crs':grid.crs,'affine':grid.affine,'width':grid.width,'height':grid.height,'pixel_area_m2':grid.pixel_area_m2},'world_transform_count':len(transforms),'transform_sha256':sha256_file(ROOT/v['paths']['source_transform_manifest']),'canonical_grid_sha256':sha256_file(ROOT/v['paths']['aligned_grid_manifest']),'intensity_resampling':'bilinear','mask_resampling':'nearest','mask_file_count':len(mask_files),'mask_manifest_sha256':sha256_file(ROOT/v['paths']['task15_root']/'stages/06_bagrn/bagrn/normalized_scenes_manifest.json'),'masks':mask_rows},'historical_output_policy':'read-only; no old method scores are evaluated here'}
    dest=ROOT/args.output/'input_audit.json';write_json_atomic(dest,report);print(json.dumps({'status':report['status'],'verified_sources':identity['verified_count'],'grid':[grid.width,grid.height], 'mask_files':len(mask_files),'report':str(dest)},ensure_ascii=False));return 0 if report['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
