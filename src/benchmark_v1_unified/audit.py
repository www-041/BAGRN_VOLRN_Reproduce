"""Read-only source identity and GeoTIFF header audit."""
from __future__ import annotations
import csv, re
from pathlib import Path
import rasterio
from .identity import sha256_file

def _record_path(source_root: Path, scene_id: str) -> Path:
    m=re.search(r'(20\d{12})',scene_id)
    if not m: raise ValueError(f'cannot derive acquisition directory from scene id {scene_id}')
    folder=m.group(1)
    return Path(source_root)/folder/scene_id/f'{scene_id}_B9.TIF'

def audit_input_sources(source_root: Path, inventory_csv: Path):
    with Path(inventory_csv).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if len(rows)!=13: raise ValueError(f'expected 13 identity rows, got {len(rows)}')
    results=[]
    for row in rows:
        path=_record_path(Path(source_root),row['scene_id'])
        current=sha256_file(path) if path.is_file() else None
        match=current==row['recorded_source_sha256_task15']
        metadata={}
        if path.is_file():
            with rasterio.open(path) as ds:
                metadata={'width':ds.width,'height':ds.height,'dtype':ds.dtypes[0],'nodata':ds.nodata,'crs':ds.crs.to_string() if ds.crs else None,'transform':tuple(ds.transform)[:6],'resolution':tuple(ds.res)}
        results.append({'scene_index':int(row['scene_index']),'scene_id':row['scene_id'],'band':'B9','path':str(path),'exists':path.is_file(),'expected_sha256':row['recorded_source_sha256_task15'],'current_sha256':current,'hash_match':match,'file_size_bytes':path.stat().st_size if path.is_file() else None,**metadata})
    return {'status':'PASS' if all(r['exists'] and r['hash_match'] for r in results) else 'BLOCKED_INPUT_IDENTITY','scene_count':len(results),'verified_count':sum(r['exists'] and r['hash_match'] for r in results),'scenes':results}
