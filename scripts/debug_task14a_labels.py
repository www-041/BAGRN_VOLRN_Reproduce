"""Short-lived diagnostic for the strict five-scene label replay."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_task13a1_source_side import _load_saved_seam
from src.seam_local.multiscene_label import aggregate_multiscene_labels, build_pairwise_preference_field
from src.seam_local.adapter import _expand_seam_to_full, _seam_domain
from src.seam_local.pipeline import PairResult
from dataclasses import replace
from types import SimpleNamespace

BASE=ROOT/'data/output/b9_five_scene_validation'
def main():
    protocol=json.loads((BASE/'seam_local_task13a/protocol.json').read_text())
    paths=[ROOT/s['path'] for s in protocol['scenes']]
    with rasterio.open(paths[0]) as t: transform=t.transform; shape=(t.height,t.width)
    masks=np.stack([rasterio.open(p).read_masks(1)>0 for p in paths])
    rows=list(csv.DictReader((BASE/'multiscene_task13b/authoritative_pair_table.csv').open(encoding='utf-8')))
    metrics={r['pair_id']:r for r in csv.DictReader((BASE/'seam_local_task13a/pair_metrics.csv').open(encoding='utf-8'))}
    fields={}
    for r in rows:
        if r['ownership_status']!='PASS': continue
        i,j=int(r['scene_i']),int(r['scene_j']); m=metrics[r['pair_id']]
        origin=tuple(json.loads(m['diagnostic_crop_origin']))
        pairdir=BASE/Path(r['refined_seam_path']).parent
        outer=(int(m['diagnostic_crop_origin'].split(',')[0]) if False else None)
        # Saved seam loader needs the crop shape; derive it from the frozen pair union crop.
        from src.seam_local.pipeline import _crop_for_pair
        va=masks[i]; vb=masks[j]; outer_slice,_,_= _crop_for_pair(None,None,va,vb)
        shp=(outer_slice[0].stop-origin[0],outer_slice[1].stop-origin[1])
        local=_load_saved_seam(pairdir/'seam_refined.geojson',origin,transform,shp)
        fake=SimpleNamespace(refined_seam=local,crop_origin=origin)
        from src.seam_local.adapter import _expand_seam_to_full
        fullseam=_expand_seam_to_full(fake,shape)
        f=build_pairwise_preference_field(masks[i],masks[j],fullseam,r['side_1_source'],scene_a=i,scene_b=j)
        dom=np.zeros(shape,bool)
        if local.orientation=='vertical': dom[origin[0]:origin[0]+len(local.row_col_path),:]=True
        else: dom[:,origin[1]:origin[1]+len(local.row_col_path)]=True
        avail=f.available & dom
        fields[(i,j)]=replace(f,available=avail,vote=np.where(avail,f.vote,0).astype(np.float32),confidence=np.where(avail,f.confidence,0).astype(np.float32))
    raw=np.stack([distance_transform_edt(m) for m in masks]).astype(np.float64)
    out=aggregate_multiscene_labels(masks,fields,interiority=raw)
    with rasterio.open(BASE/'multiscene_task13b/labels/source_label_map.tif') as src: ref=src.read(1)
    print('fields',len(fields),'base unresolved union',np.count_nonzero((out.labels<0)&masks.any(0)),'diff base/final',np.count_nonzero(out.labels!=ref))
    print('diag', {k:v for k,v in out.diagnostics.items() if isinstance(v,(int,float))})
    scores=np.zeros((5,*shape),np.float32); counts=np.zeros_like(scores)
    for (i,j),f in fields.items():
        both=f.available&masks[i]&masks[j]; vote=np.where(both,f.vote,0).astype(np.float32)
        scores[i]+=vote; scores[j]-=vote; counts[i]+=both; counts[j]+=both
    scores=np.divide(scores,counts,out=np.zeros_like(scores),where=counts>0)
    for r0,c0 in np.argwhere(out.labels!=ref):
        print((int(r0),int(c0)), 'ref', int(ref[r0,c0]), 'ours', int(out.labels[r0,c0]), 'scores', scores[:,r0,c0].tolist(), 'raw', raw[:,r0,c0].tolist(), 'geom', out.normalized_interiority[:,r0,c0].tolist())
    np.save(ROOT/'data/output/b9_13scene_task14/adapter_validation/five_scene_replay/debug_base_labels.npy',out.labels)
if __name__=='__main__': main()
