"""Mosaic coverage on frozen union support."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MosaicCoverage:
    status: str
    coverage: float | None
    holes: int
    extras: int
    seam_status: str = "NOT_APPLICABLE_NO_PROVENANCE"

def compute_mosaic_coverage(expected_union, output_valid):
    u,o=np.asarray(expected_union,bool),np.asarray(output_valid,bool)
    if u.shape!=o.shape: raise ValueError("union and output masks must share shape/grid")
    n=int(u.sum()); valid=int((u&o).sum())
    return MosaicCoverage("NO_INPUT_UNION" if not n else "EVALUABLE",None if not n else valid/n,int((u&~o).sum()),int((~u&o).sum()))

def unique_source_reference(scene_arrays, scene_masks):
    arrays=[np.asarray(a,float) for a in scene_arrays];masks=[np.asarray(m,bool) for m in scene_masks]
    if not arrays or len(arrays)!=len(masks) or len({a.shape for a in arrays+masks})!=1:raise ValueError('scene arrays/masks must have same nonempty grid')
    count=np.sum(np.stack(masks),axis=0);unique=count==1;ref=np.zeros(arrays[0].shape,float)
    for array,mask in zip(arrays,masks):ref[unique&mask]=array[unique&mask]
    return ref,unique

def compute_single_source_structure(scene_arrays,scene_masks,mosaic,mosaic_valid):
    from .structure import compute_sobel_structure
    ref,unique=unique_source_reference(scene_arrays,scene_masks);valid=unique&np.asarray(mosaic_valid,bool)
    if ref.shape!=np.shape(mosaic) or valid.shape!=np.shape(mosaic):raise ValueError('mosaic grid mismatch')
    stats=compute_sobel_structure(ref,np.asarray(mosaic,float),valid)
    return {'status':stats.status,'unique_source_pixels':int(unique.sum()),'valid_pixels':int(valid.sum()),'structure':stats.__dict__}
