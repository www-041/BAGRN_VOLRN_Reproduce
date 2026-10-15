"""Chunked construction of untouched-radiometry canonical-grid inputs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import array_bounds
from rasterio.windows import Window
from math import floor, ceil

def corrected_affine(source_transform, world_matrix):
    g=np.asarray(world_matrix,dtype=np.float64)
    t=np.array([[source_transform.a,source_transform.b,source_transform.c],[source_transform.d,source_transform.e,source_transform.f],[0,0,1]],dtype=np.float64)
    q=g@t
    return Affine(q[0,0],q[0,1],q[0,2],q[1,0],q[1,1],q[1,2])

def prepare_aligned_input(source_path, scene_index, grid, world_matrix, output_dir, *, block_size=512, verify_mask_path=None):
    source_path=Path(source_path);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    dst_path=out/f'scene_{scene_index:03d}_B9_input.tif';mask_path=out/f'scene_{scene_index:03d}_input_mask.tif'
    if dst_path.exists() or mask_path.exists(): raise FileExistsError('refusing to overwrite aligned input artifacts')
    transform=Affine(*grid.affine)
    # Match Task15's verified warp semantics: read the source as float64 and
    # reproject once into its bounded local canonical-grid footprint. Per-tile
    # reprojection was experimentally found to alter finite edge support.
    profile={'driver':'GTiff','width':grid.width,'height':grid.height,'count':1,'dtype':'float32','crs':grid.crs,'transform':transform,'nodata':None,'tiled':True,'blockxsize':512,'blockysize':512,'compress':'deflate','predictor':3,'BIGTIFF':'IF_SAFER','sparse_ok':True}
    mprofile={**profile,'dtype':'uint8','nodata':0,'predictor':2}
    with rasterio.open(source_path) as src:
        source_array=src.read(1).astype(np.float64,copy=False)
        corrected=corrected_affine(src.transform,world_matrix)
        left,bottom,right,top=array_bounds(src.height,src.width,corrected)
        col0=max(0,min(grid.width,int(floor((left-transform.c)/abs(transform.a)))))
        col1=max(0,min(grid.width,int(ceil((right-transform.c)/abs(transform.a)))))
        row0=max(0,min(grid.height,int(floor((transform.f-top)/abs(transform.e)))))
        row1=max(0,min(grid.height,int(ceil((transform.f-bottom)/abs(transform.e)))))
        if row1<=row0 or col1<=col0:raise ValueError('source has no canonical-grid footprint')
        local_transform=transform @ Affine.translation(col0,row0)
        projected=np.full((row1-row0,col1-col0),np.nan,dtype=np.float64)
        reproject(source=source_array,destination=projected,src_transform=corrected,src_crs=src.crs,dst_transform=local_transform,dst_crs=grid.crs,src_nodata=src.nodata if src.nodata is not None else 0,dst_nodata=np.nan,init_dest_nodata=True,resampling=Resampling.bilinear)
        valid=np.isfinite(projected)
        rr,cc=np.where(valid)
        if not rr.size:raise ValueError('reprojection produced no valid pixels')
        r0,r1=int(rr.min()),int(rr.max())+1;c0,c1=int(cc.min()),int(cc.max())+1
        local_data=projected[r0:r1,c0:c1].astype(np.float32,copy=False)
        local_mask=valid[r0:r1,c0:c1].astype(np.uint8,copy=False)
        crop=Window(col0+c0,row0+r0,c1-c0,r1-r0)
        with rasterio.open(dst_path,'w',**profile) as dst, rasterio.open(mask_path,'w',**mprofile) as mdst:
            dst.write(local_data,1,window=crop);mdst.write(local_mask,1,window=crop)
    if verify_mask_path is not None:
        mismatches=0
        with rasterio.open(mask_path) as a, rasterio.open(verify_mask_path) as b:
            if (a.width,a.height,a.transform,a.crs)!=(b.width,b.height,b.transform,b.crs): raise ValueError('historical input mask grid mismatch')
            for _,win in a.block_windows(1): mismatches+=int(np.count_nonzero(a.read(1,window=win)!=b.read(1,window=win)))
        if mismatches: raise ValueError(f'aligned input mask differs from persisted Task15 mask at {mismatches} pixels')
    return {'scene_index':scene_index,'aligned_path':str(dst_path),'mask_path':str(mask_path),'grid':{'crs':grid.crs,'affine':grid.affine,'width':grid.width,'height':grid.height},'mask_verified_against_task15':verify_mask_path is not None,'warp_semantics':'Task15-equivalent float64 source to local canonical footprint, bilinear, then float32 storage; explicit mask','status':'PASS'}
