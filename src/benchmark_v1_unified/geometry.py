"""Canonical grid and Task15 transform provenance parsing."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass(frozen=True)
class GridSpec:
    crs: str
    affine: tuple[float,float,float,float,float,float]
    width: int
    height: int
    pixel_area_m2: float
    pixel_center: str='col_plus_half_row_plus_half'

def load_grid(path: Path) -> GridSpec:
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    t=tuple(float(x) for x in d['transform'])
    if d['crs']!='EPSG:32650' or len(t)!=6 or abs(t[0])!=14 or abs(t[4])!=14 or t[1]!=0 or t[3]!=0:
        raise ValueError('canonical grid does not match verified north-up 14m EPSG:32650 grid')
    return GridSpec(d['crs'],t,int(d['width']),int(d['height']),abs(t[0]*t[4]))

def load_world_transforms(path: Path):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    rows=d.get('transforms')
    if not isinstance(rows,list) or len(rows)!=13: raise ValueError('Task15 world transforms must include 13 scenes')
    matrices=[]
    for i,row in enumerate(rows):
        m=row.get('matrix')
        if row.get('scene')!=i or len(m)!=3 or any(len(r)!=3 for r in m): raise ValueError(f'invalid transform record at scene {i}')
        matrices.append(m)
    return matrices
