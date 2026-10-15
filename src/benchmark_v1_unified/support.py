"""Frozen input masks and pair-support bookkeeping."""
from __future__ import annotations
from itertools import combinations
import numpy as np


def time_strata_counts(pairs):
    out={}
    for p in pairs:
        if p.get("status")=="OVERLAP": out[p["time_stratum"]]=out.get(p["time_stratum"],0)+1
    return out


def validate_support_origin(origin):
    return "PASS" if origin=="FROZEN_INPUT_MASK" else "INVALID_SCORING_SUPPORT"


def enumerate_overlap_pairs(masks, scene_ids=None, times=None, quality=None):
    if len(masks)!=13: raise ValueError("V1 source inventory must contain exactly 13 B9 masks")
    arrays=[np.asarray(m,bool) for m in masks]
    if len({a.shape for a in arrays})!=1: raise ValueError("all masks must share canonical grid")
    ids=scene_ids or [f"{i:02d}" for i in range(13)]
    rows=[]
    for i,j in combinations(range(13),2):
        n=int(np.count_nonzero(arrays[i]&arrays[j])); pid=f"{i:02d}_{j:02d}"
        rows.append({"pair_id":pid,"idx_i":i,"idx_j":j,"scene_i":ids[i],"scene_j":ids[j],"input_overlap_pixels":n,"status":"OVERLAP" if n else "NO_VALID_OVERLAP","time_stratum":(times or {}).get(pid,"UNKNOWN"),"quality_i":(quality or {}).get(ids[i],"UNKNOWN"),"quality_j":(quality or {}).get(ids[j],"UNKNOWN")})
    return rows
