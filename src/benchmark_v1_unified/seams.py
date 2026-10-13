"""Hard-label seam diagnostics; require explicit provenance."""
from __future__ import annotations
import numpy as np


def compute_seam_excess(mosaic, labels, valid=None):
    x,l=np.asarray(mosaic,float),np.asarray(labels)
    if x.shape!=l.shape or x.ndim!=2: raise ValueError("mosaic and label map must be same 2D shape")
    m=np.ones(x.shape,bool) if valid is None else np.asarray(valid,bool)
    cross=[]; within=[]
    for a,b,ma,mb,va,vb,axis in [(x[:,:-1],x[:,1:],l[:,:-1],l[:,1:],m[:,:-1],m[:,1:],1),(x[:-1,:],x[1:,:],l[:-1,:],l[1:,:],m[:-1,:],m[1:,:],0)]:
        ok=va&vb&np.isfinite(a)&np.isfinite(b)&(ma>=0)&(mb>=0)
        edge=ok&(ma!=mb)
        for r,c in np.argwhere(edge):
            jump=abs(float(a[r,c]-b[r,c])); cross.append(jump)
            if axis==1:
                if c>0 and m[r,c-1] and l[r,c-1]==ma[r,c]:within.append(abs(float(a[r,c]-x[r,c-1])))
                if c+2<x.shape[1] and m[r,c+2] and l[r,c+2]==mb[r,c+1]:within.append(abs(float(x[r,c+2]-b[r,c])))
            else:
                if r>0 and m[r-1,c] and l[r-1,c]==ma[r,c]:within.append(abs(float(a[r,c]-x[r-1,c])))
                if r+2<x.shape[0] and m[r+2,c] and l[r+2,c]==mb[r+1,c]:within.append(abs(float(x[r+2,c]-b[r,c])))
    if not cross:return {"status":"NOT_APPLICABLE_NO_SEAM","edge_count":0,"mean_excess":None}
    baseline=float(np.mean(within)) if within else 0.0
    return {"status":"EVALUABLE","edge_count":len(cross),"mean_cross_jump":float(np.mean(cross)),"mean_within_jump":baseline,"mean_excess":float(np.mean(cross)-baseline),"p95_excess":float(np.quantile(np.asarray(cross)-baseline,.95))}
