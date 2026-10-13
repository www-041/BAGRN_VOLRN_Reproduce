"""Truth scoring for arbitrary-unit synthetic S1-S4 fixtures only."""
from __future__ import annotations
import numpy as np


def score_truth(output, truth, valid=None):
    y,t=np.asarray(output,float),np.asarray(truth,float)
    if y.shape!=t.shape: raise ValueError("output and truth shapes must match")
    mask=np.ones(y.shape,bool) if valid is None else np.asarray(valid,bool)
    if mask.shape!=y.shape: raise ValueError("truth mask shape mismatch")
    if np.any(mask & (~np.isfinite(y)|~np.isfinite(t))): raise ValueError("nonfinite truth/output under valid mask")
    e=y[mask]-t[mask]
    if not e.size:return {"status":"NO_VALID_SUPPORT","n":0,"mae":None,"rmse":None}
    return {"status":"EVALUABLE","n":int(e.size),"mae":float(np.abs(e).mean()),"rmse":float(np.sqrt(np.mean(e**2)))}


def evaluate_s4_change(before_truth, after_truth, before_output, after_output, changed_mask, eval_mask=None):
    tb,ta,yb,ya=map(lambda x:np.asarray(x,float),(before_truth,after_truth,before_output,after_output))
    changed=np.asarray(changed_mask,bool); valid=np.ones(changed.shape,bool) if eval_mask is None else np.asarray(eval_mask,bool)
    if any(a.shape!=tb.shape for a in (ta,yb,ya,changed,valid)):raise ValueError("S4 arrays must share shape")
    true_delta=ta-tb; out_delta=ya-yb
    h=valid&changed; stable=valid&~changed
    if not h.any() or not np.any(np.abs(true_delta[h])>0):
        return {"status":"NOT_EVALUABLE_ZERO_TRUE_CHANGE","change_delta_mae":None,"change_delta_rmse":None,"spurious_delta_mae":None,"change_magnitude_ratio":None}
    err=out_delta[h]-true_delta[h]
    stable_err=out_delta[stable]-true_delta[stable]
    denom=float(np.abs(true_delta[h]).sum())
    return {"status":"EVALUABLE","change_delta_mae":float(np.abs(err).mean()),"change_delta_rmse":float(np.sqrt(np.mean(err**2))),"spurious_delta_mae":float(np.abs(stable_err).mean()) if stable_err.size else None,"change_magnitude_ratio":float(np.abs(out_delta[h]).sum()/denom)}
