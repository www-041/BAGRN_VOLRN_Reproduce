"""Array-level evaluation primitives reused by B1, B2 and synthetic fixtures."""
from __future__ import annotations
from dataclasses import asdict
import numpy as np
from .radiometric import compute_pair_direct_metrics,compute_pair_distribution_metrics,audit_pair_output_support

def evaluate_pair_band_arrays(pair_id, x, y, input_support, *, output_valid=None, full_support_required=True):
    support=np.asarray(input_support,bool)
    if output_valid is not None:
        audit=audit_pair_output_support(support,y,output_valid)
        if audit.status!='FULL_SUPPORT' and full_support_required:
            return {'pair_id':pair_id,'status':audit.status,'coverage':audit.coverage,'input_pixels':audit.input_pixels,'output_pixels':audit.output_pixels,'primary_eligible':False,'metrics':None}
    direct=compute_pair_direct_metrics(x,y,support);dist=compute_pair_distribution_metrics(x,y,support)
    return {'pair_id':pair_id,'status':direct.status,'coverage':1.0,'input_pixels':direct.n,'primary_eligible':direct.status=='EVALUABLE','direct':asdict(direct),'distribution':asdict(dist)}

def evaluate_fixed_window_arrays(window_ref,x,y,support):
    result=evaluate_pair_band_arrays(window_ref.pair_id,x,y,support)
    result['window_id']=window_ref.window_id;result['stratum']='UNLABELED_LOCAL_CONSISTENCY'
    return result

def evaluate_scene_macro(rows, expected_scene_ids):
    by={str(r['scene_id']):r for r in rows};items=[by.get(str(s)) for s in expected_scene_ids]
    evaluable=[r for r in items if r and r.get('status')=='EVALUABLE']
    result={}
    keys=set.intersection(*(set(r.get('metrics',{})) for r in evaluable)) if evaluable else set()
    for key in keys:
        values=[float(r['metrics'][key]) for r in evaluable if r['metrics'][key] is not None]
        result[key]=float(np.mean(values)) if values else None
    return {'expected_scenes':len(expected_scene_ids),'evaluable_scenes':len(evaluable),'status':'EVALUABLE' if len(evaluable)==len(expected_scene_ids) else 'PARTIAL_SCENE_SET','scene_macro':result}

def evaluate_synthetic_arrays(truth_before,output_before,valid_before,*,truth_after=None,output_after=None,valid_after=None,change_mask=None,anchor_matched=True):
    from .simulation import score_truth,evaluate_s4_change
    if not anchor_matched:return {'status':'UNMATCHED_OUTPUT_GAUGE','before':None,'after':None,'change':None}
    before=score_truth(output_before,truth_before,valid_before)
    after=None;change=None
    if truth_after is not None or output_after is not None:
        if any(v is None for v in (truth_after,output_after,valid_after,change_mask)):raise ValueError('S4 requires after truth/output/mask and change mask together')
        after=score_truth(output_after,truth_after,valid_after)
        valid=np.asarray(valid_before,bool)&np.asarray(valid_after,bool)
        change=evaluate_s4_change(truth_before,truth_after,output_before,output_after,change_mask,valid)
    return {'status':'EVALUABLE' if before['status']=='EVALUABLE' and (after is None or after['status']=='EVALUABLE') else 'PARTIAL_SUPPORT','before':before,'after':after,'change':change,'unit':'ARBITRARY_SYNTHETIC_UNIT'}
