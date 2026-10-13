"""Explicit pair-macro and pixel-weighted aggregations."""
from __future__ import annotations
import numpy as np


def aggregate_pair_rows(rows, expected_pair_ids):
    expected=list(expected_pair_ids); by={str(r["pair_id"]):r for r in rows}
    values=[]; weights=[]
    for pid in expected:
        r=by.get(str(pid))
        if r is None or r.get("status")!="EVALUABLE" or r.get("value") is None: continue
        values.append(float(r["value"])); weights.append(int(r.get("n",0)))
    complete=len(values)==len(expected) and bool(expected)
    statuses={}
    for pid in expected:
        status=str(by.get(str(pid),{}).get('status','MISSING_PAIR_ROW'));statuses[status]=statuses.get(status,0)+1
    return {"expected_pairs":len(expected),"evaluable_pairs":len(values),"status_counts":statuses,"pair_macro":float(np.mean(values)) if values else None,"pixel_weighted":float(np.average(values,weights=weights)) if values and sum(weights)>0 else None,"pair_median":float(np.median(values)) if values else None,"pair_p95":float(np.quantile(values,.95,method="linear")) if values else None,"all_pairs_eligible":complete,"status":"EVALUABLE" if complete else "PARTIAL_PAIR_SET"}


def aggregate_windows(rows, pair_ids):
    grouped={p:[] for p in pair_ids}
    for row in rows:
        if row.get("pair_id") in grouped and row.get("status")=="EVALUABLE" and row.get("value") is not None: grouped[row["pair_id"]].append(float(row["value"]))
    pair_stats={p:{"median":float(np.median(v)),"mean":float(np.mean(v)),"p90":float(np.quantile(v,.9)),"max":float(np.max(v)),"n":len(v)} for p,v in grouped.items() if v}
    return {"per_pair":pair_stats,"pair_macro_median":float(np.mean([v["median"] for v in pair_stats.values()])) if pair_stats else None}

def aggregate_temporal_quality(rows, metric_key='value'):
    out={}
    for dimension in ('time_stratum','quality_stratum'):
        groups={}
        for row in rows:
            value=row.get(metric_key)
            if row.get('status')!='EVALUABLE' or value is None:continue
            key=str(row.get(dimension,'UNKNOWN'));groups.setdefault(key,[]).append(float(value))
        out[dimension]={k:{'pair_count':len(v),'pair_macro':float(np.mean(v)),'median':float(np.median(v)),'p95':float(np.quantile(v,.95,method='linear'))} for k,v in sorted(groups.items())}
    return out
