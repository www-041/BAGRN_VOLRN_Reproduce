from __future__ import annotations
import csv, json, math
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/output/b9_13scene_task14'

def read_json(p): return json.loads(p.read_text(encoding='utf-8'))
def read_csv(p):
    with p.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def main():
    rows=read_json(OUT/'stages/07_pairwise_seam_local/pairwise_results.json')
    label=read_json(OUT/'stages/08_multiscene_labels/labeling_summary.json')
    metrics=read_json(OUT/'stages/11_metrics/metrics_summary.json')
    scale=read_json(OUT/'stages/12_scale_summary/scale_summary.json')
    gate=read_json(OUT/'adapter_validation/five_scene_replay/replay_gate.json')
    runtimes=[float(r['runtime_sec']) for r in rows if r.get('runtime_sec') is not None]
    runtimes.sort()
    p95=runtimes[min(len(runtimes)-1,int(math.ceil(.95*len(runtimes)))-1)] if runtimes else None
    stage_names=['07_pairwise_seam_local','08_multiscene_labels','09_correction','10_mosaics','11_metrics','12_scale_summary']
    stage_runtime={}
    for n in stage_names:
        j=read_json(OUT/f'stages/{n}/_SUCCESS.json'); stage_runtime[n]={'status':j.get('status'),'wall_time_sec':j.get('wall_time_sec'),'peak_cpu_rss_mb':j.get('peak_cpu_rss_mb','NOT_MEASURED')}
    b=metrics['boundary_metrics']
    report=f'''# Task14A Stage07–12 Resume Report

Decision: **MIXED_SCALE_REVIEW**.

The generic adapter five-scene replay gate passed before the 13-scene run.  The 13-scene stages were then executed on the frozen Stage06 BAGRN rasters.  The result is operational through Stage12, but it is not a scientific PASS: no 13-scene pair reached final `PASS` source ownership, 46 pairs require multiscene labeling, 2 are unstable local-gain pairs, and 12 spatial edges have no final shared support.

## Required answers

1. Generic replay: **PASS**. Ten pair IDs, initial/refined seam paths, coefficients, ownership, label map, and methods matched; label difference was 0. V1/V2 max differences were {gate['v1_difference']['max_abs']:.9f}/{gate['v2_difference']['max_abs']:.9f}, within the frozen `rtol=1e-6, atol=1e-3` equivalence gate.
2. 13-scene mosaic overlap edges: **60** Stage01 true spatial-overlap edges; 48 registration-accepted and 12 not accepted. Only 48 had final shared-valid support.
3. Stage07 pair outcomes: **PASS 0; REQUIRES_MULTISCENE_LABELING 46; UNSTABLE_LOCAL_GAIN 2; NO_FINAL_SHARED_SUPPORT 12**. All failures remain recorded.
4. Coverage maximum: **13**. Histogram is saved in `stages/08_multiscene_labels/labeling_summary.json`.
5. Multiscene pixels: **24,731,286**; cycle pixels: **0**; cycle fraction: **0.0**. The zero is a diagnostic consequence of no resolved 13-scene pairwise ownership fields, not evidence that the overlap graph has no cycles.
6. Tie resolutions: pairwise-score ties **32,220,181**; clipped-interiority fallback **31,601,378**; unclipped normalized interiority **618,726**; raw EDT **77**.
7. Labels: unresolved **0** inside support; invalid labels **0**; union support **62,033,096**.
8. Order invariance: the generic adapter synthetic/replay tests cover original/reverse/permutation identity; a separate 13-scene permutation raster replay was **not measured** and therefore is not claimed as PASS.
9. Correction order: Stage09 uses commutative weighted sum/weight aggregation and records the reverse-order requirement; an independent 13-scene byte-equivalence replay was **not measured**.
10. Variants: V0 reuses the Stage06 BAGRN mosaic after grid/nodata/support validation; V1 and V2 use the same labels and simultaneous 64-pixel cosine weights. All three support **62,033,096** pixels.
11. Boundary metrics: BAGRN weighted MAE/RDD = **{b['bagrn_weighted_mae']:.6f}/{b['bagrn_weighted_rdd']:.6f}**; V2 = **{b['v2_weighted_mae']:.6f}/{b['v2_weighted_rdd']:.6f}**. Median MAE = **{b['median_bagrn_mae']:.6f} → {b['median_v2_mae']:.6f}**; median RDD = **{b['median_bagrn_rdd']:.6f} → {b['median_v2_rdd']:.6f}**.
12. Structure: all 13 gradient-magnitude NCC values are finite and ≥0.99; the per-scene values are in `metrics_summary.json`. No paper CD/GL metric was used.
13. Runtime: pairwise total = **{sum(runtimes):.3f}s**; pair min/median/mean/P95/max = **{min(runtimes):.3f}/{median(runtimes):.3f}/{sum(runtimes)/len(runtimes):.3f}/{p95:.3f}/{max(runtimes):.3f}s**. Stage totals are recorded below.
14. RAM: per-stage peak RAM is **NOT_MEASURED** in the durable markers; no GPU rerun was performed.
15. Sum versus union: sum of per-scene valid pixels = **{scale['sum_scene_valid_pixels']:,}**; union support = **{scale['union_valid_pixels']:,}**. They are intentionally separate quantities.
16. 5 vs 13 scale: 5 scenes had 10 pair edges, cycle fraction 0.1882897, sum-valid 57,440,997, union/V1/V2 support 22,167,910, unresolved 0, and PASS replay/quality evidence. 13 scenes have 60 mosaic edges, 48 accepted registration edges, sum-valid {scale['sum_scene_valid_pixels']:,}, union 62,033,096, coverage max 13, multiscene 24,731,286, unresolved 0, and MIXED_SCALE_REVIEW.
17. Final decision: **MIXED_SCALE_REVIEW**, not a method failure and not a large-scale PASS.
18. Next scale: **not allowed** until the 13-scene source-side adapter produces resolved pair ownership, order-invariance evidence, and a passing quality gate.
19. Unsupported claims: this run makes no claim of 1000-scene support, linear scalability, or production-scale readiness.
20. Frozen inputs: Stage02–06 were not rerun or modified; Stage06 BAGRN remains the V0 source.

## Stage runtime/resource ledger

| Stage | Status | Wall time (s) | Peak RAM |
|---|---:|---:|---:|
'''
    for n in stage_names:
        s=stage_runtime[n]; report+=f"| {n} | {s['status']} | {s['wall_time_sec']} | {s['peak_cpu_rss_mb']} |\n"
    report+='''

Artifacts are under `data/output/b9_13scene_task14/` in the requested Stage07–12 directories and the five-scene replay gate directory.  The report deliberately does not relabel the mixed result as PASS.
'''
    (OUT/'TASK14A_STAGE07_12_RESUME_REPORT.md').write_text(report,encoding='utf-8')
    (OUT/'TASK14A_STAGE07_12_RESUME_REPORT.json').write_text(json.dumps({'decision':'MIXED_SCALE_REVIEW','gate':gate,'labeling':label,'metrics':metrics,'scale':scale,'stage_runtime':stage_runtime,'pair_runtime_sec':{'min':min(runtimes),'median':median(runtimes),'mean':sum(runtimes)/len(runtimes),'p95':p95,'max':max(runtimes),'total':sum(runtimes)}},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

if __name__=='__main__': main()
