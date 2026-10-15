"""Runtime provenance helpers; no absent log is interpreted as zero."""
from __future__ import annotations
import subprocess, time, json, os, signal
from pathlib import Path

def missing_runtime(): return {"status":"NOT_MEASURED","wall_seconds":None,"peak_rss_bytes":None,"gpu_memory_bytes":None}

def run_instrumented_method(command, cwd, env, time_limit_s, rss_limit_bytes, gpu_device_ids, output_dir, sample_interval_s=.5):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();peak_rss=0;rss_samples=0;peak_gpu=None;gpu_samples=0
    try:
        import psutil
    except ImportError:psutil=None
    with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
        proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0))
        root=psutil.Process(proc.pid) if psutil else None;timed_out=False;resource_limited=False
        while proc.poll() is None:
            if time.monotonic()-start>time_limit_s:
                timed_out=True;proc.kill();break
            if root:
                try:
                    children=root.children(recursive=True);rss=root.memory_info().rss+sum(c.memory_info().rss for c in children);peak_rss=max(peak_rss,rss);rss_samples+=1
                    if rss_limit_bytes and rss>rss_limit_bytes:resource_limited=True;proc.kill();break
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            if gpu_device_ids:
                try:
                    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,check=False)
                    pids={proc.pid}
                    if root:pids.update(c.pid for c in root.children(recursive=True))
                    values=[int(line.split(',')[1].strip())*1024*1024 for line in q.stdout.splitlines() if len(line.split(','))==2 and int(line.split(',')[0].strip()) in pids]
                    if values:peak_gpu=max(peak_gpu or 0,sum(values));gpu_samples+=1
                except (OSError,ValueError,subprocess.TimeoutExpired):pass
            time.sleep(max(.05,float(sample_interval_s)))
        proc.wait()
    elapsed=time.monotonic()-start
    status='TIMEOUT' if timed_out else 'RESOURCE_LIMIT' if resource_limited else 'COMPLETED' if proc.returncode==0 else 'METHOD_FAILED'
    return {'status':status,'return_code':proc.returncode,'wall_seconds':elapsed,'peak_rss_bytes':peak_rss if rss_samples else None,'rss_status':'MEASURED' if rss_samples else 'NOT_MEASURED','rss_sample_count':rss_samples,'peak_gpu_memory_bytes':peak_gpu,'gpu_status':'MEASURED' if gpu_samples else 'NOT_MEASURED','gpu_sample_count':gpu_samples,'stdout_path':str(out/'stdout.log'),'stderr_path':str(out/'stderr.log'),'command':list(command),'resource_sample_interval_s':float(sample_interval_s)}
