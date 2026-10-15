"""Schema-stable atomic report writing with strict JSON null handling."""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path

def _reject_nonfinite(obj):
    if isinstance(obj,float) and (obj!=obj or obj in (float('inf'),float('-inf'))): raise ValueError('nonfinite JSON number is forbidden; serialize with explicit status/null')
    if isinstance(obj,dict):
        for v in obj.values():_reject_nonfinite(v)
    elif isinstance(obj,(list,tuple)):
        for v in obj:_reject_nonfinite(v)

def write_json_atomic(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);_reject_nonfinite(payload)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)
        os.replace(tmp,path)
    except Exception:
        try:os.unlink(tmp)
        except OSError:pass
        raise
    return path
