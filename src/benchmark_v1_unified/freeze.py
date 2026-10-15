"""Immutable SHA locks; lock creation refuses overwrite."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from .identity import sha256_file

@dataclass(frozen=True)
class FreezeVerification:
    valid: bool
    errors: tuple[str,...]

def write_lock(root, paths):
    root=Path(root); lock=root/'freeze_lock.json'
    if lock.exists(): raise FileExistsError(f"refusing to overwrite freeze lock: {lock}")
    entries=[]
    for p in paths:
        path=Path(p)
        entries.append({"path":str(path.resolve()),"sha256":sha256_file(path)})
    lock.write_text(json.dumps({"schema_version":1,"files":entries},indent=2),encoding='utf-8')
    return lock

def verify_lock(root):
    root=Path(root); lock=root/'freeze_lock.json'; errors=[]
    if not lock.is_file(): return FreezeVerification(False,("MISSING_LOCK",))
    try: data=json.loads(lock.read_text(encoding='utf-8'))
    except Exception as exc:return FreezeVerification(False,(f"INVALID_LOCK:{exc}",))
    for item in data.get('files',[]):
        p=Path(item['path'])
        if not p.is_file():errors.append(f"MISSING:{p}")
        elif sha256_file(p)!=item.get('sha256'):errors.append(f"HASH_MISMATCH:{p}")
    return FreezeVerification(not errors,tuple(errors))
