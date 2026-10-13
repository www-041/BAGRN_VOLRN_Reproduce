"""Streaming identity hashes for immutable source and freeze artifacts."""
from __future__ import annotations
import hashlib
from pathlib import Path

def sha256_file(path: Path, chunk_bytes: int=8_388_608) -> str:
    if chunk_bytes<=0: raise ValueError("chunk_bytes must be positive")
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(chunk_bytes),b''):digest.update(block)
    return digest.hexdigest()
