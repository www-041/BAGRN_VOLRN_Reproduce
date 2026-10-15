"""Deterministic, input-only B2 windows anchored to the canonical grid."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    width: int = 128
    height: int = 128
    stride: int = 128
    min_valid_fraction: float = .95
    max_per_pair: int = 32
    seed: int = 20261013
    supercell: int = 512


@dataclass(frozen=True)
class WindowRef:
    window_id: str
    pair_id: str
    row_off: int
    col_off: int
    height: int
    width: int
    valid_count: int
    support_sha256: str
    stratum: str = "UNLABELED_LOCAL_CONSISTENCY"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_fixed_windows(pair_id: str, pair_mask, cfg: WindowConfig = WindowConfig(), *, time_stratum: str = "UNKNOWN") -> list[WindowRef]:
    mask = np.asarray(pair_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("pair mask must be 2D")
    if (cfg.width, cfg.height, cfg.stride) != (128,128,128) or not 0 < cfg.min_valid_fraction <= 1 or cfg.max_per_pair < 0:
        raise ValueError("V1 B2 grid and thresholds are frozen; only cap may be lowered for fixtures")
    h,w=mask.shape; candidates=[]
    for row in range(0, h-cfg.height+1, cfg.stride):
        for col in range(0, w-cfg.width+1, cfg.stride):
            tile=mask[row:row+cfg.height,col:col+cfg.width]
            n=int(tile.sum())
            if n/(cfg.width*cfg.height) < cfg.min_valid_fraction: continue
            wid=f"{pair_id}_r{row:05d}_c{col:05d}"
            digest=_sha(np.packbits(tile, bitorder="little").tobytes())
            cell=(row//cfg.supercell,col//cfg.supercell)
            key=_sha(f"{cfg.seed}|{pair_id}|{wid}".encode("ascii"))
            candidates.append((cell,key,WindowRef(wid,pair_id,row,col,cfg.height,cfg.width,n,digest)))
    by_cell={}
    for cell,key,ref in candidates: by_cell.setdefault(cell,[]).append((key,ref))
    for values in by_cell.values(): values.sort(key=lambda x:x[0])
    cells=list(by_cell)
    if len(cells)<=cfg.max_per_pair:
        chosen_cells=sorted(cells)
    else:
        # Deterministic farthest-point sampling over occupied 512px cells.
        # A seeded hash chooses the first cell and breaks equal-distance ties.
        cell_hash={cell:_sha(f"{cfg.seed}|{pair_id}|cell|{cell[0]}|{cell[1]}".encode('ascii')) for cell in cells}
        first=min(cells,key=lambda c:cell_hash[c]);chosen_cells=[first]
        while len(chosen_cells)<cfg.max_per_pair:
            remaining=[c for c in cells if c not in chosen_cells]
            nxt=max(remaining,key=lambda c:(min((c[0]-s[0])**2+(c[1]-s[1])**2 for s in chosen_cells),cell_hash[c]))
            chosen_cells.append(nxt)
    selected=[by_cell[cell][0][1] for cell in chosen_cells]
    if len(selected)<cfg.max_per_pair:
        chosen={r.window_id for r in selected}
        leftovers=sorted((key,ref) for vals in by_cell.values() for key,ref in vals if ref.window_id not in chosen)
        selected.extend(ref for _,ref in leftovers[:cfg.max_per_pair-len(selected)])
    return selected
