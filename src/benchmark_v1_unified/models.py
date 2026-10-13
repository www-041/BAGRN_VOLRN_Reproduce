"""Small typed records shared across the method-neutral evaluator."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass(frozen=True)
class SceneRef:
    scene_id:str;scene_index:int;band:str;source_path:Path;source_sha256:str;source_dtype:str;acquisition_time:str;quality_flag:str;source_nodata:float|None;source_mask_semantics:str;aligned_input_path:Path|None;aligned_mask_path:Path;alignment_transform_provenance:dict

@dataclass(frozen=True)
class PairRef:
    pair_id:str;scene_i:str;scene_j:str;time_stratum:str;days_apart:float|None;risk_stratum:str;input_overlap_pixels:int;support_mask_path:Path|None;expected_status:Literal['OVERLAP','NO_VALID_OVERLAP']

@dataclass(frozen=True)
class MetricRecord:
    value:float|None;unit:str;status:str;n_input:int;n_output:int;diagnostics:dict
