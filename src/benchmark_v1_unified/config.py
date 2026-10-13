"""Strict config loading for the isolated Unified Benchmark V1 namespace."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    values: dict

    @property
    def protocol_id(self): return self.values['protocol_id']

def load_benchmark_config(config_path: Path) -> BenchmarkConfig:
    path=Path(config_path).resolve()
    data=yaml.safe_load(path.read_text(encoding='utf-8'))
    required={'schema_version','protocol_id','status','scope','paths','geometry','identity','pair_scoring','windows','structure','simulation','runtime','status_policy'}
    if not isinstance(data,dict) or required-set(data): raise ValueError(f"missing required config keys: {sorted(required-set(data or {}))}")
    if data['schema_version']!=1 or data['scope'].get('band')!='B9' or data['scope'].get('physical_unit')!='UNKNOWN': raise ValueError('unsupported schema or scope')
    if data['pair_scoring'].get('std_ddof')!=0 or data['pair_scoring'].get('exact_wasserstein') is not True: raise ValueError('metric formula contract is not explicit/frozen')
    if data['geometry'].get('resampling_intensity') not in {'bilinear'}: raise ValueError('unverified intensity resampling')
    metric_ids=set(data['pair_scoring'].get('metric_ids',[]))
    if not metric_ids or any(not m.endswith('_V1') and m!='RDD_W1_EXACT_V1' for m in metric_ids): raise ValueError('unknown or unversioned metric ID')
    return BenchmarkConfig(path,data)
