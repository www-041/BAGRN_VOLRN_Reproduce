"""Validated immutable configuration for the Task15 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPECTED_HIERARCHY = (
    "pairwise_score",
    "clipped_normalized_interiority",
    "unclipped_normalized_interiority",
    "raw_edt",
)


@dataclass(frozen=True)
class DatasetConfig:
    input_root: Path
    band: str
    expected_scene_count: int
    scene_selection: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RegistrationConfig:
    matcher: str
    match_max_side: int
    checkpoint: Path
    global_method: str
    matcher_repo: Path | None = None
    device: str = "auto"
    precision: str = "fp32"


@dataclass(frozen=True)
class CanonicalGridConfig:
    crs: str
    resolution_m: float


@dataclass(frozen=True)
class RadiometricConfig:
    method: str
    control_scene_index: int


@dataclass(frozen=True)
class SeamConfig:
    intensity_weight: float
    gradient_weight: float
    normalization: str
    coarse_factor: int
    refine_half_width: int


@dataclass(frozen=True)
class LocalCorrectionConfig:
    corridor_half_width: int
    segment_length: int
    min_valid_pixels: int
    percentile_low: float
    percentile_high: float
    gain_min: float
    gain_max: float


@dataclass(frozen=True)
class LabelingConfig:
    preference_distance_scale: float
    tie_tolerance: float
    hierarchy: tuple[str, ...]


@dataclass(frozen=True)
class BlendConfig:
    method: str
    half_width: int


@dataclass(frozen=True)
class StreamingConfig:
    tile_size: int
    halo: int


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    write_geotiff: bool
    write_quicklooks: bool
    write_metrics: bool
    write_report: bool


@dataclass(frozen=True)
class PipelineConfig:
    dataset: DatasetConfig
    registration: RegistrationConfig
    canonical_grid: CanonicalGridConfig
    radiometric: RadiometricConfig
    seam: SeamConfig
    local_correction: LocalCorrectionConfig
    labeling: LabelingConfig
    blend: BlendConfig
    streaming: StreamingConfig
    outputs: OutputConfig

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return _paths_to_strings(value)


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_paths_to_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    return value


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing or invalid config section: {name}")
    return value


def _required(section: Mapping[str, Any], name: str) -> Any:
    if name not in section or section[name] in (None, ""):
        raise ValueError(f"missing required config field: {name}")
    return section[name]


def _as_path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a path")
    return Path(value).expanduser()


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and validate one frozen scientific pipeline configuration."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("pipeline config must be a YAML mapping")

    dataset = _section(raw, "dataset")
    registration = _section(raw, "registration")
    grid = _section(raw, "canonical_grid")
    radiometric = _section(raw, "radiometric")
    seam = _section(raw, "seam")
    correction = _section(raw, "local_correction")
    labeling = _section(raw, "labeling")
    blend = _section(raw, "blend")
    streaming = _section(raw, "streaming")
    outputs = _section(raw, "outputs")

    hierarchy = tuple(_required(labeling, "hierarchy"))
    if hierarchy != EXPECTED_HIERARCHY:
        raise ValueError(f"labeling hierarchy must be exactly {list(EXPECTED_HIERARCHY)}")
    if _required(registration, "matcher") != "efficient_loftr":
        raise ValueError("matcher must be efficient_loftr")
    if int(_required(registration, "match_max_side")) != 1024:
        raise ValueError("match_max_side must be 1024")
    if _required(registration, "global_method") != "translation_l2":
        raise ValueError("global_method must be translation_l2")
    if _required(radiometric, "method") != "bagrn":
        raise ValueError("radiometric method must be bagrn")
    if float(_required(grid, "resolution_m")) <= 0:
        raise ValueError("resolution_m must be positive")
    if int(_required(streaming, "tile_size")) <= 0 or int(_required(streaming, "halo")) < 0:
        raise ValueError("invalid streaming tile_size/halo")

    selection = dataset.get("scene_selection")
    selection_tuple = None if selection is None else tuple(int(item) for item in selection)
    return PipelineConfig(
        dataset=DatasetConfig(
            input_root=_as_path(_required(dataset, "input_root"), "input_root"),
            band=str(_required(dataset, "band")),
            expected_scene_count=int(_required(dataset, "expected_scene_count")),
            scene_selection=selection_tuple,
        ),
        registration=RegistrationConfig(
            matcher="efficient_loftr",
            match_max_side=1024,
            checkpoint=_as_path(_required(registration, "checkpoint"), "checkpoint"),
            global_method="translation_l2",
            matcher_repo=None if registration.get("matcher_repo") is None else _as_path(registration["matcher_repo"], "matcher_repo"),
            device=str(registration.get("device", "auto")),
            precision=str(registration.get("precision", "fp32")),
        ),
        canonical_grid=CanonicalGridConfig(str(_required(grid, "crs")), float(_required(grid, "resolution_m"))),
        radiometric=RadiometricConfig("bagrn", int(_required(radiometric, "control_scene_index"))),
        seam=SeamConfig(
            float(_required(seam, "intensity_weight")), float(_required(seam, "gradient_weight")),
            str(_required(seam, "normalization")), int(_required(seam, "coarse_factor")), int(_required(seam, "refine_half_width")),
        ),
        local_correction=LocalCorrectionConfig(
            int(_required(correction, "corridor_half_width")), int(_required(correction, "segment_length")),
            int(_required(correction, "min_valid_pixels")), float(_required(correction, "percentile_low")),
            float(_required(correction, "percentile_high")), float(_required(correction, "gain_min")), float(_required(correction, "gain_max")),
        ),
        labeling=LabelingConfig(float(_required(labeling, "preference_distance_scale")), float(_required(labeling, "tie_tolerance")), hierarchy),
        blend=BlendConfig(str(_required(blend, "method")), int(_required(blend, "half_width"))),
        streaming=StreamingConfig(int(_required(streaming, "tile_size")), int(_required(streaming, "halo"))),
        outputs=OutputConfig(
            _as_path(_required(outputs, "root"), "outputs.root"), bool(outputs.get("write_geotiff", True)),
            bool(outputs.get("write_quicklooks", True)), bool(outputs.get("write_metrics", True)), bool(outputs.get("write_report", True)),
        ),
    )
