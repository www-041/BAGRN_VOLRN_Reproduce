"""Scene preflight validation utilities.

Task 3 of dz01v mosaic readiness plan:
Provides validation functions for strict same-resolution enforcement.
"""

from typing import List
from rasterio.transform import Affine


def validate_strict_scene_grids(
    scene_ids: List[str],
    transforms: List[Affine],
    crs_list: List[str],
    atol: float = 1e-9,
) -> None:
    """Validate that all scenes have the same resolution and CRS in strict mode.
    
    In strict mode:
    - All scenes must have the same CRS
    - All scenes must have the same pixel size (resolution)
    - All scenes must have the same rotation/skew (transform.a, b, d, e)
    - Scene origins (c, f) may differ
    
    Parameters
    ----------
    scene_ids : List[str]
        Scene identifiers for error messages.
    transforms : List[Affine]
        Affine transforms for each scene.
    crs_list : List[str]
        CRS for each scene.
    atol : float
        Absolute tolerance for floating point comparisons.
    
    Raises
    ------
    ValueError
        If scenes have different CRS, resolution, or rotation/skew.
    """
    if len(scene_ids) < 2:
        return
    
    # Check CRS
    unique_crs = set(crs_list)
    if len(unique_crs) > 1:
        raise ValueError(
            f"strict mode requires all scenes to have the same CRS. "
            f"Found: {unique_crs}"
        )
    
    # Check resolution and rotation/skew
    ref_transform = transforms[0]
    ref_a, ref_b = ref_transform.a, ref_transform.b
    ref_d, ref_e = ref_transform.d, ref_transform.e
    
    for i, (scene_id, transform) in enumerate(zip(scene_ids[1:], transforms[1:]), start=1):
        # Check resolution (pixel size)
        if abs(transform.a - ref_a) > atol or abs(transform.e - ref_e) > atol:
            raise ValueError(
                f"strict mode requires all scenes to have the same resolution. "
                f"Scene '{scene_ids[0]}' has resolution ({abs(ref_a)}, {abs(ref_e)}), "
                f"but scene '{scene_id}' has resolution ({abs(transform.a)}, {abs(transform.e)})."
            )
        
        # Check rotation/skew
        if abs(transform.b - ref_b) > atol or abs(transform.d - ref_d) > atol:
            raise ValueError(
                f"strict mode requires all scenes to have the same rotation/skew. "
                f"Scene '{scene_ids[0]}' has (b={ref_b}, d={ref_d}), "
                f"but scene '{scene_id}' has (b={transform.b}, d={transform.d})."
            )
