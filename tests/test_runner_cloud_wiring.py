from pathlib import Path


def test_runner_generates_and_passes_cloud_masks():
    source = Path('src/multiscene_sift/runner.py').read_text(encoding='utf-8')
    assert 'from src.cloud_mask import generate_cloud_masks' in source
    assert 'cloud_masks = generate_cloud_masks(' in source
    assert 'cloud_masks=cloud_masks' in source
    assert '"cloud_mask_enabled": True' in source
