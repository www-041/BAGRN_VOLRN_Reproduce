# Task 7B - registration CUDA environment before

## Repository

- Path: `D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce_two_image_flat`
- Branch: `exp/2026-09-24-b9-five-scene-validation`
- HEAD at audit: `f663bf8e7c930838b19a08c1b916d96d384d092a`
- Tracked working tree: clean before artifact creation
- Python benchmark processes: none (`Get-Process python` returned no rows)
- Existing untracked scratch directories were preserved.

## Interpreter and pip target

- `sys.executable`: `D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce_two_image_flat\.venv-registration\Scripts\python.exe`
- `sys.prefix`: `D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce_two_image_flat\.venv-registration`
- `sys.base_prefix`: `E:\python3.11`
- pip: `pip 26.2.1 from D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce_two_image_flat\.venv-registration\Lib\site-packages\pip (python 3.11)`
- `where.exe python`: `E:\python3.11\python.exe`; `C:\Users\wang\AppData\Local\Microsoft\WindowsApps\python.exe`
- `where.exe pip`: `E:\python3\Scripts\pip.exe`; `E:\python3.11\Scripts\pip.exe`
- Gate result: all package commands will use `.venv-registration\Scripts\python.exe -m pip`; no package operation has run.

## PyTorch package origins

- torch file: `D:\科研\地质一号\文献\BAGRN_VOLRN_Reproduce_two_image_flat\.venv-registration\Lib\site-packages\torch\__init__.py`
- torch version: `2.14.0+cpu` (base version `2.14.0`)
- torch CUDA runtime: `None`
- torch CUDA available: `False`
- torchvision: installed, version `0.29.0`, same `.venv-registration` site-packages
- torchaudio: `NOT_INSTALLED`
- `pip check`: `No broken requirements found.`

## GPU/driver evidence before change

- `nvidia-smi`: NVIDIA GeForce RTX 4060 Laptop GPU detected
- Driver: `560.94`
- Driver CUDA: `12.6`
- GPU memory: `8188 MiB`
- No formal EfficientLoFTR benchmark process was running at audit time.

## Freeze

The complete pre-change package freeze is in `2026-09-25-registration-pip-freeze-before.txt`.

## Task 0 scope

No production code, CUDA package, driver, or model weight was modified in Task 0.
