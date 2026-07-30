# 实验指南

## 一、环境准备
- Python 3.14+, numpy, scipy, rasterio, scikit-image, shapely, tqdm, pyyaml
- 安装: `pip install -r requirements.txt`

## 二、实验配置文件
- 配置文件格式: YAML
- 示例: `configs/experiment.example.yaml`
- 关键字段说明

## 三、实验类型

### 1. 多波段实验 (multiband)
命令:
```
python -m src.experiment_runner --config configs/dz01_multiband.yaml multiband
```
输出:
- registered original / BAGRN / VOLRN-only / BAGRN-VOLRN / 基线方法
- 逐景多波段 GeoTIFF
- 镶嵌结果
- 评价指标 CSV

### 2. 消融实验 (ablation)
命令:
```
python -m src.experiment_runner --config configs/dz01_multiband.yaml ablation
```
比较:
- original
- BAGRN-only
- VOLRN-only
- BAGRN-VOLRN
- histogram_matching
- moment_matching
- wallis

### 3. 参数敏感性 (sensitivity)
命令:
```
python -m src.experiment_runner --config configs/dz01_multiband.yaml sensitivity
```
扫描:
- block_size: [200, 300, 400, 500, 600, 800]
- lambda: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
- rho: [0.5, 1.0, 2.0]

### 4. 多景规模扩展 (scale)
命令:
```
python -m src.experiment_runner --config configs/dz01_multiband.yaml scale --scene-count 8
```

### 5. 诊断 (diagnostics)
命令:
```
python -m src.experiment_runner --config configs/dz01_multiband.yaml diagnostics
```

### 6. 接缝评价 (seam)
命令:
```
python -m src.experiment_runner --config configs/dz01_multiband.yaml seam
```

### 7. 光谱保持 (spectral)
命令:
```
python -m src.experiment_runner --config configs/dz01_multiband.yaml spectral
```

## 四、通用参数
- --dry-run: 仅打印计划
- --smoke: 小裁剪测试
- --resume: 断点续跑
- --overwrite: 覆盖已有输出
- --seed: 随机种子
- --methods: 指定方法

## 五、输出目录结构
```
data/output/experiments/<experiment_name>/
  config_resolved.yaml
  environment.json
  registration/
  cache/
  multiband/
    original/
    bagrn/
    volrn_only/
    bagrn_volrn/
    histogram_matching/
    moment_matching/
    wallis/
  quicklooks/
  metrics/
  ablation/
  sensitivity/
  scale/
  diagnostics/
    volrn/
    rbf/
  seam/
  spectral/
  logs/
  summary.json
```

## 六、注意事项
1. 首次运行多波段实验前，确保配置文件中的路径正确
2. 大规模实验建议使用 --dry-run 先检查
3. 不要伪造实验结果
4. 所有中间计算使用 float32/float64，仅最终写出时做类型转换
5. NoData 不参与任何统计和插值
