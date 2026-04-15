# ML Registration Testing Scripts

This directory contains testing scripts for 4 different point cloud registration models on the Predator dataset.

## Available Scripts

| Model | Script | Type | PyTorch | Pretrained Weights |
|-------|--------|------|---------|-------------------|
| **PREDATOR** | `testingPredatorModelOnPredatorData.py` | Baseline | - | ROS2 service |
| **GeoTransformer** | `testingGeoTransformerOnPredatorData.py` | Standalone | 1.7.1 | Download required |
| **RegTR** | `testingRegTROnPredatorData.py` | Standalone | 1.9.1 | Download required |
| **HybridPoint** | `testingHybridPointOnPredatorData.py` | Enhanced GeoTransformer | 1.7.1 | Included |
| **PointRegGPT** | `testingPointRegGPTOnPredatorData.py` | Data Augmentation | 1.7.1 | Optional |

## Quick Start

### 1. Setup Virtual Environments

```bash
cd fsregistration
bash setup_ml_venvs.sh
```

This will create:
- `venv_geotransformer` - For GeoTransformer, HybridPoint, PointRegGPT
- `venv_regtr` - For RegTR (separate due to PyTorch version conflict)

### 2. Download Pretrained Weights

**GeoTransformer:**
```bash
cd fsregistration/ml_registration/geotransformer/weights
wget https://github.com/qinzheng93/GeoTransformer/releases/download/1.0.0/geotransformer-3dmatch.pth.tar
```

**RegTR:**
```bash
cd fsregistration/ml_registration/regtr
wget https://github.com/yewzijian/RegTR/releases/download/v1/trained_models.zip
unzip trained_models.zip
```

**HybridPoint:**
Already included in `ml_registration/HybridPoint/weights_for_hybrid/3dmatch.tar`

**PointRegGPT:**
Optional - for advanced augmentation testing:
```bash
cd fsregistration/ml_registration/PointRegGPT
wget https://github.com/Chen-Suyi/PointRegGPT/releases/download/v1/successive_ddnm_diffusion_results.zip
```

### 3. Run Testing Scripts

**GeoTransformer:**
```bash
source venv_geotransformer/bin/activate
cd fsregistration/pythonScripts/matchingProfiling3D
python testingGeoTransformerOnPredatorData.py ../../configFiles/predatorNothing.yaml low train
```

**HybridPoint:**
```bash
source venv_geotransformer/bin/activate
cd fsregistration/pythonScripts/matchingProfiling3D
python testingHybridPointOnPredatorData.py ../../configFiles/predatorNothing.yaml low train
```
Note: First run will automatically set up HybridPoint by copying necessary files to GeoTransformer.

**PointRegGPT:**
```bash
source venv_geotransformer/bin/activate
cd fsregistration/pythonScripts/matchingProfiling3D
python testingPointRegGPTOnPredatorData.py ../../configFiles/predatorNothing.yaml low train
```
Note: Tests augmentation quality using GeoTransformer as baseline.

**RegTR:**
```bash
source venv_regtr/bin/activate
cd fsregistration/pythonScripts/matchingProfiling3D
python testingRegTROnPredatorData.py ../../configFiles/predatorNothing.yaml low train
```

## Output Format

All scripts output to CSV files in the same format:

```csv
index,overlap%,GT_roll,GT_pitch,GT_yaw,GT_x,GT_y,GT_z,Est_roll,Est_pitch,Est_yaw,Est_x,Est_y,Est_z
```

Files are saved in `outputFiles/` directory:
- `outfile_geotransformer_{noise}_{dataset}.csv`
- `outfile_regtr_{noise}_{dataset}.csv`
- `outfile_hybridpoint_{noise}_{dataset}.csv`
- `outfile_pointreggpt_{noise}_{dataset}.csv`

## Arguments

All scripts accept the same 3 arguments:

1. **config**: Path to Predator config file (e.g., `../../configFiles/predatorNothing.yaml`)
2. **type_of_noise**: Noise level - `None`, `low`, or `high`
3. **type_of_data**: Dataset type - `train` or `val`

## Model Details

### GeoTransformer (CVPR 2022)
- **Architecture**: KPConvFPN + Geometric Transformer
- **Approach**: Superpoint matching with no RANSAC needed
- **Performance**: ~91% Registration Recall on 3DMatch

### HybridPoint (ICME 2023 Oral)
- **Architecture**: Enhanced GeoTransformer with hybrid sampling
- **Approach**: Salient points + uniformly distributed points
- **Performance**: ~93.4% Registration Recall on 3DMatch
- **Special**: Auto-setup on first run

### RegTR (CVPR 2022)
- **Architecture**: End-to-end transformer
- **Approach**: Direct correspondence prediction
- **Performance**: Comparable to GeoTransformer
- **Note**: Requires CUDA for PyTorch3D and MinkowskiEngine

### PointRegGPT (ECCV 2024)
- **Purpose**: Data augmentation tool (not standalone registration)
- **Approach**: Uses GeoTransformer to test augmentation quality
- **Performance**: Measures improvement from augmented data
- **Special**: Tests realistic noise and occlusion simulation

## Troubleshooting

### CUDA Issues
- If you don't have CUDA, scripts will use CPU (slower)
- RegTR requires CUDA for full functionality (PyTorch3D, MinkowskiEngine)

### Import Errors
- Ensure you're in the correct virtual environment
- Run setup script if dependencies are missing

### HybridPoint Setup Failed
- Check that `ml_registration/HybridPoint/GeoTransformer-main.zip` exists
- Run setup manually by extracting and copying files as described in HybridPoint README

### Weight Files Not Found
- Download weights as described above
- Check file paths in scripts match your directory structure

## Comparison

To compare model performance:
1. Run all 4 scripts with same noise level and dataset
2. Compare CSV output files
3. Analyze overlap percentages and transformation errors

Example comparison script:
```python
import pandas as pd
import numpy as np

# Load all results
models = ['geotransformer', 'regtr', 'hybridpoint', 'pointreggpt']
noise = 'low'
dataset = 'train'

results = {}
for model in models:
    df = pd.read_csv(f'outputFiles/outfile_{model}_{noise}_{dataset}.csv')
    results[model] = {
        'mean_overlap': df['overlap%'].mean(),
        'std_overlap': df['overlap%'].std(),
        'mean_rotation_error': np.mean(np.sqrt(
            (df['GT_roll'] - df['Est_roll'])**2 +
            (df['GT_pitch'] - df['Est_pitch'])**2 +
            (df['GT_yaw'] - df['Est_yaw'])**2
        )),
        'mean_translation_error': np.mean(np.sqrt(
            (df['GT_x'] - df['Est_x'])**2 +
            (df['GT_y'] - df['Est_y'])**2 +
            (df['GT_z'] - df['Est_z'])**2
        ))
    }

# Print comparison
for model, metrics in results.items():
    print(f"{model}: overlap={metrics['mean_overlap']:.2f}±{metrics['std_overlap']:.2f}, "
          f"rot_err={metrics['mean_rotation_error']:.4f}, trans_err={metrics['mean_translation_error']:.4f}")
```

## Notes

- All scripts mirror the structure of `testingPredatorModelOnPredatorData.py`
- No ROS2 dependencies (except original PREDATOR script)
- CPU fallback available for all models
- Same CSV output format for easy comparison
- HybridPoint auto-configures on first run
