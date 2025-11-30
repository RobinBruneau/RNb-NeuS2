# RNb-NeuS2 Meshroom Nodes

This folder contains Meshroom nodes for the RNb-NeuS2 pipeline.

## Overview

Two nodes are available:
1. **PrepareRNbNeuS2Data** - Prepares data for RNb-NeuS2 (scales to unit sphere, creates transform.json)
2. **RNbNeuS2** - Runs the RNb-NeuS2 neural reconstruction

## Installation

### Python Environment

Install the package and dependencies:

```
cd RNb-NeuS2
pip install -e .
```

Required packages:
- numpy
- opencv-python
- trimesh
- embreex
- scipy
- matplotlib

## Nodes Documentation

### 1. PrepareRNbNeuS2Data

Prepares input data for RNb-NeuS2 reconstruction.

**Key Inputs:**
- `inputNormalSfm`: SfMData file containing normal maps paths and camera information
- `inputAlbedoSfm`: SfMData file containing albedo maps paths (optional)
- `inputLandmarksSfm`: SfMData file with 3D landmarks for unit sphere scaling (optional)
- `inputMesh`: Mesh file (.obj, .ply, etc.) for unit sphere scaling (alternative to landmarks)
- `inputMaskFolder`: Folder containing mask images named by poseId
- `scaleToUnitSphere`: Automatically scale poses to fit scene in unit sphere (default: true)
- `sphereScale`: Scale factor for the unit sphere (default: 0.9)

**Outputs:**
- `outputFolder`: Folder containing prepared RNb-NeuS2 data
- `transformJson`: Generated transform.json file with camera parameters

**Features:**
- Automatic unit sphere scaling using SfM landmarks or mesh vertices
- Creates `albedos/` and `normals/` folders with proper masks

### 2. RNbNeuS2

Runs the RNb-NeuS2 neural reconstruction.

**Training Modes:**

**Key Inputs:**
- `input`: Prepared data folder from PrepareRNbNeuS2Data
- `rnbNeuS2Path`: Path to testbed executable
- `maxIter`: Maximum training iterations (default: 15000)
- `resolution`: Mesh resolution for marching cubes (default: 1024)
- `maskWeight`: Weight for mask loss (default: 1.0)
- `noAlbedo`: Disable albedo, use only normals
- `useL1Norm`: Use L1 norm
- `noRgbPlus`: Disable the RGB+ normalization
- `superNormal`: Enable SuperNormal mode (single-stage training)
- `saveMesh`: Save output mesh (default: true)

**Outputs:**
- `outputFolder`: Folder with training results
- `outputMesh`: Generated mesh (OBJ format)
- `outputSnapshot`: Training snapshot (msgpack format)

## Usage Example

1. **Prepare Data:**
   ```
   PrepareRNbNeuS2Data
   ├─ inputNormalSfm: path/to/normals_sfm.abc
   ├─ inputAlbedoSfm: path/to/albedos_sfm.abc (optional)
   ├─ inputMesh: path/to/mesh.obj (for scaling)
   └─ scaleToUnitSphere: true
   ```

2. **Run RNb-NeuS2:**
   ```
   RNbNeuS2
   ├─ input: <output from PrepareRNbNeuS2Data>
   ├─ maxIter: 15000
   ├─ resolution: 1024
   └─ superNormal: false  # Use two-stage RNb-NeuS
   ```

## Command-Line Equivalents

**RNb-NeuS (Two-Stage):**
```bash
# Stage 1
./build/testbed --scene data/ --maxiter 10000 --save-snapshot --mask-weight 1.0 --no-gui

# Stage 2
./build/testbed --scene data/ --maxiter 15000 --snapshot data/snapshot_10000.msgpack \
    --save-mesh --resolution 1024 --opti-lights --mask-weight 1.0 --no-gui
```

**SuperNormal:**
```bash
./build/testbed --scene data/ --maxiter 15000 --supernormal --save-mesh \
    --resolution 1024 --mask-weight 1.0 --no-gui
```

## Notes

- All nodes work in unit sphere coordinates
- PrepareRNbNeuS2Data automatically scales the scene using SfM landmarks or mesh vertices
- RNb-NeuS2 uses two-stage training by default (can be changed to SuperNormal)
