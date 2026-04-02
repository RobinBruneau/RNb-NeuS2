# Albedo Scaling Library

This library provides functions to scale albedos based on multi-view consistency using ray tracing through a mesh.

## Supported Camera Formats

The library automatically detects and supports three camera formats:

1. **sfmData (.sfm)** - Meshroom's SfMData format (preferred for Meshroom workflows)
2. **cameras.npz** - NumPy archive with camera matrices (legacy format)
3. **transform.json** - NeRF/Instant-NGP format

## Usage

### Python Library

```python
from scripts.utils.albedo_scaling_lib import compute_albedo_scale_ratios, scale_and_save_albedos

# Compute scaling ratios
scale_ratios = compute_albedo_scale_ratios(
    albedo_path="path/to/albedos/",
    camera_source="path/to/cameras.sfm",  # or .npz or transform.json
    mesh_path="path/to/mesh.obj",
    n_samples=2000,
    logger=None  # Optional logger
)

# Apply scaling and save
scale_and_save_albedos(
    albedo_path="path/to/albedos/",
    output_albedo_path="path/to/output/albedos/",
    scale_ratios=scale_ratios,
    bit_depth=16,
    logger=None
)
```

### Command-Line Script

```bash
# With sfmData (Meshroom workflow)
python scripts/scale_albedos.py --folder data/ --sfmdata data/albedos.sfm

# With cameras.npz (legacy)
python scripts/scale_albedos.py --folder data/ --cameras_npz data/../cameras.npz

# With transform.json (NeRF workflow)
python scripts/scale_albedos.py --folder data/ --transform_json data/transform.json

# Auto-detect (tries cameras.npz in parent, then transform.json in folder)
python scripts/scale_albedos.py --folder data/

# Additional options
python scripts/scale_albedos.py --folder data/ --sfmdata albedos.sfm \
    --mesh_path mesh.obj --n_samples 3000 --bit_depth 16
```

### Meshroom Node

The `ScaleAlbedos` node uses **sfmData** format by default:
- Input: `inputAlbedoSfm` (sfmData file with albedo images and cameras)
- No need to specify camera format, it's automatic

## Camera Format Details

### sfmData (.sfm)
- Used by Meshroom
- Contains views, poses, intrinsics
- Images matched by poseId (e.g., "46483756.png")
- Requires `pyalicevision` package

### cameras.npz
- Legacy format
- Contains `world_mat_0`, `world_mat_1`, etc.
- Simple NumPy arrays

### transform.json
- NeRF/Instant-NGP format
- Contains frames with transform matrices
- Images matched by filename in `file_path`

## Algorithm

1. Load albedo images with masks
2. Load camera parameters (auto-detect format)
3. Load mesh for ray tracing
4. For each camera:
   - Ray trace from camera through mesh
   - Find visible intersection points in neighbor cameras
   - Compare albedo values at those points
   - Compute ratios
5. Compute median ratios across all views
6. Propagate and normalize ratios
7. Apply scaling to all albedos

## Installation

The library is part of the `rnb_neus2_scripts` package. Install in development mode:

```bash
source venv/bin/activate
pip install -e .
```

This makes the library available everywhere as `from scripts.utils.albedo_scaling_lib import ...`
