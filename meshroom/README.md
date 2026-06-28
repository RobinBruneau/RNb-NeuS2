# RNb-NeuS2 — Meshroom Plugin

This folder turns RNb-NeuS2 into a [Meshroom](https://github.com/alicevision/Meshroom)
node so the whole reconstruction runs inside a photogrammetry graph.

## Overview

A single node is provided:

**`RNbNeuS2`** (category *Neural Reconstruction*) — neural surface reconstruction
from multi-view normal and reflectance maps. It runs the **entire** pipeline
internally: it reads AliceVision SfMData, converts them to the testbed format,
normalizes the scene into the unit sphere, trains the CUDA testbed (two-stage, with
optional multi-view albedo scaling), and exports the mesh in world coordinates.
No separate data-preparation node is needed.

## Installation

The node calls into the `rnb_neus2` Python package and the compiled `testbed`
binary from this repository.

1. **Build the testbed and install the package** — follow the
   [main README](../README.md#installation) (`cmake` build + `pip install -e .` in a
   Python 3.10 environment).

2. **Expose the environment as the plugin `venv/`.** Meshroom looks for a `venv/`
   directory at the plugin root and adds its `site-packages` to `PYTHONPATH`.
   - With the **`venv` install** (Option A in the main README), `venv/` already sits
     at the repository root — nothing to do.
   - With a **conda env** (Option B), symlink it from the repository root, with the
     env active:
     ```bash
     ln -s "$CONDA_PREFIX" venv
     ```

3. **Point the node to the testbed.** The path is resolved from `config.json`:
   ```json
   [
       {"key": "RNB_NEUS2_TESTBED_PATH", "type": "path", "value": "../build/testbed"}
   ]
   ```
   The default (`../build/testbed`, relative to this `meshroom/` folder) matches the
   standard build location. Edit it if your testbed lives elsewhere. An environment
   variable of the same name overrides this value.

4. **Register the plugin and launch Meshroom:**
   ```bash
   export MESHROOM_PLUGINS_PATH=/path/to/RNb-NeuS2
   meshroom
   ```

## Node: `RNbNeuS2`

### Inputs

| Parameter | Label | Default | Description |
|-----------|-------|---------|-------------|
| `inputNormalSfm` | Normal Maps SfMData | — | SfMData pointing to the normal-map images. **Required.** |
| `inputAlbedoSfm` | Albedo Maps SfMData | — | SfMData pointing to albedo images. If set, enables two-phase training with albedo scaling. |
| `inputMaskSfm` | Mask SfMData | — | SfMData pointing to mask images. |
| `inputMaskFolder` | Mask Folder | — | Folder of masks named by `viewId` (e.g. `12345.png`). Ignored when Mask SfMData is provided. |
| `maxSteps` | Max Training Steps | 15000 | Total iterations for stage 2 (stage 1 uses 2/3 of this). |
| `meshResolution` | Mesh Resolution | 1024 | Marching cubes resolution for the final mesh. |
| `scalingMode` | Scaling Mode | `auto` | Scene normalization: `auto` prefers silhouettes when masks exist, then falls back to landmarks (`pcd`) or camera centers. One of `auto`, `pcd`, `silhouettes`, `silhouettes_v2`, `cameras`, `none`. |
| `sphereScale` | Sphere Scale | 1.0 | Target scale within the unit sphere after normalization. |
| `warmupRatio` | Phase 1 Ratio | 0.1 | Fraction of `maxSteps` for the geometry-only warmup (albedo mode only). |
| `maskWeight` | Mask Weight | 1.0 | Weight of the mask loss. |
| `superNormal` | SuperNormal | false | Enable SuperNormal mode (MVPS method). |
| `useL1` | L1 Norm | false | Use L1 color loss instead of L2. |
| `useRgbPlus` | RGB+ | true | Enable the RGB+ reflectance-singularity correction. |
| `useGpu` | Use GPU | true | Use GPU for training (CUDA required). |
| `rnbNeuS2Path` | RNb-NeuS2 Testbed Path | `${RNB_NEUS2_TESTBED_PATH}` | Path to the testbed executable (advanced; from `config.json`). |
| `verboseLevel` | Verbose Level | `info` | Logging verbosity (`fatal`, `error`, `warning`, `info`, `debug`, `trace`). |

### Outputs

| Parameter | Label | Description |
|-----------|-------|-------------|
| `outputFolder` | Output Folder | Folder with all training artifacts (`nodeCacheFolder`). |
| `outputMesh` | Output Mesh | Reconstructed `mesh.obj` in world coordinates (viewable in Meshroom). |

## Usage in Meshroom

1. Produce per-view normal maps (and optionally albedo/reflectance maps) and bring
   them into the graph as SfMData.
2. Add an **`RNbNeuS2`** node and connect:
   - `Normal Maps SfMData` ← your normals SfMData (required),
   - `Albedo Maps SfMData` ← albedo SfMData (optional, for reflectance-based training),
   - `Mask SfMData` *or* `Mask Folder` ← masks (optional).
3. Compute the node. The reconstructed mesh is exposed on `Output Mesh`.

## Command-line equivalent

The node is a thin wrapper around the standalone pipeline. The same reconstruction
can be launched without Meshroom:

```bash
# From SfMData (normals only)
python ../run_pipeline.py --input normals.sfm --testbed ../build/testbed --output out/

# From SfMData with reflectance (two-phase + albedo scaling)
python ../run_pipeline.py --input normals.sfm --albedo-sfm albedos.sfm \
    --mask-sfm masks.sfm --has-albedo --testbed ../build/testbed --output out/
```

## Notes

- The node works entirely in unit-sphere coordinates internally and exports the
  mesh back to world coordinates.
- Scene normalization is automatic (`auto` prefers silhouettes when masks are
  available, then falls back to 3D landmarks or camera centers).
- Albedo scaling runs only when an Albedo SfMData is provided.
