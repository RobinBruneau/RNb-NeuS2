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
No separate data-preparation node is needed. With albedo maps, it also exports them
scaled as for training (harmonized), ready to texture the mesh with the AliceVision
`Texturing` node (see [Texturing the mesh with the albedo maps](#texturing-the-mesh-with-the-albedo-maps)).

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
| `scalingMode` | Scaling Mode | `auto` | Scene normalization: `auto` prefers silhouettes when masks exist, then falls back to landmarks (`pcd`) or camera centers. One of `auto`, `pcd`, `silhouettes` (minimum sphere enclosing the silhouettes), `cameras`, `none`. |
| `silhouetteMargin` | Silhouette Margin | `20` | Margin (pixels) between the silhouettes and the projection of the bounding sphere (`silhouettes` scaling). Advanced. |
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
| `outputAlbedoFolder` | Harmonized Albedo Folder | `albedo/<viewId>.exr`: the input albedo maps multiplied by the per-view scales used for training, as linear float EXR (not clipped), extended 8 px outside the masks. Written only when `inputAlbedoSfm` is set (the attribute is disabled otherwise). |
| `outputAlbedoSfm` | Harmonized Albedo SfMData | `albedoHarmonized.sfm`: copy of the Albedo Maps SfMData (JSON) whose views point to these maps (views without a map are dropped). Written only when `inputAlbedoSfm` is set. |

The outputs do not take part in the node UID: projects computed with a previous
version keep their cache (the harmonized albedo outputs are then missing until the
node is recomputed).

## Albedo scaling

Photometric-stereo albedo maps are relative: each view is known up to a per-channel
scale (e.g. the light powers of each pose). When an Albedo Maps SfMData is given,
the node trains a geometry-only phase 1, then estimates one scale per view and
channel on the phase-1 mesh (`rnb_neus2/albedo_scaling.py`,
`compute_albedo_scale_ratios`), rescales the albedos and trains phase 3 with them:

- 20000 masked pixels per view are ray-cast onto the mesh; each hit is compared with
  **every other view** that sees it (in front, inside the image, facing it with a
  viewing angle <= 60 deg, unoccluded, inside its mask, valid albedos in both views);
- every view pair with >= 300 shared samples gives one equation
  `log s_i - log s_j = -median(log a_i - log a_j)`, weighted by `sqrt(n_samples)`;
- the equations are solved jointly (least squares): the result does not depend on the
  order of the views and loops of views are closed; the scales are normalized to an
  arithmetic mean of 1 (same albedo level as before the scaling). Views without any
  overlap keep the scale 1 (warning in the log); separate groups of overlapping views
  are normalized independently (warning).

The log reports the median pairwise albedo mismatch before/after the scaling and, per
view, the scale and the fraction of masked pixels clipped in the 16-bit training PNGs
(the testbed reads the albedos in [0, 1], and with RGB+ its loss depends on the
absolute albedo level, so the level is kept and values above 1 are clipped; the
exported EXR maps are not clipped). `prepared_data/albedo_scales.json` keeps the
scales and the pair statistics, `prepared_data/frames.json` the mapping between the
testbed images (`albedos/00003.png`) and the input views (`pose_id`,
`albedo_view_id`, source paths).

On a 12-pose LED-getter turntable capture (biface), the previous scaling, which
chained each view to the next one in the SfMData (viewId) order, compared views on
opposite sides of the object and left a view 1.5x too bright with respect to the views
seeing the same surface; the joint solve brings the median pairwise mismatch from
22% (raw) to 5-6% (vs 12% with the chain).

## Texturing the mesh with the albedo maps

```
LedGetterPS ─Normal/Albedo SfMData, Mask Folder─► RNbNeuS2 ─Harmonized Albedo SfMData─► Texturing.input
                                                           ─Harmonized Albedo Folder──► Texturing.imagesFolder
                                                           ─Output Mesh───────────────► Texturing.inputMesh
```

The scales of the texture are those of the training. `Texturing` settings for a mesh
that does not come from `Meshing` and an SfMData without landmarks:

| Texturing parameter | Value | Why |
|---------------------|-------|-----|
| `input` | `RNbNeuS2` *Harmonized Albedo SfMData* | Cameras of the albedo maps. |
| `imagesFolder` | `RNbNeuS2` *Harmonized Albedo Folder* | Must be **non-empty**: the *Color Mapping* group is disabled otherwise, `--colorMappingFileType` is not passed and `aliceVision_texturing` writes no texture. Files are matched by `<viewId>`. |
| `inputMesh` | `RNbNeuS2` *Output Mesh* | No transform needed (see below). |
| `visibilityRemappingMethod` | `MeshItself` | The default `PullPush` remaps the visibilities of the SfM landmarks and stops with *"Cannot remap visibilities as there is no reference points"*. `MeshItself` ray-casts the visibility of every vertex. |
| `subdivisionTargetRatio` | `0` | The subdivision targets the (empty) landmark cloud: with the default 0.8 the binary writes its outputs, then aborts (*corrupted double-linked list*, exit 134) and the node fails. The RNbNeuS2 mesh is dense anyway. |
| `workingColorSpace` | `Linear` | Blend the linear albedo in linear space. |
| `outputColorSpace` | `AUTO` | EXR texture: linear albedo; PNG texture: sRGB-encoded, directly viewable. |
| `colorMapping.colorMappingFileType` | `exr` (data) or `png` (preview) | Use two `Texturing` nodes for both. |
| `correctEV` | off | The albedo maps are not exposures. |
| `unwrapMethod` | `Basic` | LSCM/ABF are limited to 600k/300k faces. |
| `textureSide` / `downscale` | 4096 / 1 | With half-resolution albedo maps, `Basic` fills ~25% of a 4096 atlas (one tile); 2048 spills over several UDIM tiles. |

Why EXR: photometric-stereo nodes write 16-bit PNG albedos with **linear** values,
but AliceVision reads any untagged 8/16-bit PNG as sRGB. The harmonized EXR maps are
read as linear data, so the texture color space is right.

The mesh needs no transform: `RNbNeuS2` exports its mesh with y and z negated with
respect to the SfMData world, which is the AliceVision mesh-file convention
(`Meshing`, `MeshFiltering`, `Texturing`), so `Texturing` places it correctly with the
SfMData cameras.

## Usage in Meshroom

1. Produce per-view normal maps (and optionally albedo/reflectance maps) and bring
   them into the graph as SfMData.
2. Add an **`RNbNeuS2`** node and connect:
   - `Normal Maps SfMData` ← your normals SfMData (required),
   - `Albedo Maps SfMData` ← albedo SfMData (optional, for reflectance-based training),
   - `Mask SfMData` *or* `Mask Folder` ← masks (optional).
3. Compute the node. The reconstructed mesh is exposed on `Output Mesh`; with
   albedos, the harmonized albedo maps on `Harmonized Albedo Folder` /
   `Harmonized Albedo SfMData` (to texture the mesh, see above).

## Command-line equivalent

The node is a thin wrapper around the standalone pipeline. The same reconstruction
can be launched without Meshroom:

```bash
# From SfMData (normals only)
python ../run_pipeline.py --input normals.sfm --testbed ../build/testbed --output out/

# From SfMData with reflectance (two-phase + albedo scaling), exporting the
# harmonized albedo maps for texturing
python ../run_pipeline.py --input normals.sfm --albedo-sfm albedos.sfm \
    --mask-sfm masks.sfm --has-albedo --testbed ../build/testbed --output out/ \
    --harmonized-albedo-folder out/albedo --harmonized-albedo-sfm out/albedoHarmonized.sfm
```

## Notes

- The node works entirely in unit-sphere coordinates internally and exports the
  mesh back to world coordinates.
- Scene normalization is automatic (`auto` prefers silhouettes when masks are
  available, then falls back to 3D landmarks or camera centers).
- Albedo scaling runs only when an Albedo SfMData is provided (see
  [Albedo scaling](#albedo-scaling)).

## Tests

CPU unit tests (albedo scaling: joint solve, order-independent ring, views without
overlap, masks; export: frame -> viewId mapping, EXR/SfMData writer):

```bash
venv/bin/python -m pytest tests/
```
