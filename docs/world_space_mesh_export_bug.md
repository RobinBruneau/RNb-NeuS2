# Bug: `run_pipeline.py` exports the mesh in the loader's normalized frame, not world

Found while benchmarking RNb-NeuS2 vs Open-RNb. Affects the standalone
`run_pipeline.py` path with `cameras.npz` (RNb format) inputs (DiLiGenT-MV,
LUCES-MV). SfM/json inputs (Martine) were already correct.

## Symptom

The mesh produced by `run_pipeline.py` comes out in a ~unit cube around the
origin (e.g. bounds `[-0.47, -0.51, -0.54] .. [0.41, 0.50, 0.72]`) instead of
world/mm coordinates that match the ground-truth mesh (e.g.
`[46.9, 39.6, 5.0] .. [122.0, 125.5, 97.7]` for DiLiGenT-MV `bear`). Any
world-space evaluation (Chamfer vs GT) then fails.

## Root cause

The C++ testbed export is correct and complete: `save_mesh`
(`src/marching_cubes.cu`) applies
`p = (vert - nerf_offset)/nerf_scale` then `p = n2w_s·p + n2w_t`, reading `n2w`
from `transform.json` (`src/nerf_loader.cu`, gated by `from_na`). So the mesh is
lifted to world space **iff `transform.json`'s `n2w` is correct**.

The bug is in `rnb_neus2/prepare.py`. It writes `n2w = inv(scale_matrix)`, where
`scale_matrix` is the normalization it computes here. But `rnb_loader` **pre-
normalizes** the poses (`P = world_mat @ scale_mat`) and returns the original
world transform in `data["scale_mat"]`. `prepare.py` recomputes its scaling from
the already-normalized cameras (≈ identity) and drops `data["scale_mat"]`, so
`n2w` ends up ≈ `0.816·I` instead of the true `≈ 84.5·I + t`. The testbed then
applies a near-identity lift → the mesh stays in the loader's normalized frame.

SfM loaders return `scale_mat = None` (their poses are already world), so
`inv(scale_matrix)` alone is correct there — which is why Martine was fine.

## Fix (`rnb_neus2/prepare.py`)

Compose the loader's world transform back into `n2w`:

```python
n2w = np.linalg.inv(scale_matrix)
loader_scale_mat = data.get("scale_mat")
if loader_scale_mat is not None:
    n2w = np.asarray(loader_scale_mat, dtype=np.float32) @ n2w
```

No C++ change / rebuild needed — the export already applies `n2w`.

## Validation

Camera-center round-trip `export(normalized_centers) == world_centers`:
- cameras.npz (DiLiGenT-MV, LUCES-MV): max error 0.0.
- sfm/json (Martine): max error 2e-5 (via `prepare`, silhouette scaling).

End-to-end pilot on `bear` (DiLiGenT-MV): world-space mesh, Chamfer 0.208,
F-score@0.5 0.93 — with no coordinate transform on the evaluation side.
