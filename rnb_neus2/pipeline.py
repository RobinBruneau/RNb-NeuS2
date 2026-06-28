"""Full RNb-NeuS2 pipeline orchestration.

Handles: load data -> prepare -> testbed -> albedo scaling -> testbed.
Can be called from CLI (run_pipeline.py) or Meshroom plugin.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


class SimpleLogger:
    """Minimal logger for standalone usage."""
    def info(self, msg):
        print("[INFO] {}".format(msg))

    def warning(self, msg):
        print("[WARN] {}".format(msg))

    def error(self, msg):
        print("[ERROR] {}".format(msg))


def run_testbed(testbed_path, scene_path, max_iter, flags,
                stage_name, logger=None):
    """Run the testbed executable."""
    if logger is None:
        logger = SimpleLogger()

    cmd = [
        testbed_path,
        "--scene", str(scene_path) + "/",
        "--maxiter", str(max_iter),
        "--no-gui",
    ] + flags

    logger.info("{} command: {}".format(stage_name, " ".join(cmd)))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            logger.info(line)

    if result.returncode != 0:
        if result.stderr:
            logger.error(result.stderr)
        raise RuntimeError(
            "{} failed with code {}".format(stage_name, result.returncode))

    logger.info("{} completed".format(stage_name))


def run_two_stage(testbed_path, data_dir, max_steps, common_flags,
                  resolution=1024, no_albedo=False, extra_flags=None,
                  logger=None):
    """Run two-stage training (stage 1 + opti-lights stage 2)."""
    if logger is None:
        logger = SimpleLogger()

    iter_stage1 = int(max_steps * 2 / 3)

    # Stage 1
    stage1_flags = list(common_flags) + ["--save-snapshot"]
    if no_albedo:
        stage1_flags.append("--no-albedo")
    if extra_flags:
        stage1_flags.extend(extra_flags)

    logger.info("Stage 1: {} iterations".format(iter_stage1))
    run_testbed(testbed_path, data_dir, iter_stage1,
                stage1_flags, "Stage 1", logger)

    # Find snapshot
    output_subdir = os.path.join(data_dir, "output")
    snapshot_path = os.path.join(
        output_subdir, "snapshot_{}.msgpack".format(iter_stage1))
    if not os.path.exists(snapshot_path):
        snapshot_path = os.path.join(
            data_dir, "snapshot_{}.msgpack".format(iter_stage1))
    if not os.path.exists(snapshot_path):
        raise RuntimeError(
            "Snapshot not found after {} iterations".format(iter_stage1))

    # Stage 2
    stage2_flags = list(common_flags) + [
        "--opti-lights",
        "--snapshot", snapshot_path,
        "--resolution", str(resolution),
        "--save-mesh",
        "--save-snapshot",
        "--free-memory",
    ]
    if no_albedo:
        stage2_flags.append("--no-albedo")
    if extra_flags:
        stage2_flags.extend(extra_flags)

    logger.info("Stage 2: {} iterations (opti-lights)".format(max_steps))
    run_testbed(testbed_path, data_dir, max_steps,
                stage2_flags, "Stage 2", logger)


def run_with_albedo_scaling(testbed_path, data_dir, max_steps,
                            common_flags, resolution=1024,
                            warmup_ratio=0.1, n_samples=2000,
                            logger=None):
    """Two-phase workflow with albedo scaling between phases."""
    if logger is None:
        logger = SimpleLogger()

    from .albedo_scaling import compute_albedo_scale_ratios, scale_and_save_albedos

    warmup_steps = max(int(max_steps * warmup_ratio), 1000)

    # Phase 1: geometry only
    logger.info("=== Phase 1: Geometry only ({} steps) ===".format(
        warmup_steps))

    phase1_flags = list(common_flags) + [
        "--no-albedo",
        "--save-mesh",
        "--resolution", "512",
        "--free-memory",
    ]
    run_testbed(testbed_path, data_dir, warmup_steps,
                phase1_flags, "Phase 1 (warmup)", logger)

    # Find intermediate mesh
    output_subdir = os.path.join(data_dir, "output")
    mesh_path = os.path.join(
        output_subdir, "mesh_{}.obj".format(warmup_steps))
    if not os.path.exists(mesh_path):
        candidates = list(Path(output_subdir).glob("mesh_*.obj"))
        if not candidates:
            raise RuntimeError(
                "Phase 1 mesh not found in {}".format(output_subdir))
        mesh_path = str(
            max(candidates, key=lambda p: p.stat().st_mtime))

    # Phase 2: scale albedos
    logger.info("=== Albedo scaling ===")

    albedo_dir = os.path.join(data_dir, "albedos")
    scaled_albedo_dir = os.path.join(data_dir, "albedos_scaled")
    transform_json = os.path.join(data_dir, "transform.json")

    scale_ratios = compute_albedo_scale_ratios(
        albedo_path=albedo_dir,
        camera_source=transform_json,
        mesh_path=mesh_path,
        n_samples=n_samples,
        logger=logger,
    )

    scale_and_save_albedos(
        albedo_path=albedo_dir,
        output_albedo_path=scaled_albedo_dir,
        scale_ratios=scale_ratios,
        logger=logger,
    )

    shutil.rmtree(albedo_dir)
    os.rename(scaled_albedo_dir, albedo_dir)
    logger.info("Albedos scaled and replaced")

    # Clean up phase 1 output
    shutil.rmtree(output_subdir, ignore_errors=True)

    # Phase 3: full training with scaled albedos
    logger.info("=== Phase 3: Full training with scaled albedos ===")
    run_two_stage(testbed_path, data_dir, max_steps, common_flags,
                  resolution=resolution, logger=logger)


def postprocess_mesh(data_dir, output_mesh_path, logger=None):
    """Post-process the output mesh (keep largest component, fix normals)."""
    if logger is None:
        logger = SimpleLogger()

    import trimesh

    # Search for mesh in output/ subdirectory first, then data_dir itself
    # Use mesh_*.o* to also match truncated filenames (e.g. mesh_50000_.o)
    output_subdir = os.path.join(data_dir, "output")
    mesh_files = list(Path(output_subdir).glob("mesh_*.o*")) if os.path.isdir(output_subdir) else []
    if not mesh_files:
        mesh_files = list(Path(data_dir).glob("mesh_*.o*"))
    mesh_files = [f for f in mesh_files if f.suffix not in ('.json', '.txt', '.msgpack')]
    if not mesh_files:
        raise RuntimeError(
            "No mesh files in {} or {}".format(output_subdir, data_dir))

    mesh_file = max(mesh_files, key=lambda p: p.stat().st_mtime)
    logger.info("Post-processing: {}".format(mesh_file.name))

    mesh = trimesh.load(str(mesh_file), process=False)

    try:
        if hasattr(mesh, "split"):
            components = mesh.split(only_watertight=False)
            if len(components) > 1:
                mesh = max(
                    components,
                    key=lambda c: c.area if hasattr(c, "area") else 0)
                logger.info("Kept largest component ({} vertices)".format(
                    len(mesh.vertices)))
    except Exception as e:
        logger.warning("Could not split mesh: {}".format(e))

    mesh.fix_normals()

    os.makedirs(os.path.dirname(output_mesh_path), exist_ok=True)
    mesh.export(output_mesh_path, file_type="obj")
    logger.info("Mesh exported to: {}".format(output_mesh_path))

    shutil.rmtree(output_subdir, ignore_errors=True)


def run_full_pipeline(input_path, testbed_path, output_dir,
                      max_steps=10000, mesh_resolution=1024,
                      scaling_mode="auto", sphere_scale=1.0,
                      margin_px=20,
                      warmup_ratio=0.1, mask_weight=1.0,
                      super_normal=False, use_l1=False,
                      use_rgb_plus=True, has_albedo=False,
                      albedo_sfm_path="", mask_sfm_path="",
                      mask_folder_path="", n_samples=2000,
                      logger=None):
    """Run the complete RNb-NeuS2 pipeline.

    Args:
        input_path: Path to input data (directory, .npz, .sfm, .json).
        testbed_path: Path to the testbed binary.
        output_dir: Root output directory.
        max_steps: Total training steps for final phase.
        mesh_resolution: Marching cubes resolution.
        scaling_mode: Scene normalization mode.
        sphere_scale: Target sphere radius.
        warmup_ratio: Fraction of steps for warmup (albedo mode).
        mask_weight: Mask loss weight.
        super_normal: Enable SuperNormal mode.
        use_l1: Use L1 norm.
        use_rgb_plus: Enable RGB+ mode.
        has_albedo: Whether albedos are available.
        albedo_sfm_path: Path to albedo SfMData.
        mask_sfm_path: Path to mask SfMData.
        mask_folder_path: Path to mask folder.
        n_samples: Samples for albedo scaling.
        logger: Logger instance.
    """
    if logger is None:
        logger = SimpleLogger()

    from .dataloaders import load_data
    from .prepare import prepare_testbed_data

    # 1. Load data
    logger.info("=== Loading data from {} ===".format(input_path))
    data = load_data(
        input_path,
        albedo_sfm_path=albedo_sfm_path,
        mask_sfm_path=mask_sfm_path,
        mask_folder_path=mask_folder_path,
        logger=logger,
    )

    # 2. Prepare testbed data
    data_dir = os.path.join(output_dir, "prepared_data")
    logger.info("=== Preparing testbed data ===")
    prepare_testbed_data(
        data, data_dir, logger,
        scaling_mode=scaling_mode,
        sphere_scale=sphere_scale,
        margin_px=margin_px,
    )

    # 3. Common testbed flags
    common_flags = ["--mask-weight", str(mask_weight)]
    if super_normal:
        common_flags.append("--supernormal")
    if use_l1:
        common_flags.append("--lone")
    if not use_rgb_plus:
        common_flags.append("--no-rgbplus")

    # 4. Run testbed
    if has_albedo:
        run_with_albedo_scaling(
            testbed_path, data_dir, max_steps, common_flags,
            resolution=mesh_resolution, warmup_ratio=warmup_ratio,
            n_samples=n_samples, logger=logger)
    else:
        run_two_stage(
            testbed_path, data_dir, max_steps, common_flags,
            resolution=mesh_resolution, no_albedo=True, logger=logger)

    # 5. Post-process mesh
    output_mesh = os.path.join(output_dir, "mesh.obj")
    postprocess_mesh(data_dir, output_mesh, logger)

    logger.info("=== Pipeline complete ===")
    return output_mesh
