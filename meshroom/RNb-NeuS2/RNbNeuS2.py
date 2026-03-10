__version__ = "2.0"

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class RNbNeuS2(desc.Node):
    """
    Neural surface reconstruction from normal/albedo maps using RNb-NeuS2.

    Accepts SfMData files for normals, albedos, and masks.
    When albedos are provided, uses two-phase training with
    automatic albedo scaling via multi-view consistency.
    Exports mesh in world coordinates.
    """

    category = "Neural Reconstruction"
    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize('inputNormalSfm')

    documentation = """
    Neural surface reconstruction from multi-view normal maps using RNb-NeuS2.
    Uses a CUDA testbed with instant neural graphics primitives.

    **Inputs:**
    - Normal maps SfMData (required)
    - Albedo maps SfMData (optional — enables two-phase training)
    - Mask SfMData or folder (optional)

    **Scene normalization:** Auto-detected from 3D landmarks or camera centers.

    **Processing:**
    - Internally converts SfMData to transform.json format
    - Two-stage training: stage 1 without opti-lights, stage 2 with opti-lights
    - Optional albedo scaling via multi-view consistency

    **Output:** OBJ mesh in world coordinates.
    """

    inputs = [
        desc.File(
            name="inputNormalSfm",
            label="Normal Maps SfMData",
            description="SfMData file pointing to normal map images. Required.",
            value="",
        ),
        desc.File(
            name="inputAlbedoSfm",
            label="Albedo Maps SfMData",
            description="SfMData file pointing to albedo images. "
                        "If provided, enables two-phase training with "
                        "albedo scaling.",
            value="",
        ),
        desc.File(
            name="inputMaskSfm",
            label="Mask SfMData",
            description="SfMData file pointing to mask images. "
                        "Used for masking during reconstruction.",
            value="",
        ),
        desc.File(
            name="inputMaskFolder",
            label="Mask Folder",
            description="Folder containing mask images with viewId in filename "
                        "(e.g. '12345.png'). "
                        "Ignored when Mask SfMData is already provided.",
            value="",
        ),
        desc.IntParam(
            name="maxSteps",
            label="Max Training Steps",
            description="Total training iterations for stage 2. "
                        "Stage 1 uses 2/3 of this value.",
            value=15000,
            range=(1000, 100000, 1000),
        ),
        desc.IntParam(
            name="meshResolution",
            label="Mesh Resolution",
            description="Marching cubes resolution for final mesh extraction.",
            value=1024,
            range=(256, 4096, 256),
        ),
        desc.ChoiceParam(
            name="scalingMode",
            label="Scaling Mode",
            description="Scene normalization: auto detects landmarks or "
                        "falls back to camera centers.",
            values=["auto", "pcd", "silhouettes", "cameras", "none"],
            value="auto",
            exclusive=True,
        ),
        desc.FloatParam(
            name="sphereScale",
            label="Sphere Scale",
            description="Target scale within unit sphere after normalization.",
            value=1.0,
            range=(0.1, 2.0, 0.1),
        ),
        desc.FloatParam(
            name="warmupRatio",
            label="Phase 1 Ratio",
            description="Fraction of maxSteps for warmup phase (geometry only, "
                        "no albedo). Only used when albedos are provided.",
            value=0.1,
            range=(0.01, 0.5, 0.01),
        ),
        desc.FloatParam(
            name="maskWeight",
            label="Mask Weight",
            description="Weight for the mask loss during training.",
            value=1.0,
            range=(0.0, 10.0, 0.1),
        ),
        desc.BoolParam(
            name="superNormal",
            label="SuperNormal",
            description="Enable SuperNormal mode (MVPS method).",
            value=False,
        ),
        desc.BoolParam(
            name="useL1",
            label="L1 Norm",
            description="Use L1 norm for color loss. If disabled, uses L2.",
            value=False,
        ),
        desc.BoolParam(
            name="useRgbPlus",
            label="RGB+",
            description="Enable RGB+ normalization mode.",
            value=True,
        ),
        desc.BoolParam(
            name="useGpu",
            label="Use GPU",
            description="Use GPU for training (CUDA required).",
            value=True,
            invalidate=False,
        ),
        desc.File(
            name="rnbNeuS2Path",
            label="RNb-NeuS2 Testbed Path",
            description="Path to the RNb-NeuS2 testbed executable. "
                        "Set via config.json key RNB_NEUS2_TESTBED_PATH.",
            value="${RNB_NEUS2_TESTBED_PATH}",
            advanced=True,
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level for logging.",
            values=VERBOSE_LEVEL,
            value="info",
            exclusive=True,
        ),
    ]

    outputs = [
        desc.File(
            name="outputFolder",
            label="Output Folder",
            description="Output folder containing all training artifacts.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="outputMesh",
            label="Output Mesh",
            description="Reconstructed mesh in world coordinates.",
            value="{nodeCacheFolder}/mesh.obj",
            semantic="mesh",
            group="",
        ),
    ]

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            # --- Input validation ---
            normal_sfm = chunk.node.inputNormalSfm.value
            if not normal_sfm:
                raise RuntimeError("inputNormalSfm is required but empty.")
            if not os.path.exists(normal_sfm):
                raise RuntimeError(
                    "Normal SfM file not found: {}".format(normal_sfm))

            albedo_sfm = chunk.node.inputAlbedoSfm.value or ""
            if albedo_sfm and not os.path.exists(albedo_sfm):
                raise RuntimeError(
                    "Albedo SfM file not found: {}".format(albedo_sfm))

            mask_sfm = chunk.node.inputMaskSfm.value or ""
            if mask_sfm and not os.path.exists(mask_sfm):
                raise RuntimeError(
                    "Mask SfM file not found: {}".format(mask_sfm))

            # Generate mask SfM from folder if needed
            mask_folder = chunk.node.inputMaskFolder.value or ""
            if not mask_sfm and mask_folder:
                if not os.path.isdir(mask_folder):
                    raise RuntimeError(
                        "Mask folder not found: {}".format(mask_folder))
                mask_sfm = self._generate_mask_sfm(
                    chunk, normal_sfm, mask_folder)

            # Validate testbed path
            testbed_path = chunk.node.rnbNeuS2Path.evalValue
            if not testbed_path or not os.path.exists(testbed_path):
                raise RuntimeError(
                    "RNB_NEUS2_TESTBED_PATH not found. "
                    "Set it in config.json. Got: '{}'".format(testbed_path))

            node_cache = chunk.node.outputFolder.value
            os.makedirs(node_cache, exist_ok=True)

            # --- Prepare data (SfMData → transform.json) ---
            data_dir = os.path.join(node_cache, "prepared_data")

            # Import _prepare_data from the same plugin directory
            plugin_dir = os.path.dirname(__file__)
            original_path = sys.path[:]
            sys.path.insert(0, plugin_dir)
            try:
                from _prepare_data import prepare_testbed_data
            finally:
                sys.path[:] = original_path

            chunk.logger.info("Preparing testbed data...")
            prep_result = prepare_testbed_data(
                normal_sfm_path=normal_sfm,
                output_folder=data_dir,
                logger=chunk.logger,
                albedo_sfm_path=albedo_sfm,
                mask_sfm_path=mask_sfm,
                mask_folder_path=mask_folder if not mask_sfm else "",
                scaling_mode=chunk.node.scalingMode.value,
                sphere_scale=chunk.node.sphereScale.value,
            )

            # --- Common testbed flags ---
            common_flags = [
                '--mask-weight', str(chunk.node.maskWeight.value),
            ]
            if chunk.node.superNormal.value:
                common_flags.append('--supernormal')
            if chunk.node.useL1.value:
                common_flags.append('--lone')
            if not chunk.node.useRgbPlus.value:
                common_flags.append('--no-rgbplus')

            max_steps = chunk.node.maxSteps.value
            has_albedo = bool(albedo_sfm)

            if has_albedo:
                self._run_with_albedo_scaling(
                    chunk, testbed_path, data_dir, max_steps,
                    common_flags, prep_result)
            else:
                self._run_two_stage(
                    chunk, testbed_path, data_dir, max_steps,
                    common_flags, no_albedo=True)

            # --- Post-process mesh ---
            self._postprocess_mesh(chunk, data_dir)

            chunk.logger.info("RNb-NeuS2 completed successfully")

        finally:
            chunk.logManager.end()

    @staticmethod
    def _check_cancelled(chunk):
        """Raise if the user cancelled the node."""
        if hasattr(chunk.node, 'stopped') and chunk.node.stopped():
            raise RuntimeError("Cancelled by user")

    def _run_testbed(self, chunk, testbed_path, scene_path, max_iter,
                     flags, stage_name):
        """Run the testbed executable."""
        cmd = [
            testbed_path,
            '--scene', str(scene_path) + '/',
            '--maxiter', str(max_iter),
            '--no-gui',
        ] + flags

        self._check_cancelled(chunk)
        chunk.logger.info("{} command: {}".format(stage_name, ' '.join(cmd)))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                chunk.logger.info(line)

        if result.returncode != 0:
            if result.stderr:
                chunk.logger.error(result.stderr)
            raise RuntimeError(
                "{} failed with code {}".format(
                    stage_name, result.returncode))

        chunk.logger.info("{} completed".format(stage_name))

    def _run_two_stage(self, chunk, testbed_path, data_dir, max_steps,
                       common_flags, no_albedo=False, extra_flags=None):
        """Run two-stage training (stage 1 + opti-lights stage 2)."""
        resolution = chunk.node.meshResolution.value
        iter_stage1 = int(max_steps * 2 / 3)

        # --- Stage 1: initial optimization ---
        stage1_flags = list(common_flags) + ['--save-snapshot']
        if no_albedo:
            stage1_flags.append('--no-albedo')
        if extra_flags:
            stage1_flags.extend(extra_flags)

        chunk.logger.info(
            "Stage 1: {} iterations".format(iter_stage1))
        self._run_testbed(
            chunk, testbed_path, data_dir, iter_stage1,
            stage1_flags, "Stage 1")

        # Find snapshot (testbed writes to <scene>/output/)
        snapshot_path = os.path.join(
            data_dir, "output",
            "snapshot_{}.msgpack".format(iter_stage1))
        if not os.path.exists(snapshot_path):
            # Fallback: check scene root
            snapshot_path = os.path.join(
                data_dir, "snapshot_{}.msgpack".format(iter_stage1))
        if not os.path.exists(snapshot_path):
            raise RuntimeError(
                "Stage 1 snapshot not found after {} iterations".format(
                    iter_stage1))

        # --- Stage 2: optimization with opti-lights ---
        stage2_flags = list(common_flags) + [
            '--opti-lights',
            '--snapshot', snapshot_path,
            '--resolution', str(resolution),
            '--save-mesh',
            '--save-snapshot',
            '--free-memory',
        ]
        if no_albedo:
            stage2_flags.append('--no-albedo')
        if extra_flags:
            stage2_flags.extend(extra_flags)

        chunk.logger.info(
            "Stage 2: {} iterations (opti-lights)".format(max_steps))
        self._run_testbed(
            chunk, testbed_path, data_dir, max_steps,
            stage2_flags, "Stage 2")

    def _run_with_albedo_scaling(self, chunk, testbed_path, data_dir,
                                 max_steps, common_flags, prep_result):
        """Two-phase workflow with albedo scaling."""
        warmup_ratio = chunk.node.warmupRatio.value
        warmup_steps = max(int(max_steps * warmup_ratio), 1000)

        # --- Phase 1: geometry only (no albedo) ---
        chunk.logger.info(
            "=== Phase 1: Geometry only ({} steps) ===".format(warmup_steps))

        phase1_flags = list(common_flags) + [
            '--no-albedo',
            '--save-mesh',
            '--resolution', '512',
            '--free-memory',
        ]
        self._run_testbed(
            chunk, testbed_path, data_dir, warmup_steps,
            phase1_flags, "Phase 1 (warmup)")

        # Find intermediate mesh
        output_subdir = os.path.join(data_dir, "output")
        mesh_path = os.path.join(
            output_subdir, "mesh_{}.obj".format(warmup_steps))
        if not os.path.exists(mesh_path):
            # Search for any mesh file
            mesh_candidates = list(
                Path(output_subdir).glob("mesh_*.obj"))
            if not mesh_candidates:
                raise RuntimeError(
                    "Phase 1 mesh not found in {}".format(output_subdir))
            mesh_path = str(
                max(mesh_candidates, key=lambda p: p.stat().st_mtime))

        # --- Phase 2: Scale albedos ---
        self._check_cancelled(chunk)
        chunk.logger.info("=== Albedo scaling ===")

        # Locate albedo_scaling_lib in the RNb-NeuS2 repo
        testbed_dir = os.path.dirname(
            os.path.realpath(chunk.node.rnbNeuS2Path.evalValue))
        # The lib is at <repo>/scripts/utils/albedo_scaling_lib.py
        repo_root = os.path.dirname(testbed_dir)
        scaling_lib_dir = os.path.join(repo_root, "scripts", "utils")

        original_path = sys.path[:]
        sys.path.insert(0, scaling_lib_dir)
        try:
            from albedo_scaling_lib import (
                compute_albedo_scale_ratios, scale_and_save_albedos,
            )
        except ImportError as e:
            chunk.logger.warning(
                "Albedo scaling library not found ({}), "
                "skipping scaling".format(e))
            # Clean up phase 1 output and run normally
            shutil.rmtree(output_subdir, ignore_errors=True)
            self._run_two_stage(
                chunk, testbed_path, data_dir, max_steps,
                common_flags)
            return
        finally:
            sys.path[:] = original_path

        albedo_dir = os.path.join(data_dir, "albedos")
        scaled_albedo_dir = os.path.join(data_dir, "albedos_scaled")
        transform_json = os.path.join(data_dir, "transform.json")

        # The testbed exports mesh in world space (applies n2w during
        # marching cubes), but cameras in transform.json are in normalized
        # space. Create a world-space transform.json for albedo scaling
        # so that ray tracing offsets work correctly at world scale.
        import numpy as np
        n2w = prep_result["n2w"]  # normalized-to-world (4x4)

        with open(transform_json, "r") as f:
            transform_data = json.load(f)

        world_frames = []
        for frame in transform_data["frames"]:
            c2w_norm = np.array(frame["transform_matrix"], dtype=np.float64)
            # Convert normalized c2w to world c2w: c2w_world = n2w @ c2w_norm
            c2w_world = n2w @ c2w_norm
            wf = dict(frame)
            wf["transform_matrix"] = c2w_world.tolist()
            world_frames.append(wf)

        world_transform = dict(transform_data)
        world_transform["frames"] = world_frames
        world_transform_path = os.path.join(data_dir, "transform_world.json")
        with open(world_transform_path, "w") as f:
            json.dump(world_transform, f, indent=4)

        chunk.logger.info(
            "Created world-space transform for albedo scaling")

        scale_ratios = compute_albedo_scale_ratios(
            albedo_path=albedo_dir,
            camera_source=world_transform_path,
            mesh_path=mesh_path,
            n_samples=2000,
            logger=chunk.logger,
        )

        scale_and_save_albedos(
            albedo_path=albedo_dir,
            output_albedo_path=scaled_albedo_dir,
            scale_ratios=scale_ratios,
            logger=chunk.logger,
        )

        # Replace albedos with scaled version
        shutil.rmtree(albedo_dir)
        os.rename(scaled_albedo_dir, albedo_dir)
        chunk.logger.info("Albedos scaled and replaced")

        # Clean up phase 1 output before main training
        shutil.rmtree(output_subdir, ignore_errors=True)

        # --- Phase 3: Full training with scaled albedos ---
        self._check_cancelled(chunk)
        chunk.logger.info(
            "=== Phase 3: Full training with scaled albedos ===")
        self._run_two_stage(
            chunk, testbed_path, data_dir, max_steps,
            common_flags)

    def _postprocess_mesh(self, chunk, data_dir):
        """Move and post-process the output mesh."""
        import trimesh

        output_subdir = os.path.join(data_dir, "output")
        if not os.path.isdir(output_subdir):
            raise RuntimeError(
                "No output directory found: {}".format(output_subdir))

        mesh_files = list(Path(output_subdir).glob("mesh_*.obj"))
        if not mesh_files:
            raise RuntimeError(
                "No mesh files found in {}".format(output_subdir))

        # Take the latest mesh
        mesh_file = max(mesh_files, key=lambda p: p.stat().st_mtime)
        chunk.logger.info(
            "Post-processing mesh: {}".format(mesh_file.name))

        mesh = trimesh.load(str(mesh_file), process=False)

        # Keep largest component
        try:
            if hasattr(mesh, 'split'):
                components = mesh.split(only_watertight=False)
                if len(components) > 1:
                    mesh = max(
                        components,
                        key=lambda c: c.area if hasattr(c, 'area') else 0)
                    chunk.logger.info(
                        "Kept largest component ({} vertices)".format(
                            len(mesh.vertices)))
        except (ImportError, Exception) as e:
            chunk.logger.warning(
                "Could not split mesh components: {}".format(e))

        # Fix normals
        mesh.fix_normals()

        # Export to final output path
        output_mesh = chunk.node.outputMesh.value
        mesh.export(output_mesh, file_type='obj')
        chunk.logger.info("Mesh exported to: {}".format(output_mesh))

        # Clean up intermediate output
        shutil.rmtree(output_subdir, ignore_errors=True)

    @staticmethod
    def _generate_mask_sfm(chunk, normal_sfm_path, mask_folder):
        """Generate a mask SfMData JSON from normal SfMData + mask folder.

        Masks are matched to views by viewId: a mask file matches if its
        filename (without extension) contains the viewId string.
        """
        import copy
        import json

        with open(normal_sfm_path, 'r') as f:
            sfm_data = json.load(f)

        mask_files = [
            entry.name for entry in os.scandir(mask_folder)
            if entry.is_file()
        ]

        matched = 0
        views_out = []
        for view in sfm_data.get('views', []):
            view_id = str(view['viewId'])
            candidates = [
                fn for fn in mask_files
                if view_id in os.path.splitext(fn)[0]
            ]
            if not candidates:
                continue
            if len(candidates) > 1:
                chunk.logger.warning(
                    "Multiple masks match viewId {}: {}. Using first.".format(
                        view_id, candidates))
            mask_path = os.path.join(mask_folder, candidates[0])
            view_copy = copy.deepcopy(view)
            view_copy['path'] = mask_path
            views_out.append(view_copy)
            matched += 1

        if matched == 0:
            raise RuntimeError(
                "No masks matched any viewId from {}.".format(
                    normal_sfm_path))

        sfm_out = copy.deepcopy(sfm_data)
        sfm_out['views'] = views_out

        node_cache = chunk.node.outputFolder.value
        os.makedirs(node_cache, exist_ok=True)
        out_path = os.path.join(node_cache, 'generated_mask_sfm.json')
        with open(out_path, 'w') as f:
            json.dump(sfm_out, f, indent=2)

        chunk.logger.info(
            "Generated mask SfM: {}/{} views matched".format(
                matched, len(sfm_data.get('views', []))))
        return out_path
