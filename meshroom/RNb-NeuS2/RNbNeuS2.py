__version__ = "2.0"

import os
import sys

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

    **Scene normalization:** Auto prefers silhouettes when masks are available, then falls back to 3D landmarks or camera centers.

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
            description="Scene normalization: auto prefers silhouettes when "
                        "masks are available, then falls back to landmarks "
                        "(pcd) or camera centres.",
            values=["auto", "pcd", "silhouettes", "silhouettes_v2",
                    "cameras", "none"],
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

            mask_folder = chunk.node.inputMaskFolder.value or ""
            if not mask_sfm and mask_folder:
                if not os.path.isdir(mask_folder):
                    raise RuntimeError(
                        "Mask folder not found: {}".format(mask_folder))

            # Validate testbed path
            testbed_path = chunk.node.rnbNeuS2Path.evalValue
            if not testbed_path or not os.path.exists(testbed_path):
                raise RuntimeError(
                    "RNB_NEUS2_TESTBED_PATH not found. "
                    "Set it in config.json. Got: '{}'".format(testbed_path))

            # Ensure rnb_neus2 package is importable
            repo_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)

            from rnb_neus2.pipeline import run_full_pipeline

            node_cache = chunk.node.outputFolder.value
            os.makedirs(node_cache, exist_ok=True)

            chunk.logger.info("Starting RNb-NeuS2 pipeline...")

            output_mesh = run_full_pipeline(
                input_path=normal_sfm,
                testbed_path=testbed_path,
                output_dir=node_cache,
                max_steps=chunk.node.maxSteps.value,
                mesh_resolution=chunk.node.meshResolution.value,
                scaling_mode=chunk.node.scalingMode.value,
                sphere_scale=chunk.node.sphereScale.value,
                warmup_ratio=chunk.node.warmupRatio.value,
                mask_weight=chunk.node.maskWeight.value,
                super_normal=chunk.node.superNormal.value,
                use_l1=chunk.node.useL1.value,
                use_rgb_plus=chunk.node.useRgbPlus.value,
                has_albedo=bool(albedo_sfm),
                albedo_sfm_path=albedo_sfm,
                mask_sfm_path=mask_sfm,
                mask_folder_path=mask_folder if not mask_sfm else "",
                logger=chunk.logger,
            )

            chunk.logger.info(
                "RNb-NeuS2 completed: {}".format(output_mesh))

        finally:
            chunk.logManager.end()
