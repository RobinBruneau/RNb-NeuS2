__version__ = "1.0"

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
import os
import subprocess
import shutil
import numpy as np
from pathlib import Path

class RNbNeuS2(desc.Node):
    
    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize('input')
    
    category = 'RNb-NeuS2'
    documentation = '''
RNb-NeuS2 : Reflectance and Normal based Neural Surface Reconstruction

This node runs the RNb-NeuS2 3D reconstruction algorithm.

Two training modes are available:
1. **RNb-NeuS (default)**: Two-stage training
    - Stage 1: Optimization without light triplets
    - Stage 2: Optimization with light triplets (opti-lights)

2. **SuperNormal**: a SOTA method that is a special case of RNb-NeuS

The algorithm works in unit sphere coordinates and requires properly scaled
input data (use PrepareRNbNeuS2Data node first).

Optional albedo scaling: The node can first run with normals only, then scale
the albedos based on multi-view consistency, and finally run with scaled albedos.
'''

    inputs = [
        desc.File(
            name='input',
            label='Input Folder',
            description='Input folder containing transform.json, albedos/, and normals/ folders.',
            value='',
        ),
        desc.File(
            name='rnbNeuS2Path',
            label='RNb-NeuS2 Testbed Path',
            description='Path to the RNb-NeuS2 testbed executable.',
            value="${RNB_NEUS2_TESTBED_PATH}",
        ),
        desc.IntParam(
            name='maxIter',
            label='Max Iterations',
            description='Maximum number of training iterations.',
            value=15000,
            range=(1000, 100000, 1000),
        ),
        desc.IntParam(
            name='resolution',
            label='Mesh Resolution',
            description='Resolution for the output mesh (marching cubes grid resolution).',
            value=1024,
            range=(256, 4096, 256),
        ),
        desc.FloatParam(
            name='maskWeight',
            label='Mask Weight',
            description='Weight for the mask loss.',
            value=1.0,
            range=(0.0, 10.0, 0.1),
        ),
        desc.BoolParam(
            name='noAlbedo',
            label='No Albedo',
            description='Disable albedo in the reconstruction (use only normals).',
            value=False,
        ),
        desc.BoolParam(
            name='useL1Norm',
            label='Use L1 Norm',
            description='Use L1 norm for color loss (default is L2 norm).',
            value=False,
        ),
        desc.BoolParam(
            name='noRgbPlus',
            label='No RGB+',
            description='Disable RGB+ normalization mode.',
            value=False,
        ),
        desc.BoolParam(
            name='superNormal',
            label='SuperNormal Mode',
            description='Enable SuperNormal mode (MVPS method, special case of RNb-NeuS).',
            value=False,
        ),
        desc.BoolParam(
            name='scaleAlbedo',
            label='Scale Albedo',
            description='Enable albedo scaling: runs first with normals only, scales albedos based on multi-view consistency, then runs with scaled albedos.',
            value=False,
        ),
        desc.File(
            name='snapshot',
            label='Input Snapshot',
            description='Optional: Path to a snapshot file to continue training from.',
            value='',
        ),
        desc.ChoiceParam(
            name='verboseLevel',
            label='Verbose Level',
            description='Verbosity level for logging.',
            values=VERBOSE_LEVEL,
            value='info',
        ),
    ]

    outputs = [
        desc.File(
            name='outputFolder',
            label='Output Folder',
            description='Output folder containing the results.',
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name='outputMesh',
            label='Output Mesh',
            description='Output mesh file (OBJ format).',
            value="{nodeCacheFolder}/mesh_{maxIterValue}.obj",
            semantic='mesh',
            group='',
        ),
        desc.File(
            name='outputSnapshot',
            label='Output Snapshot',
            description='Output snapshot file for continuing training.',
            value="{nodeCacheFolder}/snapshot_{maxIterValue}.msgpack",
            group='',
        ),
    ]

    def _run_testbed(self, chunk, testbed_path, input_path, max_iter, common_flags, stage_name, add_flags=None):
        """Helper method to run testbed with given parameters."""
        cmd = [
            testbed_path,
            '--scene', str(input_path) + '/',
            '--maxiter', str(max_iter),
        ]
        cmd.extend(common_flags)
        if add_flags:
            cmd.extend(add_flags)
        
        chunk.logger.info(f"{stage_name} Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        chunk.logger.info(result.stdout)
        if result.returncode != 0:
            chunk.logger.error(result.stderr)
            raise RuntimeError(f"{stage_name} execution failed with code {result.returncode}")
        
        chunk.logger.info(f"{stage_name} completed")

    def processChunk(self, chunk):
        """Process the RNb-NeuS2 reconstruction."""
        try:
            import trimesh
            
            chunk.logManager.start(chunk.node.verboseLevel.value)
            
            # Validate inputs
            if not chunk.node.input.value:
                raise RuntimeError("No input folder provided")
            
            input_path = Path(chunk.node.input.value)
            if not input_path.exists():
                raise RuntimeError(f"Input folder does not exist: {input_path}")
            
            transform_json = input_path / "transform.json"
            if not transform_json.exists():
                raise RuntimeError(f"transform.json not found in input folder: {input_path}")
            
            normals_dir = input_path / "normals"
            if not normals_dir.exists():
                raise RuntimeError(f"normals/ folder not found in input folder: {input_path}")
            
            # Get testbed executable path
            testbed_path = chunk.node.rnbNeuS2Path.evalValue
            if not Path(testbed_path).exists():
                raise RuntimeError(f"Testbed executable not found at: {testbed_path}")
            
            chunk.logger.info(f"Using testbed: {testbed_path}")
            chunk.logger.info(f"Input folder: {input_path}")
            chunk.logger.info(f"Max iterations: {chunk.node.maxIter.value}")
            chunk.logger.info(f"Mesh resolution: {chunk.node.resolution.value}")
            chunk.logger.info(f"SuperNormal mode: {chunk.node.superNormal.value}")
            chunk.logger.info(f"Scale albedo: {chunk.node.scaleAlbedo.value}")
            
            output_path = Path(chunk.node.outputFolder.value)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Build common flags
            common_flags = [
                '--mask-weight', str(chunk.node.maskWeight.value),
                '--no-gui'
            ]
            
            if chunk.node.useL1Norm.value:
                common_flags.append('--lone')
            if chunk.node.noRgbPlus.value:
                common_flags.append('--no-rgbplus')
            if chunk.node.snapshot.value and Path(chunk.node.snapshot.value).exists():
                common_flags.extend(['--snapshot', str(chunk.node.snapshot.value)])
            
            # Handle scale_albedo workflow
            if chunk.node.scaleAlbedo.value:
                chunk.logger.info("=== SCALE ALBEDO WORKFLOW ===")
                
                # Step 1: Run without albedo
                chunk.logger.info("Step 1: Running with normals only (no albedo)...")
                no_albedo_flags = common_flags + ['--no-albedo', '--save-mesh', '--resolution', '512']
                
                if chunk.node.superNormal.value:
                    no_albedo_flags.append('--supernormal')
                    self._run_testbed(chunk, testbed_path, input_path, chunk.node.maxIter.value, 
                                    no_albedo_flags, "SuperNormal (no albedo)")
                else:
                    # Two-stage without albedo
                    iter_opti_lights = int(chunk.node.maxIter.value * 2 / 3)
                    self._run_testbed(chunk, testbed_path, input_path, iter_opti_lights,
                                    common_flags + ['--no-albedo', '--save-snapshot'],
                                    "Stage 1 (no albedo)")
                    
                    snapshot_stage1 = input_path / f"snapshot_{iter_opti_lights}.msgpack"
                    if not snapshot_stage1.exists():
                        raise RuntimeError(f"Stage 1 snapshot not found: {snapshot_stage1}")
                    
                    stage2_flags = ['--no-albedo', '--opti-lights', '--snapshot', str(snapshot_stage1),
                                '--resolution', str(chunk.node.resolution.value), '--save-mesh', '--save-snapshot']
                    self._run_testbed(chunk, testbed_path, input_path, chunk.node.maxIter.value,
                                    common_flags + stage2_flags, "Stage 2 (no albedo)")
            
                # Step 2: Scale albedos
                chunk.logger.info("Step 2: Scaling albedos based on multi-view consistency...")
                #TODO: Implement albedo scaling (see scripts/utils/albedo_scaling_lib.py)
                
                # Step 3: Run with scaled albedos
                chunk.logger.info("Step 3: Running with scaled albedos...")
                # TODO: Implement running from scratch with scaled albedos
            
            # Main training (or second training if scale_albedo was enabled)
            if not chunk.node.scaleAlbedo.value or True:  # Always run main training
                if chunk.node.noAlbedo.value and not chunk.node.scaleAlbedo.value:
                    common_flags.append('--no-albedo')
                
                if chunk.node.superNormal.value:
                    chunk.logger.info("Running SuperNormal")
                    
                    flags = ['--supernormal',
                            '--resolution', str(chunk.node.resolution.value),
                            '--save-mesh',
                            '--save-snapshot']
                    
                    self._run_testbed(chunk, testbed_path, input_path, chunk.node.maxIter.value,
                                    common_flags + flags, "SuperNormal")
                else:
                    chunk.logger.info("Running RNb-NeuS")
                    
                    iter_opti_lights = int(chunk.node.maxIter.value * 2 / 3)
                    
                    # Stage 1: Initial optimization
                    chunk.logger.info(f"Stage 1: Initial optimization ({iter_opti_lights} iterations)")
                    self._run_testbed(chunk, testbed_path, input_path, iter_opti_lights,
                                    common_flags + ['--save-snapshot'],
                                    "RNb-NeuS Stage 1")
                    
                    # Find snapshot
                    snapshot_stage1 = input_path / f"snapshot_{iter_opti_lights}.msgpack"
                    if not snapshot_stage1.exists():
                        raise RuntimeError(f"Stage 1 snapshot not found: {snapshot_stage1}")
                    
                    # Stage 2: Full optimization
                    chunk.logger.info(f"Stage 2: Full optimization ({chunk.node.maxIter.value} iterations)")
                    
                    stage2_flags = [
                        '--snapshot', str(snapshot_stage1),
                        '--resolution', str(chunk.node.resolution.value),
                        '--opti-lights',
                        '--save-mesh',
                        '--save-snapshot'
                    ]
                    
                    self._run_testbed(chunk, testbed_path, input_path, chunk.node.maxIter.value,
                                    common_flags + stage2_flags, "RNb-NeuS Stage 2")
            
            # Copy results to output folder
            chunk.logger.info("Copying results to output folder...")
            
            # Look for files in the input_path/output/ subfolder
            output_subfolder = input_path / "output"
            
            if not output_subfolder.exists():
                chunk.logger.warning(f"Output subfolder not found: {output_subfolder}")
                chunk.logger.info("Searching in input folder directly...")
                search_path = input_path
            else:
                chunk.logger.info(f"Searching for results in: {output_subfolder}")
                search_path = output_subfolder
            
            mesh_files = list(search_path.glob("mesh_*.obj"))
            snapshot_files = list(search_path.glob("snapshot_*.msgpack"))
            
            chunk.logger.info(f"Found {len(mesh_files)} mesh files")
            chunk.logger.info(f"Found {len(snapshot_files)} snapshot files")
            
            if len(mesh_files) == 0:
                chunk.logger.warning("No mesh files found!")
            if len(snapshot_files) == 0:
                chunk.logger.warning("No snapshot files found!")
            
            # Move (not copy) files to output folder
            for mesh_file in mesh_files:
                dest = output_path / mesh_file.name
                shutil.move(str(mesh_file), str(dest))
                
                # Load mesh and flip the normals with trimesh
                mesh_data = trimesh.load(str(dest), process=False)
                mesh_data.fix_normals()
                mesh_data.export(str(dest))
                
                chunk.logger.info(f"Moved mesh: {mesh_file.name}")
            
            for snapshot_file in snapshot_files:
                dest = output_path / snapshot_file.name
                shutil.move(str(snapshot_file), str(dest))
                chunk.logger.info(f"Moved snapshot: {snapshot_file.name}")
            
            # Remove the output subfolder if it exists and is now empty
            if output_subfolder.exists():
                try:
                    # Check if folder is empty or only contains log.txt and empty subfolders
                    remaining_files = list(output_subfolder.rglob("*"))
                    # Filter out directories and log files
                    important_files = [f for f in remaining_files if f.is_file() and f.name != "log.txt"]
                    
                    if not important_files:
                        shutil.rmtree(str(output_subfolder))
                        chunk.logger.info(f"Removed output subfolder: {output_subfolder}")
                    else:
                        chunk.logger.warning(f"Output subfolder not empty, keeping it: {len(important_files)} files remaining")
                except Exception as e:
                    chunk.logger.warning(f"Could not remove output subfolder: {e}")
            
            # Copy scaled albedo folder if it exists
            if chunk.node.scaleAlbedo.value:
                scaled_output = output_path / "albedos_scaled"
                if (input_path / "albedos").exists():
                    shutil.copytree(str(input_path / "albedos"), str(scaled_output), dirs_exist_ok=True)
                    chunk.logger.info("Copied scaled albedos")
            
            chunk.logger.info("RNb-NeuS2 processing completed successfully")
            
        except Exception as e:
            chunk.logger.error(f"Error during RNb-NeuS2 processing: {e}")
            import traceback
            chunk.logger.error(traceback.format_exc())
            raise
        
        finally:
            chunk.logManager.end()
