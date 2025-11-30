__version__ = "1.0"

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
import os
from pathlib import Path

class PrepareRNbNeuS2Data(desc.Node):
    size = desc.DynamicNodeSize("inputNormalSfm")
    
    category = "RNb-NeuS2"
    documentation = """
Prepare Data for RNb-NeuS2

This node prepares input data for RNb-NeuS2 processing by:
1. Converting data format from SfMData to RNb-NeuS2 expected format (transform.json)
2. Optionally scaling the camera poses to fit within a unit sphere (required for NeRF-like techniques)
3. Creating albedos/ and normals/ folders with proper masks
4. Generating the transform.json file with proper scaling and camera information

The node expects:
- Separate SfMData files for normals and optionally albedos (from SDM-UniPS or similar)
- A mask folder with masks named by poseId (e.g., 46483756.png)
- Optionally: An SfMData file with 3D landmarks for automatic unit sphere scaling

Output will be in RNb_NeuS2 format ready for processing.
"""

    inputs = [
        desc.File(
            name="inputNormalSfm",
            label="Input Normal SfMData",
            description="Input SfMData file containing normal maps paths and camera information.",
            value="",
        ),
        desc.File(
            name="inputAlbedoSfm",
            label="Input Albedo SfMData",
            description="Input SfMData file containing albedo maps paths (optional, can use white albedo).",
            value="",
        ),
        desc.File(
            name="inputLandmarksSfm",
            label="Input Landmarks SfMData",
            description="Input SfMData file containing 3D landmarks for unit sphere scaling (optional, uses camera centers if not provided).",
            value="",
        ),
        desc.File(
            name="inputMesh",
            label="Input Mesh",
            description="Input mesh file (.obj, .ply, etc.) for unit sphere scaling. If provided, this will be used instead of landmarks SfMData.",
            value="",
        ),
        desc.File(
            name="inputMaskFolder",
            label="Mask Folder",
            description="Folder containing mask images named by poseId (e.g., 46483756.png).",
            value="",
        ),
        desc.File(
            name="inputMaskCertaintyFolder",
            label="Mask Certainty Folder",
            description="Folder containing mask certainty images (optional, will use mask if not provided).",
            value="",
        ),
        desc.BoolParam(
            name="scaleToUnitSphere",
            label="Scale to Unit Sphere",
            description="Automatically scale poses to fit the scene in a unit sphere using SfM points.",
            value=True,
        ),
        desc.FloatParam(
            name="sphereScale",
            label="Sphere Scale Factor",
            description="Scale factor for the unit sphere (1.0 = tight fit, >1.0 = larger sphere).",
            value=0.9,
            range=(0.5, 4.0, 0.1),
            enabled=lambda node: node.scaleToUnitSphere.value,
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name="outputFolder",
            label="Output Folder",
            description="Folder containing prepared RNb-NeuS2 data",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="transformJson",
            label="Transform JSON",
            description="Generated transform.json file",
            value="{nodeCacheFolder}/transform.json",
            group="",
        ),
    ]

    def visualize_scaled_scene(self, output_folder, points_3d_original, points_3d_transformed, 
                               camera_centers_original, camera_centers_transformed, 
                               scale_factor, scene_center, sphere_radius, logger):
        """
        Visualize the original and transformed 3D points and camera positions.
        
        Args:
            output_folder: Path to save the visualization
            points_3d_original: Original 3D points (Nx3)
            points_3d_transformed: Transformed 3D points (Nx3)
            camera_centers_original: Original camera centers (Mx3)
            camera_centers_transformed: Transformed camera centers (Mx3)
            scale_factor: The scaling factor applied
            scene_center: The center used for transformation
            sphere_radius: Target sphere radius
            logger: Logger for messages
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import numpy as np
            
            logger.info("Generating 3D visualization...")
            
            # Create figure with two subplots
            fig = plt.figure(figsize=(20, 10))
            
            # Original scene
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title('Original Scene', fontsize=14, fontweight='bold')
            
            # Plot original 3D points
            if len(points_3d_original) > 0:
                ax1.scatter(points_3d_original[:, 0], points_3d_original[:, 1], points_3d_original[:, 2], 
                           c='blue', marker='.', s=1, alpha=0.3, label='SfM Points')
            
            # Plot original camera centers
            if len(camera_centers_original) > 0:
                ax1.scatter(camera_centers_original[:, 0], camera_centers_original[:, 1], camera_centers_original[:, 2], 
                           c='red', marker='^', s=100, alpha=0.8, label='Camera Centers')
            
            # Plot scene center
            ax1.scatter([scene_center[0]], [scene_center[1]], [scene_center[2]], 
                       c='green', marker='o', s=200, label='Scene Center')
            
            # Set equal aspect ratio
            max_range = np.array([points_3d_original[:, 0].max() - points_3d_original[:, 0].min(),
                                 points_3d_original[:, 1].max() - points_3d_original[:, 1].min(),
                                 points_3d_original[:, 2].max() - points_3d_original[:, 2].min()]).max() / 2.0
            mid_x = (points_3d_original[:, 0].max() + points_3d_original[:, 0].min()) * 0.5
            mid_y = (points_3d_original[:, 1].max() + points_3d_original[:, 1].min()) * 0.5
            mid_z = (points_3d_original[:, 2].max() + points_3d_original[:, 2].min()) * 0.5
            ax1.set_xlim(mid_x - max_range, mid_x + max_range)
            ax1.set_ylim(mid_y - max_range, mid_y + max_range)
            ax1.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.legend()
            ax1.grid(True)
            
            # Transformed scene
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title(f'Transformed Scene (scale={scale_factor:.6f}, radius={sphere_radius})', 
                         fontsize=14, fontweight='bold')
            
            # Plot transformed 3D points
            if len(points_3d_transformed) > 0:
                ax2.scatter(points_3d_transformed[:, 0], points_3d_transformed[:, 1], points_3d_transformed[:, 2], 
                           c='blue', marker='.', s=1, alpha=0.3, label='SfM Points (scaled)')
            
            # Plot transformed camera centers
            if len(camera_centers_transformed) > 0:
                ax2.scatter(camera_centers_transformed[:, 0], camera_centers_transformed[:, 1], camera_centers_transformed[:, 2], 
                           c='red', marker='^', s=100, alpha=0.8, label='Camera Centers (scaled)')
            
            # Plot origin (new center after transformation)
            ax2.scatter([0], [0], [0], c='green', marker='o', s=200, label='Origin (0,0,0)')
            
            # Draw sphere of target radius
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan', label='Target Sphere')
            
            # Set equal aspect ratio centered at origin
            lim = sphere_radius * 1.2
            ax2.set_xlim(-lim, lim)
            ax2.set_ylim(-lim, lim)
            ax2.set_zlim(-lim, lim)
            
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.legend()
            ax2.grid(True)
            
            # Add text with statistics
            stats_text = f"Points: {len(points_3d_original)}\nCameras: {len(camera_centers_original)}\n"
            stats_text += f"Max dist (original): {np.max(np.linalg.norm(points_3d_original - scene_center, axis=1)):.4f}\n"
            stats_text += f"Max dist (transformed): {np.max(np.linalg.norm(points_3d_transformed, axis=1)):.4f}"
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Save visualization
            viz_path = output_folder / "scaling_visualization.png"
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to: {viz_path}")
            
        except ImportError as e:
            logger.warning(f"Cannot generate visualization, matplotlib not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to generate visualization: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def processChunk(self, chunk):
        """
        Process the data preparation for RNb-NeuS2.
        """
        try:
            import json
            import numpy as np
            import cv2
            from pyalicevision import sfmData, sfmDataIO, camera, numeric
            
            chunk.logManager.start(chunk.node.verboseLevel.value)
            
            if not chunk.node.inputNormalSfm.value:
                raise RuntimeError("No input Normal SfMData file provided")
            
            # Load normal SfMData (this contains the camera poses and intrinsics)
            normal_sfm_data = sfmData.SfMData()
            if not sfmDataIO.load(normal_sfm_data, chunk.node.inputNormalSfm.value, sfmDataIO.ALL):
                raise RuntimeError(f"Failed to load normal SfMData file: {chunk.node.inputNormalSfm.value}")
            
            chunk.logger.info(f"Loaded Normal SfMData from: {chunk.node.inputNormalSfm.value}")
            
            # Load albedo SfMData if provided
            albedo_sfm_data = None
            has_albedo_sfm = chunk.node.inputAlbedoSfm.value and os.path.exists(chunk.node.inputAlbedoSfm.value)
            if has_albedo_sfm:
                albedo_sfm_data = sfmData.SfMData()
                if not sfmDataIO.load(albedo_sfm_data, chunk.node.inputAlbedoSfm.value, sfmDataIO.ALL):
                    chunk.logger.warning(f"Failed to load albedo SfMData file: {chunk.node.inputAlbedoSfm.value}")
                    albedo_sfm_data = None
                else:
                    chunk.logger.info(f"Loaded Albedo SfMData from: {chunk.node.inputAlbedoSfm.value}")
            
            # Load landmarks SfMData if provided
            landmarks_sfm_data = None
            has_landmarks_sfm = chunk.node.inputLandmarksSfm.value and os.path.exists(chunk.node.inputLandmarksSfm.value)
            if has_landmarks_sfm:
                landmarks_sfm_data = sfmData.SfMData()
                if not sfmDataIO.load(landmarks_sfm_data, chunk.node.inputLandmarksSfm.value, sfmDataIO.ALL):
                    chunk.logger.warning(f"Failed to load landmarks SfMData file: {chunk.node.inputLandmarksSfm.value}")
                    landmarks_sfm_data = None
                else:
                    chunk.logger.info(f"Loaded Landmarks SfMData from: {chunk.node.inputLandmarksSfm.value}")
            
            # Get views from normal SfM
            views = normal_sfm_data.getViews()
            if len(views) == 0:
                raise RuntimeError("No views found in Normal SfMData")
            
            chunk.logger.info(f"Found {len(views)} views in Normal SfMData")
            
            # Check mask folders
            mask_folder = Path(chunk.node.inputMaskFolder.value) if chunk.node.inputMaskFolder.value else None
            mask_certainty_folder = Path(chunk.node.inputMaskCertaintyFolder.value) if chunk.node.inputMaskCertaintyFolder.value else None
            
            has_mask = mask_folder and mask_folder.exists()
            has_mask_certainty = mask_certainty_folder and mask_certainty_folder.exists()
            
            chunk.logger.info(f"Has albedo SfM: {has_albedo_sfm}")
            chunk.logger.info(f"Has mask folder: {has_mask}")
            chunk.logger.info(f"Has mask certainty folder: {has_mask_certainty}")
            
            # Prepare output folders
            output_folder = Path(chunk.node.outputFolder.value)
            output_folder.mkdir(parents=True, exist_ok=True)
            
            albedos_output = output_folder / "albedos"
            normals_output = output_folder / "normals"
            albedos_output.mkdir(parents=True, exist_ok=True)
            normals_output.mkdir(parents=True, exist_ok=True)
            
            # Compute scale matrix if scaling to unit sphere
            scale_matrix = np.eye(4, dtype=np.float32)
            scene_center = np.zeros(3, dtype=np.float32)
            scale_factor = 1.0
            
            if chunk.node.scaleToUnitSphere.value:
                chunk.logger.info("Computing transformation to fit SfM points in unit sphere...")
                
                # Check if mesh is provided
                has_mesh = chunk.node.inputMesh.value and os.path.exists(chunk.node.inputMesh.value)
                
                if has_mesh:
                    # Load mesh using trimesh
                    chunk.logger.info(f"Loading mesh from: {chunk.node.inputMesh.value}")
                    try:
                        import trimesh
                        mesh = trimesh.load(chunk.node.inputMesh.value)
                        
                        # Extract vertices from mesh (no Y/Z flip)
                        points_3d = np.array(mesh.vertices, dtype=np.float32)
                        chunk.logger.info(f"Loaded mesh with {len(points_3d)} vertices")
                        
                    except ImportError:
                        chunk.logger.error("trimesh library not available. Please install it with: pip install trimesh")
                        raise RuntimeError("trimesh library is required to load mesh files")
                    except Exception as e:
                        chunk.logger.error(f"Failed to load mesh: {e}")
                        raise RuntimeError(f"Failed to load mesh file: {chunk.node.inputMesh.value}")
                else:
                    # Get 3D points from landmarks SfM if available, otherwise from normal SfM
                    source_sfm = landmarks_sfm_data if landmarks_sfm_data else normal_sfm_data
                    landmarks = source_sfm.getLandmarks()
                    
                    if len(landmarks) == 0:
                        chunk.logger.warning("No 3D points found in SfMData.")
                        # Raise error or fallback to camera centers
                        raise RuntimeError("Cannot perform unit sphere scaling without 3D points.")
                    else:
                        # Extract 3D point coordinates
                        points_3d = []
                        for landmark_id, landmark in landmarks.items():
                            coord = landmark.X
                            points_3d.append([coord[0], -coord[1], -coord[2]])
                        points_3d = np.array(points_3d).squeeze()
                
                chunk.logger.info(f"Using {len(points_3d)} 3D points for computing the transformation")
                
                # Compute centroid (mean) of the 3D points
                scene_center = np.mean(points_3d, axis=0)
                
                # Compute distances from centroid to each point
                distances = np.linalg.norm(points_3d - scene_center, axis=1)

                # Define the max_dist as the 99th percentile to avoid outliers
                max_dist = np.percentile(distances, 99)
                
                # Estimate the scene center without the outliers
                inlier_points = points_3d[distances <= max_dist]
                scene_center = np.mean(inlier_points, axis=0)
                max_dist = np.max(np.linalg.norm(inlier_points - scene_center, axis=1))
                
                # Scale factor to fit in sphere of radius sphereScale
                # After transformation: new_point = scale * (point - centroid)
                # We want max(||new_point||) = sphereScale
                scale_factor = chunk.node.sphereScale.value / max_dist
                
                chunk.logger.info(f"3D Points statistics:")
                chunk.logger.info(f"  Centroid (mean): {scene_center}")
                chunk.logger.info(f"  Max distance from centroid: {max_dist:.4f}")
                chunk.logger.info(f"  Target sphere radius: {chunk.node.sphereScale.value}")
                chunk.logger.info(f"  Scale factor: {scale_factor:.6f}")
                
                # Create transformation matrix
                # Transformation: new = scale * (old - centroid)
                # Which is: new = scale * old - scale * centroid
                # In homogeneous coordinates:
                # [scale   0      0      -scale*centroid_x]   [x]
                # [0       scale  0      -scale*centroid_y] * [y]
                # [0       0      scale  -scale*centroid_z]   [z]
                # [0       0      0      1                 ]   [1]
                scale_matrix[0, 0] = scale_factor
                scale_matrix[1, 1] = scale_factor
                scale_matrix[2, 2] = scale_factor
                scale_matrix[0, 3] = -scene_center[0] * scale_factor
                scale_matrix[1, 3] = -scene_center[1] * scale_factor
                scale_matrix[2, 3] = -scene_center[2] * scale_factor
                
                chunk.logger.info(f"Transformation matrix computed successfully")
                chunk.logger.info(f"Matrix n2w (world-to-normalized with Y/Z flip):")
                chunk.logger.info(f"{scale_matrix}")
                
                # Verify the transformation on the points
                transformed_points = scale_factor * (inlier_points - scene_center)
                max_dist_transformed = np.max(np.linalg.norm(transformed_points, axis=1))
                chunk.logger.info(f"  Verification: max distance after transform: {max_dist_transformed:.4f}")
            else:
                chunk.logger.info("Skipping unit sphere scaling (disabled)")
            
            # Process images and create transform.json
            frames = []
            image_width = None
            image_height = None
            
            # Get sorted list of pose IDs from normal SfM (only representative views: viewId == poseId)
            pose_ids = sorted([view_id for view_id, view in views.items() if view_id == view.getPoseId()])
            
            chunk.logger.info(f"Processing {len(pose_ids)} poses...")
            
            for idx, pose_id in enumerate(pose_ids):
                view = views[pose_id]
                intrinsic_id = view.getIntrinsicId()
                
                # Get pose
                if not normal_sfm_data.isPoseAndIntrinsicDefined(pose_id):
                    chunk.logger.warning(f"Pose or intrinsic not defined for pose {pose_id}, skipping")
                    continue
                
                pose = normal_sfm_data.getPose(view)
                transform = pose.getTransform()
                
                # Get rotation and center from AliceVision (already in OpenCV)
                R = transform.rotation()  # 3x3 rotation matrix (w2c)
                center = transform.center().squeeze()  # camera center in world coordinates
                
                # Build camera-to-world matrix
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = R.transpose()  # Invert rotation to get c2w
                c2w[:3, 3] = center
                
                # Apply coordinate system flip
                flip_yz = np.array([[1,  0,  0, 0],
                                    [0, -1,  0, 0],
                                    [0,  0, -1, 0],
                                    [0,  0,  0, 1]], dtype=np.float32)
                c2w = flip_yz @ c2w
                
                # Transform the camera center
                center = c2w[:3, 3]
                new_center = scale_factor * (center - scene_center)
                c2w[:3, 3] = new_center
                                
                # Get intrinsics using proper AliceVision functions
                intrinsic = normal_sfm_data.getIntrinsics()[intrinsic_id]
                
                # Try to cast to Pinhole camera
                cam = camera.Pinhole.cast(intrinsic)
                
                K = np.eye(4, dtype=np.float32)
                
                if cam is not None:
                    # Get focal length
                    # focal = cam.getFocalLength()
                    K[0, 0] = cam.getFocalLengthPixX()
                    K[1, 1] = cam.getFocalLengthPixY()
                    
                    # Get principal point using numeric functions
                    pp = cam.getPrincipalPoint()
                    pp_x = numeric.getX(pp)
                    pp_y = numeric.getY(pp)
                    K[0, 2] = pp_x  # cx
                    K[1, 2] = pp_y  # cy
                    
                    chunk.logger.debug(f"Pose {pose_id}: focal=({K[0, 0]}, {K[1, 1]}) pp=({pp_x}, {pp_y})")
                else:
                    # Fallback for non-pinhole cameras
                    chunk.logger.warning(f"Pose {pose_id}: Not a pinhole camera, using fallback intrinsics extraction")
                    intrinsic_params = intrinsic.getParams()
                    if len(intrinsic_params) >= 1:
                        K[0, 0] = intrinsic_params[0]  # focal length or fx
                        K[1, 1] = intrinsic_params[0] if len(intrinsic_params) < 2 else intrinsic_params[1]  # fy
                    
                    # Get principal point
                    principal_point = intrinsic.getPrincipalPoint()
                    K[0, 2] = numeric.getX(principal_point)  # cx
                    K[1, 2] = numeric.getY(principal_point)  # cy
                
                # Get image dimensions
                if image_width is None:
                    image_width = view.getImage().getWidth()
                    image_height = view.getImage().getHeight()
                
                # Get normal image path from SfM
                normal_path = Path(view.getImage().getImagePath())
                
                if not normal_path.exists():
                    chunk.logger.warning(f"Normal image not found for pose {pose_id}: {normal_path}, skipping")
                    continue
                
                # Get albedo image path from albedo SfM if available
                albedo_path = None
                if albedo_sfm_data:
                    albedo_views = albedo_sfm_data.getViews()
                    if pose_id in albedo_views:
                        albedo_view = albedo_views[pose_id]
                        albedo_path = Path(albedo_view.getImage().getImagePath())
                        if not albedo_path.exists():
                            chunk.logger.warning(f"Albedo image not found for pose {pose_id}: {albedo_path}")
                            albedo_path = None
                
                # Find mask image (named by poseId)
                mask_path = None
                if has_mask:
                    for ext in ['.png', '.jpg', '.jpeg']:
                        candidate = mask_folder / f"{pose_id}{ext}"
                        if candidate.exists():
                            mask_path = candidate
                            break
                
                # Find mask certainty image
                mask_certainty_path = None
                if has_mask_certainty:
                    for ext in ['.png', '.jpg', '.jpeg']:
                        candidate = mask_certainty_folder / f"{pose_id}{ext}"
                        if candidate.exists():
                            mask_certainty_path = candidate
                            break
                
                # If no mask certainty, use mask
                if not mask_certainty_path and mask_path:
                    mask_certainty_path = mask_path
                
                chunk.logger.debug(f"Pose {pose_id}:")
                chunk.logger.debug(f"  Normal: {normal_path}")
                chunk.logger.debug(f"  Albedo: {albedo_path}")
                chunk.logger.debug(f"  Mask: {mask_path}")
                chunk.logger.debug(f"  Mask certainty: {mask_certainty_path}")
                
                # Load and process images
                # Load normal image
                normal_img = cv2.imread(str(normal_path), cv2.IMREAD_UNCHANGED)
                if normal_img is None:
                    chunk.logger.warning(f"Could not read normal image: {normal_path}")
                    continue
                
                # Ensure RGB (no alpha)
                if len(normal_img.shape) == 3 and normal_img.shape[2] == 4:
                    normal_img = normal_img[:, :, :3]
                
                # Determine bit depth
                if normal_img.dtype == np.uint8:
                    n_bits = 8
                    max_val = 255
                elif normal_img.dtype == np.uint16:
                    n_bits = 16
                    max_val = 65535
                else:
                    chunk.logger.warning(f"Unsupported image dtype: {normal_img.dtype}, converting to uint8")
                    normal_img = (normal_img * 255).astype(np.uint8)
                    n_bits = 8
                    max_val = 255
                
                # Load or create albedo image
                if albedo_path:
                    albedo_img = cv2.imread(str(albedo_path), cv2.IMREAD_UNCHANGED)
                    if albedo_img is not None:
                        if len(albedo_img.shape) == 3 and albedo_img.shape[2] == 4:
                            albedo_img = albedo_img[:, :, :3]
                    else:
                        chunk.logger.warning(f"Could not read albedo image: {albedo_path}, using white")
                        albedo_img = None
                else:
                    albedo_img = None
                
                # Create white albedo if not available
                if albedo_img is None:
                    if n_bits == 8:
                        albedo_img = np.ones_like(normal_img, dtype=np.uint8) * 255
                    else:
                        albedo_img = np.ones_like(normal_img, dtype=np.uint16) * 65535
                
                # Load or create mask
                if mask_path:
                    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    if mask_img is not None:
                        if len(mask_img.shape) == 3:
                            mask_img = mask_img[:, :, 0]
                        
                        # Threshold mask
                        if mask_img.dtype == np.uint8:
                            mask_img = np.where(mask_img > 125, 1.0, 0.0)
                        else:
                            mask_img = np.where(mask_img > 30000, 1.0, 0.0)
                        
                        # Convert to appropriate bit depth
                        if n_bits == 8:
                            mask_img = (mask_img * 255).astype(np.uint8)
                        else:
                            mask_img = (mask_img * 65535).astype(np.uint16)
                    else:
                        chunk.logger.warning(f"Could not read mask image: {mask_path}, using full mask")
                        mask_img = None
                else:
                    mask_img = None
                
                # Create full mask if not available
                if mask_img is None:
                    if n_bits == 8:
                        mask_img = np.ones((normal_img.shape[0], normal_img.shape[1]), dtype=np.uint8) * 255
                    else:
                        mask_img = np.ones((normal_img.shape[0], normal_img.shape[1]), dtype=np.uint16) * 65535
                
                # Load or create mask certainty
                if mask_certainty_path:
                    mask_certainty_img = cv2.imread(str(mask_certainty_path), cv2.IMREAD_UNCHANGED)
                    if mask_certainty_img is not None:
                        if len(mask_certainty_img.shape) == 3:
                            mask_certainty_img = mask_certainty_img[:, :, 0]
                        
                        # Threshold mask certainty
                        if mask_certainty_img.dtype == np.uint8:
                            mask_certainty_img = np.where(mask_certainty_img > 125, 1.0, 0.0)
                        else:
                            mask_certainty_img = np.where(mask_certainty_img > 30000, 1.0, 0.0)
                        
                        # Convert to appropriate bit depth
                        if n_bits == 8:
                            mask_certainty_img = (mask_certainty_img * 255).astype(np.uint8)
                        else:
                            mask_certainty_img = (mask_certainty_img * 65535).astype(np.uint16)
                    else:
                        mask_certainty_img = mask_img.copy()
                else:
                    mask_certainty_img = mask_img.copy()
                
                # Combine albedo with mask certainty (4 channels)
                albedo_with_mask = np.concatenate([albedo_img, mask_certainty_img[:, :, np.newaxis]], axis=-1)
                
                # Combine normal with mask (4 channels)
                normal_with_mask = np.concatenate([normal_img, mask_img[:, :, np.newaxis]], axis=-1)
                
                # Save processed images
                output_filename = f"{idx:05d}.png"
                albedo_output_path = albedos_output / output_filename
                normal_output_path = normals_output / output_filename
                
                cv2.imwrite(str(albedo_output_path), albedo_with_mask)
                cv2.imwrite(str(normal_output_path), normal_with_mask)
                
                chunk.logger.debug(f"Saved albedo: {albedo_output_path}")
                chunk.logger.debug(f"Saved normal: {normal_output_path}")
                
                # Create frame entry
                frame = {
                    "albedo_path": f"albedos/{output_filename}",
                    "normal_path": f"normals/{output_filename}",
                    "transform_matrix": c2w.tolist(),
                    "intrinsic_matrix": K.tolist()
                }
                
                frames.append(frame)
            
            if len(frames) == 0:
                raise RuntimeError("No valid frames could be processed")
            
            chunk.logger.info(f"Successfully processed {len(frames)} frames")
            
            # Create transform.json
            transform_data = {
                "w": image_width,
                "h": image_height,
                "aabb_scale": 1.0,
                "scale": 0.5,
                "offset": [0.5, 0.5, 0.5],  # neus: [-1,1] ngp[0,1]
                "from_na": True,
                "n2w": np.linalg.inv(scale_matrix).tolist(),
                "frames": frames
            }
            
            # Save transform.json
            transform_json_path = output_folder / "transform.json"
            with open(transform_json_path, 'w') as f:
                json.dump(transform_data, f, indent=4)
            
            chunk.logger.info(f"Transform JSON saved to: {transform_json_path}")
            chunk.logger.info("Data preparation completed successfully")
            
        except Exception as e:
            chunk.logger.error(f"Error preparing RNb-NeuS2 data: {e}")
            import traceback
            chunk.logger.error(traceback.format_exc())
            raise
        
        finally:
            chunk.logManager.end()
