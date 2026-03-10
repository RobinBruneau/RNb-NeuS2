"""
Albedo Scaling Library

This library provides functions to scale albedos based on multi-view consistency.
It computes the relative scaling factors between neighboring views by ray tracing
through a mesh and comparing albedo values at intersection points.

Supports multiple camera formats:
- sfmData (.sfm): Meshroom's SfMData format
- cameras.npz: NumPy archive with camera matrices
- transform.json: NeRF/Instant-NGP format
"""

import os
import json
import numpy as np
import trimesh
import cv2
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path


def load_image(path):
    """Load an image from disk and normalize to [0, 1]."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    # Read as 8 or 16 bits
    if image.dtype == np.uint8:
        bit_depth = 8
    elif image.dtype == np.uint16:
        bit_depth = 16
    else:
        raise ValueError(f"Unsupported bit depth: {image.dtype}")
    
    # Check if the image has a fourth channel
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image / np.float32(2**bit_depth - 1)


def save_image(image, path, bit_depth=8):
    """Save an image to disk with specified bit depth."""
    # Clamp and convert to 8 or 16 bits
    image = np.nan_to_num(image, nan=0.0)
    image = np.clip(image, 0.0, 1.0)
    image = (image * np.float32(2**bit_depth - 1))
    if bit_depth == 8:
        image = image.astype(np.uint8)
    elif bit_depth == 16:
        image = image.astype(np.uint16)

    # Convert to BGR or BGRA
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Write the image
    cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def load_K_Rt_from_P(P):
    """Decompose projection matrix into intrinsics and pose."""
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def load_cameras_from_sfmdata(sfmdata_path, albedo_images, logger=None):
    """
    Load camera parameters from Meshroom's sfmData format.
    
    Args:
        sfmdata_path: Path to .sfm file
        albedo_images: List of albedo image filenames (for matching views)
        logger: Optional logger
        
    Returns:
        K_array: Array of intrinsics matrices (n_views, 3, 3)
        R_c2w_array: Array of rotation matrices camera-to-world (n_views, 3, 3)
        centers_array: Array of camera centers (n_views, 3, 1)
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log(f"Loading cameras from sfmData: {sfmdata_path}")
    
    from pyalicevision import sfmData, sfmDataIO
    
    # Load sfmData
    sfm_data = sfmData.SfMData()
    if not sfmDataIO.load(sfm_data, str(sfmdata_path), sfmDataIO.ALL):
        raise RuntimeError(f"Failed to load sfmData file: {sfmdata_path}")
    
    views = sfm_data.getViews()
    poses = sfm_data.getPoses()
    intrinsics_dict = sfm_data.getIntrinsics()
    
    # Match albedo images to views by poseId
    # Albedo images are named like "46483756.png" where 46483756 is the poseId
    image_pose_ids = [int(Path(img).stem) for img in albedo_images]
    
    K_array = []
    R_c2w_array = []
    centers_array = []
    
    for pose_id in image_pose_ids:
        # Find the view with this poseId
        view = None
        for v in views:
            if v.getPoseId() == pose_id:
                view = v
                break
        
        if view is None:
            raise RuntimeError(f"No view found with poseId {pose_id}")
        
        # Get the pose
        if pose_id not in poses:
            raise RuntimeError(f"No pose found for poseId {pose_id}")
        pose = poses[pose_id]
        
        # Get intrinsics
        intrinsic_id = view.getIntrinsicId()
        if intrinsic_id not in intrinsics_dict:
            raise RuntimeError(f"No intrinsics found for intrinsicId {intrinsic_id}")
        intrinsic = intrinsics_dict[intrinsic_id]
        
        # Extract K matrix
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = intrinsic.getScale().x()  # fx
        K[1, 1] = intrinsic.getScale().y()  # fy
        K[0, 2] = intrinsic.getOffset().x()  # cx
        K[1, 2] = intrinsic.getOffset().y()  # cy
        
        # Extract pose (world to camera)
        pose_matrix = pose.getTransform().getHomogeneous()
        
        # Convert to numpy and get camera-to-world
        T_w2c = np.array([[pose_matrix.coeff(i, j) for j in range(4)] for i in range(4)], dtype=np.float32)
        T_c2w = np.linalg.inv(T_w2c)
        
        R_c2w = T_c2w[:3, :3]
        center = T_c2w[:3, [3]]
        
        K_array.append(K)
        R_c2w_array.append(R_c2w)
        centers_array.append(center)
    
    log(f"Loaded {len(K_array)} cameras from sfmData")
    
    return np.array(K_array), np.array(R_c2w_array), np.array(centers_array)


def load_cameras_from_npz(npz_path, n_views, logger=None):
    """
    Load camera parameters from cameras.npz format.
    
    Args:
        npz_path: Path to cameras.npz file
        n_views: Number of views (to match with albedo images)
        logger: Optional logger
        
    Returns:
        K_array: Array of intrinsics matrices (n_views, 3, 3)
        R_c2w_array: Array of rotation matrices camera-to-world (n_views, 3, 3)
        centers_array: Array of camera centers (n_views, 3, 1)
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log(f"Loading cameras from NPZ: {npz_path}")
    
    data_cam = np.load(npz_path)
    K_array = []
    R_c2w_array = []
    centers_array = []
    
    for k in range(n_views):
        K, RT_c2w = load_K_Rt_from_P(data_cam[f"world_mat_{k}"][:3, :])
        R_c2w_array.append(RT_c2w[:3, :3])
        centers_array.append(RT_c2w[:3, [3]])
        K_array.append(K[:3, :3])
    
    log(f"Loaded {len(K_array)} cameras from NPZ")
    
    return np.array(K_array), np.array(R_c2w_array), np.array(centers_array)


def load_cameras_from_transform_json(json_path, albedo_images, logger=None):
    """
    Load camera parameters from transform.json format (NeRF/Instant-NGP).
    
    Args:
        json_path: Path to transform.json file
        albedo_images: List of albedo image filenames (for ordering)
        logger: Optional logger
        
    Returns:
        K_array: Array of intrinsics matrices (n_views, 3, 3)
        R_c2w_array: Array of rotation matrices camera-to-world (n_views, 3, 3)
        centers_array: Array of camera centers (n_views, 3, 1)
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    log(f"Loading cameras from transform.json: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    
    # Match frames to albedo images
    # Albedo images are in order, frames should match by filename
    K_array = []
    R_c2w_array = []
    centers_array = []
    
    # Get common intrinsics (fl_x/fl_y/cx/cy format)
    global_fx = data.get('fl_x', None)
    global_fy = data.get('fl_y', global_fx)
    global_cx = data.get('cx', None)
    global_cy = data.get('cy', None)

    for albedo_img in albedo_images:
        # Find matching frame
        frame = None
        for f in frames:
            if Path(f['albedo_path']).stem == Path(albedo_img).stem:
                frame = f
                break

        if frame is None:
            raise RuntimeError(f"No frame found for albedo image: {albedo_img}")

        # Get intrinsics: try intrinsic_matrix first, then per-frame, then global
        K = np.eye(3, dtype=np.float32)
        if 'intrinsic_matrix' in frame:
            K_full = np.array(frame['intrinsic_matrix'], dtype=np.float32)
            K[:3, :3] = K_full[:3, :3]
        else:
            fx = frame.get('fl_x', global_fx or 500.0)
            fy = frame.get('fl_y', global_fy or fx)
            cx = frame.get('cx', global_cx or data.get('w', 512) / 2)
            cy = frame.get('cy', global_cy or data.get('h', 512) / 2)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy
        
        # Get transform matrix (camera-to-world)
        transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
        
        R_c2w = transform_matrix[:3, :3]
        center = transform_matrix[:3, [3]]
        
        K_array.append(K)
        R_c2w_array.append(R_c2w)
        centers_array.append(center)
    
    log(f"Loaded {len(K_array)} cameras from transform.json")
    
    return np.array(K_array), np.array(R_c2w_array), np.array(centers_array)


def load_cameras(camera_source, albedo_images, logger=None):
    """
    Load camera parameters from various formats.
    
    Args:
        camera_source: Path to camera file (.sfm, .npz, or transform.json) or dict with camera data
        albedo_images: List of albedo image filenames
        logger: Optional logger
        
    Returns:
        K_array: Array of intrinsics matrices (n_views, 3, 3)
        R_c2w_array: Array of rotation matrices camera-to-world (n_views, 3, 3)
        centers_array: Array of camera centers (n_views, 3, 1)
    """
    camera_path = Path(camera_source)
    n_views = len(albedo_images)
    
    if camera_path.suffix == '.sfm':
        return load_cameras_from_sfmdata(camera_path, albedo_images, logger)
    elif camera_path.suffix == '.npz':
        return load_cameras_from_npz(camera_path, n_views, logger)
    elif camera_path.suffix == '.json' or camera_path.name == 'transform.json':
        return load_cameras_from_transform_json(camera_path, albedo_images, logger)
    else:
        raise ValueError(f"Unsupported camera format: {camera_path.suffix}. Supported: .sfm, .npz, .json")


def compute_albedo_scale_ratios(
    albedo_path,
    camera_source,
    mesh_path,
    n_samples=2000,
    logger=None
):
    """
    Compute albedo scaling ratios based on multi-view consistency.
    
    Args:
        albedo_path: Path to folder containing albedo images
        camera_source: Path to camera file (.sfm, .npz, or transform.json)
        mesh_path: Path to the mesh file (OBJ)
        n_samples: Number of samples per image for ratio computation
        logger: Optional logger for progress messages
        
    Returns:
        median_ratio_prop_norm: Array of shape (n_views, 3) with scaling factors per view
    """
    
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # Load albedos and masks
    list_names = sorted([f for f in os.listdir(albedo_path) if f.endswith(".png")])
    n_views = len(list_names)
    
    log(f"Loading {n_views} albedo images...")
    albedos = []
    masks = []
    for name in list_names:
        albedo = load_image(os.path.join(albedo_path, name))
        mask = albedo[:, :, 3] if albedo.shape[2] == 4 else np.ones(albedo.shape[:2])
        albedo = albedo[:, :, :3]
        albedos.append(albedo)
        masks.append(mask)
    
    albedos = np.array(albedos)
    masks = np.array(masks)
    n_views, h, w, _ = albedos.shape
    
    # Load camera parameters (auto-detect format)
    K_array, R_c2w_array, centers_array = load_cameras(camera_source, list_names, logger)
    
    # Load mesh
    log(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load_mesh(mesh_path)
    
    # Initialize storage for ratios
    ratios = np.zeros((n_views, n_samples, 3, 2), dtype=np.float32)
    intersection_found = np.zeros((n_views, n_samples, 2), dtype=np.bool_)
    
    log("Computing ratios between neighboring views...")
    for cam_id in range(n_views):
        log(f"Processing camera {cam_id}/{n_views}...")
        
        # Get masked pixels
        mask = masks[cam_id].astype(bool)
        ind_mask = np.where(mask)
        pixels = np.stack([ind_mask[1], ind_mask[0]], axis=1)
        
        # Get albedo values
        albedo_values = albedos[cam_id, ind_mask[0], ind_mask[1], :]
        
        # Sample pixels
        current_K = K_array[cam_id]
        current_R_c2w = R_c2w_array[cam_id]
        current_center = centers_array[cam_id]
        
        n_samples_good = min(n_samples, pixels.shape[0])
        if n_samples_good < n_samples:
            log(f"Warning: only {n_samples_good} valid pixels in image {cam_id}")
        
        ind = np.random.choice(pixels.shape[0], n_samples_good, replace=False)
        pixels = pixels[ind]
        albedo_values = albedo_values[ind]
        
        # Create rays and intersect with mesh
        rays_origin = np.tile(current_center.T, (n_samples_good, 1))
        point_on_rays = (current_R_c2w @ (np.linalg.inv(current_K) @ 
                        np.concatenate((pixels, np.ones((n_samples_good, 1))), axis=1).T) + 
                        current_center).T
        rays_direction = point_on_rays - rays_origin
        rays_direction /= np.linalg.norm(rays_direction, axis=1)[:, None]
        
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=rays_origin,
            ray_directions=rays_direction,
            multiple_hits=False
        )
        
        pixels = pixels[index_ray]
        albedo_values = albedo_values[index_ray]
        
        # Check both neighbors
        right_cam_id = (cam_id + 1) % n_views
        left_cam_id = (cam_id - 1) % n_views
        
        for kk, neigh_cam_id in enumerate([right_cam_id, left_cam_id]):
            neighbor_K = K_array[neigh_cam_id]
            neighbor_R_c2w = R_c2w_array[neigh_cam_id]
            neighbor_center = centers_array[neigh_cam_id]
            
            # Check visibility from neighbor
            neighbor_rays_direction = neighbor_center.T - locations
            neighbor_rays_direction /= np.linalg.norm(neighbor_rays_direction, axis=1)[:, None]
            neighbor_rays_origin = locations + 1e-3 * neighbor_rays_direction
            hit = mesh.ray.intersects_any(
                ray_origins=neighbor_rays_origin,
                ray_directions=neighbor_rays_direction
            )
            
            # Keep only visible points
            intersection_points = locations[~hit]
            index_ray_kk = index_ray[~hit]
            pixels_no_hit = pixels[~hit]
            albedo_values_no_hit = albedo_values[~hit]
            
            # Project to neighbor camera
            neighbor_R_w2c = neighbor_R_c2w.T
            intersection_points_in_neighbor_cam = (neighbor_R_w2c @ 
                                                   (intersection_points.T - neighbor_center))
            intersection_points_in_neighbor_cam = (neighbor_K @ 
                                                   intersection_points_in_neighbor_cam).T
            intersection_points_in_neighbor_cam /= intersection_points_in_neighbor_cam[:, 2][:, None]
            intersection_points_in_neighbor_cam = intersection_points_in_neighbor_cam[:, :2]
            
            # Filter points inside image
            valid_indices = (
                (0 <= intersection_points_in_neighbor_cam[:, 1]) & 
                (intersection_points_in_neighbor_cam[:, 1] < h-1) &
                (0 <= intersection_points_in_neighbor_cam[:, 0]) & 
                (intersection_points_in_neighbor_cam[:, 0] < w-1)
            )
            
            intersection_points_in_neighbor_cam = intersection_points_in_neighbor_cam[valid_indices]
            index_ray_kk = index_ray_kk[valid_indices]
            albedo_values_valid = albedo_values_no_hit[valid_indices]
            
            # Interpolate albedo values in neighbor image
            albedo_in_neighbor = albedos[neigh_cam_id].astype(np.float32)
            rows_inds = np.arange(0, h, 1)
            cols_inds = np.arange(0, w, 1)
            interpR = RegularGridInterpolator((rows_inds, cols_inds), albedo_in_neighbor[:, :, 0])
            interpG = RegularGridInterpolator((rows_inds, cols_inds), albedo_in_neighbor[:, :, 1])
            interpB = RegularGridInterpolator((rows_inds, cols_inds), albedo_in_neighbor[:, :, 2])
            
            pts_yx = np.stack([intersection_points_in_neighbor_cam[:, 1], 
                              intersection_points_in_neighbor_cam[:, 0]], axis=1)
            albedo_val = np.stack([interpR(pts_yx), interpG(pts_yx), interpB(pts_yx)], axis=1)
            
            # Filter out zero values
            zero_indices = np.any(albedo_val == 0, axis=1)
            index_ray_kk = index_ray_kk[~zero_indices]
            albedo_val = albedo_val[~zero_indices]
            albedo_values_valid = albedo_values_valid[~zero_indices]
            
            # Compute ratios
            ratios[cam_id, index_ray_kk, :, kk] = albedo_values_valid / albedo_val
            intersection_found[cam_id, index_ray_kk, kk] = True
    
    # Compute median ratios
    log("Computing final scaling factors...")
    median_ratios = np.zeros((n_views, 3))
    right_ratios = ratios[:, :, :, 0]
    right_ind = intersection_found[:, :, 0]
    left_ratios = np.roll(ratios[:, :, :, 1], -1, axis=0)
    left_ind = np.roll(intersection_found[:, :, 1], -1, axis=0)
    
    for cam_id in range(n_views):
        right_ratio = right_ratios[cam_id, right_ind[cam_id]]
        left_ratio = 1 / left_ratios[cam_id, left_ind[cam_id]]
        all_ratio = np.concatenate((right_ratio, left_ratio), axis=0)
        median_ratios[cam_id] = np.median(all_ratio, axis=0)
    
    # Propagate ratios
    median_ratio_prop = np.ones((n_views, 3))
    for ii in range(n_views - 1):
        median_ratio_prop[ii + 1] = median_ratio_prop[ii] * median_ratios[ii]
    
    # Normalize
    mean_median_ratio_prop = np.mean(median_ratio_prop, axis=0)
    median_ratio_prop_norm = median_ratio_prop / mean_median_ratio_prop
    
    log(f"Scale ratios computed: {median_ratio_prop_norm}")
    
    return median_ratio_prop_norm


def scale_and_save_albedos(
    albedo_path,
    output_albedo_path,
    scale_ratios,
    bit_depth=None,
    logger=None
):
    """
    Apply scaling to albedo images and save them.
    
    Args:
        albedo_path: Input albedo folder
        output_albedo_path: Output albedo folder
        scale_ratios: Array of shape (n_views, 3) with scaling factors
        bit_depth: Output bit depth (8 or 16). If None, auto-detect from first image.
        logger: Optional logger
    """
    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    os.makedirs(output_albedo_path, exist_ok=True)
    
    list_names = sorted([f for f in os.listdir(albedo_path) if f.endswith(".png")])
    
    # Auto-detect bit depth from first image if not specified
    if bit_depth is None:
        first_image_path = os.path.join(albedo_path, list_names[0])
        first_image = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
        if first_image.dtype == np.uint8:
            bit_depth = 8
        elif first_image.dtype == np.uint16:
            bit_depth = 16
        else:
            log(f"Warning: Unknown image dtype {first_image.dtype}, defaulting to 16-bit")
            bit_depth = 16
        log(f"Auto-detected bit depth: {bit_depth}")
    
    log(f"Scaling and saving {len(list_names)} albedo images (bit depth: {bit_depth})...")
    for ii, name in enumerate(list_names):
        # Load albedo
        albedo = load_image(os.path.join(albedo_path, name))
        mask = albedo[:, :, 3] if albedo.shape[2] == 4 else np.ones(albedo.shape[:2])
        albedo_rgb = albedo[:, :, :3]
        
        # Scale
        albedo_rgb *= scale_ratios[ii]
        
        # Combine with mask
        albedo_to_save = np.concatenate((albedo_rgb, mask[:, :, np.newaxis]), axis=-1)
        
        # Save
        output_path = os.path.join(output_albedo_path, name)
        save_image(albedo_to_save, output_path, bit_depth=bit_depth)
        log(f"Saved scaled albedo {ii+1}/{len(list_names)}: {name}")
