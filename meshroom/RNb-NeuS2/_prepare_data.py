"""
Data preparation helpers for RNb-NeuS2.

Converts SfMData (normals, albedos, masks) into the transform.json format
expected by the RNb-NeuS2 testbed, with optional unit sphere scaling.
"""

import json
import os


def compute_unit_sphere_scaling(points_3d, sphere_scale=1.0):
    """Compute centroid and scale factor to fit points in a unit sphere.

    Uses 99th percentile for outlier rejection, then recomputes centroid
    on inliers.

    Returns:
        scene_center: (3,) array
        scale_factor: float
        scale_matrix: (4,4) homogeneous transformation matrix
    """
    import numpy as np

    centroid = np.mean(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - centroid, axis=1)
    threshold = np.percentile(distances, 99)

    inliers = points_3d[distances <= threshold]
    scene_center = np.mean(inliers, axis=0)
    max_dist = np.max(np.linalg.norm(inliers - scene_center, axis=1))

    scale_factor = sphere_scale / max_dist

    scale_matrix = np.eye(4, dtype=np.float32)
    for i in range(3):
        scale_matrix[i, i] = scale_factor
        scale_matrix[i, 3] = -scene_center[i] * scale_factor

    return scene_center, scale_factor, scale_matrix


def extract_3d_points_from_sfm(sfm_data, mode="pcd"):
    """Extract 3D points from a pyalicevision SfMData object.

    Args:
        sfm_data: pyalicevision SfMData object.
        mode: "pcd" for landmarks, "cameras" for camera centers.

    Returns:
        (N, 3) array in Y/Z-flipped coordinates, or None if no points.
    """
    import numpy as np

    flip = np.array([1, -1, -1], dtype=np.float32)

    if mode == "pcd":
        landmarks = sfm_data.getLandmarks()
        if len(landmarks) == 0:
            return None
        points = []
        for lm_id in landmarks.keys():
            coord = landmarks[lm_id].X
            points.append([coord[0], coord[1], coord[2]])
        return np.array(points, dtype=np.float32) * flip

    if mode == "cameras":
        views = sfm_data.getViews()
        centers = []
        for view_id in views.keys():
            view = views[view_id]
            if sfm_data.isPoseAndIntrinsicDefined(view_id):
                pose = sfm_data.getPose(view)
                c = pose.getTransform().center().squeeze()
                centers.append(
                    np.array([c[0], c[1], c[2]], dtype=np.float32) * flip)
        if not centers:
            return None
        return np.array(centers, dtype=np.float32)

    return None


def _load_mask(pose_id, mask_sfm, mask_folder_path, img_shape, bit_depth):
    """Load a mask from SfMData or folder, or create a full mask."""
    import cv2
    import numpy as np

    max_val = 65535 if bit_depth == 16 else 255
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    h, w = img_shape

    mask_img = None

    # Try mask SfMData
    if mask_sfm is not None:
        mask_views = mask_sfm.getViews()
        if pose_id in mask_views:
            mask_view = mask_views[pose_id]
            mask_path = mask_view.getImage().getImagePath()
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Try mask folder
    if mask_img is None and mask_folder_path and os.path.isdir(mask_folder_path):
        for ext in ('.png', '.jpg', '.jpeg', '.exr'):
            candidate = os.path.join(
                mask_folder_path, "{}{}".format(pose_id, ext))
            if os.path.exists(candidate):
                mask_img = cv2.imread(candidate, cv2.IMREAD_UNCHANGED)
                break

    if mask_img is not None:
        # Convert to single channel
        if len(mask_img.shape) == 3:
            mask_img = mask_img[:, :, 0]
        # Threshold to binary
        threshold = 125 if mask_img.dtype == np.uint8 else 30000
        mask_binary = np.where(mask_img > threshold, 1.0, 0.0)
        return (mask_binary * max_val).astype(dtype)

    # Full mask
    return np.ones((h, w), dtype=dtype) * max_val


def _extract_intrinsics(intrinsic, camera_module, numeric_module):
    """Extract intrinsic matrix from a pyalicevision intrinsic object.

    Returns:
        K: (4, 4) intrinsic matrix.
    """
    import numpy as np

    K = np.eye(4, dtype=np.float32)

    cam = camera_module.Pinhole.cast(intrinsic)
    if cam is not None:
        try:
            K[0, 0] = cam.getFocalLengthPixX()
            K[1, 1] = cam.getFocalLengthPixY()
        except AttributeError:
            scale = intrinsic.getScale()
            K[0, 0] = scale.x()
            K[1, 1] = scale.y()
        pp = cam.getPrincipalPoint()
        K[0, 2] = numeric_module.getX(pp)
        K[1, 2] = numeric_module.getY(pp)
    else:
        scale = intrinsic.getScale()
        offset = intrinsic.getOffset()
        K[0, 0] = scale.x()
        K[1, 1] = scale.y()
        K[0, 2] = offset.x()
        K[1, 2] = offset.y()

    return K


def prepare_testbed_data(normal_sfm_path, output_folder, logger,
                         albedo_sfm_path="", mask_sfm_path="",
                         mask_folder_path="", scaling_mode="auto",
                         sphere_scale=1.0):
    """Prepare all data for the RNb-NeuS2 testbed.

    Loads SfMData files, computes scaling, processes images,
    and writes transform.json + albedos/ + normals/ folders.

    Returns:
        dict with scene_center, scale_factor, scale_matrix, n2w, n_frames.
    """
    import cv2
    import numpy as np
    from pyalicevision import (
        sfmData as sfmDataModule, sfmDataIO, camera, numeric,
    )

    # Y/Z flip for NeRF coordinate convention
    flip_yz = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=np.float32)

    # --- Load SfMData files ---
    normal_sfm = sfmDataModule.SfMData()
    if not sfmDataIO.load(normal_sfm, normal_sfm_path, sfmDataIO.ALL):
        raise RuntimeError(
            "Failed to load normal SfMData: {}".format(normal_sfm_path))
    logger.info("Loaded normal SfMData: {}".format(normal_sfm_path))

    albedo_sfm = None
    if albedo_sfm_path and os.path.exists(albedo_sfm_path):
        albedo_sfm = sfmDataModule.SfMData()
        if not sfmDataIO.load(albedo_sfm, albedo_sfm_path, sfmDataIO.ALL):
            logger.warning("Failed to load albedo SfMData, using white albedos")
            albedo_sfm = None
        else:
            logger.info("Loaded albedo SfMData: {}".format(albedo_sfm_path))

    mask_sfm = None
    if mask_sfm_path and os.path.exists(mask_sfm_path):
        mask_sfm = sfmDataModule.SfMData()
        if not sfmDataIO.load(mask_sfm, mask_sfm_path, sfmDataIO.ALL):
            logger.warning("Failed to load mask SfMData, continuing without")
            mask_sfm = None
        else:
            logger.info("Loaded mask SfMData: {}".format(mask_sfm_path))

    views = normal_sfm.getViews()
    if len(views) == 0:
        raise RuntimeError("No views found in normal SfMData")

    # --- Compute unit sphere scaling ---
    scene_center = np.zeros(3, dtype=np.float32)
    scale_factor = 1.0
    scale_matrix = np.eye(4, dtype=np.float32)

    if scaling_mode != "none":
        modes_to_try = {
            "auto": ["pcd", "cameras"],
            "pcd": ["pcd"],
            "cameras": ["cameras"],
            "silhouettes": ["pcd", "cameras"],
        }.get(scaling_mode, ["pcd", "cameras"])

        if scaling_mode == "silhouettes":
            logger.warning(
                "Silhouette scaling not supported for RNb-NeuS2, "
                "falling back to auto")

        points_3d = None
        for mode in modes_to_try:
            points_3d = extract_3d_points_from_sfm(normal_sfm, mode)
            if points_3d is not None and len(points_3d) > 0:
                logger.info("Scaling mode '{}': {} points".format(
                    mode, len(points_3d)))
                break

        if points_3d is None or len(points_3d) == 0:
            raise RuntimeError(
                "No 3D points for scaling. Use scalingMode='none' to disable.")

        scene_center, scale_factor, scale_matrix = (
            compute_unit_sphere_scaling(points_3d, sphere_scale))
        logger.info("Scene center: {}".format(scene_center.tolist()))
        logger.info("Scale factor: {:.6f}".format(scale_factor))

    # --- Create output directories ---
    albedos_dir = os.path.join(output_folder, "albedos")
    normals_dir = os.path.join(output_folder, "normals")
    os.makedirs(albedos_dir, exist_ok=True)
    os.makedirs(normals_dir, exist_ok=True)

    # Representative views: viewId == poseId
    pose_ids = sorted([
        vid for vid in views.keys()
        if vid == views[vid].getPoseId()
    ])
    logger.info("Processing {} poses...".format(len(pose_ids)))

    # --- Process each pose ---
    frames = []
    image_width = None
    image_height = None

    for idx, pose_id in enumerate(pose_ids):
        view = views[pose_id]

        if not normal_sfm.isPoseAndIntrinsicDefined(pose_id):
            logger.warning(
                "Pose/intrinsic not defined for {}, skipping".format(pose_id))
            continue

        # Camera-to-world matrix
        pose = normal_sfm.getPose(view)
        transform = pose.getTransform()
        R = transform.rotation()
        center = transform.center().squeeze()

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.transpose()
        c2w[:3, 3] = [center[0], center[1], center[2]]

        # Apply Y/Z flip
        c2w = flip_yz @ c2w

        # Apply unit sphere scaling to translation
        cam_center = c2w[:3, 3].copy()
        c2w[:3, 3] = scale_factor * (cam_center - scene_center)

        # Intrinsics
        intrinsic_id = view.getIntrinsicId()
        intrinsic = normal_sfm.getIntrinsics()[intrinsic_id]
        K = _extract_intrinsics(intrinsic, camera, numeric)

        # Image dimensions
        if image_width is None:
            image_width = view.getImage().getWidth()
            image_height = view.getImage().getHeight()

        # Load normal image
        normal_path = view.getImage().getImagePath()
        if not os.path.exists(normal_path):
            logger.warning(
                "Normal image not found: {}, skipping".format(normal_path))
            continue

        normal_img = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        if normal_img is None:
            logger.warning(
                "Could not read normal image: {}".format(normal_path))
            continue

        # Strip alpha if present
        if len(normal_img.shape) == 3 and normal_img.shape[2] == 4:
            normal_img = normal_img[:, :, :3]

        bit_depth = 16 if normal_img.dtype == np.uint16 else 8
        max_val = 65535 if bit_depth == 16 else 255

        # Load albedo image
        albedo_img = None
        if albedo_sfm is not None:
            albedo_views = albedo_sfm.getViews()
            if pose_id in albedo_views:
                albedo_view = albedo_views[pose_id]
                albedo_path = albedo_view.getImage().getImagePath()
                if os.path.exists(albedo_path):
                    albedo_img = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
                    if (albedo_img is not None
                            and len(albedo_img.shape) == 3
                            and albedo_img.shape[2] == 4):
                        albedo_img = albedo_img[:, :, :3]

        if albedo_img is None:
            albedo_img = (np.ones_like(normal_img) * max_val).astype(
                normal_img.dtype)

        # Load mask
        mask_img = _load_mask(
            pose_id, mask_sfm, mask_folder_path,
            normal_img.shape[:2], bit_depth)

        # Create RGBA images (RGB + mask alpha)
        normal_rgba = np.concatenate(
            [normal_img, mask_img[:, :, np.newaxis]], axis=-1)
        albedo_rgba = np.concatenate(
            [albedo_img, mask_img[:, :, np.newaxis]], axis=-1)

        # Save
        filename = "{:05d}.png".format(idx)
        cv2.imwrite(os.path.join(normals_dir, filename), normal_rgba)
        cv2.imwrite(os.path.join(albedos_dir, filename), albedo_rgba)

        frames.append({
            "albedo_path": "albedos/{}".format(filename),
            "normal_path": "normals/{}".format(filename),
            "transform_matrix": c2w.tolist(),
            "intrinsic_matrix": K.tolist(),
        })

    if not frames:
        raise RuntimeError("No valid frames could be processed")

    logger.info("Processed {} frames".format(len(frames)))

    # --- Write transform.json ---
    n2w = np.linalg.inv(scale_matrix)
    transform_data = {
        "w": image_width,
        "h": image_height,
        "aabb_scale": 1.0,
        "scale": 0.5,
        "offset": [0.5, 0.5, 0.5],
        "from_na": True,
        "n2w": n2w.tolist(),
        "frames": frames,
    }

    transform_path = os.path.join(output_folder, "transform.json")
    with open(transform_path, "w") as f:
        json.dump(transform_data, f, indent=4)
    logger.info("Saved transform.json to {}".format(transform_path))

    return {
        "scene_center": scene_center,
        "scale_factor": scale_factor,
        "scale_matrix": scale_matrix,
        "n2w": n2w,
        "n_frames": len(frames),
    }
