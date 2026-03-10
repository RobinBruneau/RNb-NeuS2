"""
Data preparation helpers for RNb-NeuS2.

Converts SfMData (normals, albedos, masks) into the transform.json format
expected by the RNb-NeuS2 testbed, with optional unit sphere scaling.
"""

import json
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


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


def compute_scaling_from_silhouettes(cameras, masks, sphere_scale=1.0,
                                     fg_area_ratio=5):
    """Compute (scene_center, scale_factor) from silhouettes (MVSCPS method).

    Center is estimated via mask center-of-mass triangulation (least-squares).
    Radius is estimated via projected sphere area matching.

    Args:
        cameras: list of dicts with keys fx, fy, cx, cy,
                 R_cam2world (3x3), center (3,).
        masks:   list of (H, W) float arrays (values in [0, 1]).
        sphere_scale:  target sphere radius.
        fg_area_ratio: ratio of sphere area to foreground area.

    Returns:
        scene_center: (3,) array
        scale_factor: float
    """
    import numpy as np
    from scipy.ndimage import center_of_mass

    A = np.zeros((3, 3))
    b = np.zeros(3)

    cam_data = []
    for cam, mask in zip(cameras, masks):
        fx, fy = cam['fx'], cam['fy']
        cx, cy = cam['cx'], cam['cy']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K_inv = np.linalg.inv(K)

        R_c2w = cam['R_cam2world']
        center_cam = cam['center']

        com = center_of_mass(mask.astype(np.float64))
        com_pixel = np.array([com[1], com[0], 1.0])

        dir_cam = K_inv @ com_pixel
        dir_cam = dir_cam / np.linalg.norm(dir_cam)

        m = R_c2w @ dir_cam
        o = center_cam

        I_mmT = np.eye(3) - np.outer(m, m)
        A += I_mmT
        b += I_mmT @ o

        cam_data.append((fx, fy, R_c2w, center_cam, mask))

    scene_center = np.linalg.lstsq(A, b, rcond=None)[0]

    total_fg_area = 0
    sum_fz2 = 0
    for fx, fy, R_c2w, center_cam, mask in cam_data:
        total_fg_area += mask.sum()
        R_w2c = R_c2w.T
        center_in_cam = R_w2c @ (scene_center - center_cam)
        Z = center_in_cam[2]
        if abs(Z) < 1e-8:
            Z = 1e-8
        sum_fz2 += (fx / Z) ** 2

    radius = np.sqrt(fg_area_ratio * total_fg_area / (np.pi * sum_fz2))
    if radius < 1e-8:
        radius = 1.0
    scale_factor = float(sphere_scale / radius)

    return scene_center, scale_factor


def _extract_cameras_for_scaling(sfm_data, camera_module, numeric_module,
                                 mask_sfm, mask_folder_path):
    """Extract camera dicts and masks for silhouette-based scaling.

    Returns:
        cameras: list of dicts with fx, fy, cx, cy, R_cam2world, center
        masks: list of (H, W) float arrays
    """
    import cv2
    import numpy as np

    flip = np.array([1, -1, -1], dtype=np.float32)

    views = sfm_data.getViews()
    pose_ids = sorted([
        vid for vid in views.keys()
        if vid == views[vid].getPoseId()
    ])

    cameras = []
    masks = []

    for pose_id in pose_ids:
        view = views[pose_id]
        if not sfm_data.isPoseAndIntrinsicDefined(pose_id):
            continue

        # Camera pose
        pose = sfm_data.getPose(view)
        transform = pose.getTransform()
        R = transform.rotation()
        center = transform.center().squeeze()
        center_world = np.array(
            [center[0], center[1], center[2]], dtype=np.float64) * flip

        # R_cam2world with Y/Z flip
        R_c2w = R.transpose()
        flip_mat = np.diag(flip).astype(np.float64)
        R_c2w = flip_mat @ R_c2w

        # Intrinsics
        intrinsic_id = view.getIntrinsicId()
        intrinsic = sfm_data.getIntrinsics()[intrinsic_id]
        K = _extract_intrinsics(intrinsic, camera_module, numeric_module)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # Load mask
        mask_img = None
        if mask_sfm is not None:
            mask_views = mask_sfm.getViews()
            if pose_id in mask_views:
                mask_view = mask_views[pose_id]
                mask_path = mask_view.getImage().getImagePath()
                if os.path.exists(mask_path):
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask_img is None and mask_folder_path and os.path.isdir(
                mask_folder_path):
            for ext in ('.png', '.jpg', '.jpeg', '.exr'):
                candidate = os.path.join(
                    mask_folder_path, "{}{}".format(pose_id, ext))
                if os.path.exists(candidate):
                    mask_img = cv2.imread(candidate, cv2.IMREAD_UNCHANGED)
                    break

        if mask_img is None:
            continue

        # Convert to single-channel float [0, 1]
        if len(mask_img.shape) == 3:
            mask_img = mask_img[:, :, 0]
        threshold = 125 if mask_img.dtype == np.uint8 else 30000
        mask_float = (mask_img > threshold).astype(np.float32)

        cameras.append({
            'fx': float(fx), 'fy': float(fy),
            'cx': float(cx), 'cy': float(cy),
            'R_cam2world': R_c2w.astype(np.float64),
            'center': center_world.astype(np.float64),
        })
        masks.append(mask_float)

    return cameras, masks


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
        scaled = False

        # --- Try silhouettes first (more reliable for PS/neural reconstruction) ---
        if not scaled and scaling_mode in ("auto", "silhouettes"):
            sil_cams, sil_masks = _extract_cameras_for_scaling(
                normal_sfm, camera, numeric, mask_sfm, mask_folder_path)
            if sil_cams and sil_masks:
                logger.info("Scaling from silhouettes: {} views".format(
                    len(sil_cams)))
                scene_center, scale_factor = (
                    compute_scaling_from_silhouettes(
                        sil_cams, sil_masks,
                        sphere_scale=sphere_scale))
                scene_center = scene_center.astype(np.float32)
                scale_matrix = np.eye(4, dtype=np.float32)
                for i in range(3):
                    scale_matrix[i, i] = scale_factor
                    scale_matrix[i, 3] = -scene_center[i] * scale_factor
                scaled = True

        # --- Fall back to pcd (landmarks) ---
        if not scaled and scaling_mode in ("auto", "pcd"):
            points_3d = extract_3d_points_from_sfm(normal_sfm, "pcd")
            if points_3d is not None and len(points_3d) > 0:
                logger.info("Scaling from landmarks: {} points".format(
                    len(points_3d)))
                scene_center, scale_factor, scale_matrix = (
                    compute_unit_sphere_scaling(points_3d, sphere_scale))
                scaled = True

        # --- Try camera centers ---
        if not scaled and scaling_mode in ("auto", "cameras"):
            points_3d = extract_3d_points_from_sfm(normal_sfm, "cameras")
            if points_3d is not None and len(points_3d) > 0:
                logger.info("Scaling from camera centers: {} cameras".format(
                    len(points_3d)))
                scene_center, scale_factor, scale_matrix = (
                    compute_unit_sphere_scaling(points_3d, sphere_scale))
                scaled = True

        if not scaled:
            raise RuntimeError(
                "No data for scaling. Use scalingMode='none' to disable.")

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
