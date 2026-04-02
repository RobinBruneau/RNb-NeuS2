"""Scene normalization (unit sphere scaling) for RNb-NeuS2.

Provides scaling from 3D landmarks, camera centers, or silhouettes.
"""

import numpy as np


def compute_unit_sphere_scaling(points_3d, sphere_scale=1.0):
    """Fit points into a unit sphere.

    Uses 99th percentile for outlier rejection.

    Returns:
        scene_center: (3,)
        scale_factor: float
        scale_matrix: (4, 4) homogeneous transform
    """
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


def compute_scaling_from_silhouettes(cameras, masks, sphere_scale=1.0,
                                     fg_area_ratio=5):
    """Compute (scene_center, scale_factor) from silhouettes (MVSCPS).

    Center via mask center-of-mass triangulation.
    Radius via projected sphere area matching.

    Args:
        cameras: list of dicts with fx, fy, cx, cy,
                 R_cam2world (3x3), center (3,).
        masks: list of (H, W) float arrays in [0, 1].
        sphere_scale: target sphere radius.
        fg_area_ratio: ratio of sphere area to foreground area.

    Returns:
        scene_center: (3,)
        scale_factor: float
    """
    from scipy.ndimage import center_of_mass

    A = np.zeros((3, 3))
    b = np.zeros(3)

    cam_data = []
    for cam, mask in zip(cameras, masks):
        fx, fy = cam["fx"], cam["fy"]
        cx, cy = cam["cx"], cam["cy"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K_inv = np.linalg.inv(K)

        R_c2w = cam["R_cam2world"]
        center_cam = cam["center"]

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


def extract_cameras_for_scaling(data, mask_folder_path=""):
    """Extract camera dicts and masks from loaded data for silhouette scaling.

    Args:
        data: standardized dict from a dataloader
        mask_folder_path: fallback folder for masks

    Returns:
        cameras: list of scaling-compatible camera dicts
        masks: list of (H, W) float arrays
    """
    import cv2

    flip = np.array([1, -1, -1], dtype=np.float32)
    cameras = []
    masks = []

    for view in data["views"]:
        c2w = view["c2w"]
        K = view["K"]

        # Camera center and rotation (already Y/Z-flipped in c2w)
        R_c2w = c2w[:3, :3].astype(np.float64)
        center = c2w[:3, 3].astype(np.float64)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # Load mask
        mask_path = view["mask_path"]
        mask_img = None
        if mask_path and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask_img is None:
            continue

        if len(mask_img.shape) == 3:
            mask_img = mask_img[:, :, 0]
        threshold = 125 if mask_img.dtype == np.uint8 else 30000
        mask_float = (mask_img > threshold).astype(np.float32)

        cameras.append({
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "R_cam2world": R_c2w,
            "center": center,
        })
        masks.append(mask_float)

    return cameras, masks


# Need os for extract_cameras_for_scaling
import os
