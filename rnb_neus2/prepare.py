"""Prepare testbed data from any supported input format.

Converts loaded data (from any dataloader) into the transform.json +
normals/ + albedos/ structure expected by the RNb-NeuS2 testbed.
"""

import json
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

from .scaling import (
    compute_unit_sphere_scaling,
    compute_scaling_from_silhouettes,
    compute_scaling_from_silhouettes_v2,
    extract_cameras_for_scaling,
)


def _load_mask_image(mask_path, img_shape, bit_depth):
    """Load and threshold a mask image, or return a full mask."""
    max_val = 65535 if bit_depth == 16 else 255
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    h, w = img_shape

    if mask_path and os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is not None:
            if len(mask_img.shape) == 3:
                mask_img = mask_img[:, :, 0]
            if mask_img.dtype == np.float32:
                mask_binary = (mask_img > 0.5).astype(np.float64)
            else:
                threshold = 125 if mask_img.dtype == np.uint8 else 30000
                mask_binary = np.where(mask_img > threshold, 1.0, 0.0)
            return (mask_binary * max_val).astype(dtype)

    return np.ones((h, w), dtype=dtype) * max_val


def _compute_scaling(data, scaling_mode, sphere_scale, margin_px, logger):
    """Compute scene scaling from loaded data.

    Returns:
        scene_center, scale_factor, scale_matrix
    """
    scene_center = np.zeros(3, dtype=np.float32)
    scale_factor = 1.0
    scale_matrix = np.eye(4, dtype=np.float32)

    if scaling_mode == "none":
        return scene_center, scale_factor, scale_matrix

    scaled = False

    # Try silhouettes first (more reliable for PS/neural reconstruction)
    if not scaled and scaling_mode in ("auto", "silhouettes", "silhouettes_v2"):
        sil_cams, sil_masks = extract_cameras_for_scaling(data)
        if sil_cams and sil_masks:
            if scaling_mode in ("auto", "silhouettes_v2"):
                logger.info("Scaling from silhouettes_v2 (min enclosing sphere): {} views".format(
                    len(sil_cams)))
                scene_center, scale_factor = (
                    compute_scaling_from_silhouettes_v2(
                        sil_cams, sil_masks, sphere_scale=sphere_scale,
                        margin_px=margin_px))
            else:
                logger.info("Scaling from silhouettes: {} views".format(
                    len(sil_cams)))
                scene_center, scale_factor = (
                    compute_scaling_from_silhouettes(
                        sil_cams, sil_masks, sphere_scale=sphere_scale))
            scene_center = scene_center.astype(np.float32)
            scale_matrix = np.eye(4, dtype=np.float32)
            for i in range(3):
                scale_matrix[i, i] = scale_factor
                scale_matrix[i, 3] = -scene_center[i] * scale_factor
            scaled = True

    # Fall back to landmarks (pcd)
    if not scaled and scaling_mode in ("auto", "pcd"):
        landmarks = data.get("landmarks")
        if landmarks is not None and len(landmarks) > 0:
            logger.info("Scaling from landmarks: {} points".format(
                len(landmarks)))
            scene_center, scale_factor, scale_matrix = (
                compute_unit_sphere_scaling(landmarks, sphere_scale))
            scaled = True

    # Try camera centers
    if not scaled and scaling_mode in ("auto", "cameras"):
        centers = []
        for v in data["views"]:
            centers.append(v["c2w"][:3, 3].copy())
        if centers:
            points_3d = np.array(centers, dtype=np.float32)
            logger.info("Scaling from camera centers: {} cameras".format(
                len(points_3d)))
            scene_center, scale_factor, scale_matrix = (
                compute_unit_sphere_scaling(points_3d, sphere_scale))
            scaled = True

    if not scaled:
        raise RuntimeError(
            "No data for scaling. Use scaling_mode='none' to disable.")

    logger.info("Scene center: {}".format(scene_center.tolist()))
    logger.info("Scale factor: {:.6f}".format(scale_factor))

    return scene_center, scale_factor, scale_matrix


def prepare_testbed_data(data, output_folder, logger,
                         scaling_mode="auto", sphere_scale=1.0,
                         margin_px=20):
    """Prepare testbed data from a loaded data dict.

    Args:
        data: Standardized dict from any dataloader.
        output_folder: Where to write transform.json + image folders.
        logger: Logger with .info() method.
        scaling_mode: "auto", "pcd", "silhouettes", "silhouettes_v2", "cameras", "none".
        sphere_scale: Target sphere radius.
        margin_px: Pixel margin for silhouettes_v2 mode.

    Returns:
        dict with scene_center, scale_factor, scale_matrix, n2w, n_frames.
    """
    scene_center, scale_factor, scale_matrix = _compute_scaling(
        data, scaling_mode, sphere_scale, margin_px, logger)

    # Output directories
    albedos_dir = os.path.join(output_folder, "albedos")
    normals_dir = os.path.join(output_folder, "normals")
    os.makedirs(albedos_dir, exist_ok=True)
    os.makedirs(normals_dir, exist_ok=True)

    image_width = data["image_width"]
    image_height = data["image_height"]

    frames = []
    for idx, view in enumerate(data["views"]):
        c2w = view["c2w"].copy()

        # Apply scaling to camera center
        cam_center = c2w[:3, 3].copy()
        c2w[:3, 3] = scale_factor * (cam_center - scene_center)

        K = view["K"]

        # Load normal image
        normal_path = view["normal_path"]
        if not os.path.exists(normal_path):
            logger.warning("Normal not found: {}, skipping".format(
                normal_path))
            continue

        normal_img = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        if normal_img is None:
            logger.warning("Cannot read: {}".format(normal_path))
            continue

        # Handle EXR normals (float32 in [-1,1]) -> 16-bit PNG [0, 65535]
        if normal_img.dtype == np.float32:
            # Normals in [-1, 1] -> [0, 1] -> uint16
            normal_img = np.clip((normal_img + 1.0) / 2.0, 0, 1)
            normal_img = (normal_img * 65535).astype(np.uint16)

        # Strip alpha if present
        if len(normal_img.shape) == 3 and normal_img.shape[2] == 4:
            normal_img = normal_img[:, :, :3]

        bit_depth = 16 if normal_img.dtype == np.uint16 else 8
        max_val = 65535 if bit_depth == 16 else 255

        # Load albedo
        albedo_path = view.get("albedo_path")
        albedo_img = None
        if albedo_path and os.path.exists(albedo_path):
            albedo_img = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
            if albedo_img is not None:
                # Handle EXR albedos
                if albedo_img.dtype == np.float32:
                    albedo_img = np.clip(albedo_img, 0, 1)
                    albedo_img = (albedo_img * 65535).astype(np.uint16)
                if len(albedo_img.shape) == 3 and albedo_img.shape[2] == 4:
                    albedo_img = albedo_img[:, :, :3]

        if albedo_img is None:
            albedo_img = (np.ones_like(normal_img) * max_val).astype(
                normal_img.dtype)

        # Load mask
        mask_img = _load_mask_image(
            view.get("mask_path"), normal_img.shape[:2], bit_depth)

        # RGBA = RGB + mask alpha
        normal_rgba = np.concatenate(
            [normal_img, mask_img[:, :, np.newaxis]], axis=-1)
        albedo_rgba = np.concatenate(
            [albedo_img, mask_img[:, :, np.newaxis]], axis=-1)

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

    # Write transform.json
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
