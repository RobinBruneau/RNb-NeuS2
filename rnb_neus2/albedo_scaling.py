"""Albedo scaling via multi-view consistency.

Computes per-view scale ratios by ray-tracing through a mesh and
comparing albedo values at shared surface points between neighbors.

Supports camera loading from:
- transform.json (with automatic n2w world-space conversion)
- cameras.npz
- sfmData (.sfm via pyalicevision)
"""

import json
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import trimesh
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from .image_io import load_image, save_image


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
    """Load cameras from Meshroom sfmData format."""
    def log(msg):
        if logger:
            logger.info(msg)

    log("Loading cameras from sfmData: {}".format(sfmdata_path))

    from pyalicevision import sfmData, sfmDataIO

    sfm_data = sfmData.SfMData()
    if not sfmDataIO.load(sfm_data, str(sfmdata_path), sfmDataIO.ALL):
        raise RuntimeError(
            "Failed to load sfmData: {}".format(sfmdata_path))

    views = sfm_data.getViews()
    poses = sfm_data.getPoses()
    intrinsics_dict = sfm_data.getIntrinsics()

    image_pose_ids = [int(Path(img).stem) for img in albedo_images]

    K_array, R_c2w_array, centers_array = [], [], []

    for pose_id in image_pose_ids:
        view = None
        for v in views:
            if v.getPoseId() == pose_id:
                view = v
                break
        if view is None:
            raise RuntimeError("No view for poseId {}".format(pose_id))

        if pose_id not in poses:
            raise RuntimeError("No pose for poseId {}".format(pose_id))
        pose = poses[pose_id]

        intrinsic_id = view.getIntrinsicId()
        intrinsic = intrinsics_dict[intrinsic_id]

        K = np.eye(3, dtype=np.float32)
        try:
            K[0, 0] = intrinsic.getScale().x()
            K[1, 1] = intrinsic.getScale().y()
            K[0, 2] = intrinsic.getOffset().x()
            K[1, 2] = intrinsic.getOffset().y()
        except Exception:
            K[0, 0] = K[1, 1] = 500.0
            K[0, 2] = K[1, 2] = 256.0

        pose_matrix = pose.getTransform().getHomogeneous()
        T_w2c = np.array([
            [pose_matrix.coeff(i, j) for j in range(4)] for i in range(4)
        ], dtype=np.float32)
        T_c2w = np.linalg.inv(T_w2c)

        K_array.append(K)
        R_c2w_array.append(T_c2w[:3, :3])
        centers_array.append(T_c2w[:3, [3]])

    log("Loaded {} cameras from sfmData".format(len(K_array)))
    return np.array(K_array), np.array(R_c2w_array), np.array(centers_array)


def load_cameras_from_npz(npz_path, n_views, logger=None):
    """Load cameras from cameras.npz format."""
    def log(msg):
        if logger:
            logger.info(msg)

    log("Loading cameras from NPZ: {}".format(npz_path))

    data_cam = np.load(npz_path)
    K_array, R_c2w_array, centers_array = [], [], []

    for k in range(n_views):
        K, RT_c2w = load_K_Rt_from_P(data_cam["world_mat_{}".format(k)][:3, :])
        R_c2w_array.append(RT_c2w[:3, :3])
        centers_array.append(RT_c2w[:3, [3]])
        K_array.append(K[:3, :3])

    log("Loaded {} cameras from NPZ".format(len(K_array)))
    return np.array(K_array), np.array(R_c2w_array), np.array(centers_array)


def load_cameras_from_transform_json(json_path, albedo_images, logger=None):
    """Load cameras from transform.json (NeRF/Instant-NGP format).

    If the JSON contains an ``n2w`` matrix, cameras are automatically
    transformed from normalized space to world space.
    """
    def log(msg):
        if logger:
            logger.info(msg)

    log("Loading cameras from transform.json: {}".format(json_path))

    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data["frames"]

    # n2w: normalized-to-world transform (if present)
    n2w = None
    if "n2w" in data:
        n2w = np.array(data["n2w"], dtype=np.float64)
        log("Found n2w, converting cameras to world space")

    K_array, R_c2w_array, centers_array = [], [], []

    global_fx = data.get("fl_x", None)
    global_fy = data.get("fl_y", global_fx)
    global_cx = data.get("cx", None)
    global_cy = data.get("cy", None)

    for albedo_img in albedo_images:
        frame = None
        for f in frames:
            if Path(f["albedo_path"]).stem == Path(albedo_img).stem:
                frame = f
                break
        if frame is None:
            raise RuntimeError(
                "No frame for albedo image: {}".format(albedo_img))

        K = np.eye(3, dtype=np.float32)
        if "intrinsic_matrix" in frame:
            K_full = np.array(frame["intrinsic_matrix"], dtype=np.float32)
            K[:3, :3] = K_full[:3, :3]
        else:
            fx = frame.get("fl_x", global_fx or 500.0)
            fy = frame.get("fl_y", global_fy or fx)
            cx = frame.get("cx", global_cx or data.get("w", 512) / 2)
            cy = frame.get("cy", global_cy or data.get("h", 512) / 2)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        if n2w is not None:
            c2w = n2w @ c2w

        R_c2w = c2w[:3, :3].astype(np.float32)
        center = c2w[:3, [3]].astype(np.float32)

        K_array.append(K)
        R_c2w_array.append(R_c2w)
        centers_array.append(center)

    log("Loaded {} cameras from transform.json".format(len(K_array)))
    return np.array(K_array), np.array(R_c2w_array), np.array(centers_array)


def load_cameras(camera_source, albedo_images, logger=None):
    """Load cameras from any supported format (auto-detect)."""
    camera_path = Path(camera_source)
    n_views = len(albedo_images)

    if camera_path.suffix == ".sfm":
        return load_cameras_from_sfmdata(camera_path, albedo_images, logger)
    elif camera_path.suffix == ".npz":
        return load_cameras_from_npz(camera_path, n_views, logger)
    elif camera_path.suffix == ".json" or camera_path.name == "transform.json":
        return load_cameras_from_transform_json(
            camera_path, albedo_images, logger)
    else:
        raise ValueError(
            "Unsupported camera format: {}".format(camera_path.suffix))


def compute_albedo_scale_ratios(albedo_path, camera_source, mesh_path,
                                n_samples=2000, logger=None):
    """Compute per-view albedo scaling ratios via multi-view consistency.

    Returns:
        (n_views, 3) array of scale factors.
    """
    def log(msg):
        if logger:
            logger.info(msg)

    # Load albedos
    list_names = sorted([
        f for f in os.listdir(albedo_path)
        if f.lower().endswith((".png", ".exr"))
    ])
    n_views = len(list_names)

    log("Loading {} albedo images...".format(n_views))
    albedos, masks = [], []
    for name in list_names:
        albedo = load_image(os.path.join(albedo_path, name))
        mask = albedo[:, :, 3] if albedo.shape[2] == 4 else np.ones(
            albedo.shape[:2])
        albedos.append(albedo[:, :, :3])
        masks.append(mask)

    albedos = np.array(albedos)
    masks = np.array(masks)
    n_views, h, w, _ = albedos.shape

    K_array, R_c2w_array, centers_array = load_cameras(
        camera_source, list_names, logger)

    log("Loading mesh from {}...".format(mesh_path))
    mesh = trimesh.load_mesh(mesh_path)

    ratios = np.zeros((n_views, n_samples, 3, 2), dtype=np.float32)
    intersection_found = np.zeros((n_views, n_samples, 2), dtype=np.bool_)

    log("Computing ratios between neighboring views...")
    for cam_id in range(n_views):
        log("Processing camera {}/{}...".format(cam_id, n_views))

        mask = masks[cam_id].astype(bool)
        ind_mask = np.where(mask)
        pixels = np.stack([ind_mask[1], ind_mask[0]], axis=1)
        albedo_values = albedos[cam_id, ind_mask[0], ind_mask[1], :]

        current_K = K_array[cam_id]
        current_R_c2w = R_c2w_array[cam_id]
        current_center = centers_array[cam_id]

        n_good = min(n_samples, pixels.shape[0])
        if n_good < n_samples:
            log("Warning: only {} valid pixels in image {}".format(
                n_good, cam_id))

        ind = np.random.choice(pixels.shape[0], n_good, replace=False)
        pixels = pixels[ind]
        albedo_values = albedo_values[ind]

        rays_origin = np.tile(current_center.T, (n_good, 1))
        point_on_rays = (
            current_R_c2w @ (
                np.linalg.inv(current_K) @ np.concatenate(
                    (pixels, np.ones((n_good, 1))), axis=1
                ).T
            ) + current_center
        ).T
        rays_direction = point_on_rays - rays_origin
        rays_direction /= np.linalg.norm(
            rays_direction, axis=1)[:, None]

        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=rays_origin,
            ray_directions=rays_direction,
            multiple_hits=False,
        )

        pixels = pixels[index_ray]
        albedo_values = albedo_values[index_ray]

        right_cam_id = (cam_id + 1) % n_views
        left_cam_id = (cam_id - 1) % n_views

        for kk, neigh_cam_id in enumerate([right_cam_id, left_cam_id]):
            neighbor_K = K_array[neigh_cam_id]
            neighbor_R_c2w = R_c2w_array[neigh_cam_id]
            neighbor_center = centers_array[neigh_cam_id]

            neighbor_rays_direction = neighbor_center.T - locations
            neighbor_dists = np.linalg.norm(
                neighbor_rays_direction, axis=1, keepdims=True)
            neighbor_rays_direction = (
                neighbor_rays_direction / neighbor_dists)
            eps = np.maximum(neighbor_dists.flatten() * 1e-4, 1e-2)
            neighbor_rays_origin = (
                locations + eps[:, None] * neighbor_rays_direction)

            hit_locs, hit_ray_idx, _ = mesh.ray.intersects_location(
                ray_origins=neighbor_rays_origin,
                ray_directions=neighbor_rays_direction,
                multiple_hits=True,
            )
            hit = np.zeros(len(locations), dtype=bool)
            if len(hit_locs) > 0:
                hit_d = np.linalg.norm(
                    hit_locs - neighbor_rays_origin[hit_ray_idx], axis=1)
                blocked = hit_d < (
                    neighbor_dists.flatten()[hit_ray_idx] - eps[hit_ray_idx])
                for idx in hit_ray_idx[blocked]:
                    hit[idx] = True

            intersection_points = locations[~hit]
            index_ray_kk = index_ray[~hit]
            albedo_values_no_hit = albedo_values[~hit]

            neighbor_R_w2c = neighbor_R_c2w.T
            pts_cam = (
                neighbor_R_w2c @ (intersection_points.T - neighbor_center)
            )
            pts_proj = (neighbor_K @ pts_cam).T
            pts_proj /= pts_proj[:, 2][:, None]
            pts_proj = pts_proj[:, :2]

            valid = (
                (0 <= pts_proj[:, 1]) & (pts_proj[:, 1] < h - 1)
                & (0 <= pts_proj[:, 0]) & (pts_proj[:, 0] < w - 1)
            )

            pts_proj = pts_proj[valid]
            index_ray_kk = index_ray_kk[valid]
            albedo_valid = albedo_values_no_hit[valid]

            alb_neigh = albedos[neigh_cam_id].astype(np.float32)
            rows = np.arange(0, h, 1)
            cols = np.arange(0, w, 1)
            interpR = RegularGridInterpolator(
                (rows, cols), alb_neigh[:, :, 0])
            interpG = RegularGridInterpolator(
                (rows, cols), alb_neigh[:, :, 1])
            interpB = RegularGridInterpolator(
                (rows, cols), alb_neigh[:, :, 2])

            pts_yx = np.stack([pts_proj[:, 1], pts_proj[:, 0]], axis=1)
            albedo_val = np.stack(
                [interpR(pts_yx), interpG(pts_yx), interpB(pts_yx)],
                axis=1)

            zero_idx = np.any(albedo_val == 0, axis=1)
            index_ray_kk = index_ray_kk[~zero_idx]
            albedo_val = albedo_val[~zero_idx]
            albedo_valid = albedo_valid[~zero_idx]

            ratios[cam_id, index_ray_kk, :, kk] = (
                albedo_valid / albedo_val)
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

    median_ratio_prop = np.ones((n_views, 3))
    for ii in range(n_views - 1):
        median_ratio_prop[ii + 1] = (
            median_ratio_prop[ii] * median_ratios[ii])

    mean_prop = np.mean(median_ratio_prop, axis=0)
    median_ratio_prop_norm = median_ratio_prop / mean_prop

    log("Scale ratios: {}".format(median_ratio_prop_norm))
    return median_ratio_prop_norm


def scale_and_save_albedos(albedo_path, output_albedo_path, scale_ratios,
                           bit_depth=None, logger=None):
    """Apply scaling to albedo images and save them."""
    def log(msg):
        if logger:
            logger.info(msg)

    os.makedirs(output_albedo_path, exist_ok=True)

    list_names = sorted([
        f for f in os.listdir(albedo_path)
        if f.lower().endswith((".png", ".exr"))
    ])

    if bit_depth is None:
        first_path = os.path.join(albedo_path, list_names[0])
        first_image = cv2.imread(first_path, cv2.IMREAD_UNCHANGED)
        if first_image.dtype == np.uint8:
            bit_depth = 8
        elif first_image.dtype == np.uint16:
            bit_depth = 16
        else:
            bit_depth = 16
        log("Auto-detected bit depth: {}".format(bit_depth))

    log("Scaling {} albedos ({}bit)...".format(len(list_names), bit_depth))
    for ii, name in enumerate(list_names):
        albedo = load_image(os.path.join(albedo_path, name))
        mask = (albedo[:, :, 3] if albedo.shape[2] == 4
                else np.ones(albedo.shape[:2]))
        albedo_rgb = albedo[:, :, :3] * scale_ratios[ii]
        albedo_to_save = np.concatenate(
            (albedo_rgb, mask[:, :, np.newaxis]), axis=-1)
        save_image(
            albedo_to_save,
            os.path.join(output_albedo_path, name),
            bit_depth=bit_depth)
        log("Saved {}/{}: {}".format(ii + 1, len(list_names), name))
