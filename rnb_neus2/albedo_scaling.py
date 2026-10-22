"""Albedo scaling via multi-view consistency.

Photometric-stereo albedo maps are known per view up to a per-channel scale (e.g. the light powers of each pose).
The per-view scales are estimated by ray-tracing masked pixels of every view onto a mesh and comparing their albedo
with every other view that sees the same surface point (facing, unoccluded, inside both masks). The median log
ratio of every overlapping view pair gives one equation log s_i - log s_j = -median(log a_i - log a_j); the
equations are solved jointly (weighted least squares), so the result does not depend on the view order and loops
of views are closed. Views without overlap keep the scale 1.

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

from .image_io import load_image, save_image

# Minimum number of surface points seen by both views for a view pair to constrain the scales
MIN_PAIR_SAMPLES = 300
# Albedo values kept for the ratios: not empty, not clipped
MIN_ALBEDO, MAX_ALBEDO = 1e-4, 0.999


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
            # absolute principal point (getOffset() is relative to the image center), converted from the
            # AliceVision pixel convention to the NeuS2 one (center of pixel i at i + 0.5)
            K[0, 0] = intrinsic.getScale().x()
            K[1, 1] = intrinsic.getScale().y()
            pp = intrinsic.getPrincipalPoint()
            K[0, 2] = pp[0] + 0.5
            K[1, 2] = pp[1] + 0.5
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
            # n2w is a similarity: the camera basis is its linear part divided by its scale magnitude (a reflection
            # keeps the cameras looking forward, as the mesh export does)
            c2w = n2w @ c2w
            c2w[:3, :3] /= np.cbrt(abs(np.linalg.det(n2w[:3, :3])))

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


def list_albedo_names(albedo_path):
    """Albedo image names of a folder, in the order of the scale ratios (sorted)."""
    return sorted(f for f in os.listdir(albedo_path) if f.lower().endswith((".png", ".exr")))


def view_groups(n_views, edges):
    """Connected components (lists of view indices) of the view-overlap graph."""
    parent = list(range(n_views))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, j in edges:
        parent[root(i)] = root(j)
    groups = {}
    for k in range(n_views):
        groups.setdefault(root(k), []).append(k)
    return list(groups.values())


def solve_pairwise_log_scales(pair_log_ratios, n_views, min_pair_samples=MIN_PAIR_SAMPLES):
    """Per-view, per-channel scales s such that s_i a_i ~ s_j a_j on the points seen by views i and j.

    Every view pair with at least min_pair_samples samples gives one equation
    log s_i - log s_j = -median(log a_i - log a_j), weighted by sqrt(n_samples). The equations are solved jointly
    (least squares), with the geometric mean of the scales fixed to 1 in every connected group of overlapping views.
    Views without any pair keep the scale 1.

    Args:
        pair_log_ratios: dict (i, j) with i < j -> (n, 3) array of log a_i - log a_j.
        n_views: number of views.

    Returns:
        (scales (n_views, 3), pairs: list of (i, j, n_samples, median log ratio (3,)), unconstrained view indices).
    """
    rows, rhs, weights, pairs = [], [], [], []
    for (i, j), log_ratios in sorted(pair_log_ratios.items()):
        log_ratios = np.asarray(log_ratios).reshape(-1, 3)
        if len(log_ratios) < min_pair_samples:
            continue
        median = np.median(log_ratios, axis=0)
        row = np.zeros(n_views)
        row[i], row[j] = 1.0, -1.0
        rows.append(row)
        rhs.append(-median)
        weights.append(np.sqrt(len(log_ratios)))
        pairs.append((i, j, len(log_ratios), median))
    constrained = sorted({k for i, j, _, _ in pairs for k in (i, j)})
    unconstrained = [k for k in range(n_views) if k not in constrained]
    scales = np.ones((n_views, 3))
    if not pairs:
        return scales, pairs, unconstrained
    weights = np.array(weights)[:, None]
    a = np.array(rows)[:, constrained] * weights
    b = np.array(rhs) * weights
    # gauge: sum of the log scales = 0 in every group of overlapping views
    index = {k: c for c, k in enumerate(constrained)}
    groups = view_groups(len(constrained), [(index[i], index[j]) for i, j, _, _ in pairs])
    gauge = np.zeros((len(groups), len(constrained)))
    for g, group in enumerate(groups):
        gauge[g, group] = 10.0 * weights.max()
    a = np.vstack([a, gauge])
    b = np.vstack([b, np.zeros((len(groups), 3))])
    scales[constrained] = np.exp(np.linalg.lstsq(a, b, rcond=None)[0])
    return scales, pairs, unconstrained


def pair_mismatch(pairs, scales):
    """Median over the view pairs of the mean (over channels) |log ratio|, before and after scaling, in %."""
    if not pairs:
        return 0.0, 0.0
    before = [np.abs(median).mean() for _, _, _, median in pairs]
    after = [np.abs(median + np.log(scales[i]) - np.log(scales[j])).mean() for i, j, _, median in pairs]
    return float(100.0 * (np.exp(np.median(before)) - 1.0)), float(100.0 * (np.exp(np.median(after)) - 1.0))


def _bilinear(image, xy):
    """Bilinear lookup of an (h, w, c) image at (n, 2) pixel-index coordinates inside [0, w-1] x [0, h-1]."""
    h, w = image.shape[:2]
    x0 = np.clip(np.floor(xy[:, 0]).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(xy[:, 1]).astype(int), 0, h - 2)
    fx = (xy[:, 0] - x0)[:, None]
    fy = (xy[:, 1] - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def collect_pair_log_ratios(mesh, albedos, masks, K_array, R_c2w_array, centers_array, n_samples=20000,
                            max_angle=60.0, seed=0):
    """Log albedo ratios of every view pair on shared surface points.

    For every view i, n_samples masked pixels are ray-cast onto the mesh (first hit, viewing angle <= max_angle);
    each hit is compared with every other view j that sees it: in front of j and inside its image, facing j
    (angle <= max_angle), unoccluded, inside j's mask, with valid (non-empty, non-clipped) albedos in both views.

    Args:
        mesh: trimesh.Trimesh in the camera world frame.
        albedos: (V, h, w, 3) float albedo maps; masks: (V, h, w) bool.
        K_array: (V, 3, 3) intrinsics in the NeuS2 pixel convention (center of pixel i at i + 0.5).
        R_c2w_array: (V, 3, 3); centers_array: (V, 3, 1) camera centers.

    Returns:
        dict (i, j) with i < j -> (n, 3) float array of log a_i - log a_j.
    """
    rng = np.random.default_rng(seed)
    n_views, h, w, _ = albedos.shape
    cos_max = np.cos(np.radians(max_angle))
    centers = centers_array.reshape(n_views, 3).astype(np.float64)
    ratios = {}
    for i in range(n_views):
        ys, xs = np.where(masks[i])
        if len(xs) == 0:
            continue
        pick = rng.choice(len(xs), min(n_samples, len(xs)), replace=False)
        ys, xs = ys[pick], xs[pick]
        pixels = np.stack([xs + 0.5, ys + 0.5, np.ones(len(xs))], axis=1)  # NeuS2 pixel centers
        directions = (R_c2w_array[i] @ np.linalg.inv(K_array[i]) @ pixels.T).T
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        origins = np.tile(centers[i], (len(directions), 1))
        points, ray_ids, tri_ids = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        if len(points) == 0:
            continue
        # face normals oriented towards camera i (independent of the mesh winding)
        normals = mesh.face_normals[tri_ids]
        to_i = centers[i] - points
        to_i /= np.linalg.norm(to_i, axis=1, keepdims=True)
        cos_i = np.einsum("ij,ij->i", normals, to_i)
        normals *= np.sign(cos_i)[:, None]
        keep = np.abs(cos_i) > cos_max
        points, normals = points[keep], normals[keep]
        albedo_i = albedos[i][ys[ray_ids[keep]], xs[ray_ids[keep]]]
        good_i = np.all((albedo_i > MIN_ALBEDO) & (albedo_i < MAX_ALBEDO), axis=1)
        points, normals, albedo_i = points[good_i], normals[good_i], albedo_i[good_i]
        for j in range(n_views):
            if j == i or len(points) == 0:
                continue
            to_j = centers[j] - points
            dist_j = np.linalg.norm(to_j, axis=1)
            to_j /= dist_j[:, None]
            pts_cam = (R_c2w_array[j].T @ (points - centers[j]).T).T
            front = pts_cam[:, 2] > 0
            proj = (K_array[j] @ pts_cam.T).T
            proj = proj[:, :2] / np.where(front, proj[:, 2], 1.0)[:, None] - 0.5  # -> pixel indices
            ok = (front & (np.einsum("ij,ij->i", normals, to_j) > cos_max)
                  & (proj[:, 0] >= 0) & (proj[:, 0] <= w - 1) & (proj[:, 1] >= 0) & (proj[:, 1] <= h - 1))
            ok_ids = np.where(ok)[0]
            if len(ok_ids) == 0:
                continue
            ui, vi = np.round(proj[ok_ids, 0]).astype(int), np.round(proj[ok_ids, 1]).astype(int)
            ok_ids = ok_ids[masks[j][vi, ui]]
            if len(ok_ids) == 0:
                continue
            # occlusion: the first hit of the ray towards camera j must be beyond the camera
            eps = (1e-4 * dist_j[ok_ids])[:, None]
            origins_j = points[ok_ids] + to_j[ok_ids] * eps
            hits, hit_rays, _ = mesh.ray.intersects_location(origins_j, to_j[ok_ids], multiple_hits=False)
            occluded = np.zeros(len(ok_ids), bool)
            if len(hits):
                hit_dist = np.linalg.norm(hits - origins_j[hit_rays], axis=1)
                occluded[hit_rays[hit_dist < dist_j[ok_ids][hit_rays] - eps[hit_rays, 0]]] = True
            ok_ids = ok_ids[~occluded]
            if len(ok_ids) == 0:
                continue
            albedo_j = _bilinear(albedos[j], proj[ok_ids])
            good = np.all((albedo_j > MIN_ALBEDO) & (albedo_j < MAX_ALBEDO), axis=1)
            log_ratio = np.log(albedo_i[ok_ids[good]]) - np.log(albedo_j[good])
            key, sign = ((i, j), 1.0) if i < j else ((j, i), -1.0)
            ratios.setdefault(key, []).append(sign * log_ratio)
    return {key: np.concatenate(parts) for key, parts in ratios.items()}


def compute_albedo_scale_ratios(albedo_path, camera_source, mesh_path,
                                n_samples=20000, max_angle=60.0,
                                min_pair_samples=MIN_PAIR_SAMPLES,
                                logger=None, return_report=False):
    """Compute per-view albedo scaling ratios via multi-view consistency.

    Args:
        albedo_path: folder of albedo images (RGBA: the alpha channel is the mask), in the order of
            list_albedo_names.
        camera_source: transform.json, cameras.npz or SfMData of these images.
        mesh_path: mesh in the camera world frame.
        n_samples: masked pixels ray-cast per view.
        max_angle: maximum viewing angle (degrees) of a surface point.
        min_pair_samples: minimum number of samples for a view pair to constrain the scales.
        return_report: also return a report dict (pairs, mismatch, unconstrained views).

    Returns:
        (n_views, 3) array of scale factors (arithmetic mean 1 over the views constrained by an overlap,
        1 for the others), and the report if return_report.
    """
    def log(msg):
        if logger:
            logger.info(msg)

    list_names = list_albedo_names(albedo_path)
    log("Loading {} albedo images...".format(len(list_names)))
    albedos, masks = [], []
    for name in list_names:
        albedo = load_image(os.path.join(albedo_path, name))
        masks.append(albedo[:, :, 3] > 0.5 if albedo.shape[2] == 4 else np.ones(albedo.shape[:2], bool))
        albedos.append(np.ascontiguousarray(albedo[:, :, :3]))
    albedos = np.array(albedos)
    masks = np.array(masks)
    n_views = len(list_names)

    K_array, R_c2w_array, centers_array = load_cameras(camera_source, list_names, logger)

    log("Loading mesh from {}...".format(mesh_path))
    mesh = trimesh.load_mesh(mesh_path)

    log("Comparing the albedo of all the view pairs ({} samples per view)...".format(n_samples))
    pair_log_ratios = collect_pair_log_ratios(mesh, albedos, masks, K_array, R_c2w_array, centers_array,
                                              n_samples=n_samples, max_angle=max_angle)
    scales, pairs, unconstrained = solve_pairwise_log_scales(pair_log_ratios, n_views, min_pair_samples)
    constrained = [k for k in range(n_views) if k not in unconstrained]
    if constrained:
        # same albedo level as before the scaling: arithmetic mean 1 over the constrained views
        scales[constrained] /= scales[constrained].mean(axis=0)
    before, after = pair_mismatch(pairs, scales)
    log("{} overlapping view pairs (>= {} samples); median pairwise albedo mismatch {:.1f}% -> {:.1f}%".format(
        len(pairs), min_pair_samples, before, after))
    if unconstrained:
        log("WARNING: no overlap with the other views, scale kept at 1 for: {}".format(
            ", ".join(list_names[k] for k in unconstrained)))
    groups = [g for g in view_groups(n_views, [(i, j) for i, j, _, _ in pairs]) if len(g) > 1]
    if len(groups) > 1:
        log("WARNING: {} separate groups of overlapping views, scaled independently: {}".format(
            len(groups), [[list_names[k] for k in g] for g in groups]))
    log("Scale ratios: {}".format(scales))
    if not return_report:
        return scales
    report = {
        "names": list_names,
        "scales": scales.tolist(),
        "pairs": [{"views": [list_names[i], list_names[j]], "samples": n, "medianLogRatio": median.tolist()}
                  for i, j, n, median in pairs],
        "mismatchBeforePercent": round(before, 2),
        "mismatchAfterPercent": round(after, 2),
        "unconstrained": [list_names[k] for k in unconstrained],
    }
    return scales, report


def scale_and_save_albedos(albedo_path, output_albedo_path, scale_ratios,
                           bit_depth=None, logger=None):
    """Apply scaling to albedo images and save them.

    The images keep the bit depth of the inputs (what the testbed reads), so scaled values above 1 are clipped;
    the clipped fraction of the masked pixels is logged per view.
    """
    def log(msg):
        if logger:
            logger.info(msg)

    os.makedirs(output_albedo_path, exist_ok=True)

    list_names = list_albedo_names(albedo_path)

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
        inside = mask > 0.5
        clipped = float(np.any(albedo_rgb[inside] > 1.0, axis=1).mean()) if inside.any() else 0.0
        log("Saved {}/{}: {} (scale {}, {:.2f}% of the masked pixels clipped)".format(
            ii + 1, len(list_names), name, np.round(scale_ratios[ii], 3).tolist(), 100.0 * clipped))
