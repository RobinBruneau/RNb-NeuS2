"""Test multiple projection hypotheses to find misalignment root cause.

Hypotheses:
1. Standard projection (pyalicevisionlib conventions, with WORLD_CORRECTION)
2. Without WORLD_CORRECTION (raw AliceVision cameras)
3. Centered principal point (cx=W/2, cy=H/2, ignoring PP offset)
4. Sign-flipped principal point Y offset
5. Sign-flipped principal point X offset
6. Both PP offsets sign-flipped
7. Using cameras from transform.json (reconstructed to world space)
"""

import json
import os

import cv2
import numpy as np
import trimesh

from pyalicevisionlib import load_sfmdata, Camera, WORLD_CORRECTION

SFM_PATH = (
    "/home/babrument/dev/alicevision/dataset_temp/objects/01_rock"
    "/09_unimsps_d2/sfm.json"
)
MASK_DIR = (
    "/home/babrument/dev/alicevision/dataset_temp/objects/01_rock"
    "/09_unimsps_d2/masks"
)
MESH_PATH = (
    "/home/babrument/dev/alicevision/dataset_temp/objects/01_rock"
    "/10_rnbneus2/unimsps_d2/mesh.obj"
)
TRANSFORM_JSON = (
    "/home/babrument/dev/alicevision/dataset_temp/objects/01_rock"
    "/10_rnbneus2/unimsps_d2/prepared_data/transform.json"
)


def load_mesh_verts(path, n=30000):
    mesh = trimesh.load(path, process=False)
    verts = np.array(mesh.vertices, dtype=np.float64)
    if len(verts) > n:
        verts = verts[np.random.default_rng(42).choice(len(verts), n, replace=False)]
    return verts


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(m.shape) == 3:
        m = m[:, :, 0]
    thr = 125 if m.dtype == np.uint8 else 30000
    return (m > thr).astype(np.uint8)


def project_custom(verts, R_c2w, center, fx, fy, cx, cy, w, h):
    """Project using explicit parameters."""
    R_w2c = R_c2w.T
    pts_cam = (R_w2c @ (verts - center).T).T
    in_front = pts_cam[:, 2] > 0.1
    pts_f = pts_cam[in_front]
    if len(pts_f) == 0:
        return 0.0
    u = fx * pts_f[:, 0] / pts_f[:, 2] + cx
    v = fy * pts_f[:, 1] / pts_f[:, 2] + cy
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[in_img], v[in_img]


def overlap(u, v, mask):
    w, h = mask.shape[1], mask.shape[0]
    ui = np.clip(u.astype(int), 0, w - 1)
    vi = np.clip(v.astype(int), 0, h - 1)
    n_in = mask[vi, ui].sum()
    return n_in / len(u) * 100 if len(u) > 0 else 0.0


def find_mask(cam, mask_dir):
    if cam.image_path:
        stem = os.path.splitext(os.path.basename(cam.image_path))[0]
        p = os.path.join(mask_dir, stem + ".png")
        if os.path.exists(p):
            return p
    p = os.path.join(mask_dir, cam.view_id + ".png")
    if os.path.exists(p):
        return p
    return None


def main():
    verts = load_mesh_verts(MESH_PATH)
    print(f"Mesh: {len(verts)} vertices")

    # Load cameras via pyalicevisionlib
    sfmdata = load_sfmdata(SFM_PATH)
    cams_corrected = sfmdata.get_cameras(apply_world_correction=True)
    cams_raw = sfmdata.get_cameras(apply_world_correction=False)
    print(f"Cameras: {len(cams_corrected)}")

    # Load transform.json
    with open(TRANSFORM_JSON) as f:
        tj = json.load(f)
    n2w = np.array(tj["n2w"])
    n2w_s = n2w[0, 0]
    n2w_t = n2w[:3, 3]

    # Test views (use first 20 for speed)
    n_test = min(20, len(cams_corrected))

    hypotheses = {
        "H1_standard": [],
        "H2_no_world_corr": [],
        "H3_centered_pp": [],
        "H4_flip_pp_y": [],
        "H5_flip_pp_x": [],
        "H6_flip_pp_both": [],
        "H7_transform_json": [],
    }

    for i in range(n_test):
        cam = cams_corrected[i]
        cam_raw = cams_raw[i]
        mask_path = find_mask(cam, MASK_DIR)
        if mask_path is None:
            continue
        mask = load_mask(mask_path)

        w, h = cam.width, cam.height
        fx = fy = cam.focal_length_pixels
        pp = cam.principal_point

        # H1: Standard (pyalicevisionlib with WORLD_CORRECTION)
        uv = project_custom(verts, cam.rotation_cam2world, cam.center,
                            fx, fy, cam.cx, cam.cy, w, h)
        if isinstance(uv, float):
            continue
        hypotheses["H1_standard"].append(overlap(*uv, mask))

        # H2: No WORLD_CORRECTION (raw cameras, mesh still in corrected space)
        uv = project_custom(verts, cam_raw.rotation_cam2world, cam_raw.center,
                            fx, fy, cam.cx, cam.cy, w, h)
        if not isinstance(uv, float):
            hypotheses["H2_no_world_corr"].append(overlap(*uv, mask))

        # H3: Centered principal point
        uv = project_custom(verts, cam.rotation_cam2world, cam.center,
                            fx, fy, w / 2.0, h / 2.0, w, h)
        if not isinstance(uv, float):
            hypotheses["H3_centered_pp"].append(overlap(*uv, mask))

        # H4: Flip PP Y sign
        cy_flipped = h / 2.0 - pp[1]  # instead of h/2 + pp[1]
        uv = project_custom(verts, cam.rotation_cam2world, cam.center,
                            fx, fy, cam.cx, cy_flipped, w, h)
        if not isinstance(uv, float):
            hypotheses["H4_flip_pp_y"].append(overlap(*uv, mask))

        # H5: Flip PP X sign
        cx_flipped = w / 2.0 - pp[0]
        uv = project_custom(verts, cam.rotation_cam2world, cam.center,
                            fx, fy, cx_flipped, cam.cy, w, h)
        if not isinstance(uv, float):
            hypotheses["H5_flip_pp_x"].append(overlap(*uv, mask))

        # H6: Flip both PP signs
        uv = project_custom(verts, cam.rotation_cam2world, cam.center,
                            fx, fy, cx_flipped, cy_flipped, w, h)
        if not isinstance(uv, float):
            hypotheses["H6_flip_pp_both"].append(overlap(*uv, mask))

        # H7: Using transform.json cameras (back to world via n2w)
        if i < len(tj["frames"]):
            frame = tj["frames"][i]
            c2w_tj = np.array(frame["transform_matrix"])
            K_tj = np.array(frame["intrinsic_matrix"])
            # Camera center in normalized space
            center_norm = c2w_tj[:3, 3]
            R_c2w_tj = c2w_tj[:3, :3]
            # Convert center to world
            center_world = n2w_s * center_norm + n2w_t
            fx_tj = K_tj[0, 0]
            fy_tj = K_tj[1, 1]
            cx_tj = K_tj[0, 2]
            cy_tj = K_tj[1, 2]
            uv = project_custom(verts, R_c2w_tj, center_world,
                                fx_tj, fy_tj, cx_tj, cy_tj, w, h)
            if not isinstance(uv, float):
                hypotheses["H7_transform_json"].append(overlap(*uv, mask))

    print("\n" + "=" * 60)
    print("HYPOTHESIS RESULTS (mean overlap %)")
    print("=" * 60)
    for name, vals in hypotheses.items():
        if vals:
            print(f"  {name:25s}: {np.mean(vals):6.2f}% "
                  f"(min={np.min(vals):.1f}, max={np.max(vals):.1f}, "
                  f"std={np.std(vals):.1f})")
        else:
            print(f"  {name:25s}: NO DATA")

    # Find best hypothesis
    best = max(hypotheses.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)
    print(f"\n  >>> Best: {best[0]} ({np.mean(best[1]):.2f}%)")


if __name__ == "__main__":
    main()
