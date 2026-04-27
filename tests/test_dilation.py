"""Test how much mask dilation is needed to reach 99% overlap.

This tells us the effective boundary error in pixels.
Also separately measure: mesh-only error (centered PP) vs PP+mesh error.
"""

import json
import os

import cv2
import numpy as np
import trimesh

from pyalicevisionlib import load_sfmdata

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
OUTPUT_DIR = "/tmp/mesh_projection_debug"


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


def find_mask(cam, mask_dir):
    if cam.image_path:
        stem = os.path.splitext(os.path.basename(cam.image_path))[0]
        p = os.path.join(mask_dir, stem + ".png")
        if os.path.exists(p):
            return p
    return None


def overlap_at_dilation(uv, mask, dilation_px):
    """Compute overlap after dilating mask by N pixels."""
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
        mask_d = cv2.dilate(mask, kernel, iterations=1)
    else:
        mask_d = mask

    w, h = mask_d.shape[1], mask_d.shape[0]
    u = np.clip(uv[:, 0].astype(int), 0, w - 1)
    v = np.clip(uv[:, 1].astype(int), 0, h - 1)
    n_in = mask_d[v, u].sum()
    return n_in / len(uv) * 100


def project(cam, verts, cx_override=None, cy_override=None):
    """Project with optional PP override."""
    R_w2c = cam.rotation_cam2world.T
    pts_cam = (R_w2c @ (verts - cam.center).T).T
    in_front = pts_cam[:, 2] > 0.1
    pts_f = pts_cam[in_front]
    if len(pts_f) == 0:
        return None

    fx = fy = cam.focal_length_pixels
    cx = cx_override if cx_override is not None else cam.cx
    cy = cy_override if cy_override is not None else cam.cy

    u = fx * pts_f[:, 0] / pts_f[:, 2] + cx
    v = fy * pts_f[:, 1] / pts_f[:, 2] + cy

    in_img = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return np.column_stack([u[in_img], v[in_img]])


def main():
    verts = load_mesh_verts(MESH_PATH)
    sfmdata = load_sfmdata(SFM_PATH)
    cams = sfmdata.get_cameras(apply_world_correction=True)

    dilations = [0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    n_test = min(20, len(cams))

    print("=" * 70)
    print("MASK DILATION ANALYSIS")
    print("=" * 70)

    # Test 1: Standard projection with increasing dilation
    print("\n--- Standard projection (actual PP) ---")
    for d in dilations:
        overlaps = []
        for i in range(n_test):
            cam = cams[i]
            mp = find_mask(cam, MASK_DIR)
            if mp is None:
                continue
            mask = load_mask(mp)
            uv = project(cam, verts)
            if uv is None:
                continue
            overlaps.append(overlap_at_dilation(uv, mask, d))
        if overlaps:
            mean_o = np.mean(overlaps)
            print(f"  Dilation {d:3d}px: {mean_o:6.2f}% "
                  f"(min={np.min(overlaps):.1f})")
            if mean_o >= 99.0:
                print(f"  >>> 99% reached at dilation = {d}px <<<")
                break

    # Test 2: Centered PP with increasing dilation
    print("\n--- Centered PP projection ---")
    for d in dilations:
        overlaps = []
        for i in range(n_test):
            cam = cams[i]
            mp = find_mask(cam, MASK_DIR)
            if mp is None:
                continue
            mask = load_mask(mp)
            uv = project(cam, verts,
                         cx_override=cam.width / 2.0,
                         cy_override=cam.height / 2.0)
            if uv is None:
                continue
            overlaps.append(overlap_at_dilation(uv, mask, d))
        if overlaps:
            mean_o = np.mean(overlaps)
            print(f"  Dilation {d:3d}px: {mean_o:6.2f}% "
                  f"(min={np.min(overlaps):.1f})")
            if mean_o >= 99.0:
                print(f"  >>> 99% reached at dilation = {d}px <<<")
                break

    # Test 3: per-view analysis of outside-mask point distribution
    print("\n--- Outside-mask point distribution (view 0) ---")
    cam = cams[0]
    mp = find_mask(cam, MASK_DIR)
    mask = load_mask(mp)
    uv = project(cam, verts)
    if uv is not None:
        u_int = np.clip(uv[:, 0].astype(int), 0, cam.width - 1)
        v_int = np.clip(uv[:, 1].astype(int), 0, cam.height - 1)
        outside = ~mask[v_int, u_int].astype(bool)

        uv_out = uv[outside]
        uv_in = uv[~outside]

        print(f"  Outside: {outside.sum()} points")
        print(f"  Outside centroid: ({uv_out.mean(0)[0]:.1f}, {uv_out.mean(0)[1]:.1f})")
        print(f"  Inside centroid:  ({uv_in.mean(0)[0]:.1f}, {uv_in.mean(0)[1]:.1f})")
        print(f"  All centroid:     ({uv.mean(0)[0]:.1f}, {uv.mean(0)[1]:.1f})")

        # Mask centroid
        ys, xs = np.where(mask > 0)
        print(f"  Mask centroid:    ({xs.mean():.1f}, {ys.mean():.1f})")

        # Distance of outside points to nearest mask boundary
        # Use distance transform
        dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
        distances = dist[v_int[outside], u_int[outside]]
        print(f"  Outside-point distance to mask boundary: "
              f"mean={distances.mean():.1f}px, "
              f"median={np.median(distances):.1f}px, "
              f"p95={np.percentile(distances, 95):.1f}px, "
              f"max={distances.max():.1f}px")

        # Save enhanced visualization showing distance
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        scale = 0.25
        h_vis = int(cam.height * scale)
        w_vis = int(cam.width * scale)
        vis = np.zeros((h_vis, w_vis, 3), dtype=np.uint8)
        mask_s = cv2.resize(mask * 255, (w_vis, h_vis))
        contours, _ = cv2.findContours(mask_s, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (128, 128, 128), 1)

        # Color outside points by distance
        for j in range(len(uv_out)):
            x = int(uv_out[j, 0] * scale)
            y = int(uv_out[j, 1] * scale)
            if 0 <= x < w_vis and 0 <= y < h_vis:
                d_px = distances[j]
                # Color: blue=close, red=far
                b = max(0, min(255, int(255 - d_px * 3)))
                r = max(0, min(255, int(d_px * 3)))
                cv2.circle(vis, (x, y), 1, (b, 0, r), -1)

        # Draw inside points in dim green
        for j in range(0, len(uv_in), 3):  # subsample for speed
            x = int(uv_in[j, 0] * scale)
            y = int(uv_in[j, 1] * scale)
            if 0 <= x < w_vis and 0 <= y < h_vis:
                cv2.circle(vis, (x, y), 1, (0, 80, 0), -1)

        cv2.putText(vis, f"Outside: mean dist={distances.mean():.0f}px",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "dist_analysis.png"), vis)
        print(f"\n  Saved: {OUTPUT_DIR}/dist_analysis.png")


if __name__ == "__main__":
    main()
