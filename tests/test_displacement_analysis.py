"""Analyze the systematic displacement between mesh projection and masks.

Computes the optimal 2D pixel shift per view to maximize overlap,
then back-projects to determine if there's a consistent 3D offset.
"""

import json
import os
import sys

import cv2
import numpy as np
import trimesh

from pyalicevisionlib import load_sfmdata, Camera

# ── Paths (same as diagnostic) ─────────────────────────────────────────
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


def load_mesh_vertices(mesh_path, max_vertices=50000):
    mesh = trimesh.load(mesh_path, process=False)
    verts = np.array(mesh.vertices, dtype=np.float64)
    if len(verts) > max_vertices:
        idx = np.random.default_rng(42).choice(
            len(verts), max_vertices, replace=False)
        verts = verts[idx]
    return verts


def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    if mask.dtype == np.float32:
        return (mask > 0.5).astype(np.uint8)
    threshold = 125 if mask.dtype == np.uint8 else 30000
    return (mask > threshold).astype(np.uint8)


def find_mask_for_camera(cam, mask_dir):
    if cam.image_path:
        stem = os.path.splitext(os.path.basename(cam.image_path))[0]
        for ext in (".png", ".jpg", ".exr"):
            candidate = os.path.join(mask_dir, stem + ext)
            if os.path.exists(candidate):
                return candidate
    for ext in (".png", ".jpg"):
        candidate = os.path.join(mask_dir, cam.view_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def compute_overlap_at_shift(uv, mask, dx, dy):
    """Compute mask overlap after shifting projected points by (dx, dy)."""
    h, w = mask.shape
    u_shifted = np.clip((uv[:, 0] + dx).astype(int), 0, w - 1)
    v_shifted = np.clip((uv[:, 1] + dy).astype(int), 0, h - 1)
    return mask[v_shifted, u_shifted].sum() / len(uv)


def find_optimal_2d_shift(uv, mask, search_range=100, step=2):
    """Brute-force search for optimal 2D pixel shift to maximize overlap."""
    best_overlap = 0.0
    best_dx, best_dy = 0, 0

    # Coarse search
    for dx in range(-search_range, search_range + 1, step):
        for dy in range(-search_range, search_range + 1, step):
            overlap = compute_overlap_at_shift(uv, mask, dx, dy)
            if overlap > best_overlap:
                best_overlap = overlap
                best_dx, best_dy = dx, dy

    # Fine search around best coarse result
    for dx in range(best_dx - step, best_dx + step + 1):
        for dy in range(best_dy - step, best_dy + step + 1):
            overlap = compute_overlap_at_shift(uv, mask, dx, dy)
            if overlap > best_overlap:
                best_overlap = overlap
                best_dx, best_dy = dx, dy

    return best_dx, best_dy, best_overlap


def compute_centroid_shift(uv, mask):
    """Compare centroids of projected points vs mask foreground."""
    proj_centroid = uv.mean(axis=0)

    # Mask centroid
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0
    mask_centroid = np.array([xs.mean(), ys.mean()])

    return mask_centroid - proj_centroid


def main():
    print("=" * 70)
    print("DISPLACEMENT ANALYSIS")
    print("=" * 70)

    # Load data
    vertices = load_mesh_vertices(MESH_PATH, max_vertices=50000)
    sfmdata = load_sfmdata(SFM_PATH)
    cameras = sfmdata.get_cameras(apply_world_correction=True)

    print(f"Mesh: {len(vertices)} vertices")
    print(f"Cameras: {len(cameras)}")

    shifts = []
    centroid_shifts = []

    for i, cam in enumerate(cameras):
        mask_path = find_mask_for_camera(cam, MASK_DIR)
        if mask_path is None:
            continue

        mask = load_mask(mask_path)

        # Project
        pts_cam = cam.world_to_camera(vertices)
        in_front = pts_cam[:, 2] > 0.1
        uv = cam.project_points(vertices[in_front])
        in_image = (
            (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) &
            (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
        )
        uv_valid = uv[in_image]

        if len(uv_valid) < 100:
            continue

        # Method 1: Centroid comparison
        cs = compute_centroid_shift(uv_valid, mask)
        centroid_shifts.append(cs)

        # Method 2: Optimal 2D shift (only for a few views for speed)
        if i < 10:
            dx, dy, overlap = find_optimal_2d_shift(uv_valid, mask)
            shifts.append((i, dx, dy, overlap))
            print(f"  View {i:3d}: optimal shift = ({dx:+4d}, {dy:+4d}) px, "
                  f"overlap after shift = {overlap*100:.1f}%, "
                  f"centroid shift = ({cs[0]:+.1f}, {cs[1]:+.1f})")

    # Summary
    centroid_shifts = np.array(centroid_shifts)
    print(f"\n--- Centroid Shift Summary (mask_center - proj_center) ---")
    print(f"  Mean:   ({centroid_shifts[:, 0].mean():+.1f}, "
          f"{centroid_shifts[:, 1].mean():+.1f}) px")
    print(f"  Std:    ({centroid_shifts[:, 0].std():.1f}, "
          f"{centroid_shifts[:, 1].std():.1f}) px")
    print(f"  Median: ({np.median(centroid_shifts[:, 0]):+.1f}, "
          f"{np.median(centroid_shifts[:, 1]):+.1f}) px")

    if shifts:
        shifts_arr = np.array([(s[1], s[2]) for s in shifts])
        print(f"\n--- Optimal 2D Shift (first 10 views) ---")
        print(f"  Mean:   ({shifts_arr[:, 0].mean():+.1f}, "
              f"{shifts_arr[:, 1].mean():+.1f}) px")
        print(f"  Individual: {[(s[1], s[2]) for s in shifts]}")

    # Check: is the shift view-dependent (3D offset) or constant (2D offset)?
    print(f"\n--- View-Dependence Analysis ---")
    if centroid_shifts[:, 0].std() > 5 or centroid_shifts[:, 1].std() > 5:
        print("  Shift VARIES across views → likely a 3D translation offset")
    else:
        print("  Shift is CONSTANT across views → likely a 2D offset "
              "(principal point or image origin)")

    # Save visualization of first view with optimal shift applied
    if shifts:
        cam = cameras[shifts[0][0]]
        mask_path = find_mask_for_camera(cam, MASK_DIR)
        mask = load_mask(mask_path)
        pts_cam = cam.world_to_camera(vertices)
        in_front = pts_cam[:, 2] > 0.1
        uv = cam.project_points(vertices[in_front])
        in_image = (
            (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) &
            (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
        )
        uv_valid = uv[in_image]
        dx, dy = shifts[0][1], shifts[0][2]

        scale = 0.25
        h_vis = int(cam.height * scale)
        w_vis = int(cam.width * scale)
        vis = np.zeros((h_vis, w_vis, 3), dtype=np.uint8)

        mask_small = cv2.resize(mask * 255, (w_vis, h_vis))
        contours, _ = cv2.findContours(
            mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

        # Draw shifted points
        uv_shifted = uv_valid + np.array([dx, dy])
        uv_s = uv_shifted * scale
        for j in range(len(uv_s)):
            x, y = int(uv_s[j, 0]), int(uv_s[j, 1])
            if 0 <= x < w_vis and 0 <= y < h_vis:
                cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)

        cv2.putText(vis, f"After shift ({dx:+d}, {dy:+d}) px",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        save_path = os.path.join(OUTPUT_DIR, "proj_000_shifted.png")
        cv2.imwrite(save_path, vis)
        print(f"\n  Saved shifted visualization: {save_path}")


if __name__ == "__main__":
    main()
