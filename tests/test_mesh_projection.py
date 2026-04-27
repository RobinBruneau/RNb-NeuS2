"""Test mesh-to-mask projection alignment.

Projects RNb-NeuS2 mesh vertices onto camera views using pyalicevisionlib
conventions and checks that 99%+ of visible projected vertices fall within
the corresponding masks.

The mesh (mesh.obj) is in WORLD_CORRECTION-applied world space (output of
marching cubes with n2w transform). Cameras from sfm.json are loaded with
apply_world_correction=True (default), putting them in the same space.

Projection follows pyalicevisionlib exactly:
    P_cam = R_w2c @ (P_world - center)
    u = fx * P_cam.x / P_cam.z + cx
    v = fy * P_cam.y / P_cam.z + cy
"""

import json
import os
import sys

import cv2
import numpy as np
import trimesh

# Use pyalicevisionlib for camera loading (the reference implementation)
from pyalicevisionlib import load_sfmdata, Camera


# ── Paths ──────────────────────────────────────────────────────────────
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
OUTPUT_DIR = "/tmp/mesh_projection_debug"


def load_mesh_vertices(mesh_path, max_vertices=50000):
    """Load mesh and sample vertices (subsample for speed)."""
    mesh = trimesh.load(mesh_path, process=False)
    verts = np.array(mesh.vertices, dtype=np.float64)
    if len(verts) > max_vertices:
        idx = np.random.default_rng(42).choice(len(verts), max_vertices, replace=False)
        verts = verts[idx]
    return verts


def load_mask(mask_path, width, height):
    """Load a binary mask image."""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    if mask.dtype == np.float32:
        return mask > 0.5
    threshold = 125 if mask.dtype == np.uint8 else 30000
    return mask > threshold


def find_mask_for_camera(cam, mask_dir):
    """Find mask file matching camera view_id or pose_id."""
    # Masks are typically named by the image filename stem
    if cam.image_path:
        stem = os.path.splitext(os.path.basename(cam.image_path))[0]
        for ext in (".png", ".jpg", ".exr"):
            candidate = os.path.join(mask_dir, stem + ext)
            if os.path.exists(candidate):
                return candidate
    # Fallback: try view_id
    for ext in (".png", ".jpg"):
        candidate = os.path.join(mask_dir, cam.view_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def project_and_check(cam, vertices, mask, save_path=None):
    """Project vertices and compute mask overlap.

    Returns dict with stats:
        n_in_front: vertices with z > 0
        n_in_image: vertices within image bounds
        n_in_mask: vertices within mask foreground
        pct_in_mask: n_in_mask / n_in_image (the key metric)
    """
    # Transform to camera coordinates: P_cam = R_w2c @ (P_world - C)
    pts_cam = cam.world_to_camera(vertices)

    # Keep only points in front of camera
    in_front = pts_cam[:, 2] > 0.1
    pts_front = vertices[in_front]

    if len(pts_front) == 0:
        return {"n_in_front": 0, "n_in_image": 0, "n_in_mask": 0,
                "pct_in_mask": 0.0}

    # Project to 2D
    uv = cam.project_points(pts_front)

    # Filter to image bounds
    in_image = (
        (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    )
    uv_valid = uv[in_image]

    if len(uv_valid) == 0:
        return {"n_in_front": int(in_front.sum()), "n_in_image": 0,
                "n_in_mask": 0, "pct_in_mask": 0.0}

    # Check mask overlap
    u_px = np.clip(uv_valid[:, 0].astype(int), 0, cam.width - 1)
    v_px = np.clip(uv_valid[:, 1].astype(int), 0, cam.height - 1)
    in_mask = mask[v_px, u_px]
    n_in_mask = int(in_mask.sum())
    pct = n_in_mask / len(uv_valid) * 100.0

    # Save debug visualization
    if save_path is not None:
        scale = 0.25  # Downsample for visualization
        h_vis = int(cam.height * scale)
        w_vis = int(cam.width * scale)
        vis = np.zeros((h_vis, w_vis, 3), dtype=np.uint8)

        # Draw mask outline
        mask_small = cv2.resize(mask.astype(np.uint8) * 255,
                                (w_vis, h_vis))
        contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

        # Draw projected points
        uv_s = uv_valid * scale
        for i in range(len(uv_s)):
            x, y = int(uv_s[i, 0]), int(uv_s[i, 1])
            if 0 <= x < w_vis and 0 <= y < h_vis:
                color = (0, 255, 0) if in_mask[i] else (0, 0, 255)
                cv2.circle(vis, (x, y), 1, color, -1)

        # Add text
        cv2.putText(vis, f"{pct:.1f}% in mask ({n_in_mask}/{len(uv_valid)})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imwrite(save_path, vis)

    return {
        "n_in_front": int(in_front.sum()),
        "n_in_image": int(in_image.sum()),
        "n_in_mask": n_in_mask,
        "pct_in_mask": pct,
    }


def run_diagnostic():
    """Main diagnostic: project mesh onto all views and report alignment."""
    print("=" * 70)
    print("MESH-TO-MASK PROJECTION ALIGNMENT DIAGNOSTIC")
    print("=" * 70)

    # Load mesh
    print(f"\nLoading mesh: {MESH_PATH}")
    vertices = load_mesh_vertices(MESH_PATH, max_vertices=50000)
    print(f"  Sampled {len(vertices)} vertices")
    print(f"  Bounds: min={vertices.min(axis=0)}, max={vertices.max(axis=0)}")
    print(f"  Center: {vertices.mean(axis=0)}")

    # Load cameras via pyalicevisionlib (with world correction = default)
    print(f"\nLoading cameras from: {SFM_PATH}")
    sfmdata = load_sfmdata(SFM_PATH)
    cameras = sfmdata.get_cameras(apply_world_correction=True)
    print(f"  Loaded {len(cameras)} cameras")

    if cameras:
        cam0 = cameras[0]
        print(f"  Image size: {cam0.width}x{cam0.height}")
        print(f"  Focal (px): {cam0.focal_length_pixels:.2f}")
        print(f"  Principal point: ({cam0.cx:.2f}, {cam0.cy:.2f})")
        print(f"  First camera center: {cam0.center}")

    # Load n2w from transform.json for reference
    print(f"\nLoading transform.json: {TRANSFORM_JSON}")
    with open(TRANSFORM_JSON) as f:
        transform = json.load(f)
    n2w = np.array(transform["n2w"])
    print(f"  n2w scale: {n2w[0, 0]:.4f}")
    print(f"  n2w translation: [{n2w[0, 3]:.4f}, {n2w[1, 3]:.4f}, {n2w[2, 3]:.4f}]")

    # Compare camera centers: pyalicevisionlib vs transform.json
    print("\n--- Camera Center Comparison (first 3) ---")
    for i, frame in enumerate(transform["frames"][:3]):
        c2w_tj = np.array(frame["transform_matrix"])
        # transform.json centers are in normalized (scaled) space
        center_tj_normalized = c2w_tj[:3, 3]
        # Convert back to world via n2w
        center_tj_world = n2w[0, 0] * center_tj_normalized + n2w[:3, 3]

        if i < len(cameras):
            cam = cameras[i]
            print(f"  View {i}: pyav center = {cam.center}")
            print(f"           tj→world    = {center_tj_world}")
            diff = np.linalg.norm(cam.center - center_tj_world)
            print(f"           diff = {diff:.6f}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Project onto each camera
    print("\n--- Projection Results ---")
    results = []
    for i, cam in enumerate(cameras):
        mask_path = find_mask_for_camera(cam, MASK_DIR)
        if mask_path is None:
            continue

        mask = load_mask(mask_path, cam.width, cam.height)

        # Save visualization for first 5 cameras
        save_path = None
        if i < 5:
            save_path = os.path.join(OUTPUT_DIR, f"proj_{i:03d}.png")

        stats = project_and_check(cam, vertices, mask, save_path)

        if stats["n_in_image"] > 0:
            results.append(stats)
            status = "OK" if stats["pct_in_mask"] >= 99.0 else "FAIL"
            print(f"  [{status}] View {i:3d}: {stats['pct_in_mask']:6.2f}% "
                  f"in mask ({stats['n_in_mask']}/{stats['n_in_image']})")

    if not results:
        print("  ERROR: No views with projected points!")
        return

    # Summary
    pcts = [r["pct_in_mask"] for r in results]
    print(f"\n--- Summary ({len(results)} views) ---")
    print(f"  Mean:   {np.mean(pcts):.2f}%")
    print(f"  Median: {np.median(pcts):.2f}%")
    print(f"  Min:    {np.min(pcts):.2f}%")
    print(f"  Max:    {np.max(pcts):.2f}%")
    print(f"  Std:    {np.std(pcts):.2f}%")
    n_pass = sum(1 for p in pcts if p >= 99.0)
    print(f"  Views >= 99%: {n_pass}/{len(results)}")
    print(f"\n  Debug visualizations saved to: {OUTPUT_DIR}")

    if np.mean(pcts) < 99.0:
        print("\n  >>> MISALIGNMENT DETECTED <<<")
        print("  Investigating potential causes...")
        _investigate_misalignment(cameras, vertices, transform)


def _investigate_misalignment(cameras, vertices, transform):
    """Try to identify the source of misalignment."""
    n2w = np.array(transform["n2w"])

    # Check 1: Is it a pure translation offset?
    print("\n  [Check 1] Pure translation offset test")
    print("    Mesh centroid:", vertices.mean(axis=0))
    cam_centers = np.array([c.center for c in cameras])
    print("    Camera centroid:", cam_centers.mean(axis=0))

    # Check 2: Compare RNb-NeuS2 vs pyalicevisionlib camera params
    print("\n  [Check 2] Camera parameter comparison")
    # Load sfm.json directly to compare with pyalicevisionlib
    with open(SFM_PATH) as f:
        sfm_data = json.load(f)

    # Build pose lookup
    poses = {p["poseId"]: p["pose"]["transform"]
             for p in sfm_data.get("poses", [])}
    intrinsics = {i["intrinsicId"]: i
                  for i in sfm_data.get("intrinsics", [])}

    # Compare first view
    view0 = sfm_data["views"][0]
    pose_id = view0["poseId"]
    intr_id = view0["intrinsicId"]
    intr = intrinsics[intr_id]
    transform_data = poses[pose_id]

    # Raw rotation and center from sfm.json
    rot_flat = [float(r) for r in transform_data["rotation"]]
    R_raw = np.array(rot_flat).reshape(3, 3)
    center_raw = np.array([float(c) for c in transform_data["center"]])

    # WORLD_CORRECTION
    W = np.diag([1.0, -1.0, -1.0])
    R_corrected = W @ R_raw
    center_corrected = W @ center_raw

    # RNb-NeuS2 sfm_json_loader does the same thing
    # Check: does pyalicevisionlib give the same result?
    cam0 = cameras[0]
    print(f"    Raw center: {center_raw}")
    print(f"    Corrected center: {center_corrected}")
    print(f"    pyav center: {cam0.center}")
    print(f"    Center match: {np.allclose(center_corrected, cam0.center, atol=1e-3)}")

    print(f"    Corrected R:\n{R_corrected}")
    print(f"    pyav R:\n{cam0.rotation_cam2world}")
    print(f"    R match: {np.allclose(R_corrected, cam0.rotation_cam2world, atol=1e-6)}")

    # Check 3: principal point
    print("\n  [Check 3] Principal point")
    pp = intr.get("principalPoint", ["0", "0"])
    pp_x, pp_y = float(pp[0]), float(pp[1])
    width = int(intr["width"])
    height = int(intr["height"])
    cx_rnb = width / 2.0 + pp_x
    cy_rnb = height / 2.0 + pp_y
    print(f"    sfm.json PP offset: ({pp_x}, {pp_y})")
    print(f"    RNb cx,cy: ({cx_rnb}, {cy_rnb})")
    print(f"    pyav cx,cy: ({cam0.cx}, {cam0.cy})")
    print(f"    Match: cx={abs(cx_rnb - cam0.cx) < 0.01}, "
          f"cy={abs(cy_rnb - cam0.cy) < 0.01}")

    # Check 4: focal length
    print("\n  [Check 4] Focal length")
    if "pxFocalLength" in intr:
        fx_rnb = float(intr["pxFocalLength"])
    else:
        fx_rnb = float(intr["focalLength"]) * width / float(intr["sensorWidth"])
    print(f"    RNb fx: {fx_rnb}")
    print(f"    pyav fx: {cam0.focal_length_pixels}")
    print(f"    Match: {abs(fx_rnb - cam0.focal_length_pixels) < 0.01}")

    # Check 5: Try without WORLD_CORRECTION to see if mesh is in raw space
    print("\n  [Check 5] Projection WITHOUT world correction")
    # Project with raw (uncorrected) cameras
    cam0_raw = Camera(
        view_id=cam0.view_id,
        width=cam0.width, height=cam0.height,
        focal_length_mm=cam0.focal_length_mm,
        sensor_width=cam0.sensor_width,
        principal_point=cam0.principal_point.copy(),
        center=center_raw,
        rotation_cam2world=R_raw,
    )
    pts_cam_raw = cam0_raw.world_to_camera(vertices)
    in_front_raw = (pts_cam_raw[:, 2] > 0.1).sum()
    print(f"    Points in front (no correction): {in_front_raw}/{len(vertices)}")

    pts_cam_corr = cam0.world_to_camera(vertices)
    in_front_corr = (pts_cam_corr[:, 2] > 0.1).sum()
    print(f"    Points in front (with correction): {in_front_corr}/{len(vertices)}")


if __name__ == "__main__":
    run_diagnostic()
