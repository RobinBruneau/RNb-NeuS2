"""Pure-JSON SfMData loader (no pyalicevision dependency).

Parses the AliceVision SfMData JSON format directly. Supports both
.sfm and .json files. Ported from Open-RNb datasets/sfm.py.
"""

import json
import os
import warnings

import numpy as np

from .base import BaseDataLoader

# Y/Z flip: AliceVision Y-down/Z-forward -> Y-up world
WORLD_CORRECTION = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _resolve_path(path, sfm_dir):
    """Resolve relative paths against sfm_dir."""
    if not path or os.path.isabs(path) or sfm_dir is None:
        return path
    return os.path.join(sfm_dir, path)


def parse_sfm_json(data, sfm_dir=None):
    """Parse SfMData JSON dict into standardized camera list + landmarks.

    Returns:
        cameras: list of dicts with view_id, pose_id, image_path,
            R_cam2world, center, fx, fy, cx, cy, width, height
        landmarks: (N, 3) array or None
    """
    intrinsics = {
        i["intrinsicId"]: i for i in data.get("intrinsics", [])
    }
    poses = {
        p["poseId"]: p["pose"]["transform"]
        for p in data.get("poses", [])
    }

    cameras = []
    for view in data.get("views", []):
        view_id = view["viewId"]
        intr_id = view["intrinsicId"]
        pose_id = view["poseId"]

        if intr_id not in intrinsics or pose_id not in poses:
            continue

        intr = intrinsics[intr_id]
        transform = poses[pose_id]

        width = int(intr["width"])
        height = int(intr["height"])

        # Focal length
        if "pxFocalLength" in intr:
            pxf = intr["pxFocalLength"]
            if isinstance(pxf, list):
                fx, fy = float(pxf[0]), float(pxf[1])
            else:
                fx = fy = float(pxf)
        else:
            focal_mm = float(intr["focalLength"])
            sensor_width = float(intr.get("sensorWidth", 36.0))
            if "sensorWidth" not in intr:
                warnings.warn(
                    "sensorWidth not found, using default 36.0mm")
            fx = fy = focal_mm * width / sensor_width

        # Principal point (offset from image center)
        pp = intr.get("principalPoint", ["0", "0"])
        cx = width / 2.0 + float(pp[0])
        cy = height / 2.0 + float(pp[1])

        # Rotation cam2world (row-major flat)
        rotation_flat = [float(r) for r in transform["rotation"]]
        R_cam2world = np.array(rotation_flat).reshape(3, 3)

        center = np.array([float(c) for c in transform["center"]])

        # Apply world correction
        R_cam2world = WORLD_CORRECTION @ R_cam2world
        center = WORLD_CORRECTION @ center

        cameras.append({
            "view_id": view_id,
            "pose_id": pose_id,
            "image_path": _resolve_path(view.get("path", ""), sfm_dir),
            "R_cam2world": R_cam2world,
            "center": center,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": width, "height": height,
        })

    # Landmarks
    landmarks = None
    structure = data.get("structure", [])
    if structure:
        pts = []
        for s in structure:
            coord = s.get("X", None)
            if coord is not None:
                pts.append([float(coord[0]), float(coord[1]),
                            float(coord[2])])
        if pts:
            landmarks = (WORLD_CORRECTION @ np.array(pts).T).T

    return cameras, landmarks


class SfmJsonDataLoader(BaseDataLoader):
    """Load SfMData from a JSON file without pyalicevision.

    Args:
        sfm_path: Path to the .sfm or .json file.
        normal_sfm_path: Path to the normal maps SfMData (same or different).
        albedo_sfm_path: Path to the albedo maps SfMData (optional).
        mask_sfm_path: Path to the mask SfMData (optional).
        mask_folder_path: Path to a folder with mask images (optional).
    """

    def __init__(self, sfm_path, normal_sfm_path=None,
                 albedo_sfm_path="", mask_sfm_path="",
                 mask_folder_path=""):
        self.sfm_path = sfm_path
        self.normal_sfm_path = normal_sfm_path or sfm_path
        self.albedo_sfm_path = albedo_sfm_path
        self.mask_sfm_path = mask_sfm_path
        self.mask_folder_path = mask_folder_path

    def load(self):
        sfm_dir = os.path.dirname(os.path.abspath(self.normal_sfm_path))

        with open(self.normal_sfm_path, "r") as f:
            normal_data = json.load(f)
        normal_cams, landmarks = parse_sfm_json(normal_data, sfm_dir)

        if not normal_cams:
            raise RuntimeError(
                "No valid views in {}".format(self.normal_sfm_path))

        # Albedo SfMData
        albedo_by_pose = {}
        if self.albedo_sfm_path and os.path.exists(self.albedo_sfm_path):
            a_dir = os.path.dirname(os.path.abspath(self.albedo_sfm_path))
            with open(self.albedo_sfm_path, "r") as f:
                albedo_data = json.load(f)
            albedo_cams, _ = parse_sfm_json(albedo_data, a_dir)
            albedo_by_pose = {c["pose_id"]: c for c in albedo_cams}

        # Mask SfMData
        mask_by_pose = {}
        if self.mask_sfm_path and os.path.exists(self.mask_sfm_path):
            m_dir = os.path.dirname(os.path.abspath(self.mask_sfm_path))
            with open(self.mask_sfm_path, "r") as f:
                mask_data = json.load(f)
            mask_cams, _ = parse_sfm_json(mask_data, m_dir)
            mask_by_pose = {c["pose_id"]: c for c in mask_cams}

        first = normal_cams[0]
        image_width = first["width"]
        image_height = first["height"]

        views = []
        for cam in normal_cams:
            # Build c2w (4x4)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = cam["R_cam2world"]
            c2w[:3, 3] = cam["center"]

            # Build K (4x4)
            K = np.eye(4, dtype=np.float32)
            K[0, 0] = cam["fx"]
            K[1, 1] = cam["fy"]
            K[0, 2] = cam["cx"]
            K[1, 2] = cam["cy"]

            pose_id = cam["pose_id"]

            # Albedo path
            albedo_path = None
            if pose_id in albedo_by_pose:
                albedo_path = albedo_by_pose[pose_id]["image_path"]

            # Mask path
            mask_path = None
            if pose_id in mask_by_pose:
                mask_path = mask_by_pose[pose_id]["image_path"]
            elif self.mask_folder_path and os.path.isdir(
                    self.mask_folder_path):
                for ext in (".png", ".jpg", ".jpeg", ".exr"):
                    candidate = os.path.join(
                        self.mask_folder_path,
                        "{}{}".format(pose_id, ext))
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break

            views.append({
                "c2w": c2w,
                "K": K,
                "normal_path": cam["image_path"],
                "albedo_path": albedo_path,
                "mask_path": mask_path,
                "pose_id": pose_id,
            })

        return {
            "views": views,
            "landmarks": landmarks,
            "image_width": image_width,
            "image_height": image_height,
            "scale_mat": None,
        }
