"""RNb format dataloader (cameras.npz + normal/albedo/mask folders).

Expects::

    data_dir/
        cameras.npz
        normal/   000.png or 0000.png
        albedo/   (optional)
        mask/
"""

import os

import cv2
import numpy as np

from .base import BaseDataLoader


def load_K_Rt_from_P(P):
    """Decompose projection matrix into intrinsics and camera-to-world."""
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


class RnbDataLoader(BaseDataLoader):
    """Load from the RNb cameras.npz format.

    Args:
        data_dir: Directory containing cameras.npz and image folders.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load(self):
        npz_path = os.path.join(self.data_dir, "cameras.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                "cameras.npz not found in {}".format(self.data_dir))

        camera_dict = np.load(npz_path)

        n_images = max(
            int(k.split("_")[-1]) for k in camera_dict.keys()
        ) + 1

        # Detect naming convention
        normal_dir = os.path.join(self.data_dir, "normal")
        if not os.path.isdir(normal_dir):
            raise FileNotFoundError(
                "normal/ folder not found in {}".format(self.data_dir))
        first_img = sorted(os.listdir(normal_dir))[0]
        n_digits = len(first_img.split(".")[0])

        sample = cv2.imread(os.path.join(normal_dir, first_img))
        image_height, image_width = sample.shape[:2]

        albedo_dir = os.path.join(self.data_dir, "albedo")
        has_albedo = os.path.isdir(albedo_dir)
        mask_dir = os.path.join(self.data_dir, "mask")

        scale_mat_0 = camera_dict["scale_mat_0"].astype(np.float32)

        views = []
        for i in range(n_images):
            world_mat = camera_dict["world_mat_{}".format(i)].astype(
                np.float32)
            scale_mat = camera_dict["scale_mat_{}".format(i)].astype(
                np.float32)

            P = (world_mat @ scale_mat)[:3, :4]
            K, c2w = load_K_Rt_from_P(P)

            filename = "{:0{n}d}.png".format(i, n=n_digits)

            normal_path = os.path.join(normal_dir, filename)
            albedo_path = (
                os.path.join(albedo_dir, filename) if has_albedo else None
            )
            mask_path = os.path.join(mask_dir, filename)

            views.append({
                "c2w": c2w,
                "K": K.astype(np.float32),
                "normal_path": normal_path,
                "albedo_path": albedo_path,
                "mask_path": mask_path if os.path.exists(mask_path) else None,
                "pose_id": str(i),
            })

        return {
            "views": views,
            "landmarks": None,
            "image_width": image_width,
            "image_height": image_height,
            "scale_mat": scale_mat_0,
        }
