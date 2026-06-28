"""SfMData loader using pyalicevision C++ bindings.

Falls back gracefully if pyalicevision is not installed — use the
factory in __init__.py which handles the fallback to sfm_json_loader.
"""

import os

import numpy as np

from .base import BaseDataLoader

# Y/Z flip: AliceVision Y-down/Z-forward -> Y-up world
FLIP_YZ = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float32)


def _extract_intrinsics(intrinsic, camera_module, numeric_module):
    """Extract 4x4 intrinsic matrix from a pyalicevision intrinsic."""
    K = np.eye(4, dtype=np.float32)

    cam = camera_module.Pinhole.cast(intrinsic)
    if cam is not None:
        try:
            K[0, 0] = cam.getFocalLengthPixX()
            K[1, 1] = cam.getFocalLengthPixY()
        except AttributeError:
            scale = intrinsic.getScale()
            K[0, 0] = scale.x()
            K[1, 1] = scale.y()
        pp = cam.getPrincipalPoint()
        K[0, 2] = numeric_module.getX(pp)
        K[1, 2] = numeric_module.getY(pp)
    else:
        scale = intrinsic.getScale()
        offset = intrinsic.getOffset()
        K[0, 0] = scale.x()
        K[1, 1] = scale.y()
        K[0, 2] = offset.x()
        K[1, 2] = offset.y()

    return K


class SfmPyavDataLoader(BaseDataLoader):
    """Load SfMData via pyalicevision bindings.

    Args:
        normal_sfm_path: Path to the normal maps SfMData file.
        albedo_sfm_path: Path to albedo SfMData (optional).
        mask_sfm_path: Path to mask SfMData (optional).
        mask_folder_path: Folder with mask images (optional).
        logger: Optional logger.
    """

    def __init__(self, normal_sfm_path, albedo_sfm_path="",
                 mask_sfm_path="", mask_folder_path="", logger=None):
        self.normal_sfm_path = normal_sfm_path
        self.albedo_sfm_path = albedo_sfm_path
        self.mask_sfm_path = mask_sfm_path
        self.mask_folder_path = mask_folder_path
        self.logger = logger

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def load(self):
        from pyalicevision import (
            sfmData as sfmDataModule, sfmDataIO, camera, numeric,
        )

        # Load normal SfMData
        normal_sfm = sfmDataModule.SfMData()
        if not sfmDataIO.load(
                normal_sfm, self.normal_sfm_path, sfmDataIO.ALL):
            raise RuntimeError(
                "Failed to load normal SfMData: {}".format(
                    self.normal_sfm_path))
        self._log("Loaded normal SfMData: {}".format(self.normal_sfm_path))

        # Load albedo SfMData
        albedo_sfm = None
        if self.albedo_sfm_path and os.path.exists(self.albedo_sfm_path):
            albedo_sfm = sfmDataModule.SfMData()
            if not sfmDataIO.load(
                    albedo_sfm, self.albedo_sfm_path, sfmDataIO.ALL):
                self._log("Failed to load albedo SfMData, using white")
                albedo_sfm = None
            else:
                self._log("Loaded albedo SfMData: {}".format(
                    self.albedo_sfm_path))

        # Load mask SfMData
        mask_sfm = None
        if self.mask_sfm_path and os.path.exists(self.mask_sfm_path):
            mask_sfm = sfmDataModule.SfMData()
            if not sfmDataIO.load(
                    mask_sfm, self.mask_sfm_path, sfmDataIO.ALL):
                self._log("Failed to load mask SfMData")
                mask_sfm = None
            else:
                self._log("Loaded mask SfMData: {}".format(
                    self.mask_sfm_path))

        views = normal_sfm.getViews()
        if len(views) == 0:
            raise RuntimeError("No views in normal SfMData")

        # Extract landmarks
        landmarks = self._extract_landmarks(normal_sfm)

        # Representative views: viewId == poseId
        pose_ids = sorted([
            vid for vid in views.keys()
            if vid == views[vid].getPoseId()
        ])

        result_views = []
        image_width = None
        image_height = None

        for pose_id in pose_ids:
            view = views[pose_id]

            if not normal_sfm.isPoseAndIntrinsicDefined(pose_id):
                continue

            # Camera pose
            pose = normal_sfm.getPose(view)
            transform = pose.getTransform()
            R = transform.rotation()
            center = transform.center().squeeze()

            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R.transpose()
            c2w[:3, 3] = [center[0], center[1], center[2]]
            c2w = FLIP_YZ @ c2w

            # Intrinsics
            intrinsic_id = view.getIntrinsicId()
            intrinsic = normal_sfm.getIntrinsics()[intrinsic_id]
            K = _extract_intrinsics(intrinsic, camera, numeric)

            # Image dimensions
            if image_width is None:
                image_width = view.getImage().getWidth()
                image_height = view.getImage().getHeight()

            # Normal path
            normal_path = view.getImage().getImagePath()

            # Albedo path
            albedo_path = None
            if albedo_sfm is not None:
                albedo_views = albedo_sfm.getViews()
                if pose_id in albedo_views:
                    albedo_view = albedo_views[pose_id]
                    albedo_path = albedo_view.getImage().getImagePath()

            # Mask path
            mask_path = self._find_mask_path(
                pose_id, mask_sfm, self.mask_folder_path)

            result_views.append({
                "c2w": c2w,
                "K": K,
                "normal_path": normal_path,
                "albedo_path": albedo_path,
                "mask_path": mask_path,
                "pose_id": str(pose_id),
            })

        if not result_views:
            raise RuntimeError("No valid views could be loaded")

        self._log("Loaded {} views".format(len(result_views)))

        # Store raw pyalicevision objects for scaling (silhouettes need them)
        return {
            "views": result_views,
            "landmarks": landmarks,
            "image_width": image_width,
            "image_height": image_height,
            "scale_mat": None,
            # Extra: keep pyav objects for silhouette scaling
            "_pyav_sfm": normal_sfm,
            "_pyav_mask_sfm": mask_sfm,
            "_pyav_camera_module": camera,
            "_pyav_numeric_module": numeric,
        }

    @staticmethod
    def _extract_landmarks(sfm_data):
        """Extract 3D landmarks with Y/Z flip."""
        flip = np.array([1, -1, -1], dtype=np.float32)
        landmarks = sfm_data.getLandmarks()
        if len(landmarks) == 0:
            return None
        points = []
        for lm_id in landmarks.keys():
            coord = landmarks[lm_id].X
            points.append([coord[0], coord[1], coord[2]])
        return np.array(points, dtype=np.float32) * flip

    @staticmethod
    def _find_mask_path(pose_id, mask_sfm, mask_folder_path):
        """Find mask image path from SfMData or folder."""
        if mask_sfm is not None:
            mask_views = mask_sfm.getViews()
            if pose_id in mask_views:
                mask_view = mask_views[pose_id]
                path = mask_view.getImage().getImagePath()
                if os.path.exists(path):
                    return path

        if mask_folder_path and os.path.isdir(mask_folder_path):
            for ext in (".png", ".jpg", ".jpeg", ".exr"):
                candidate = os.path.join(
                    mask_folder_path, "{}{}".format(pose_id, ext))
                if os.path.exists(candidate):
                    return candidate

        return None
