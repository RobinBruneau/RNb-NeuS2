"""Base dataloader interface for RNb-NeuS2.

All dataloaders return a standardized dict:

    {
        "views": [
            {
                "c2w": np.array (4, 4),      camera-to-world (OpenCV camera axes), in the pose frame
                "K": np.array (4, 4),         intrinsic matrix, NeuS2 pixel convention
                "normal_path": str,           path to normal image
                "albedo_path": str or None,   path to albedo image
                "albedo_view_id": str or None, viewId of the albedo SfMData view (SfMData loaders)
                "mask_path": str or None,     path to mask image
                "pose_id": str,               unique identifier
            },
            ...
        ],
        "landmarks": np.array (N, 3) or None,   3D points in the pose frame
        "image_width": int,
        "image_height": int,
        "scale_mat": np.array (4, 4) or None,   pose frame -> world (None: identity)
    }

Output frame: the exported mesh is in the WORLD frame of the input cameras, whatever the loader. "scale_mat" maps
the pose frame of "c2w" and "landmarks" to that world; prepare_testbed_data composes it with its own normalization
into the "n2w" of transform.json, which the testbed applies to the mesh. The world is:

- RNb / IDR (cameras.npz): the frame of world_mat. The poses come pre-normalized (P = world_mat @ scale_mat), so
  "scale_mat" is the IDR scale_mat;
- SfMData (.sfm, .abc, .json): the SfMData frame flipped in Y and Z (diag(1, -1, -1), as AliceVision meshes).
  The loaders flip the poses and the landmarks, so "scale_mat" is None.
"""

from abc import ABC, abstractmethod

# Pixel convention of "K": NeuS2 / instant-ngp put the center of pixel i at coordinate i + 0.5 (rays are cast
# through (pixel + 0.5), see testbed_nerf.cu), whereas AliceVision puts it at coordinate i. Dataloaders reading
# AliceVision cameras add this offset to the principal point, and code casting rays from pixel indices with "K"
# (scaling, albedo scaling) uses the pixel centers (index + 0.5).
ALICEVISION_TO_NEUS2_PIXEL_OFFSET = 0.5


class BaseDataLoader(ABC):
    @abstractmethod
    def load(self):
        """Load data and return standardized dict."""
        raise NotImplementedError
