"""Base dataloader interface for RNb-NeuS2.

All dataloaders return a standardized dict:

    {
        "views": [
            {
                "c2w": np.array (4, 4),      camera-to-world, Y/Z-flipped
                "K": np.array (4, 4),         intrinsic matrix, NeuS2 pixel convention
                "normal_path": str,           path to normal image
                "albedo_path": str or None,   path to albedo image
                "mask_path": str or None,     path to mask image
                "pose_id": str,               unique identifier
            },
            ...
        ],
        "landmarks": np.array (N, 3) or None,   Y/Z-flipped 3D points
        "image_width": int,
        "image_height": int,
        "scale_mat": np.array (4, 4) or None,   RNb format only
    }
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
