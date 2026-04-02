"""Base dataloader interface for RNb-NeuS2.

All dataloaders return a standardized dict:

    {
        "views": [
            {
                "c2w": np.array (4, 4),      camera-to-world, Y/Z-flipped
                "K": np.array (4, 4),         intrinsic matrix
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


class BaseDataLoader(ABC):
    @abstractmethod
    def load(self):
        """Load data and return standardized dict."""
        raise NotImplementedError
