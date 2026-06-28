"""Dataloader factory for RNb-NeuS2.

Auto-detects input format and returns the appropriate loader.
"""

import os

from .base import BaseDataLoader
from .rnb_loader import RnbDataLoader
from .sfm_json_loader import SfmJsonDataLoader


def create_loader(input_path, **kwargs):
    """Create the appropriate dataloader based on input path.

    Args:
        input_path: Path to input data. Can be:
            - A directory containing cameras.npz -> RnbDataLoader
            - A .npz file -> RnbDataLoader (uses parent dir)
            - A .sfm or .abc file -> SfmPyavDataLoader (fallback to JSON)
            - A .json file -> SfmJsonDataLoader
        **kwargs: Extra arguments forwarded to the loader constructor
            (e.g. albedo_sfm_path, mask_sfm_path, mask_folder_path).

    Returns:
        BaseDataLoader instance.
    """
    # Directory with cameras.npz
    if os.path.isdir(input_path):
        if os.path.exists(os.path.join(input_path, "cameras.npz")):
            return RnbDataLoader(input_path)
        raise FileNotFoundError(
            "No cameras.npz found in {}. "
            "Provide a .sfm or .json file instead.".format(input_path))

    ext = os.path.splitext(input_path)[1].lower()

    # cameras.npz file directly
    if ext == ".npz":
        return RnbDataLoader(os.path.dirname(input_path))

    # SfMData via pyalicevision (with JSON fallback)
    if ext in (".sfm", ".abc"):
        try:
            from .sfm_pyav_loader import SfmPyavDataLoader
            return SfmPyavDataLoader(
                normal_sfm_path=input_path,
                albedo_sfm_path=kwargs.get("albedo_sfm_path", ""),
                mask_sfm_path=kwargs.get("mask_sfm_path", ""),
                mask_folder_path=kwargs.get("mask_folder_path", ""),
                logger=kwargs.get("logger"),
            )
        except ImportError:
            # Fall through to JSON loader
            pass

    # Pure JSON
    if ext in (".json", ".sfm"):
        return SfmJsonDataLoader(
            sfm_path=input_path,
            normal_sfm_path=input_path,
            albedo_sfm_path=kwargs.get("albedo_sfm_path", ""),
            mask_sfm_path=kwargs.get("mask_sfm_path", ""),
            mask_folder_path=kwargs.get("mask_folder_path", ""),
        )

    raise ValueError(
        "Unsupported input format: {}. "
        "Supported: directory with cameras.npz, .npz, .sfm, .abc, .json"
        .format(ext))


def load_data(input_path, **kwargs):
    """Convenience: create loader and call load().

    Returns the standardized data dict.
    """
    loader = create_loader(input_path, **kwargs)
    return loader.load()
