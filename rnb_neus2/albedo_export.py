"""Export of the harmonized albedo maps for texturing.

After the albedo scaling, the input albedo maps are written once more with the per-view scales used for training
(phase 3), for the AliceVision Texturing node:

- as linear float EXR named <viewId>.exr (viewId of the albedo SfMData view): Texturing reads EXR as linear data,
  whereas it reads untagged 8/16-bit PNG albedos as sRGB; the values are not clipped;
- extended a few pixels outside the masks, so that texels on the silhouettes do not pick up the black background;
- with a copy of the albedo SfMData whose views point to them (only the exported views are kept, Texturing needs
  an image for every view with a pose).
"""

import json
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

from .image_io import load_image, save_exr

BLEED_PIXELS = 8


def frame_view_ids(frames_info, names):
    """Albedo SfMData viewId of each albedo image name ("00003.png" -> viewId), None if unknown.

    Args:
        frames_info: list of dicts with "name" (image stem) and "albedo_view_id", from prepare_testbed_data.
        names: albedo image names (e.g. list_albedo_names of the prepared albedo folder).
    """
    by_stem = {frame["name"]: frame.get("albedo_view_id") for frame in frames_info}
    return [by_stem.get(os.path.splitext(name)[0]) for name in names]


def bleed(image, mask, pixels):
    """Extend image values `pixels` px outside the mask (mean of the known 8-neighbours, ring by ring); the other
    pixels outside the mask are set to 0 and the pixels inside are unchanged."""
    known = mask.astype(np.float32)
    out = (image * known[:, :, None]).astype(np.float32)
    kernel = np.ones((3, 3), np.float32)
    for _ in range(pixels):
        total = cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        count = cv2.filter2D(known, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        grow = (known == 0) & (count > 0)
        if not grow.any():
            break
        out[grow] = total[grow] / count[grow][:, None]
        known[grow] = 1.0
    return out


def read_mask(mask_path, shape):
    """Boolean mask (same thresholds as prepare_testbed_data), resized to shape (h, w); None if no mask."""
    if not mask_path or not os.path.exists(mask_path):
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.dtype == np.float32:
        binary = mask > 0.5
    else:
        binary = mask > (125 if mask.dtype == np.uint8 else 30000)
    if binary.shape != tuple(shape):
        binary = cv2.resize(binary.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return binary


def is_json_sfm(path):
    """True if path is a JSON SfMData (the harmonized albedo SfMData can only be written from one)."""
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def write_sfm_copy(albedo_sfm_path, new_paths, output_sfm_path):
    """Copy of the albedo SfMData (JSON) keeping only the views of new_paths (viewId -> image path), pointing to
    them."""
    try:
        with open(albedo_sfm_path) as f:
            sfm = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("The albedo SfMData '{}' is not a JSON SfMData (.sfm/.json): the harmonized albedo "
                           "SfMData cannot be written".format(albedo_sfm_path))
    sfm["views"] = [view for view in sfm["views"] if str(view["viewId"]) in new_paths]
    for view in sfm["views"]:
        view["path"] = new_paths[str(view["viewId"])]
    with open(output_sfm_path, "w") as f:
        json.dump(sfm, f, indent=4)


def export_harmonized_albedos(albedo_sfm_path, frames_info, scales_by_name, output_folder, output_sfm_path,
                              bleed_pixels=BLEED_PIXELS, logger=None):
    """Write the input albedo maps scaled by the training scales, for texturing.

    Args:
        albedo_sfm_path: the input albedo SfMData.
        frames_info: list of dicts with "name", "albedo_view_id", "albedo_path", "mask_path" (prepare_testbed_data).
        scales_by_name: image stem ("00003") -> RGB scale used for training.
        output_folder: receives <viewId>.exr.
        output_sfm_path: copy of the albedo SfMData pointing to the exported maps.

    Returns:
        dict viewId -> exported path.
    """
    def log(msg):
        if logger:
            logger.info(msg)

    os.makedirs(output_folder, exist_ok=True)
    new_paths = {}
    for frame in frames_info:
        view_id, name = frame.get("albedo_view_id"), frame["name"]
        if view_id is None or not frame.get("albedo_path") or name not in scales_by_name:
            continue
        albedo = load_image(frame["albedo_path"])
        if albedo.ndim == 2:
            albedo = np.repeat(albedo[:, :, None], 3, axis=2)
        albedo = albedo[:, :, :3]
        mask = read_mask(frame.get("mask_path"), albedo.shape[:2])
        if mask is None:
            mask = np.any(albedo > 0, axis=2)
        scaled = albedo * np.asarray(scales_by_name[name], np.float32)[None, None, :]
        path = os.path.join(os.path.abspath(output_folder), "{}.exr".format(view_id))
        save_exr(bleed(scaled, mask, bleed_pixels), path)
        if not os.path.exists(path):
            raise RuntimeError("Cannot write '{}'".format(path))
        new_paths[str(view_id)] = path
    if not new_paths:
        raise RuntimeError("No harmonized albedo map to export")
    write_sfm_copy(albedo_sfm_path, new_paths, output_sfm_path)
    log("Exported {} harmonized albedo maps to {} and {}".format(len(new_paths), output_folder, output_sfm_path))
    return new_paths
