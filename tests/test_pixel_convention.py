"""Pixel convention of the intrinsics read from AliceVision SfMData.

AliceVision puts the center of pixel i at coordinate i, NeuS2 at i + 0.5 (rays cast through pixel + 0.5): the
dataloaders must return K in the NeuS2 convention, i.e. principal point + 0.5.

Run: venv/bin/python -m pytest tests/test_pixel_convention.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rnb_neus2.dataloaders.base import ALICEVISION_TO_NEUS2_PIXEL_OFFSET  # noqa: E402
from rnb_neus2.dataloaders.sfm_json_loader import parse_sfm_json  # noqa: E402

WIDTH, HEIGHT, OFFSET = 4144, 2760, (10.25, -3.5)


def _sfm():
    return {
        "views": [{"viewId": "1", "poseId": "1", "intrinsicId": "2", "path": "/n/1.png"}],
        "intrinsics": [{"intrinsicId": "2", "type": "pinhole", "width": str(WIDTH), "height": str(HEIGHT),
                        "sensorWidth": "36", "sensorHeight": "24", "focalLength": "50",
                        "principalPoint": [str(OFFSET[0]), str(OFFSET[1])]}],
        "poses": [{"poseId": "1", "pose": {"transform": {"rotation": ["1", "0", "0", "0", "1", "0", "0", "0", "1"],
                                                         "center": ["0", "0", "0"]}}}],
    }


def test_offset_value():
    assert ALICEVISION_TO_NEUS2_PIXEL_OFFSET == 0.5


def test_json_loader_principal_point():
    cameras, _ = parse_sfm_json(_sfm())
    assert cameras[0]["cx"] == pytest.approx(WIDTH / 2 + OFFSET[0] + 0.5)
    assert cameras[0]["cy"] == pytest.approx(HEIGHT / 2 + OFFSET[1] + 0.5)


def test_pyalicevision_loader_principal_point():
    camera = pytest.importorskip("pyalicevision.camera")
    numeric = pytest.importorskip("pyalicevision.numeric")
    from rnb_neus2.dataloaders.sfm_pyav_loader import _extract_intrinsics
    focal = 50.0 * WIDTH / 36.0
    pinhole = camera.Pinhole(WIDTH, HEIGHT, focal, focal, OFFSET[0], OFFSET[1])
    K = _extract_intrinsics(pinhole, camera, numeric)
    cameras, _ = parse_sfm_json(_sfm())
    assert K[0, 2] == pytest.approx(WIDTH / 2 + OFFSET[0] + 0.5)
    assert K[1, 2] == pytest.approx(HEIGHT / 2 + OFFSET[1] + 0.5)
    assert K[0, 2] == pytest.approx(cameras[0]["cx"]) and K[1, 2] == pytest.approx(cameras[0]["cy"])


def test_pixel_center_projects_to_neus2_center():
    """A 3D point that AliceVision images at the center of pixel (i, j) must be at (i + 0.5, j + 0.5) with K."""
    cameras, _ = parse_sfm_json(_sfm())
    cam = cameras[0]
    i, j = 1000, 700
    # AliceVision: x = f * X / Z + pp (pp absolute, pixel i centered at i)
    ppx, ppy = WIDTH / 2 + OFFSET[0], HEIGHT / 2 + OFFSET[1]
    X = np.array([(i - ppx) / cam["fx"], (j - ppy) / cam["fy"], 1.0])
    u = cam["fx"] * X[0] / X[2] + cam["cx"]
    v = cam["fy"] * X[1] / X[2] + cam["cy"]
    assert u == pytest.approx(i + 0.5) and v == pytest.approx(j + 0.5)
