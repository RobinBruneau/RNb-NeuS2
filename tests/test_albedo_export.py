"""Export of the harmonized albedo maps for texturing (rnb_neus2/albedo_export.py).

The testbed frames ("00003.png") must map back to the viewIds of the albedo SfMData, and the exported maps must be
the input albedos times the training scales, as linear float EXR named <viewId>.exr, bled outside the masks, with an
SfMData copy pointing to them. CPU only.

Run: venv/bin/python -m pytest tests/test_albedo_export.py
"""
import json
import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rnb_neus2 import albedo_export  # noqa: E402
from rnb_neus2.dataloaders.sfm_json_loader import SfmJsonDataLoader  # noqa: E402
from rnb_neus2.prepare import prepare_testbed_data  # noqa: E402

H, W = 12, 16


class _Log:
    def info(self, msg):
        pass

    def warning(self, msg):
        pass


def _sfm(views):
    """AliceVision JSON SfMData; views: list of (viewId, poseId, path)."""
    return {
        "version": ["1", "2", "8"],
        "views": [{"viewId": v, "poseId": p, "intrinsicId": "9", "path": path, "width": str(W), "height": str(H)}
                  for v, p, path in views],
        "intrinsics": [{"intrinsicId": "9", "type": "pinhole", "width": str(W), "height": str(H),
                        "sensorWidth": "36", "sensorHeight": "27", "focalLength": "50",
                        "principalPoint": ["0", "0"]}],
        "poses": [{"poseId": p, "pose": {"transform": {"rotation": ["1", "0", "0", "0", "1", "0", "0", "0", "1"],
                                                       "center": ["0", "0", str(-3 - k)]}}}
                  for k, (_, p, _) in enumerate(views)],
    }


def _write_png16(path, rgb):
    cv2.imwrite(str(path), np.round(np.clip(rgb, 0, 1) * 65535).astype(np.uint16)[:, :, ::-1])


def test_frame_view_ids():
    frames = [{"name": "00000", "albedo_view_id": "11"}, {"name": "00002", "albedo_view_id": "33"},
              {"name": "00003", "albedo_view_id": None}]
    assert albedo_export.frame_view_ids(frames, ["00000.png", "00002.png", "00003.png", "00009.png"]) == \
        ["11", "33", None, None]


def test_prepare_maps_frames_to_albedo_view_ids(tmp_path):
    """Normal views by pose; albedo views with their own viewIds; the second normal map is missing (skipped
    frame): the frame names keep the view index and frames.json maps them back to the albedo viewIds."""
    normal = np.full((H, W, 3), 128, np.uint8)
    for name in ("n1.png", "n3.png"):
        cv2.imwrite(str(tmp_path / name), normal)
    for k in (1, 2, 3):
        _write_png16(tmp_path / "a{}.png".format(k), np.full((H, W, 3), 0.1 * k))
    normals = _sfm([("101", "101", str(tmp_path / "n1.png")), ("102", "102", str(tmp_path / "missing.png")),
                    ("103", "103", str(tmp_path / "n3.png"))])
    albedos = _sfm([("901", "101", str(tmp_path / "a1.png")), ("902", "102", str(tmp_path / "a2.png")),
                    ("903", "103", str(tmp_path / "a3.png"))])
    for name, sfm in (("normals.sfm", normals), ("albedos.sfm", albedos)):
        with open(tmp_path / name, "w") as f:
            json.dump(sfm, f)
    data = SfmJsonDataLoader(str(tmp_path / "normals.sfm"), albedo_sfm_path=str(tmp_path / "albedos.sfm")).load()
    assert [v["albedo_view_id"] for v in data["views"]] == ["901", "902", "903"]

    result = prepare_testbed_data(data, str(tmp_path / "prepared"), _Log(), scaling_mode="cameras")
    info = result["frames_info"]
    assert [(f["name"], f["pose_id"], f["albedo_view_id"]) for f in info] == \
        [("00000", "101", "901"), ("00002", "103", "903")]
    with open(tmp_path / "prepared" / "frames.json") as f:
        assert json.load(f) == info
    with open(tmp_path / "prepared" / "transform.json") as f:
        frames = json.load(f)["frames"]
    assert [os.path.basename(fr["albedo_path"]) for fr in frames] == ["00000.png", "00002.png"]
    assert info[1]["albedo_path"] == str(tmp_path / "a3.png")


def test_export_harmonized_albedos(tmp_path):
    albedo = np.full((H, W, 3), 0.4, np.float32)
    albedo[:, W // 2:] = 0.0
    mask = np.zeros((H, W), np.uint8)
    mask[:, : W // 2] = 255
    _write_png16(tmp_path / "a1.png", albedo)
    _write_png16(tmp_path / "a2.png", albedo * 0.5)
    _write_png16(tmp_path / "a3.png", albedo)
    cv2.imwrite(str(tmp_path / "m1.png"), mask)
    sfm = _sfm([("901", "101", str(tmp_path / "a1.png")), ("902", "102", str(tmp_path / "a2.png")),
                ("903", "103", str(tmp_path / "a3.png"))])
    with open(tmp_path / "albedos.sfm", "w") as f:
        json.dump(sfm, f)
    frames = [{"name": "00000", "albedo_view_id": "901", "albedo_path": str(tmp_path / "a1.png"),
               "mask_path": str(tmp_path / "m1.png")},
              {"name": "00001", "albedo_view_id": "902", "albedo_path": str(tmp_path / "a2.png"), "mask_path": None}]
    # 903 has no frame (e.g. no normal map): it is not exported and dropped from the SfMData copy
    scales = {"00000": [2.0, 3.0, 0.5], "00001": [4.0, 4.0, 4.0]}
    out = tmp_path / "out"
    paths = albedo_export.export_harmonized_albedos(str(tmp_path / "albedos.sfm"), frames, scales, str(out / "albedo"),
                                                    str(out / "albedoHarmonized.sfm"), bleed_pixels=2)
    assert sorted(paths) == ["901", "902"]

    exr = cv2.imread(str(out / "albedo" / "901.exr"), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    assert exr.dtype == np.float32
    np.testing.assert_allclose(exr[:, : W // 2], np.broadcast_to([0.8, 1.2, 0.2], (H, W // 2, 3)), atol=1e-4)
    # bled 2 px outside the mask, zero beyond; values above 1 are not clipped
    np.testing.assert_allclose(exr[:, W // 2: W // 2 + 2], np.broadcast_to([0.8, 1.2, 0.2], (H, 2, 3)), atol=1e-4)
    assert np.all(exr[:, W // 2 + 2:] == 0)
    # no mask: the non-zero albedo pixels are the mask
    exr2 = cv2.imread(str(out / "albedo" / "902.exr"), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    np.testing.assert_allclose(exr2[0, 0], [0.8, 0.8, 0.8], atol=1e-4)

    with open(out / "albedoHarmonized.sfm") as f:
        copy = json.load(f)
    assert [(v["viewId"], v["path"]) for v in copy["views"]] == \
        [("901", str(out / "albedo" / "901.exr")), ("902", str(out / "albedo" / "902.exr"))]
    assert copy["intrinsics"] == sfm["intrinsics"] and copy["poses"] == sfm["poses"]


def test_is_json_sfm(tmp_path):
    (tmp_path / "a.sfm").write_text(json.dumps({"views": []}))
    (tmp_path / "b.abc").write_bytes(b"\x00\x01Ogawa")
    assert albedo_export.is_json_sfm(str(tmp_path / "a.sfm"))
    assert not albedo_export.is_json_sfm(str(tmp_path / "b.abc"))
    assert not albedo_export.is_json_sfm(str(tmp_path / "missing.sfm"))
