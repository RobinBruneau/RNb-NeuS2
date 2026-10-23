"""Output frame contract: the n2w of every transform.json maps the testbed frames back to the input cameras' world.

Whatever the input (IDR cameras.npz, AliceVision sfm.json / .sfm, NeuS cameras_sphere.npz) and the scene
normalization, ``n2w @ transform_matrix`` must give the world cameras: the frame of ``world_mat`` for IDR/NeuS (the
``scale_mat`` undone), the SfMData frame flipped in Y and Z for AliceVision (as AliceVision meshes). The testbed
applies the same n2w to the exported mesh (tests/gpu/check_world_frame.py checks the meshes).

Run: python -m pytest tests/test_world_frame.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnb_neus2.dataloaders import load_data  # noqa: E402
from rnb_neus2.prepare import prepare_testbed_data  # noqa: E402
from synthetic_scene import Scene, rotation, similarity, world_cameras, write_idr, write_neus, write_sfm_json  # noqa: E402

SCALE_MAT = similarity(60.0, np.eye(3), (32.0, -153.0, -899.0))  # IDR normalized -> world (sphere around the object)


class _Log:
    def info(self, msg):
        pass

    warning = error = info


@pytest.fixture(scope="module")
def scene():
    return Scene(n_views=12, width=80, height=60, focal=170.0)


def _check_world(transform, scene, tol_mm=1e-2):
    R, C = world_cameras(transform)
    np.testing.assert_allclose(C, np.array(scene.C), atol=tol_mm)
    np.testing.assert_allclose(R, np.array(scene.R_c2w), atol=1e-5)


def _prepare(data, out, mode):
    prepare_testbed_data(data, str(out), _Log(), scaling_mode=mode)
    with open(os.path.join(str(out), "transform.json")) as f:
        return json.load(f)


@pytest.mark.parametrize("mode", ["none", "silhouettes", "cameras"])
def test_idr(scene, tmp_path, mode):
    folder = write_idr(scene, str(tmp_path / "idr"), SCALE_MAT)
    _check_world(_prepare(load_data(folder), tmp_path / "prep", mode), scene)


@pytest.mark.parametrize("mode", ["silhouettes", "pcd", "cameras"])
def test_sfm_json(scene, tmp_path, mode):
    sfm, masks = write_sfm_json(scene, str(tmp_path / "sfm"))
    _check_world(_prepare(load_data(sfm, mask_folder_path=masks), tmp_path / "prep", mode), scene)


def test_sfm_pyalicevision(scene, tmp_path):
    pytest.importorskip("pyalicevision")
    from rnb_neus2.dataloaders.sfm_pyav_loader import SfmPyavDataLoader
    sfm, masks = write_sfm_json(scene, str(tmp_path / "sfm"))
    os.rename(sfm, sfm[:-5] + ".sfm")
    data = SfmPyavDataLoader(normal_sfm_path=sfm[:-5] + ".sfm", mask_folder_path=masks).load()
    _check_world(_prepare(data, tmp_path / "prep", "silhouettes"), scene)


def test_neus_tool(scene, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("neus_tool", os.path.join(ROOT, "tools", "data_format_from_neus.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    write_neus(scene, str(tmp_path / "in" / "obj"), SCALE_MAT)
    monkeypatch.chdir(tmp_path)
    tool.generate("obj", str(tmp_path / "in"), copy_image=True)
    with open(tmp_path / "data" / "neus" / "obj" / "transform_train.json") as f:
        train = json.load(f)
    with open(tmp_path / "data" / "neus" / "obj" / "transform_test.json") as f:
        test = json.load(f)
    n_test = len(test["frames"])
    test_ids = [i for i in range(len(scene.C)) if i in (8, 13, 16, 21, 26, 31, 34, 56)]
    train_ids = [i for i in range(len(scene.C)) if i not in test_ids]
    assert n_test == len(test_ids)
    for transform, ids in ((train, train_ids), (test, test_ids)):
        R, C = world_cameras(transform)
        np.testing.assert_allclose(C, np.array(scene.C)[ids], atol=1e-2)
        np.testing.assert_allclose(R, np.array(scene.R_c2w)[ids], atol=1e-5)


def test_albedo_scaling_cameras_are_rigid(scene, tmp_path):
    """Cameras read back from transform.json (albedo scaling) are world cameras with orthonormal rotations."""
    from rnb_neus2.albedo_scaling import load_cameras_from_transform_json
    folder = write_idr(scene, str(tmp_path / "idr"), SCALE_MAT)
    _prepare(load_data(folder), tmp_path / "prep", "silhouettes")
    names = ["{:05d}.png".format(i) for i in range(len(scene.C))]
    _, R, C = load_cameras_from_transform_json(str(tmp_path / "prep" / "transform.json"), names)
    np.testing.assert_allclose(C[:, :, 0], np.array(scene.C), atol=1e-2)
    np.testing.assert_allclose(R, np.array(scene.R_c2w), atol=1e-5)


def test_albedo_scaling_cameras_reflected_n2w(scene, tmp_path):
    """An orientation-reversing n2w (det < 0) keeps the world cameras looking forward, as the C++ mesh export does."""
    from rnb_neus2.albedo_scaling import load_cameras_from_transform_json
    folder = write_idr(scene, str(tmp_path / "idr"), SCALE_MAT)
    tj = _prepare(load_data(folder), tmp_path / "prep", "silhouettes")
    F = np.diag([-1.0, 1.0, 1.0, 1.0])  # reflection of the normalized frame (F = F^-1)
    tj["n2w"] = (np.array(tj["n2w"]) @ F).tolist()
    for frame in tj["frames"]:
        frame["transform_matrix"] = (F @ np.array(frame["transform_matrix"])).tolist()
    path = tmp_path / "prep" / "transform_reflected.json"
    with open(path, "w") as f:
        json.dump(tj, f)
    names = ["{:05d}.png".format(i) for i in range(len(scene.C))]
    _, R, C = load_cameras_from_transform_json(str(path), names)
    np.testing.assert_allclose(C[:, :, 0], np.array(scene.C), atol=1e-2)
    np.testing.assert_allclose(R, np.array(scene.R_c2w), atol=1e-5)


def test_world_cameras_helper(scene):
    """Self-check of the helper: a rotated (or reflected) normalized frame, with and without from_na (NeRF axes)."""
    for reflect in (False, True):
        G = similarity(55.0, rotation(0.5, 1.0, -0.4), (10.0, 20.0, -30.0))
        if reflect:
            G = G @ np.diag([-1.0, 1.0, 1.0, 1.0])
        _check_helper(scene, G)


def _check_helper(scene, G):
    Ginv = np.linalg.inv(G)
    for from_na in (True, False):
        frames = []
        for R, C in zip(scene.R_c2w, scene.C):
            T = np.eye(4)
            T[:3, :3] = Ginv[:3, :3] @ R * 55.0
            T[:3, 3] = Ginv[:3, :3] @ C + Ginv[:3, 3]
            if not from_na:
                T[:3, 1:3] *= -1
            frames.append({"transform_matrix": T.tolist()})
        _check_world({"from_na": from_na, "n2w": G.tolist(), "frames": frames}, scene)
