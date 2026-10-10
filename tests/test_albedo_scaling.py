"""Albedo scaling via multi-view consistency (rnb_neus2/albedo_scaling.py).

The per-view albedo scales are solved jointly over all the overlapping view pairs: they must be recovered up to the
global normalization whatever the view order (no ring assumption), views without overlap keep 1, and the masked-out
pixels are ignored. CPU only.

Run: venv/bin/python -m pytest tests/test_albedo_scaling.py
"""
import json
import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rnb_neus2 import albedo_scaling as scaling  # noqa: E402


def _geometric_normalized(scales):
    return scales / np.exp(np.log(scales).mean(axis=0))


def _pair_ratios(true_scales, pairs, n_samples=1000, noise=0.03, seed=0):
    """log a_i - log a_j samples of view pairs whose albedos are albedo / true_scales[view]."""
    rng = np.random.default_rng(seed)
    out = {}
    for i, j in pairs:
        albedo = rng.uniform(0.1, 0.8, (n_samples, 3))
        a_i = albedo / true_scales[i] * np.exp(noise * rng.standard_normal((n_samples, 3)))
        a_j = albedo / true_scales[j] * np.exp(noise * rng.standard_normal((n_samples, 3)))
        out[(i, j)] = np.log(a_i) - np.log(a_j)
    return out


# --- joint solve ---------------------------------------------------------------------------------------------------

def test_solve_recovers_known_scales():
    rng = np.random.default_rng(1)
    true_scales = np.exp(rng.uniform(-0.7, 0.7, (10, 3)))
    pairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    scales, used, unconstrained = scaling.solve_pairwise_log_scales(_pair_ratios(true_scales, pairs), 10)
    assert len(used) == len(pairs) and not unconstrained
    np.testing.assert_allclose(np.exp(np.log(scales).mean(axis=0)), 1.0, atol=1e-6)
    np.testing.assert_allclose(scales, _geometric_normalized(true_scales), rtol=0.01)
    before, after = scaling.pair_mismatch(used, scales)
    assert after < 1.0 < before


def test_solve_ring_is_order_independent():
    """12 views on a ring listed in a shuffled order: only neighbours on the ring overlap (the old chain over the
    list order compared views that do not overlap)."""
    n = 12
    rng = np.random.default_rng(2)
    true_scales = np.exp(rng.uniform(-0.7, 0.7, (n, 3)))
    ring = rng.permutation(n)  # ring[k] = index in the list of the k-th view around the object
    pairs = [tuple(sorted((int(ring[k]), int(ring[(k + 1) % n])))) for k in range(n)]
    scales, used, _ = scaling.solve_pairwise_log_scales(_pair_ratios(true_scales, pairs), n)
    assert len(used) == n
    np.testing.assert_allclose(scales, _geometric_normalized(true_scales), rtol=0.01)
    # same result for another listing order of the same views
    perm = rng.permutation(n)
    inverse = np.argsort(perm)
    pairs_perm = [tuple(sorted((int(inverse[i]), int(inverse[j])))) for i, j in pairs]
    ratios = _pair_ratios(true_scales[perm], pairs_perm)
    scales_perm, _, _ = scaling.solve_pairwise_log_scales(ratios, n)
    np.testing.assert_allclose(scales_perm, _geometric_normalized(true_scales[perm]), rtol=0.01)


def test_solve_views_without_overlap_and_groups():
    true_scales = np.array([[1.0] * 3, [2.0] * 3, [0.5] * 3, [4.0] * 3, [3.0] * 3])
    ratios = _pair_ratios(true_scales, [(0, 1), (2, 3)], noise=0.0)
    scales, _, unconstrained = scaling.solve_pairwise_log_scales(ratios, 5)
    assert unconstrained == [4]
    np.testing.assert_allclose(scales[4], 1.0)
    np.testing.assert_allclose(scales[:2], _geometric_normalized(true_scales[:2]), rtol=1e-6)
    np.testing.assert_allclose(scales[2:4], _geometric_normalized(true_scales[2:4]), rtol=1e-6)
    assert sorted(map(sorted, scaling.view_groups(5, [(0, 1), (2, 3)]))) == [[0, 1], [2, 3], [4]]


def test_solve_ignores_pairs_with_few_samples():
    true_scales = np.array([[1.0] * 3, [2.0] * 3, [3.0] * 3])
    ratios = _pair_ratios(true_scales, [(0, 1), (1, 2)], n_samples=1000)
    ratios[(0, 2)] = np.full((10, 3), 5.0)  # few, wrong samples
    scales, used, _ = scaling.solve_pairwise_log_scales(ratios, 3, min_pair_samples=300)
    assert [(i, j) for i, j, _, _ in used] == [(0, 1), (1, 2)]
    np.testing.assert_allclose(scales, _geometric_normalized(true_scales), rtol=0.01)


# --- end to end on a synthetic scene --------------------------------------------------------------------------------

RADIUS, SIZE, FOCAL = 1.0, 120, 150.0


def _albedo_field(points):
    """Smooth albedo on the sphere, in [0.2, 0.7]."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([0.45 + 0.2 * np.sin(3 * x + z), 0.4 + 0.2 * np.cos(3 * y), 0.3 + 0.1 * np.sin(2 * (x + y))], -1)


def _camera(azimuth):
    """OpenCV camera (x right, y down, z forward) on a ring of radius 3 looking at the origin."""
    center = 3.0 * np.array([np.cos(azimuth), np.sin(azimuth), 0.3])
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, down, forward], axis=1)
    c2w[:3, 3] = center
    return c2w


def _write_scene(folder, true_scales, azimuths, garbage_view=None):
    """Sphere mesh, transform.json and RGBA 16-bit albedos (alpha = mask) of views at the given azimuths."""
    import trimesh
    mesh_path = os.path.join(folder, "sphere.obj")
    trimesh.creation.icosphere(subdivisions=5, radius=RADIUS).export(mesh_path)
    os.makedirs(os.path.join(folder, "albedos"))
    K = np.array([[FOCAL, 0, SIZE / 2], [0, FOCAL, SIZE / 2], [0, 0, 1.0]])
    frames = []
    for k, (scale, azimuth) in enumerate(zip(true_scales, azimuths)):
        c2w = _camera(azimuth)
        xs, ys = np.meshgrid(np.arange(SIZE) + 0.5, np.arange(SIZE) + 0.5)  # NeuS2 pixel centers
        rays = np.stack([xs, ys, np.ones_like(xs)], -1) @ np.linalg.inv(K).T @ c2w[:3, :3].T
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        origin = c2w[:3, 3]
        b = rays @ origin
        disc = b ** 2 - (origin @ origin - RADIUS ** 2)
        hit = disc > 0
        t = -b - np.sqrt(np.maximum(disc, 0))
        points = origin + t[..., None] * rays
        albedo = np.where(hit[..., None], _albedo_field(points) / scale, 0.0)
        mask = hit.copy()
        if k == garbage_view:
            # the right 60% of the object is masked out and holds wrong values
            cols = np.where(hit.any(axis=0))[0]
            cut = int(cols.min() + 0.4 * (cols.max() - cols.min()))
            mask[:, cut:] = False
            albedo[:, cut:] = 0.95
        rgba = np.concatenate([np.clip(albedo, 0, 1), mask[..., None]], -1)
        png = np.round(rgba * 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(folder, "albedos", "{:05d}.png".format(k)), cv2.cvtColor(png, cv2.COLOR_RGBA2BGRA))
        frames.append({"albedo_path": "albedos/{:05d}.png".format(k), "transform_matrix": c2w.tolist(),
                       "intrinsic_matrix": K.tolist()})
    with open(os.path.join(folder, "transform.json"), "w") as f:
        json.dump({"w": SIZE, "h": SIZE, "frames": frames}, f)
    return mesh_path


def test_compute_scales_on_a_ring_in_shuffled_order(tmp_path):
    pytest.importorskip("trimesh")
    rng = np.random.default_rng(3)
    n = 8
    true_scales = np.exp(rng.uniform(-0.6, 0.6, (n, 3)))
    azimuths = 2 * np.pi * rng.permutation(n) / n  # list order != order around the object
    mesh_path = _write_scene(str(tmp_path), true_scales, azimuths, garbage_view=0)
    scales, report = scaling.compute_albedo_scale_ratios(
        str(tmp_path / "albedos"), str(tmp_path / "transform.json"), mesh_path, n_samples=4000,
        return_report=True)
    expected = true_scales / true_scales.mean(axis=0)  # albedo level kept: arithmetic mean 1
    np.testing.assert_allclose(scales.mean(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(scales, expected, rtol=0.02)
    # views 180 deg apart see opposite hemispheres: only views close on the ring are paired
    paired = {tuple(p["views"]) for p in report["pairs"]}
    names = report["names"]
    for i in range(n):
        for j in range(i + 1, n):
            gap = abs(np.angle(np.exp(1j * (azimuths[i] - azimuths[j]))))
            if gap > 0.9 * np.pi:
                assert (names[i], names[j]) not in paired
    assert report["mismatchAfterPercent"] < 2.0 < report["mismatchBeforePercent"]
    assert not report["unconstrained"]


def test_scale_and_save_albedos_keeps_the_mask(tmp_path):
    folder = tmp_path / "albedos"
    folder.mkdir()
    rgba = np.zeros((4, 4, 4), np.float32)
    rgba[..., :3] = 0.6
    rgba[:2, :, 3] = 1.0
    png = np.round(rgba * 65535).astype(np.uint16)
    cv2.imwrite(str(folder / "00000.png"), cv2.cvtColor(png, cv2.COLOR_RGBA2BGRA))
    scaling.scale_and_save_albedos(str(folder), str(tmp_path / "out"), np.array([[0.5, 1.0, 2.0]]))
    out = cv2.cvtColor(cv2.imread(str(tmp_path / "out" / "00000.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    out = out.astype(np.float32) / 65535
    np.testing.assert_allclose(out[0, 0], [0.3, 0.6, 1.0, 1.0], atol=1e-4)  # 1.2 clipped to 1 (16-bit PNG)
    np.testing.assert_allclose(out[3, 0, 3], 0.0)
