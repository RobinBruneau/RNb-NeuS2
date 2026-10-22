"""Synthetic scene for the output-frame tests: a tilted, off-center ellipsoid seen by cameras on a sphere.

The scene is defined in the WORLD frame, i.e. the frame the exported mesh must be in. Its normal maps and masks are
analytic (ray casting, NeuS2 pixel convention: the ray of pixel i goes through i + 0.5). It can be written as:

- an IDR folder (``cameras.npz`` + ``normal/`` + ``mask/``), ``world_mat`` in the world frame and ``scale_mat`` a
  normalized -> world similarity;
- an AliceVision ``sfm.json`` (+ mask folder), whose raw frame is the world flipped in Y and Z (the SfMData loaders
  flip it back, as for AliceVision meshes);
- a NeuS folder (``cameras_sphere.npz`` + ``image/`` + ``mask/``) for ``tools/data_format_from_neus.py``.

Normal maps follow the converters' RNb-NeuS2 format: camera frame, OpenGL axes (x right, y up, z backward), 16-bit
PNG ``round((n + 1) / 2 * 65535)`` with R G B = x y z, 0 outside the mask.
"""
import json
import os

import cv2
import numpy as np

FLIP_YZ = np.diag([1.0, -1.0, -1.0])


def rotation(rx, ry, rz):
    """Rotation matrix Rz @ Ry @ Rx (radians)."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def similarity(scale, R, t):
    """4x4 matrix x -> scale * R x + t."""
    M = np.eye(4)
    M[:3, :3] = scale * np.asarray(R, dtype=np.float64)
    M[:3, 3] = t
    return M


class Scene:
    """Ellipsoid {x : |diag(1/axes) Q^T (x - center)| = 1} and pinhole cameras (OpenCV axes) looking at it."""

    def __init__(self, n_views=30, width=200, height=150, focal=420.0, distance=320.0,
                 center=(30.0, -150.0, -900.0), axes=(40.0, 25.0, 15.0), angles=(0.4, -0.3, 0.7)):
        self.width, self.height = width, height
        self.center = np.array(center, dtype=np.float64)
        self.axes = np.array(axes, dtype=np.float64)
        self.Q = rotation(*angles)
        self.K = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]])
        self.R_c2w, self.C = [], []  # camera -> world rotations (columns = camera axes), centers
        golden = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_views):  # Fibonacci sphere
            z = 1.0 - 2.0 * (i + 0.5) / n_views
            r = np.sqrt(1.0 - z * z)
            d = np.array([r * np.cos(golden * i), r * np.sin(golden * i), z])
            C = self.center + distance * d
            fwd = -d
            up = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = np.cross(fwd, up)
            x /= np.linalg.norm(x)
            y = np.cross(fwd, x)
            self.R_c2w.append(np.stack([x, y, fwd], axis=1))
            self.C.append(C)

    def implicit(self, points):
        """|diag(1/axes) Q^T (x - center)| - 1 for (N, 3) world points (0 on the surface)."""
        local = (np.asarray(points, dtype=np.float64) - self.center) @ self.Q / self.axes
        return np.linalg.norm(local, axis=1) - 1.0

    def distance(self, points):
        """Approximate Euclidean distance (mm) of world points to the surface (first-order, |f| / |grad f|)."""
        p = np.asarray(points, dtype=np.float64)
        local = (p - self.center) @ self.Q / self.axes
        norm = np.linalg.norm(local, axis=1)
        grad = (local / (norm[:, None] * self.axes)) @ self.Q.T
        return np.abs(norm - 1.0) / np.linalg.norm(grad, axis=1)

    def w2c(self, i):
        M = np.eye(4)
        M[:3, :3] = self.R_c2w[i].T
        M[:3, 3] = -self.R_c2w[i].T @ self.C[i]
        return M

    def render(self, i):
        """Normal map (H, W, 3) in the camera frame, OpenCV axes, and mask (H, W) bool."""
        u, v = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        d_cam = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u)], -1)
        d = d_cam.reshape(-1, 3) @ self.R_c2w[i].T
        o = (self.C[i] - self.center) @ self.Q / self.axes
        dl = d @ self.Q / self.axes
        a = np.sum(dl * dl, 1)
        b = 2.0 * dl @ o
        c = o @ o - 1.0
        disc = b * b - 4 * a * c
        hit = disc > 0
        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2 * a)
        local = o + t[:, None] * dl
        n_world = (local / self.axes) @ self.Q.T
        n_world /= np.linalg.norm(n_world, axis=1, keepdims=True)
        n_cam = n_world @ self.R_c2w[i]
        n_cam[~hit] = 0.0
        return n_cam.reshape(self.height, self.width, 3), hit.reshape(self.height, self.width)

    def surface_points(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        s = rng.normal(size=(n, 3))
        s /= np.linalg.norm(s, axis=1, keepdims=True)
        return self.center + (s * self.axes) @ self.Q.T


def _write_rgb16(path, rgb):
    cv2.imwrite(path, np.ascontiguousarray(rgb[..., ::-1]))


def _normal_png(n_cv, mask):
    enc = np.round((np.clip(n_cv @ FLIP_YZ, -1.0, 1.0) + 1.0) * 0.5 * 65535.0).astype(np.uint16)
    enc[~mask] = 0
    return enc


def write_idr(scene, folder, scale_mat):
    """IDR folder: cameras.npz (world_mat_i = K [R | t] in the world frame, scale_mat_i), normal/, mask/."""
    os.makedirs(os.path.join(folder, "normal"), exist_ok=True)
    os.makedirs(os.path.join(folder, "mask"), exist_ok=True)
    K4 = np.eye(4)
    K4[:3, :3] = scene.K
    cams = {}
    for i in range(len(scene.C)):
        cams["world_mat_{}".format(i)] = K4 @ scene.w2c(i)
        cams["scale_mat_{}".format(i)] = np.asarray(scale_mat, dtype=np.float64)
        n_cv, mask = scene.render(i)
        _write_rgb16(os.path.join(folder, "normal", "{:03d}.png".format(i)), _normal_png(n_cv, mask))
        cv2.imwrite(os.path.join(folder, "mask", "{:03d}.png".format(i)), mask.astype(np.uint8) * 255)
    np.savez(os.path.join(folder, "cameras.npz"), **cams)
    return folder


def write_sfm_json(scene, folder, landmarks=True):
    """AliceVision sfm.json in the raw frame (world flipped in Y and Z) + normals/ and masks/<poseId>.png.

    Returns (sfm_path, mask_folder)."""
    normal_dir, mask_dir = os.path.join(folder, "normals"), os.path.join(folder, "masks")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    sensor = 36.0
    # AliceVision puts the center of pixel i at coordinate i: principal point offset - 0.5 (see the loaders)
    intr = {"intrinsicId": "1", "type": "pinhole", "width": str(scene.width), "height": str(scene.height),
            "sensorWidth": str(sensor), "sensorHeight": str(sensor * scene.height / scene.width),
            "focalLength": repr(float(scene.K[0, 0] * sensor / scene.width)),
            "principalPoint": [repr(float(scene.K[0, 2] - scene.width / 2.0 - 0.5)),
                               repr(float(scene.K[1, 2] - scene.height / 2.0 - 0.5))],
            # remaining fields required by AliceVision's own reader (pyalicevision)
            "serialNumber": "synthetic", "initializationMode": "unknown", "initialFocalLength": "-1",
            "pixelRatio": "1", "pixelRatioLocked": "true", "offsetLocked": "false", "scaleLocked": "false",
            "distortionInitializationMode": "none", "distortionParams": "", "undistortionOffset": ["0", "0"],
            "undistortionParams": "", "distortionType": "none", "undistortionType": "none", "locked": "false"}
    views, poses = [], []
    for i in range(len(scene.C)):
        pid = str(1000 + i)
        n_cv, mask = scene.render(i)
        _write_rgb16(os.path.join(normal_dir, "{}.png".format(pid)), _normal_png(n_cv, mask))
        cv2.imwrite(os.path.join(mask_dir, "{}.png".format(pid)), mask.astype(np.uint8) * 255)
        views.append({"viewId": pid, "poseId": pid, "frameId": str(i), "intrinsicId": "1",
                      # absolute, as Meshroom writes them (the pyalicevision loader does not resolve relative paths)
                      "path": os.path.abspath(os.path.join(normal_dir, "{}.png".format(pid))),
                      "width": str(scene.width), "height": str(scene.height)})
        R_raw = FLIP_YZ @ scene.R_c2w[i]
        poses.append({"poseId": pid, "pose": {"transform": {
            "rotation": [repr(float(x)) for x in R_raw.reshape(-1)],
            "center": [repr(float(x)) for x in FLIP_YZ @ scene.C[i]]}, "locked": "false"}})
    data = {"version": ["1", "2", "12"], "featuresFolders": [], "matchesFolders": [], "views": views,
            "intrinsics": [intr], "poses": poses}
    if landmarks:
        data["structure"] = [{"landmarkId": str(k), "descType": "sift", "color": ["255", "255", "255"],
                              "X": [repr(float(x)) for x in FLIP_YZ @ p], "observations": []}
                             for k, p in enumerate(scene.surface_points())]
    sfm_path = os.path.join(folder, "sfm.json")
    with open(sfm_path, "w") as f:
        json.dump(data, f, indent=1)
    return sfm_path, mask_dir


def write_neus(scene, folder, scale_mat):
    """NeuS folder for tools/data_format_from_neus.py: cameras_sphere.npz, image/ (RGB), mask/."""
    os.makedirs(os.path.join(folder, "image"), exist_ok=True)
    os.makedirs(os.path.join(folder, "mask"), exist_ok=True)
    K4 = np.eye(4)
    K4[:3, :3] = scene.K
    cams = {}
    for i in range(len(scene.C)):
        cams["world_mat_{}".format(i)] = K4 @ scene.w2c(i)
        cams["scale_mat_{}".format(i)] = np.asarray(scale_mat, dtype=np.float64)
        n_cv, mask = scene.render(i)
        cv2.imwrite(os.path.join(folder, "image", "{:03d}.png".format(i)), ((n_cv + 1) * 127.5).astype(np.uint8))
        cv2.imwrite(os.path.join(folder, "mask", "{:03d}.png".format(i)), mask.astype(np.uint8) * 255)
    np.savez(os.path.join(folder, "cameras_sphere.npz"), **cams)
    return folder


def world_cameras(transform):
    """Camera -> world rotations (OpenCV axes) and centers of a transform.json dict, through its n2w.

    The frames are in the testbed's normalized frame (OpenCV camera axes with "from_na", NeRF/OpenGL ones without);
    n2w (identity if absent) is the normalized -> world similarity."""
    n2w = np.array(transform.get("n2w", np.eye(4)), dtype=np.float64)
    A = n2w[:3, :3]
    s = np.cbrt(abs(np.linalg.det(A)))  # scale magnitude: a reflection must not reverse the viewing direction
    Rs, Cs = [], []
    for frame in transform["frames"]:
        T = np.array(frame["transform_matrix"], dtype=np.float64)
        R = A @ T[:3, :3] / s
        if not transform.get("from_na", False):
            R = R @ FLIP_YZ
        Rs.append(R)
        Cs.append(A @ T[:3, 3] + n2w[:3, 3])
    return np.array(Rs), np.array(Cs)
