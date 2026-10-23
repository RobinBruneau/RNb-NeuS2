"""GPU check of the output frame: the meshes of every input route land on the synthetic ellipsoid, in the world frame.

Trains the testbed on the synthetic scene of tests/synthetic_scene.py through each route and checks the distance of
the mesh vertices to the world ellipsoid:

- idr       IDR folder (cameras.npz, scale_mat) -> run_full_pipeline
- sfm_json  AliceVision sfm.json + masks -> run_full_pipeline (world = SfMData frame flipped in Y and Z)
- rotated   transform.json written directly, normalized frame rotated w.r.t. the world (n2w with a rotation)
- nerf      same as rotated without "from_na" (NeRF camera axes; the testbed cycles the axes internally)
- snapshot  mesh exported from the snapshot of "rotated" alone (no --scene): n2w comes from the snapshot

Usage (GPU node): python tests/gpu/check_world_frame.py --testbed build/testbed --work /tmp/wf [--steps 3000]
Exit code 0 iff every route passes.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
from rnb_neus2.pipeline import postprocess_mesh, run_full_pipeline, run_two_stage  # noqa: E402
from synthetic_scene import Scene, rotation, similarity, write_idr, write_sfm_json  # noqa: E402

SCALE_MAT = similarity(60.0, np.eye(3), (32.0, -153.0, -899.0))
Q = rotation(0.5, 1.0, -0.4)  # rotation of the normalized frame for "rotated" / "nerf"


class Log:
    """Pipeline logger that keeps the stage lines and the errors only (the testbed output is long)."""

    def info(self, msg):
        if msg.startswith("===") or "command:" in msg or "completed" in msg:
            print(msg, flush=True)

    def warning(self, msg):
        print("WARNING " + msg, flush=True)

    def error(self, msg):
        print("ERROR " + msg[-3000:], flush=True)


def rotated_copy(prepared, out, from_na):
    """Copy of a prepared_data folder whose normalized frame is rotated by Q^T (n2w composed with Q)."""
    os.makedirs(out, exist_ok=True)
    for d in ("normals", "albedos"):
        if not os.path.exists(os.path.join(out, d)):
            os.symlink(os.path.join(prepared, d), os.path.join(out, d))
    with open(os.path.join(prepared, "transform.json")) as f:
        tj = json.load(f)
    G = np.eye(4)
    G[:3, :3] = Q
    tj["n2w"] = (np.array(tj["n2w"]) @ G).tolist()
    for frame in tj["frames"]:
        T = np.array(frame["transform_matrix"])
        T = np.linalg.inv(G) @ T
        if not from_na:
            T[:3, 1:3] *= -1  # OpenCV -> NeRF camera axes
        frame["transform_matrix"] = T.tolist()
    if not from_na:
        del tj["from_na"]  # the testbed tests the presence of the key, not its value
    with open(os.path.join(out, "transform.json"), "w") as f:
        json.dump(tj, f, indent=1)
    return out


def train(testbed, data_dir, steps, resolution, keep_snapshot=None):
    run_two_stage(testbed, data_dir, steps, ["--mask-weight", "1.0"], resolution=resolution, no_albedo=True,
                  logger=Log())
    if keep_snapshot:
        shutil.copy(os.path.join(data_dir, "output", "snapshot_{}.msgpack".format(steps)), keep_snapshot)
    mesh = os.path.join(data_dir, "mesh.obj")
    postprocess_mesh(data_dir, mesh, Log())
    return mesh


def check(name, mesh_path, scene, median_mm, p90_mm):
    import trimesh
    v = np.asarray(trimesh.load(mesh_path, process=False).vertices)
    d = scene.distance(v)
    med, p90 = float(np.median(d)), float(np.percentile(d, 90))
    ok = med < median_mm and p90 < p90_mm
    print("{:9s} {}  vertices {:7d}  median {:8.3f} mm  p90 {:8.3f} mm  bbox {} .. {}".format(
        name, "OK  " if ok else "FAIL", len(v), med, p90, np.round(v.min(0), 1), np.round(v.max(0), 1)), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--median-mm", type=float, default=1.0)
    ap.add_argument("--p90-mm", type=float, default=3.0)
    ap.add_argument("--routes", default="idr,sfm_json,rotated,nerf,snapshot")
    args = ap.parse_args()
    testbed, work = os.path.abspath(args.testbed), os.path.abspath(args.work)
    routes = args.routes.split(",")
    scene = Scene()
    lo = scene.center - np.abs(scene.Q) @ scene.axes
    hi = scene.center + np.abs(scene.Q) @ scene.axes
    print("ellipsoid bbox (approx.) {} .. {}".format(np.round(lo, 1), np.round(hi, 1)), flush=True)
    results = {}

    def run(name, fn):
        try:
            results[name] = check(name, fn(), scene, args.median_mm, args.p90_mm)
        except Exception as e:  # a crash of one route must not hide the others
            print("{:9s} FAIL  {}: {}".format(name, type(e).__name__, e), flush=True)
            results[name] = False

    idr = write_idr(scene, os.path.join(work, "idr_in"), SCALE_MAT)
    if "idr" in routes:
        run("idr", lambda: run_full_pipeline(idr, testbed, os.path.join(work, "idr"), max_steps=args.steps,
                                             mesh_resolution=args.resolution, scaling_mode="silhouettes",
                                             logger=Log()))
    if "sfm_json" in routes:
        sfm, masks = write_sfm_json(scene, os.path.join(work, "sfm_in"))
        run("sfm_json", lambda: run_full_pipeline(sfm, testbed, os.path.join(work, "sfm_json"), max_steps=args.steps,
                                                  mesh_resolution=args.resolution, scaling_mode="silhouettes",
                                                  mask_folder_path=masks, logger=Log()))
    if any(r in routes for r in ("rotated", "nerf", "snapshot")):
        from rnb_neus2.dataloaders import load_data
        from rnb_neus2.prepare import prepare_testbed_data
        prepared = os.path.join(work, "prepared")
        prepare_testbed_data(load_data(idr), prepared, Log(), scaling_mode="silhouettes")
        snap = os.path.join(work, "snapshot", "rotated.msgpack")
        os.makedirs(os.path.dirname(snap), exist_ok=True)
        if "rotated" in routes or "snapshot" in routes:
            run("rotated", lambda: train(testbed, rotated_copy(prepared, os.path.join(work, "rotated"), True),
                                         args.steps, args.resolution, keep_snapshot=snap))
        if "nerf" in routes:
            run("nerf", lambda: train(testbed, rotated_copy(prepared, os.path.join(work, "nerf"), False),
                                      args.steps, args.resolution))
        if "snapshot" in routes:
            def from_snapshot():
                cmd = [testbed, "--snapshot", snap, "--no-train", "--no-gui", "--save-mesh",
                       "--resolution", str(args.resolution), "--maxiter", str(args.steps)]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError("testbed failed ({}): {}".format(r.returncode, r.stderr[-2000:]))
                raw = os.path.join(os.path.dirname(snap), "mesh_{}.obj".format(args.steps))
                mesh = os.path.join(os.path.dirname(snap), "mesh.obj")
                import trimesh
                m = trimesh.load(raw, process=False)
                max(m.split(only_watertight=False), key=lambda c: c.area).export(mesh)
                return mesh
            run("snapshot", from_snapshot)

    print("RESULT {}".format("PASS" if results and all(results.values()) else "FAIL"),
          {k: ("ok" if v else "FAIL") for k, v in results.items()}, flush=True)
    sys.exit(0 if results and all(results.values()) else 1)


if __name__ == "__main__":
    main()
