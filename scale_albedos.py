#!/usr/bin/env python3
"""Standalone albedo scaling via multi-view consistency.

Supports all camera formats: cameras.npz, sfmData (.sfm), transform.json.

Usage examples::

    # Auto-detect camera format
    python scale_albedos.py --folder data/RNb-NeuS2/ --mesh mesh.obj

    # Explicit camera source
    python scale_albedos.py --folder data/RNb-NeuS2/ --mesh mesh.obj \\
        --cameras transform.json

    # From SfMData
    python scale_albedos.py --folder data/RNb-NeuS2/ --mesh mesh.obj \\
        --cameras albedos.sfm
"""

import argparse
import os
import shutil

import numpy as np

from rnb_neus2.albedo_scaling import (
    compute_albedo_scale_ratios,
    scale_and_save_albedos,
)


def main():
    parser = argparse.ArgumentParser(
        description="Scale albedos based on multi-view consistency "
                    "using ray tracing through a mesh.")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing albedos/.")
    parser.add_argument("--mesh", "--mesh_path", type=str, default=None,
                        help="Path to mesh (OBJ). Auto-detected if omitted.")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Camera source: .npz, .sfm, .json, or "
                             "transform.json. Auto-detected if omitted.")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of samples per image (default: 2000).")
    parser.add_argument("--bit-depth", type=int, default=16,
                        choices=[8, 16],
                        help="Output bit depth (default: 16).")
    parser.add_argument("--in-place", action="store_true",
                        help="Scale albedos in-place instead of creating "
                             "a new folder.")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    np.random.seed(args.seed)

    folder = args.folder.rstrip("/")

    # Find mesh
    mesh_path = args.mesh
    if mesh_path is None:
        mesh_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.startswith("mesh_") and f.endswith(".obj")
        ]
        if not mesh_files:
            raise RuntimeError(
                "No mesh found in {}. Use --mesh.".format(folder))
        mesh_path = mesh_files[0]

    albedo_path = os.path.join(folder, "albedos")
    if not os.path.isdir(albedo_path):
        raise RuntimeError("No albedos/ folder in {}".format(folder))

    # Find camera source
    camera_source = args.cameras
    if camera_source is None:
        # Auto-detect
        candidates = [
            os.path.join(folder, "transform.json"),
            os.path.join(folder, "..", "cameras.npz"),
            os.path.join(folder, "..", "transform.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                camera_source = c
                print("Auto-detected cameras: {}".format(c))
                break
        if camera_source is None:
            raise RuntimeError(
                "No camera source found. Use --cameras.")

    print("Computing albedo scaling ratios...")
    print("  Albedos:  {}".format(albedo_path))
    print("  Mesh:     {}".format(mesh_path))
    print("  Cameras:  {}".format(camera_source))
    print("  Samples:  {}".format(args.n_samples))

    scale_ratios = compute_albedo_scale_ratios(
        albedo_path=albedo_path,
        camera_source=camera_source,
        mesh_path=mesh_path,
        n_samples=args.n_samples,
    )

    if args.in_place:
        output_albedo_path = albedo_path + "_tmp"
        scale_and_save_albedos(
            albedo_path=albedo_path,
            output_albedo_path=output_albedo_path,
            scale_ratios=scale_ratios,
            bit_depth=args.bit_depth,
        )
        shutil.rmtree(albedo_path)
        os.rename(output_albedo_path, albedo_path)
        print("Scaled albedos in-place: {}".format(albedo_path))
    else:
        exp_name = os.path.basename(folder)
        output_path = os.path.join(
            os.path.dirname(folder), exp_name + "-albedoscaled")
        os.makedirs(output_path, exist_ok=True)

        # Copy transform.json and normals
        transform_path = os.path.join(folder, "transform.json")
        if os.path.exists(transform_path):
            shutil.copyfile(
                transform_path,
                os.path.join(output_path, "transform.json"))
        normal_path = os.path.join(folder, "normals")
        if os.path.isdir(normal_path):
            shutil.copytree(
                normal_path,
                os.path.join(output_path, "normals"),
                dirs_exist_ok=True)

        np.save(os.path.join(output_path, "ratios.npy"), scale_ratios)

        output_albedo_path = os.path.join(output_path, "albedos")
        scale_and_save_albedos(
            albedo_path=albedo_path,
            output_albedo_path=output_albedo_path,
            scale_ratios=scale_ratios,
            bit_depth=args.bit_depth,
        )
        print("Scaled albedos saved to {}".format(output_path))


if __name__ == "__main__":
    main()
