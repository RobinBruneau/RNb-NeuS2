#!/usr/bin/env python3
"""Standalone RNb-NeuS2 pipeline.

Usage examples::

    # From cameras.npz (RNb format)
    python run_pipeline.py --input data/my_scene/ --testbed build-ubuntu/testbed

    # From SfMData JSON (no pyalicevision needed)
    python run_pipeline.py --input normals.sfm --testbed build-ubuntu/testbed \\
        --albedo-sfm albedos.sfm --mask-sfm masks.sfm --has-albedo

    # From SfMData with pyalicevision
    python run_pipeline.py --input normals.sfm --testbed build-ubuntu/testbed
"""

import argparse
import numpy as np

from rnb_neus2.pipeline import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="RNb-NeuS2: Neural surface reconstruction pipeline")

    parser.add_argument("--input", "-i", required=True,
                        help="Input data: directory (cameras.npz), "
                             ".npz, .sfm, or .json")
    parser.add_argument("--testbed", "-t", required=True,
                        help="Path to the testbed binary")
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Max training steps (default: 10000)")
    parser.add_argument("--mesh-resolution", type=int, default=1024,
                        help="Marching cubes resolution (default: 1024)")
    parser.add_argument("--scaling-mode", default="auto",
                        choices=["auto", "pcd", "silhouettes",
                                 "silhouettes_v2", "cameras", "none"],
                        help="Scene normalization mode (default: auto)")
    parser.add_argument("--sphere-scale", type=float, default=1.0,
                        help="Target sphere radius (default: 1.0)")
    parser.add_argument("--margin-px", type=int, default=20,
                        help="Pixel margin for silhouettes_v2 (default: 20)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Phase 1 ratio for albedo mode (default: 0.1)")
    parser.add_argument("--mask-weight", type=float, default=1.0,
                        help="Mask loss weight (default: 1.0)")
    parser.add_argument("--has-albedo", action="store_true",
                        help="Enable two-phase training with albedo scaling")
    parser.add_argument("--albedo-sfm", default="",
                        help="Path to albedo SfMData (SfM mode)")
    parser.add_argument("--mask-sfm", default="",
                        help="Path to mask SfMData (SfM mode)")
    parser.add_argument("--mask-folder", default="",
                        help="Folder with mask images")
    parser.add_argument("--supernormal", action="store_true",
                        help="Enable SuperNormal mode")
    parser.add_argument("--l1", action="store_true",
                        help="Use L1 norm for color loss")
    parser.add_argument("--no-rgbplus", action="store_true",
                        help="Disable RGB+ normalization")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Samples for albedo scaling (default: 2000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")

    args = parser.parse_args()

    np.random.seed(args.seed)

    run_full_pipeline(
        input_path=args.input,
        testbed_path=args.testbed,
        output_dir=args.output,
        max_steps=args.max_steps,
        mesh_resolution=args.mesh_resolution,
        scaling_mode=args.scaling_mode,
        sphere_scale=args.sphere_scale,
        margin_px=args.margin_px,
        warmup_ratio=args.warmup_ratio,
        mask_weight=args.mask_weight,
        super_normal=args.supernormal,
        use_l1=args.l1,
        use_rgb_plus=not args.no_rgbplus,
        has_albedo=args.has_albedo,
        albedo_sfm_path=args.albedo_sfm,
        mask_sfm_path=args.mask_sfm,
        mask_folder_path=args.mask_folder,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
