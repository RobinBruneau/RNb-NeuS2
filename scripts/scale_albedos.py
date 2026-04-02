import os
import numpy as np
import argparse
import shutil
from pathlib import Path

# Clean import - no path manipulation needed!
from scripts.utils.albedo_scaling_lib import compute_albedo_scale_ratios, scale_and_save_albedos


if __name__ == "__main__":

    # fix the seed
    np.random.seed(0)

    parser = argparse.ArgumentParser(
        description="Scale albedos based on multi-view consistency using ray tracing through a mesh."
    )
    parser.add_argument("--folder", type=str, required=True, help="Folder path containing albedos/.")
    parser.add_argument("--mesh_path", type=str, default=None, help="Path to the mesh (OBJ file).")
    
    # Camera format options (mutually exclusive)
    camera_group = parser.add_mutually_exclusive_group(required=False)
    camera_group.add_argument("--cameras_npz", type=str, help="Path to cameras.npz file.")
    camera_group.add_argument("--sfmdata", type=str, help="Path to sfmData file (.sfm).")
    camera_group.add_argument("--transform_json", type=str, help="Path to transform.json file.")
    
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples per image.")
    parser.add_argument("--bit_depth", type=int, default=16, choices=[8, 16], help="Output bit depth.")
    
    args = parser.parse_args()

    # Inputs
    folder = args.folder
    if args.mesh_path is not None:
        mesh_path = args.mesh_path
    else:
        # Search for mesh in folder
        mesh_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("mesh_") and f.endswith(".obj")]
        if not mesh_files:
            raise RuntimeError(f"No mesh file found in folder {folder}. Please provide --mesh_path.")
        mesh_path = mesh_files[0]
    
    albedo_path = os.path.join(folder, "albedos")
    normal_path = os.path.join(folder, "normals")
    transform_path = os.path.join(folder, "transform.json")
    
    # Determine camera source (auto-detect if not specified)
    camera_source = None
    if args.cameras_npz:
        camera_source = args.cameras_npz
    elif args.sfmdata:
        camera_source = args.sfmdata
    elif args.transform_json:
        camera_source = args.transform_json
    else:
        # Auto-detect: try cameras.npz in parent folder, then transform.json in folder
        cameras_npz_path = os.path.join(folder, "..", "cameras.npz")
        if os.path.exists(cameras_npz_path):
            camera_source = cameras_npz_path
            print(f"Auto-detected camera source: {cameras_npz_path}")
        elif os.path.exists(transform_path):
            camera_source = transform_path
            print(f"Auto-detected camera source: {transform_path}")
        else:
            raise RuntimeError(
                "No camera source found. Please provide one of: --cameras_npz, --sfmdata, or --transform_json. "
                "Alternatively, place cameras.npz in parent folder or transform.json in the data folder."
            )

    # Outputs
    if folder.endswith("/"):
        folder = folder[:-1]
    exp_name = os.path.basename(folder)
    output_path = os.path.join(folder, "..", exp_name + "-albedoscaled")
    os.makedirs(output_path, exist_ok=True)
    shutil.copyfile(transform_path, os.path.join(output_path, "transform.json"))
    if os.path.exists(normal_path):
        shutil.copytree(normal_path, os.path.join(output_path, "normals"), dirs_exist_ok=True)

    print(f"Computing albedo scaling ratios...")
    print(f"  Albedo path: {albedo_path}")
    print(f"  Mesh path: {mesh_path}")
    print(f"  Camera source: {camera_source}")
    print(f"  Number of samples: {args.n_samples}")
    
    # Compute scale ratios using the library
    scale_ratios = compute_albedo_scale_ratios(
        albedo_path=albedo_path,
        camera_source=camera_source,
        mesh_path=mesh_path,
        n_samples=args.n_samples,
        logger=None  # Will use print statements
    )

    # Save ratios
    np.save(os.path.join(output_path, "ratios.npy"), scale_ratios)
    print(f"Saved ratios to {os.path.join(output_path, 'ratios.npy')}")

    # Scale and save albedos
    print("Scaling and saving albedos...")
    output_albedo_path = os.path.join(output_path, "albedos")
    scale_and_save_albedos(
        albedo_path=albedo_path,
        output_albedo_path=output_albedo_path,
        scale_ratios=scale_ratios,
        bit_depth=args.bit_depth,
        logger=None  # Will use print statements
    )
    
    print(f"Done! Scaled albedos saved to {output_path}")

