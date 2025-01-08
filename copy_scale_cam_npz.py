import argparse
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cameras-npz", default="cameras.npz", help="input path of the .npz file")
    parser.add_argument("--copy-npz", default="copy.npz", help="output path of the .npz file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    CAMERAS_NPZ_PATH = args.cameras_npz
    COPY_NPZ_PATH = args.copy_npz

    # Load npz file
    cameras = np.load(CAMERAS_NPZ_PATH)
    copy = np.load(COPY_NPZ_PATH)
    n_cameras = len(cameras.files)

    # Copy scale matrix
    scale_matrix = copy['scale_mat_0']
    new_cameras = {}
    for ii in range(n_cameras):
        new_cameras[f'scale_mat_{ii}'] = scale_matrix
        new_cameras[f'world_mat_{ii}'] = cameras[f'world_mat_{ii}']

    # Save npz file
    np.savez(CAMERAS_NPZ_PATH, **new_cameras)