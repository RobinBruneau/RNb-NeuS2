import numpy as np
import cv2
import argparse
import os
import glob
import shutil
import trimesh
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Select images for IDR')
    parser.add_argument('--data_path', type=str, default='data/DiLiGenT-MV/buddhaPNG', help='Path to the DiLiGenT-MV dataset')
    parser.add_argument('--output_path', type=str, default='/tmp/buddhaPNG', help='Path to the output folder')
    parser.add_argument('--ind_images', type=int, nargs='+', default=[18, 19, 0, 1, 2], help='Indices of images to keep')
    ## ex: python select_ind_idr.py --data_path data/DiLiGenT-MV/buddhaPNG --output_path /tmp/buddhaPNG --ind_images 18 19 0 1 2 
    return parser.parse_args()

if __name__ == '__main__':

    # Parse arguments
    arg = parse_args()
    DATA_PATH = arg.data_path
    OUTPUT_PATH = arg.output_path
    ind_images = arg.ind_images

    # Get camera poses
    CAMERAS_NPZ = os.path.join(DATA_PATH, 'cameras.npz')
    ALBEDOS_PATH = os.path.join(DATA_PATH, 'albedo')
    # IMAGES_PATH = os.path.join(DATA_PATH, 'image')
    NORMALS_PATH = os.path.join(DATA_PATH, 'normal')
    MASKS_PATH = os.path.join(DATA_PATH, 'mask')

    OUT_CAMERAS_NPZ = os.path.join(OUTPUT_PATH, 'cameras.npz')
    
    OUT_NORMALS_PATH = os.path.join(OUTPUT_PATH, 'normal')
    OUT_MASKS_PATH = os.path.join(OUTPUT_PATH, 'mask')
    # OUT_IMAGES_PATH = os.path.join(OUTPUT_PATH, 'image')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    # if not os.path.exists(OUT_IMAGES_PATH):
    #     os.makedirs(OUT_IMAGES_PATH)
    if not os.path.exists(OUT_MASKS_PATH):
        os.makedirs(OUT_MASKS_PATH)
    if not os.path.exists(OUT_NORMALS_PATH):
        os.makedirs(OUT_NORMALS_PATH)

    if os.path.exists(ALBEDOS_PATH):
        OUT_ALBEDOS_PATH = os.path.join(OUTPUT_PATH, 'albedo')
        if not os.path.exists(OUT_ALBEDOS_PATH):
            os.makedirs(OUT_ALBEDOS_PATH)
    
    # Settings
    n_images_ = len(ind_images)

    # Copy images and masks
    if os.path.exists(ALBEDOS_PATH):
        albedo_files = glob.glob(os.path.join(ALBEDOS_PATH, '*.png'))
        albedo_files.sort()
    normal_files = glob.glob(os.path.join(NORMALS_PATH, '*.png'))
    normal_files.sort()
    mask_files = glob.glob(os.path.join(MASKS_PATH, '*.png'))
    mask_files.sort()
    # image_files = glob.glob(os.path.join(IMAGES_PATH, '*.png'))
    # image_files.sort()
    for ii in range(n_images_):
        ind = ind_images[ii]
        if os.path.exists(ALBEDOS_PATH):

            # copy with a symbolic link with relative path
            albedo_file = os.path.relpath(albedo_files[ind], OUT_ALBEDOS_PATH)
            if not os.path.exists(os.path.join(OUT_ALBEDOS_PATH, "%03d.png" % ii)):
                os.symlink(albedo_file, os.path.join(OUT_ALBEDOS_PATH, "%03d.png" % ii))

        normal_file = os.path.relpath(normal_files[ind], OUT_NORMALS_PATH)
        if not os.path.exists(os.path.join(OUT_NORMALS_PATH, "%03d.png" % ii)):
            os.symlink(normal_file, os.path.join(OUT_NORMALS_PATH, "%03d.png" % ii))
        
        mask_file = os.path.relpath(mask_files[ind], OUT_MASKS_PATH)
        if not os.path.exists(os.path.join(OUT_MASKS_PATH, "%03d.png" % ii)):
            os.symlink(mask_file, os.path.join(OUT_MASKS_PATH, "%03d.png" % ii))

        # image_file = os.path.relpath(image_files[ind], OUT_IMAGES_PATH)
        # if not os.path.exists(os.path.join(OUT_IMAGES_PATH, "%03d.png" % ii)):
        #     os.symlink(image_file, os.path.join(OUT_IMAGES_PATH, "%03d.png" % ii))

    # Load camera poses    
    camera_dict = np.load(CAMERAS_NPZ)
    n_images = len([key for key in camera_dict.files if 'world_mat' in key and not 'inv' in key])
    
    if "camera_mat_0" in camera_dict.files:
        camera_mats_np = [camera_dict['camera_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    if "camera_mat_inv_0" in camera_dict.files:
        camera_mats_inv_np = [camera_dict['camera_mat_inv_%d' % idx].astype(np.float32) for idx in range(n_images)]
    if "world_mat_0" in camera_dict.files:
        world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    if "world_mat_inv_0" in camera_dict.files:
        world_mats_inv_np = [camera_dict['world_mat_inv_%d' % idx].astype(np.float32) for idx in range(n_images)]
    if "scale_mat_0" in camera_dict.files:
        scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    if "scale_mat_inv_0" in camera_dict.files:
        scale_mats_inv_np = [camera_dict['scale_mat_inv_%d' % idx].astype(np.float32) for idx in range(n_images)]

    # Keep only the selected images
    new_dict = {}
    for ii in range(n_images_):
        ind = ind_images[ii]

        if "camera_mat_0" in camera_dict.files:
            new_dict['camera_mat_%d' % ii] = camera_mats_np[ind]
        if "camera_mat_inv_0" in camera_dict.files:
            new_dict['camera_mat_inv_%d' % ii] = camera_mats_inv_np[ind]
        if "world_mat_0" in camera_dict.files:
            new_dict['world_mat_%d' % ii] = world_mats_np[ind]
        if "world_mat_inv_0" in camera_dict.files:
            new_dict['world_mat_inv_%d' % ii] = world_mats_inv_np[ind]
        if "scale_mat_0" in camera_dict.files:
            new_dict['scale_mat_%d' % ii] = scale_mats_np[ind]
        if "scale_mat_inv_0" in camera_dict.files:
            new_dict['scale_mat_inv_%d' % ii] = scale_mats_inv_np[ind]

        np.savez(OUT_CAMERAS_NPZ, **new_dict)