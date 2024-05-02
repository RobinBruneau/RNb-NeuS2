import argparse
import json
import numpy as np
import os
import cv2
from glob import glob
import shutil
import matplotlib.pyplot as plt

def load_K_Rt_from_P(P=None):

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose # K, RT_c2w

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.dtype == "uint8":
        bit_depth = 8
    elif image.dtype == "uint16":
        bit_depth = 16
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/np.float32(2**bit_depth - 1)

def load_normal(path):
    image = load_image(path)
    normal = image * 2.0 - 1.0  # Convert to range [-1, 1]
    normal[:,:,1] = -normal[:,:,1] # y axis is flipped
    normal[:,:,2] = -normal[:,:,2] # z axis is flipped
    return normal

def save_image(path, image, bit_depth=8, mask=None):
    
    image_cp = np.copy(image)
    if mask is not None:
        mask = mask[:,:,np.newaxis]
        image_cp = np.concatenate([image_cp, mask], axis=-1)
    image_cp = (image_cp * (2**bit_depth - 1))
    
    if bit_depth == 8:
        image_cp = image_cp.astype(np.uint8)
    elif bit_depth == 16:
        image_cp = image_cp.astype(np.uint16)
    
    if mask is not None:
        image_cp = cv2.cvtColor(image_cp, cv2.COLOR_RGBA2BGRA)
    else:
        image_cp = cv2.cvtColor(image_cp, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_cp, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def save_normal(path, normal, bit_depth=8):
    normal_flipped = np.copy(normal)
    normal_flipped[:,:,1] = -normal_flipped[:,:,1] # y axis is flipped
    normal_flipped[:,:,2] = -normal_flipped[:,:,2] # z axis is flipped
    image = (normal_flipped + 1) / 2
    save_image(path, image, bit_depth=bit_depth)

def load_images_folder(folder):
    images = []
    for image_path in sorted(glob(os.path.join(folder, "*.png"))):
        images.append(load_image(image_path))
    return images

def load_normals_folder(folder):
    normals = []
    for normal_path in sorted(glob(os.path.join(folder, "*.png"))):
        normals.append(load_normal(normal_path))
    return normals

def gen_light_directions(normal=None):
    tilt = np.radians([0, 120, 240])
    slant = np.radians([30, 30, 30]) if normal is None else np.radians([54.74, 54.74, 54.74])
    n_lights = tilt.shape[0]

    u = -np.array([
        np.sin(slant) * np.cos(tilt),
        np.sin(slant) * np.sin(tilt),
        np.cos(slant)
    ]) # [3, n_lights]

    if normal is not None:
        n_rows, n_cols, _, _ = normal.shape # [H, W, 1, 3]
        light_directions = np.zeros((n_rows, n_cols, 3, n_lights))
        outer_prod = np.einsum('...j,...k->...jk', normal[:,:,0,:], normal[:,:,0,:]) # [H, W, 3, n_lights
        U, _, _ = np.linalg.svd(outer_prod)
        det_U = np.linalg.det(U)
        det_U_sign = np.where(det_U < 0, -1, 1)[..., np.newaxis, np.newaxis]
        R = np.where(det_U_sign < 0, 
                    np.einsum('...ij,jk->...ik', U, np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])), 
                    np.einsum('...ij,jk->...ik', U, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])))
        R_22 = (R[..., 2, 2] < 0)[..., np.newaxis, np.newaxis]
        R = np.where(R_22, np.einsum('...ij,jk->...ik', R, np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])), R)
        light_directions_all = np.einsum('...lm,mn->...ln', R, u) # [H, W, 3, n_lights]
        light_directions = light_directions_all.transpose(0, 1, 3, 2) # [H, W, n_lights, 3]
    else:
        light_directions = u.transpose(1, 0)[np.newaxis, np.newaxis, ...] # [1, 1, n_lights, 3]

    return light_directions

def export_NeuS2(folder, W, H, intrinsics_all, pose_all, scale_mats_np, lights=[]):

    output = {
        "w": W,
        "h": H,
        "aabb_scale": 1.0,
        "scale": 0.5,
        "offset": [  # neus: [-1,1] ngp[0,1]
            0.5,
            0.5,
            0.5
        ],
        "from_na": True,
    }

    output.update({"n2w": scale_mats_np[0].tolist()})

    output['frames'] = []
    all_image_dir = sorted(os.listdir(os.path.join(folder, "images")))
    mult = 3
    image_num = len(all_image_dir) // mult
    camera_num = intrinsics_all.shape[0]
    assert image_num == camera_num, "The number of cameras should be equal to the number of images!"
    for i in range(image_num):
        for j in range(mult):
            rgb_dir = os.path.join("images", all_image_dir[mult*i+j])
            ixt = intrinsics_all[i]

            # add one_frame
            one_frame = {}
            one_frame["file_path"] = rgb_dir
            one_frame["transform_matrix"] = pose_all[i].tolist()
            if len(lights) != 0 :
                one_frame["light"] = lights[i][j].tolist()
            else :
                one_frame["light"] = [0,0,0]

            one_frame["intrinsic_matrix"] = ixt.tolist()
            output['frames'].append(one_frame)

    file_dir = os.path.join(folder, 'transform.json')
    with open(file_dir, 'w') as f:
        json.dump(output, f, indent=4)

def preprocess(folder):

    # Get inputs folders
    print("Getting input data...")
    normalFolder = os.path.join(folder,"normal")
    maskFolder = os.path.join(folder,"mask")
    albedoFolder = os.path.join(folder,"albedo")
    camerasNpzFile = os.path.join(folder,"cameras.npz")

    # Get normal, mask and albedo data
    normal_data = load_normals_folder(normalFolder)
    mask_data = load_images_folder(maskFolder)
    if os.path.exists(albedoFolder):
        albedo_data = load_images_folder(albedoFolder)
    n_images = len(normal_data)

    # Get camera data
    camera_dict = np.load(camerasNpzFile)
    # world_mat is a projection matrix from world to image
    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    scale_mats_np = []
    # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(P) # K, RT_c2w
        intrinsics_all.append(intrinsics)
        pose_all.append(pose)

    # Create folders for NeuS2
    print("Creating folders for Neus2...")
    mainFolder = os.path.join(folder,"NeuS2_python")
    mainFolder30 = os.path.join(mainFolder,"NeuS2_l30")
    mainFolderOpti = os.path.join(mainFolder,"NeuS2_lopti")
    imagesFolder30 = os.path.join(mainFolder30,"images")
    imagesFolderOpti = os.path.join(mainFolderOpti,"images")
    if os.path.exists(mainFolder):
        shutil.rmtree(mainFolder)
    os.makedirs(mainFolder, exist_ok=True)
    os.makedirs(mainFolder30, exist_ok=True)
    os.makedirs(mainFolderOpti, exist_ok=True)
    os.makedirs(imagesFolder30, exist_ok=True)
    os.makedirs(imagesFolderOpti, exist_ok=True)

    # Generate images
    print("Generating rendered images...")
    nImages = len(normal_data)
    lights_30 = []
    lights = []
    for i in range(nImages):

        print(f"Generating image {i+1}/{nImages}...")

        # Get data
        normal = normal_data[i][:,:,np.newaxis,:] # [H, W, 1, 3]
        mask = mask_data[i][:,:,0] # [H, W]
        if os.path.exists(albedoFolder):
            albedo = albedo_data[i][:,:,np.newaxis,:] # [H, W, 1, 3]
        else:
            albedo = np.ones_like(normal)[:,:,np.newaxis,:] # [H, W, 1, 3]
        H, W, _, _ = normal.shape

        # Generate 30 degrees light directions and optimal light directions
        light_directions_30 = gen_light_directions() # [1, 1, n_lights, 3]
        light_directions = gen_light_directions(normal) # [H, W, n_lights, 3]

        # Generate images
        images_30 = albedo * np.maximum(np.sum(normal * light_directions_30, axis=-1)[..., np.newaxis], 0) # [H, W, n_lights, 3]
        images = albedo * np.maximum(np.sum(normal * light_directions, axis=-1)[..., np.newaxis], 0) # [H, W, n_lights, 3]

        # Convert lights to the world frame
        R_c2w = np.transpose(pose_all[i][:3,:3])
        light_directions_30_world = np.einsum('...ij,...kj->...ki', R_c2w, light_directions_30) # [1, 1, n_lights, 3]
        light_directions_world = np.einsum('...ij,...kj->...ki', R_c2w, light_directions) # [H, W, n_lights, 3]

        # Save images
        for j in range(images.shape[2]):
            image = images[:,:,j,:]
            image_30 = images_30[:,:,j,:]
            save_image(os.path.join(imagesFolderOpti, f"{i:03d}_{j:03d}.png"), image, bit_depth=16, mask=mask)
            save_image(os.path.join(imagesFolder30, f"{i:03d}_{j:03d}.png"), image_30, bit_depth=16, mask=mask)

        # Save lights
        lights_30.append(light_directions_30_world[0,0,:,:])
        light_directions_world = light_directions_world * mask[:,:,np.newaxis,np.newaxis] # [H, W, n_lights, 3]
        lights.extend(light_directions_world.ravel().tolist())

    # Save lights
    with open(os.path.join(mainFolderOpti,"lights.json"),'w') as f:
        json.dump({"lights":lights},f)

    print("Data convertion from NeuS to NeuS2...")
    export_NeuS2(mainFolder30, W, H, np.array(intrinsics_all), np.array(pose_all), np.array(scale_mats_np), lights_30) 
    shutil.copy(os.path.join(mainFolder30,"transform.json"),mainFolderOpti)

    print("Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)  # Parse the argument
    args = parser.parse_args()

    folder = args.folder
    preprocess(folder)



