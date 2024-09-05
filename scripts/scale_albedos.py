import os
import numpy as np
import argparse
import shutil
import trimesh
import cv2
import json
from pyoctree import pyoctree as ot
from scipy.interpolate import RegularGridInterpolator
import tqdm

def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Read as 8 or 16 bits
    if image.dtype == "uint8":
        bit_depth = 8
    elif image.dtype == "uint16":
        bit_depth = 16
    
    # Check if the image has a fourth channel
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image/np.float32(2**bit_depth - 1)

def save_image(image, path, bit_depth=8):

    # Convert to 8 or 16 bits
    image = (image * np.float32(2**bit_depth - 1))
    if bit_depth == 8:
        image = image.astype(np.uint8)
    elif bit_depth == 16:
        image = image.astype(np.uint16)

    # Convert to BGR or BGRA
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Write the image
    cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def get_ray(pixel, K, R_c2w, center):
    pixel = pixel.reshape(-1, 1)
    pixel = np.concatenate((pixel, np.ones((1, 1))), axis=0)
    K_inv = np.linalg.inv(K)
    pixel_c2w = 500*K_inv @ pixel
    pixel_w2c = R_c2w @ pixel_c2w + center
    ray = np.concatenate((center.T, pixel_w2c.T), axis=0)
    return ray

def load_K_Rt_from_P(P):
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

    return intrinsics, pose


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder path.")
    args = parser.parse_args()

    # Inputs
    folder = args.folder
    mesh_path = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("mesh_") and f.endswith(".obj")][0]
    albedo_path = os.path.join(folder, "albedos")
    normal_path = os.path.join(folder, "normals")
    transform_path = os.path.join(folder, "transform.json")
    cameras_npz_path = os.path.join(folder, "../cameras.npz")

    # Outputs
    if folder.endswith("/"):
        folder = folder[:-1]
    exp_name = os.path.basename(folder)
    output_path = os.path.join(folder, "..", exp_name + "-albedoscaled")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    shutil.copyfile(transform_path, os.path.join(output_path, "transform.json"))
    shutil.copytree(normal_path, os.path.join(output_path, "normals"))
    os.makedirs(os.path.join(output_path, "albedos"), exist_ok=True)

    # Load albedos and masks
    n_views = len(os.listdir(albedo_path))
    albedos = []
    masks = []
    for i in range(n_views):
        name = f"{i:03d}.png"
        albedo = load_image(os.path.join(albedo_path, name))
        mask = albedo[:, :, 3]
        albedo = albedo[:, :, :3]
        albedos.append(albedo)
        masks.append(mask)
    albedos = np.array(albedos)
    masks = np.array(masks)
    n_views, h, w, _ = albedos.shape

    # Load camera parameters
    data_cam = np.load(cameras_npz_path)
    K_array = []
    R_c2w_array = []
    centers_array = []
    for k in range(n_views):
        K, RT_c2w = load_K_Rt_from_P(data_cam[f"world_mat_{k}"][:3, :])
        R_c2w_array.append(RT_c2w[:3, :3])
        centers_array.append(RT_c2w[:3, [3]])
        K_array.append(K[:3, :3])
    K_array = np.array(K_array)
    R_c2w_array = np.array(R_c2w_array)
    centers_array = np.array(centers_array)

    # Load mesh 
    print("Loading mesh...")
    mesh = trimesh.load_mesh(mesh_path)
    vertices = mesh.vertices
    faces = mesh.faces.astype(np.int32)
    print("Creating octree...")
    octree = ot.PyOctree(vertices, faces)

    # For all cameras, loop
    print("Computing number of samples...")
    sum_masks = np.sum(np.sum(masks, axis=1), axis=1)
    mean_sum_masks = np.mean(sum_masks)
    min_sum_masks = np.min(sum_masks)
    n_samples = 2000
    print(f"Number of samples per image: {n_samples}")
    ratios = np.zeros((n_views, n_samples, 3, 2), dtype=np.float32)
    intersection_found = np.zeros((n_views, n_samples, 2), dtype=np.bool_)
    print("Computing ratios...")
    for cam_id in range(n_views):

        print(f"Processing Camera {cam_id}...")

        # Get all pixels of the first image in the mask
        mask = masks[cam_id, :, :]
        mask = mask.astype(np.bool_)
        ind_mask = np.where(mask)
        pixels = np.zeros((ind_mask[0].shape[0], 2))
        pixels[:, 0] = ind_mask[1] + 0.5
        pixels[:, 1] = ind_mask[0] + 0.5

        # Get albedo values for all pixels
        albedo = albedos[cam_id, :, :, :]
        albedo_values = albedo[ind_mask[0], ind_mask[1], :]
        albedo_values = albedo_values.reshape(-1, 3)

        # Precompute the current camera parameters
        current_K = K_array[cam_id]
        current_R_c2w = R_c2w_array[cam_id]
        current_center = centers_array[cam_id]

        # Check number of pixels
        n_samples_good = np.min((n_samples, pixels.shape[0]))
        if n_samples_good < n_samples:
            print(f"Warning: not enough pixels in image {cam_id} to get {n_samples} samples.")
            print(f"Only {n_samples_good} samples will be taken.")

        # Get a random set of pixels and their corresponding albedo values
        ind = np.random.choice(pixels.shape[0], n_samples_good, replace=False)
        pixels = pixels[ind, :]
        albedo_values = albedo_values[ind, :]

        # For each pixel, find the closest point on the mesh (tqdm)
        for i, pixel in enumerate(tqdm.tqdm(pixels, desc=f"Camera {cam_id}")):
            ray = get_ray(pixel, K_array[cam_id], R_c2w_array[cam_id], centers_array[cam_id])
            ray = ray.astype(np.float32)

            # Get the intersection point
            intersection = octree.rayIntersection(ray)
            if len(intersection) == 0:
                continue # print(f"No intersection for pixel {i}")
            
            intersection_point = np.array(intersection[0].p, dtype=np.float32).reshape(3, 1)

            # Get neighbor camera id
            right_cam_id = (cam_id + 1) % n_views
            left_cam_id = (cam_id - 1) % n_views
            for kk, neigh_cam_id in enumerate([right_cam_id, left_cam_id]):

                # Precompute neighbor camera parameters
                neighbor_K = K_array[neigh_cam_id]
                neighbor_R_c2w = R_c2w_array[neigh_cam_id]
                neighbor_center = centers_array[neigh_cam_id]

                # Convert intersection point from world to camera space
                ray_neighbor = np.concatenate((intersection_point.T, neighbor_center.T), axis=0).astype(np.float32)

                # Check if ray intersects any objects in the octree
                neighbor_intersection = octree.rayIntersection(ray_neighbor)
                bol_intersection_between_point_and_cam = False # np.any([nb.s for nb in neighbor_intersection] > 0)
                for nb in neighbor_intersection:
                    if nb.s > 0:
                        bol_intersection_between_point_and_cam = True
                        break
                if bol_intersection_between_point_and_cam:
                    continue

                # Convert intersection point from world to camera space
                neighbor_R_w2c = neighbor_R_c2w.T
                neighbor_t_w2c = -neighbor_R_w2c @ neighbor_center
                intersection_in_neighbor = neighbor_R_w2c @ intersection_point + neighbor_t_w2c
                intersection_in_neighbor = neighbor_K @ intersection_in_neighbor
                intersection_in_neighbor /= intersection_in_neighbor[2]
                intersection_in_neighbor = intersection_in_neighbor[:2]

                # Ensure pixel_in_neighbor is in the correct shape for interpolation
                pixel_in_neighbor = np.array([intersection_in_neighbor[1]-0.5, intersection_in_neighbor[0]-0.5]).reshape(1, 2)

                if 0 <= pixel_in_neighbor[0, 0] < h and 0 <= pixel_in_neighbor[0, 1] < w:
                    # Create interpolation function for albedo values
                    albedo_in_neighbor = albedos[neigh_cam_id, :, :, :].astype(np.float32)
                    rows_inds = np.linspace(0, albedo_in_neighbor.shape[0], albedo_in_neighbor.shape[0])
                    cols_inds = np.linspace(0, albedo_in_neighbor.shape[1], albedo_in_neighbor.shape[1])
                    interpR = RegularGridInterpolator((rows_inds, cols_inds), albedo_in_neighbor[:,:,0])
                    interpG = RegularGridInterpolator((rows_inds, cols_inds), albedo_in_neighbor[:,:,1])
                    interpB = RegularGridInterpolator((rows_inds, cols_inds), albedo_in_neighbor[:,:,2])
                    
                    albedo_val = np.array([interpR(pixel_in_neighbor), interpG(pixel_in_neighbor), interpB(pixel_in_neighbor)])

                    if np.all(albedo_val != 0):
                        ratios[cam_id, i, :, kk] = albedo_values[i] / albedo_val[:, 0]
                        intersection_found[cam_id, i, kk] = True

    # Get concatenated ratios
    median_ratios = np.zeros((n_views, 3))
    right_ratios = ratios[:, :, :, 0]
    right_ind = intersection_found[:, :, 0]
    left_ratios = np.roll(ratios[:, :, :, 1], -1, axis=0)
    left_ind = np.roll(intersection_found[:, :, 1], -1, axis=0)
    for cam_id in range(n_views):

        right_ratio = right_ratios[cam_id, :, :]
        left_ratio = left_ratios[cam_id, :, :]

        right_ind_cam = right_ind[cam_id, :]
        left_ind_cam = left_ind[cam_id, :]

        right_ratio = right_ratio[right_ind_cam, :]
        left_ratio = 1 / left_ratio[left_ind_cam, :]
        
        all_ratio = np.concatenate((right_ratio, left_ratio), axis=0)
        median_ratios[cam_id, :] = np.median(all_ratio, axis=0)

    # Update the median ratios
    median_ratio_prop = np.ones((n_views, 3))
    for ii in range(n_views-1):
        median_ratio_prop[ii+1, :] = median_ratio_prop[ii, :] * median_ratios[ii, :]
        # print(ii+1, median_ratio_prop[ii+1, :])

    # Compute the mean ratio
    mean_median_ratio_prop = np.mean(median_ratio_prop, axis=0)
    # print(f"Mean ratio: {mean_median_ratio_prop}")
    median_ratio_prop_norm = median_ratio_prop / mean_median_ratio_prop
    print(f"Scale ratios to apply to each albedo: {median_ratio_prop_norm}")

    # Load albedo images
    print("Scaling and saving albedos...")
    for ii in range(n_views):

        # Scale albedo
        albedo = albedos[ii, ...]
        mask = masks[ii, ...]
        albedo *= median_ratio_prop_norm[ii, :]
        albedo_to_save = np.concatenate((albedo, mask[:, :, np.newaxis]), axis=-1)

        # Save albedo
        output_im_path = os.path.join(output_path, "albedos", f"{ii:03d}.png")
        save_image(albedo_to_save, output_im_path, bit_depth=16)
        print(f"Saved {output_im_path}")
    
    