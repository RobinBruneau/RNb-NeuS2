import os
import numpy as np
import argparse
import shutil
import trimesh
import cv2
from scipy.interpolate import RegularGridInterpolator
import tqdm
import matplotlib.pyplot as plt

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

def plot_projected_points():

    #
    selection = np.random.choice(pixels_valid.shape[0], 10, replace=False)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
    for jj, ind_jj in enumerate(selection):
        print(f"Color: {colors[jj]}")
        print(f"Albedo value in neighbor image: {albedo_val[ind_jj, :]}")
        print(f"Albedo value in current image: {albedo_values_valid[ind_jj, :]}")
        print(f"Ratio: {albedo_values_valid[ind_jj, :] / albedo_val[ind_jj, :]}")
        print("")

    # Plot points that are zero
    plt.subplot(1, 3, 1)
    plt.imshow(albedos[cam_id, :, :, :])
    plt.title(f"Current image (#{cam_id})")
    for jj, ind_jj in enumerate(selection):
        plt.scatter(pixels_valid[ind_jj, 0], pixels_valid[ind_jj, 1], c=colors[jj])
    
    plt.subplot(1, 3, 2)
    plt.imshow(albedos[neigh_cam_id, :, :, :])
    plt.title(f"Neighbor image (#{neigh_cam_id})")
    for jj, ind_jj in enumerate(selection):
        plt.scatter(intersection_points_in_neighbor_cam[ind_jj, 0], intersection_points_in_neighbor_cam[ind_jj, 1], c=colors[jj])
    
    plt.subplot(1, 3, 3)
    plt.imshow(albedos[neigh_cam_id, :, :, :])
    for jj, ind_jj in enumerate(selection):
        plt.scatter(intersection_points_in_neighbor_cam_yx[ind_jj, 0], intersection_points_in_neighbor_cam_yx[ind_jj, 1], c=colors[jj])
    
    plt.show()


if __name__ == "__main__":

    # fix the seed
    np.random.seed(0)

    # fix the seed
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder path.")
    parser.add_argument("--mesh_path", type=str, default=None, help="Path to the mesh.")
    args = parser.parse_args()

    # Inputs
    folder = args.folder
    if args.mesh_path is not None:
        mesh_path = args.mesh_path
    else:
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
    os.makedirs(output_path, exist_ok=True)
    shutil.copyfile(transform_path, os.path.join(output_path, "transform.json"))
    shutil.copytree(normal_path, os.path.join(output_path, "normals"), dirs_exist_ok=True)
    os.makedirs(os.path.join(output_path, "albedos"), exist_ok=True)

    # Load albedos and masks
    n_views = len(os.listdir(albedo_path))
    albedos = []
    masks = []
    list_names = sorted([f for f in os.listdir(albedo_path) if f.endswith(".png")])
    for i in range(n_views):
        name = list_names[i]
        albedo = load_image(os.path.join(albedo_path, name))
        mask = albedo[:, :, 3]
        # mask = load_image(os.path.join(albedo_path, name).replace("albedo", "mask"))[:, :, 0]
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

    # For all cameras, loop
    print("Computing number of samples...")
    sum_masks = np.sum(np.sum(masks, axis=1), axis=1)
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
        pixels[:, 0] = ind_mask[1]
        pixels[:, 1] = ind_mask[0]

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

        # Create rays
        rays_origin = np.tile(current_center.T, (n_samples_good, 1))
        point_on_rays = (current_R_c2w @ (np.linalg.inv(current_K) @ np.concatenate((pixels, np.ones((n_samples_good, 1))), axis=1).T) + current_center).T
        rays_direction = point_on_rays - rays_origin
        rays_direction /= np.linalg.norm(rays_direction, axis=1)[:, None]

        # Get the intersection point
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=rays_origin,
            ray_directions=rays_direction,
            multiple_hits=False
        )
        # locations = array([[-0.28622812,  0.33651689,  0.05616308],
        #    [ 0.13977279,  0.24709683, -0.09688516],
        #    [ 0.01342933,  0.24768316, -0.19124962],
        #    ...,
        #    [-0.21286387, -0.00275792, -0.17479006],
        #    [-0.26448901, -0.02001748, -0.15498513],
        #    [-0.22350254,  0.03773047, -0.18866993]])
        # index_ray = array([   0,    1,    2, ...,  650, 1604, 1726])

        # Get first intersection point for each ray
        # index_ray = np.unique(index_ray)
        # locations = locations[index_ray, :]
        pixels = pixels[index_ray, :]
        albedo_values = albedo_values[index_ray, :]

        # Plot two figures, one with the albedo and the pixels and the other with the vertices
        display_ = False
        if display_:

            selection = np.random.choice(pixels.shape[0], 10, replace=False)
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']

            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.imshow(albedos[cam_id, :, :, :])
            for jj, ind_jj in enumerate(selection):
                plt.scatter(pixels[ind_jj, 0], pixels[ind_jj, 1], c=colors[jj % 10])

            # Plot in 3D some vertices and intersection points
            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            vertices_selected = vertices[np.random.choice(vertices.shape[0], 1000, replace=False), :]
            ax.scatter(vertices_selected[:, 0], vertices_selected[:, 1], vertices_selected[:, 2], c='b', s=1)
            for jj, ind_jj in enumerate(selection):
                ax.scatter(locations[ind_jj, 0], locations[ind_jj, 1], locations[ind_jj, 2], c=colors[jj % 10], s=50)

            plt.show()

        # Get neighbor camera id
        right_cam_id = (cam_id + 1) % n_views
        left_cam_id = (cam_id - 1) % n_views
        for kk, neigh_cam_id in enumerate([right_cam_id, left_cam_id]):

            # Precompute neighbor camera parameters
            neighbor_K = K_array[neigh_cam_id] # 3x3
            neighbor_R_c2w = R_c2w_array[neigh_cam_id] # 3x3
            neighbor_center = centers_array[neigh_cam_id]

            # Create ray from intersection point to neighbor camera center
            neighbor_rays_direction = neighbor_center.T - locations
            neighbor_rays_direction /= np.linalg.norm(neighbor_rays_direction, axis=1)[:, None]
            neighbor_rays_origin = locations + 1e-3 * neighbor_rays_direction
            hit = mesh.ray.intersects_any(
                ray_origins=neighbor_rays_origin,
                ray_directions=neighbor_rays_direction
            ) # bool (n_samples_good, )

            # Keep only the intersection points that do not hit the mesh
            intersection_points = locations[~hit, :]
            index_ray_kk = index_ray[~hit]
            pixels_no_hit = pixels[~hit, :]
            albedo_values_no_hit = albedo_values[~hit, :]
            assert intersection_points.shape[0] == index_ray_kk.shape[0]

            display_ = False
            if display_:

                selection = np.random.choice(pixels_no_hit.shape[0], 10, replace=False)
                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']

                import matplotlib.pyplot as plt
                plt.figure(1)
                plt.imshow(albedos[cam_id, :, :, :])
                for jj, ind_jj in enumerate(selection):
                    plt.scatter(pixels_no_hit[ind_jj, 0], pixels_no_hit[ind_jj, 1], c=colors[jj % 10])

                # Plot in 3D some vertices and intersection points
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                vertices_selected = vertices[np.random.choice(vertices.shape[0], 1000, replace=False), :]
                ax.scatter(vertices_selected[:, 0], vertices_selected[:, 1], vertices_selected[:, 2], c='b', s=1)
                for jj, ind_jj in enumerate(selection):
                    ax.scatter(intersection_points[ind_jj, 0], intersection_points[ind_jj, 1], intersection_points[ind_jj, 2], c=colors[jj % 10], s=50)

                for jj, ind_jj in enumerate(selection):
                    ax.scatter(intersection_points[ind_jj, 0], intersection_points[ind_jj, 1], intersection_points[ind_jj, 2], c=colors[jj % 10], s=50)

                plt.show()

            # Project intersection points to neighbor camera
            neighbor_R_w2c = neighbor_R_c2w.T
            intersection_points_in_neighbor_cam = (neighbor_R_w2c @ (intersection_points.T - neighbor_center)) # (3, n_samples_good)
            intersection_points_in_neighbor_cam = (neighbor_K @ intersection_points_in_neighbor_cam).T # (n_samples_good, 3)
            intersection_points_in_neighbor_cam /= intersection_points_in_neighbor_cam[:, 2][:, None] # (n_samples_good, 3)
            intersection_points_in_neighbor_cam = intersection_points_in_neighbor_cam[:, :2] # (n_samples_good, 2) (xy-coordinates)

            # Check if the intersection points are inside the image
            valid_indices = (
                (0 <= intersection_points_in_neighbor_cam[:, 1]) & (intersection_points_in_neighbor_cam[:, 1] < h-1) &
                (0 <= intersection_points_in_neighbor_cam[:, 0]) & (intersection_points_in_neighbor_cam[:, 0] < w-1)
            )
            intersection_points_in_neighbor_cam = intersection_points_in_neighbor_cam[valid_indices, :]
            index_ray_kk = index_ray_kk[valid_indices]
            pixels_valid = pixels_no_hit[valid_indices, :]
            albedo_values_valid = albedo_values_no_hit[valid_indices, :]
            assert intersection_points_in_neighbor_cam.shape[0] == index_ray_kk.shape[0]

            display_ = False
            if display_:

                selection = np.random.choice(pixels_valid.shape[0], 10, replace=False)
                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']

                # Plot points on the image and the point on the neighbor image (subplot)
                plt.subplot(1, 2, 1)
                plt.imshow(albedos[cam_id, :, :, :])
                for jj, ind_jj in enumerate(selection):
                    plt.scatter(pixels_valid[ind_jj, 0], pixels_valid[ind_jj, 1], c=colors[jj])

                plt.subplot(1, 2, 2)
                plt.imshow(albedos[neigh_cam_id, :, :, :])
                for jj, ind_jj in enumerate(selection):
                    plt.scatter(intersection_points_in_neighbor_cam[ind_jj, 0], intersection_points_in_neighbor_cam[ind_jj, 1], c=colors[jj])

                plt.show()

            # Get albedo values in neighbor image
            # if 0 <= intersection_points_in_neighbor_cam[0, 0] < h  0 <= intersection_points_in_neighbor_cam[0, 1] < w:
            # Create interpolation function for albedo values
            albedo_in_neighbor = albedos[neigh_cam_id, :, :, :].astype(np.float32)
            rows_inds = np.arange(0, albedo_in_neighbor.shape[0], 1)
            cols_inds = np.arange(0, albedo_in_neighbor.shape[1], 1)
            interpR = RegularGridInterpolator((rows_inds, cols_inds), albedo_in_neighbor[:,:,0])
            interpG = RegularGridInterpolator((rows_inds, cols_inds), albedo_in_neighbor[:,:,1])
            interpB = RegularGridInterpolator((rows_inds, cols_inds), albedo_in_neighbor[:,:,2])

            # Interpolate for each channel
            intersection_points_in_neighbor_cam_yx = np.concatenate((intersection_points_in_neighbor_cam[:, 1][:, None], intersection_points_in_neighbor_cam[:, 0][:, None]), axis=1)
            albedo_R = interpR(intersection_points_in_neighbor_cam_yx)
            albedo_G = interpG(intersection_points_in_neighbor_cam_yx)
            albedo_B = interpB(intersection_points_in_neighbor_cam_yx)

            # Stack the interpolated values to get the final albedo values (n,3)
            albedo_val = np.stack([albedo_R, albedo_G, albedo_B], axis=1)
            assert albedo_val.shape[0] == intersection_points_in_neighbor_cam.shape[0]
            assert albedo_val.shape[0] == index_ray_kk.shape[0]

            # Compute the ratio between the albedo values for all non-zero values
            zero_indices = np.any(albedo_val == 0, axis=1)
            index_ray_kk = index_ray_kk[~zero_indices]
            pixels_valid = pixels_valid[~zero_indices, :]
            albedo_val = albedo_val[~zero_indices, :]
            intersection_points_in_neighbor_cam = intersection_points_in_neighbor_cam[~zero_indices, :]
            albedo_values_valid = albedo_values_valid[~zero_indices, :]

            display_ = False
            if display_:

                #
                selection = np.random.choice(pixels_valid.shape[0], 10, replace=False)
                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
                for jj, ind_jj in enumerate(selection):
                    print(f"Color: {colors[jj]}")
                    print(f"Albedo value in neighbor image: {albedo_val[ind_jj, :]}")
                    print(f"Albedo value in current image: {albedo_values_valid[ind_jj, :]}")
                    print(f"Ratio: {albedo_values_valid[ind_jj, :] / albedo_val[ind_jj, :]}")
                    print("")

                # Plot points that are zero
                plt.subplot(1, 3, 1)
                plt.imshow(albedos[cam_id, :, :, :])
                plt.title(f"Current image (#{cam_id})")
                for jj, ind_jj in enumerate(selection):
                    plt.scatter(pixels_valid[ind_jj, 0], pixels_valid[ind_jj, 1], c=colors[jj])
                
                plt.subplot(1, 3, 2)
                plt.imshow(albedos[neigh_cam_id, :, :, :])
                plt.title(f"Neighbor image (#{neigh_cam_id})")
                for jj, ind_jj in enumerate(selection):
                    plt.scatter(intersection_points_in_neighbor_cam[ind_jj, 0], intersection_points_in_neighbor_cam[ind_jj, 1], c=colors[jj])
                
                plt.subplot(1, 3, 3)
                plt.imshow(albedos[neigh_cam_id, :, :, :])
                for jj, ind_jj in enumerate(selection):
                    plt.scatter(intersection_points_in_neighbor_cam_yx[ind_jj, 0], intersection_points_in_neighbor_cam_yx[ind_jj, 1], c=colors[jj])
                
                plt.show()
                plt.close()

            ratios[cam_id, index_ray_kk, :, kk] = albedo_values_valid / albedo_val
            intersection_found[cam_id, index_ray_kk, kk] = True

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

    # Save ratios
    np.save(os.path.join(output_path, "ratios.npy"), median_ratio_prop_norm)

    # Load albedo images
    print("Scaling and saving albedos...")
    for ii in range(n_views):

        # Scale albedo
        albedo = albedos[ii, ...]
        mask = masks[ii, ...]
        albedo *= median_ratio_prop_norm[ii, :]
        albedo_to_save = np.concatenate((albedo, mask[:, :, np.newaxis]), axis=-1)

        # Save albedo
        output_im_path = os.path.join(output_path, "albedos", list_names[ii])
        save_image(albedo_to_save, output_im_path, bit_depth=16)
        print(f"Saved {output_im_path}")
    
    