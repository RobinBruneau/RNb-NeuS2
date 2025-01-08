import argparse
import os
import numpy as np
import cv2

def _cv_to_gl(cv_matrix):
    """
    Convert a camera matrix from OpenCV convention to OpenGL convention.
    """
    cv_to_gl = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)
    gl_matrix = cv_to_gl @ cv_matrix
    return gl_matrix

def _gl_to_cv(gl_matrix):
    """
    Convert a camera matrix from OpenGL convention to OpenCV convention.
    """
    return _cv_to_gl(gl_matrix)

def load_image_opencv(path):
    
    # Load image
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Convert to float32
    if image.dtype == "uint8":
        bit_depth = 8
    elif image.dtype == "uint16":
        bit_depth = 16
    image = np.float32(image) / (2**bit_depth - 1)

    # Handle alpha channel if present
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def load_image_exr(path):

    # Load image
    image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    image = np.float32(image)

    # Convert from linear to sRGB
    def linear_to_srgb(linear):
        a = 0.055
        threshold = 0.0031308
        srgb = np.where(linear <= threshold,
                        linear * 12.92,
                        (1.0 + a) * np.power(linear, 1.0 / 2.4) - a)
        return np.clip(srgb, 0.0, 1.0)

    image = linear_to_srgb(image)

    # Handle alpha channel if present
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def load_image(path):

    # If .png, .jpg, or .jpeg, load with OpenCV
    if path.endswith((".png")):
        return load_image_opencv(path)
    elif path.endswith((".jpg", ".jpeg")):
        return load_image_opencv(path)
    elif path.endswith(".exr"):
        return load_image_exr(path)
    

def save_image(image, path, bit_depth=8):

    # Convert to uint8 or uint16
    image = (image * np.float32(2**bit_depth - 1))
    if bit_depth == 8:
        image = image.astype(np.uint8)
    elif bit_depth == 16:
        image = image.astype(np.uint16)

    # Handle alpha channel if present
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        image = image[...,:3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save image
    cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def save_mask(mask, path):
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(path, mask)


def get_view_parameters(view, intrinsics, poses):

    # Get image path
    path = view['path']
    
    # Get pose and intrinsic ids
    pose_id = view['poseId']
    intrinsic_id = view['intrinsicId']

    # Get intrinsic and pose data
    intrinsic = intrinsics[intrinsic_id]
    pose = poses[pose_id]
            
    # Get width, height
    width = float(intrinsic['width'])
    height = float(intrinsic['height'])

    # Get focal length
    if 'pxFocalLength' in intrinsic:
        fx = float(intrinsic['pxFocalLength'][0])
        fy = float(intrinsic['pxFocalLength'][1])
    else:
        sensor_width = float(intrinsic['sensorWidth'])
        sensor_height = float(intrinsic['sensorHeight'])
        focal_length = float(intrinsic['focalLength'])
        fx = focal_length * width / sensor_width
        fy = focal_length * height / sensor_height

    # Get principal point
    cx = width / 2 + float(intrinsic['principalPoint'][0])
    cy = height / 2 + float(intrinsic['principalPoint'][1])

    # Get intrinsics matrix
    K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    # Get rotation matrix and center in OpenGL convention
    R_c2w = np.array(pose['pose']['transform']['rotation'], dtype=np.float32).reshape([3,3]) # orientation
    center = np.expand_dims(np.array(pose['pose']['transform']['center'], dtype=np.float32), axis=1) # center
    Rt_c2w_gl = np.eye(4)
    Rt_c2w_gl[:3,:3] = R_c2w
    Rt_c2w_gl[:3,3] = center[:,0]

    return K, Rt_c2w_gl, path


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

def get_intrinsic_id(intrinsic_data, intrinsics_data):
    thr = 1e-2
    for intrinsicId, intrinsic in intrinsics_data.items():
        if "pxFocalLength" in intrinsic_data and "pxFocalLength" in intrinsic:
            if intrinsic_data["width"] == intrinsic["width"] and \
                intrinsic_data["height"] == intrinsic["height"] and \
                abs(float(intrinsic_data["pxFocalLength"][0]) - float(intrinsic["pxFocalLength"][0])) < thr and \
                abs(float(intrinsic_data["pxFocalLength"][1]) - float(intrinsic["pxFocalLength"][1]) < thr) and \
                abs(float(intrinsic_data["principalPoint"][0]) - float(intrinsic["principalPoint"][0]) < thr) and \
                abs(float(intrinsic_data["principalPoint"][1]) - float(intrinsic["principalPoint"][1]) < thr):
                return intrinsicId
        elif "focalLength" in intrinsic_data and "focalLength" in intrinsic:
            if intrinsic_data["width"] == intrinsic["width"] and \
                intrinsic_data["height"] == intrinsic["height"] and \
                abs(float(intrinsic_data["focalLength"]) - float(intrinsic["focalLength"]) < thr) and \
                abs(float(intrinsic_data["principalPoint"][0]) - float(intrinsic["principalPoint"][0]) < thr) and \
                abs(float(intrinsic_data["principalPoint"][1]) - float(intrinsic["principalPoint"][1]) < thr):
                return intrinsicId
        else:
            if intrinsic_data["width"] == intrinsic["width"] and \
                intrinsic_data["height"] == intrinsic["height"] and \
                abs(float(intrinsic_data["principalPoint"][0]) - float(intrinsic["principalPoint"][0]) < thr) and \
                abs(float(intrinsic_data["principalPoint"][1]) - float(intrinsic["principalPoint"][1]) < thr):
                return intrinsicId
    intrinsicId = intrinsic_data["intrinsicId"]
    return intrinsicId

def get_pose_id(pose_data, poses_data):
    thr = 1e-2
    for poseId, pose in poses_data.items():
        if np.allclose(np.array(pose_data["pose"]["transform"]["rotation"], dtype=np.float32), np.array(pose["pose"]["transform"]["rotation"], dtype=np.float32), rtol=thr) and \
            np.allclose(np.array(pose_data["pose"]["transform"]["center"], dtype=np.float32), np.array(pose["pose"]["transform"]["center"], dtype=np.float32), rtol=thr):
            return poseId
    poseId = pose_data["poseId"]
    return poseId

def read_idr_data(idr_folder_path, use_scale_matrix=False, size=None):

    # Find folders and cameras.npz
    image_folder = os.path.join(idr_folder_path, "normal")
    mask_folder = os.path.join(idr_folder_path, "mask")
    camera_path = os.path.join(idr_folder_path, "cameras.npz")

    if not os.path.exists(image_folder):
        raise ValueError("Image folder not found")
    if not os.path.exists(camera_path):
        raise ValueError("Camera file not found")
    
    # Get image extension and paths
    image_extension = os.path.splitext(os.listdir(image_folder)[0])[1]
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(image_extension)])
    n_images = len(image_paths)

    # Get mask paths
    mask_extension = os.path.splitext(os.listdir(mask_folder)[0])[1]
    mask_paths = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith(mask_extension)])
    if len(mask_paths) != n_images:
        mask_paths = [None for _ in range(n_images)]

    # Load camera data 
    camera_dict = np.load(camera_path)
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    if use_scale_matrix:
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    else:
        scale_mats = [np.eye(4, dtype=np.float32) for idx in range(n_images)]

    # Set views, intrinsics and poses
    views_data = {}
    intrinsics_data = {}
    poses_data = {}
    for i, (image_path, world_mat, scale_mat) in enumerate(zip(image_paths, world_mats, scale_mats)):

        # Set view id
        # viewId = f"{random.randint(10000000, 1000000000)}"
        # while viewId in views_data.keys():
        #     viewId = f"{random.randint(10000000, 1000000000)}"
        viewId = f"{i}"

        image_path = os.path.abspath(image_path)
        
        # Load image
        if size is None:
            image = load_image(image_path)
            width = image.shape[1]
            height = image.shape[0]
        else:
            width, height = size

        # Get width, height
        print(f"Image {i+1}/{n_images}: {image_path} ({width}x{height})")
        if abs(width/height - 4/3) < 0.1:
            sensor_width = 6.4
            sensor_height = 4.8
            sensor_width_str = f"{sensor_width:0.2f}"
            sensor_height_str = f"{sensor_height:0.2f}"
        elif abs(width/height - 3/2) < 0.1:
            sensor_width = 36
            sensor_height = 24
            sensor_width_str = f"{sensor_width:0.2f}"
            sensor_height_str = f"{sensor_height:0.2f}"
        else:
            print("Image aspect ratio is not 4:3 or 3:2")
            sensor_width = None
            sensor_height = None
            sensor_width_str = None
            sensor_height_str = None

        # Get pose and intrinsics data
        P = world_mat @ scale_mat
        intrinsics, pose = load_K_Rt_from_P(P[:3,:4])
        pose = _cv_to_gl(pose)

        # Get focal and principal point
        focal_px = float(intrinsics[0, 0])
        focal_py = float(intrinsics[1, 1])

        focal_px_mm = None
        focal_px_mm_str = None
        if sensor_width is not None:
            focal_px_mm = focal_px * sensor_width / width
            focal_px_mm_str = f"{focal_px_mm:0.20f}"

        ppx = float(intrinsics[0, 2])
        ppy = float(intrinsics[1, 2])
        cx = ppx - width / 2
        cy = ppy - height / 2

        # Set intrinsic and pose data
        intrinsic_data = {
            "intrinsicId": viewId,
            "width": f"{width:0.0f}",
            "height": f"{height:0.0f}",
            "sensorWidth": sensor_width_str,
            "sensorHeight": sensor_height_str,
            "serialNumber": "-1",
            "type": "pinhole",
            "initializationMode": "unknown",
            "pxFocalLength": [f"{focal_px:0.20f}",
                            f"{focal_py:0.20f}"
                            ],
            "pxInitialFocalLength": "-1",
            "focalLength": focal_px_mm_str,
            "initialFocalLength": "-1",
            "pixelRatio": "1",
            "pixelRatioLocked": "1",
            "principalPoint": [f"{cx:0.20f}",
                            f"{cy:0.20f}"
                            ],
            "distortionInitializationMode": "none",
            "distortionParams": ["0", "0", "0"],
            "undistortionOffset": ["0", "0"],
            "undistortionParams": "",
            "distortionType" : "radialk3",
            "undistortionType": "none",
            "locked": "0"
        }
        pose_data = {
            "poseId": viewId,
            "pose": {
                "transform": {
                    "rotation": [f"{x:0.20f}" for x in pose[:3,:3].ravel().tolist()],
                    "center": [f"{x:0.20f}" for x in pose[:3,3].tolist()]
                },
                "scale": {
                    "scaleBol": use_scale_matrix,
                    "scaleMat": [f"{x:0.20f}" for x in scale_mat[:3,:4].ravel().tolist()]
                }
            }
        }

        # Check if intrinsic and pose data already exist
        intrinsicId = get_intrinsic_id(intrinsic_data, intrinsics_data)
        poseId = get_pose_id(pose_data, poses_data)

        # Update intrinsic and pose ids
        intrinsic_data["intrinsicId"] = intrinsicId
        pose_data["poseId"] = poseId

        # Set view data
        view_data = {
            "viewId": viewId,
            "poseId": poseId,
            "frameId": f"{i+1}",
            "intrinsicId": intrinsicId,
            "resectionId": "",
            "path": image_path,
            "width": f"{width:0.0f}",
            "height": f"{height:0.0f}",
            "metadata" : "",
            "maskPath": mask_paths[i],
        }

        # Add data to dictionaries
        intrinsics_data[intrinsicId] = intrinsic_data
        poses_data[poseId] = pose_data
        views_data[viewId] = view_data

    return views_data, intrinsics_data, poses_data



if __name__ == "__main__":

    IDR_FOLDER_PATH = "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/data/unimsps/green_tea_boxes"
    OUTPUT_PATH = "/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/skoltech3d/data/unimsps/green_tea_boxes/cropped"
    USE_SCALE_MATRIX = False
    SIZE = (2368, 1952)

    # Read IDR data
    views_data, intrinsics_data, poses_data = read_idr_data(IDR_FOLDER_PATH, USE_SCALE_MATRIX, size=SIZE)

    # Define bbox of masks
    bbox = np.array([np.inf, np.inf, -np.inf, -np.inf])
    for id, view in views_data.items():

        mask = load_image(view["maskPath"])
        mask_where = np.where(mask > 0)
        x_min = mask_where[1].min()
        y_min = mask_where[0].min()
        x_max = mask_where[1].max()
        y_max = mask_where[0].max()

        bbox[0] = min(bbox[0], x_min)
        bbox[1] = min(bbox[1], y_min)
        bbox[2] = max(bbox[2], x_max)
        bbox[3] = max(bbox[3], y_max)

        # plot the bbox for this mask and the global bbox
        if False:
            import matplotlib.pyplot as plt
            plt.imshow(mask)
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='r', linewidth=2))
            plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor='g', linewidth=2))
            plt.show()
        
    bbox = bbox.astype(int)

    # plot the bbox for this mask and the global bbox
    if True:
        import matplotlib.pyplot as plt
        plt.imshow(mask)
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='r', linewidth=2))
        plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor='g', linewidth=2))
        plt.show()

    # Crop images and masks
    idr_dict = {}
    for i, (view_id, view_data) in enumerate(views_data.items()):

        # Get view parameters
        K, c2w_gl, _ = get_view_parameters(view_data, intrinsics_data, poses_data)
        c2w_cv = _gl_to_cv(c2w_gl)

        # Apply crop on K
        K[0, 2] -= bbox[0]
        K[1, 2] -= bbox[1]

        # Save projection matrix
        idr_dict['world_mat_%d'%i] = K @ np.linalg.inv(c2w_cv)

        # Load image and mask
        image = load_image(view_data['path'])
        mask = load_image(view_data['maskPath'])

        # Crop image and mask
        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # Save image
        output_image_path = os.path.join(OUTPUT_PATH, "image")
        if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)
        save_image(image[:, :, :3], os.path.join(output_image_path, f"{i:08d}.png"), bit_depth=8)

        # Save mask
        output_mask_path = os.path.join(OUTPUT_PATH, "mask")
        if not os.path.exists(output_mask_path):
            os.makedirs(output_mask_path)
        save_mask(mask, os.path.join(output_mask_path, f"{i:08d}.png"))

    # Save npz file
    output_npz_path = os.path.join(OUTPUT_PATH, "cameras.npz")
    np.savez(output_npz_path,**idr_dict)