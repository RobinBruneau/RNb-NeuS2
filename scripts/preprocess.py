import argparse
import json
from os.path import join
import numpy as np
import os
import cv2
from glob import glob
import shutil
from scipy.spatial.transform import Rotation

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

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        self.conf = conf

        self.data_dir = conf['data_dir']
        self.render_cameras_name = conf['render_cameras_name']

        self.camera_outside_sphere = True
        self.scale_mat_scale = 1.1

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.n_images = len(self.images_lis)

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(P)
            #intrinsics = np.array([[3759.07747101073,0,305.500000000000,0],
            #[0,3759.00543107133,255.500000000000,0],
            #        [0,0,1,0],
            #        [0,0,0,1]])
            self.intrinsics_all.append(intrinsics)
            # root_determinant = np.linalg.det(pose[:3,:3])**(1/3)
            # pose = pose / root_determinant
            self.pose_all.append(pose)

        self.intrinsics_all = np.array(self.intrinsics_all)  # [n_images, 4, 4]
        self.intrinsics_all_inv = np.linalg.inv(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = np.array(self.pose_all)  # [n_images, 4, 4]



def NeuS_to_NeuS2(inputFolder, outputFolder, mask_certainty_name):
    conf = {
        "data_dir": inputFolder,
        "render_cameras_name": "cameras.npz",
    }
    dataset = Dataset(conf)

    base_albedo_dir = join(inputFolder, "albedo")
    albedo_folder_exist = os.path.exists(base_albedo_dir)
    base_normal_dir = join(inputFolder, "normal")
    base_msk_dir = join(inputFolder, "mask")
    base_msk_certainty_dir = join(inputFolder, mask_certainty_name)
    
    if albedo_folder_exist :
        all_images_albedo = sorted(os.listdir(base_albedo_dir))
    else :
        all_images_albedo = sorted(os.listdir(base_normal_dir))
    all_images_normal = sorted(os.listdir(base_normal_dir))
    all_masks = sorted(os.listdir(base_msk_dir))

    msk_certainty_folder_exist = os.path.exists(base_msk_certainty_dir)
    if not msk_certainty_folder_exist :
        base_msk_certainty_dir = base_msk_dir
    all_masks_certainty = sorted(os.listdir(base_msk_certainty_dir))

    def copy_directories(root_src_dir, root_dst_dir):
        for src_dir, dirs, files in os.walk(root_src_dir):
            dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for file_ in files:
                src_file = os.path.join(src_dir, file_)
                dst_file = os.path.join(dst_dir, file_)
                if os.path.exists(dst_file):
                    os.remove(dst_file)
                shutil.copy(src_file, dst_dir)

    new_albedo_dir = join(outputFolder, "albedos")
    new_normal_dir = join(outputFolder, "normals")
    os.makedirs(new_albedo_dir, exist_ok=True)
    os.makedirs(new_normal_dir, exist_ok=True)
    for i in range(len(all_masks)):
        img_albedo_name = all_images_albedo[i]
        img_normal_name = all_images_normal[i]
        msk_name = all_masks[i]
        msk_certainty_name = all_masks_certainty[i]
        img_albedo_path = join(base_albedo_dir, img_albedo_name)
        img_normal_path = join(base_normal_dir, img_normal_name)
        msk_path = join(base_msk_dir, msk_name)
        msk_certainty_path = join(base_msk_certainty_dir, msk_certainty_name)

        img_normal = cv2.imread(img_normal_path, -1)[:, :, :3]
        if img_normal.dtype == np.uint8:
            n_bits = 8
        elif img_normal.dtype == np.uint16:
            n_bits = 16
        else:
            raise ValueError("Invalid image type")
        
        if albedo_folder_exist :
            img_albedo = cv2.imread(img_albedo_path,-1)[:,:,:3]
        else :

            if n_bits == 8:
                img_albedo = (np.ones_like(img_normal, dtype=float)*(2**8-1)).astype(np.uint8)
            elif n_bits == 16:
                img_albedo = (np.ones_like(img_normal, dtype=float)*(2**16-1)).astype(np.uint16)

        msk = cv2.imread(msk_path, -1)
        if len(msk.shape) > 2 :
            msk = msk[:,:,0]
        if msk.dtype == np.uint8:
            msk = np.where(msk > 125, 1.0, 0.0)
        else:
            msk = np.where(msk > 30000, 1.0, 0.0)

        if n_bits == 8:
            msk = (msk*(2**8-1)).astype(np.uint8)
        elif n_bits == 16:
            msk = (msk*(2**16-1)).astype(np.uint16)

        msk_certainty = cv2.imread(msk_certainty_path, -1)
        if len(msk_certainty.shape) > 2:
            msk_certainty = msk_certainty[:, :, 0]

        if msk_certainty.dtype == np.uint8:
            msk_certainty = np.where(msk_certainty > 125, 1.0, 0.0)
        else:
            msk_certainty = np.where(msk_certainty > 30000, 1.0, 0.0)
        
        if n_bits == 8:
            msk_certainty = (msk_certainty * (2 ** 8 - 1)).astype(np.uint8)
        elif n_bits == 16:
            msk_certainty = (msk_certainty * (2 ** 16 - 1)).astype(np.uint16)


        # if img_albedo.dtype == np.uint8 :
        #     img_albedo = (img_albedo/255*(2**16-1)).astype(np.uint16)
        # if img_normal.dtype == np.uint8 :
        #     img_normal = (img_normal/255*(2**16-1)).astype(np.uint16)

        image_albedo = np.concatenate([img_albedo, msk_certainty[:, :, np.newaxis]], axis=-1)
        H, W = image_albedo.shape[0], image_albedo.shape[1]
        cv2.imwrite(join(new_albedo_dir, img_albedo_name), image_albedo)

        image_normal = np.concatenate([img_normal, msk[:, :, np.newaxis]], axis=-1)
        H, W = image_normal.shape[0], image_normal.shape[1]
        cv2.imwrite(join(new_normal_dir, img_normal_name), image_normal)

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

    output.update({"n2w": dataset.scale_mats_np[0].tolist()})

    output['frames'] = []
    all_mask_dir = sorted(os.listdir(join(inputFolder, "mask")))
    if albedo_folder_exist :
        all_albedo_dir = sorted(os.listdir(join(inputFolder, "albedo")))
    else :
        all_albedo_dir = sorted(os.listdir(join(inputFolder, "normal")))
    all_normal_dir = sorted(os.listdir(join(inputFolder, "normal")))
    mask_num = len(all_mask_dir)
    camera_num = dataset.intrinsics_all.shape[0]
    assert mask_num == camera_num, "The number of cameras should be equal to the number of images!"
    for i in range(mask_num):
            albedo_dir = join("albedos", all_albedo_dir[i])
            normal_dir = join("normals", all_normal_dir[i])
            ixt = dataset.intrinsics_all[i]

            # add one_frame
            one_frame = {}
            one_frame["albedo_path"] = albedo_dir
            one_frame["normal_path"] = normal_dir
            one_frame["transform_matrix"] = dataset.pose_all[i].tolist()

            one_frame["intrinsic_matrix"] = ixt.tolist()
            output['frames'].append(one_frame)

    file_dir = join(outputFolder, f'transform.json')
    with open(file_dir, 'w') as f:
        json.dump(output, f, indent=4)

def cameras_npz_to_json(folder="",camera_file=""):
    if folder != "" :
        if os.path.exists(folder) :
            if os.path.exists(camera_file):
                data_cam = np.load(camera_file)
                nb_views = len(data_cam.files)//2
                lk = []
                lr = []
                lt = []
                lr_euler = []
                for k in range(nb_views):
                    P_k = data_cam["world_mat_{}".format(k)]
                    K,RT = load_K_Rt_from_P(P_k[:3,:])
                    lr.append(RT[:3,:3].T.tolist())
                    lt.append((-RT[:3,:3].T @ RT[:3,[3]]).tolist())
                    lk.append(K[:3,:3].tolist())
                    rb = np.eye(3)
                    rb[1,1] = -1
                    rb[2,2] = -1
                    r = Rotation.from_matrix((rb @ RT[:3,:3].T).T)
                    euler_rot = r.as_euler('xyz', degrees=True)
                    lr_euler.append(euler_rot.tolist())
                data_out = {"K":lk,"R":lr,"T":lt,"R_euler":lr_euler}
                f=open(folder+"cameras.json",'w')
                json.dump(data_out,f,indent=4)
                f.close()
            else :
                raise("There is no cameras.npz in your folder")
        else :
            raise("Your folder doesn't exist !")
    else :
        raise("You need to add the folder : --folder name")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)  # Parse the argument
    parser.add_argument('--exp_name', type=str, required=False, default="RNb-NeuS2")
    parser.add_argument('--mask_certainty_name', type=str, required=False, default="mask")
    args = parser.parse_args()

    folder = args.folder
    exp_name = args.exp_name
    mask_certainty_name = args.mask_certainty_name
    mainFolder = os.path.join(folder, exp_name)
        
    os.makedirs(mainFolder, exist_ok=True)
    NeuS_to_NeuS2(folder, mainFolder, mask_certainty_name)

    print("-DONE-")