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
            self.intrinsics_all.append(intrinsics)
            self.pose_all.append(pose)

        self.intrinsics_all = np.array(self.intrinsics_all)  # [n_images, 4, 4]
        self.intrinsics_all_inv = np.linalg.inv(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = np.array(self.pose_all)  # [n_images, 4, 4]



def NeuS_to_NeuS2(inputFolder,outputFolder):
    conf = {
        "data_dir": inputFolder,
        "render_cameras_name": "cameras.npz",
    }
    dataset = Dataset(conf)

    lights = []
    try :
        f=open(inputFolder+"/lights_60.json",'r')
        dataLight_60 = json.load(f)
        f.close()
        for k in range(dataset.n_images) :
            lights.append([[dataLight_60[9*k],dataLight_60[9*k+1],dataLight_60[9*k+2]],
                           [dataLight_60[9*k+3],dataLight_60[9*k+4],dataLight_60[9*k+5]],
                           [dataLight_60[9*k+6],dataLight_60[9*k+7],dataLight_60[9*k+8]]])
        a=0
    except :
        a=0

    base_rgb_dir = join(inputFolder, "image")
    base_msk_dir = join(inputFolder, "mask")
    all_images = sorted(os.listdir(base_rgb_dir))
    all_masks = sorted(os.listdir(base_msk_dir))
    mult = len(all_images) // len(all_masks)
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

    new_image_dir = join(outputFolder, "images")
    os.makedirs(new_image_dir, exist_ok=True)
    for i in range(len(all_masks)):
        for j in range(mult):
            img_name = all_images[mult*i+j]
            msk_name = all_masks[i]
            img_path = join(base_rgb_dir, img_name)
            msk_path = join(base_msk_dir, msk_name)

            img = cv2.imread(img_path,-1)
          
            msk = ((cv2.imread(msk_path, -1).astype(np.uint8)/255)*(2**16 -1)).astype(np.uint16)
            if len(msk.shape)> 2 : 
            	msk = msk[:,:,0]
            
            image = np.concatenate([img, msk[:, :, np.newaxis]], axis=-1)
            H, W = image.shape[0], image.shape[1]
            cv2.imwrite(join(new_image_dir, img_name), image)

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
    all_mask_dir = sorted(os.listdir(join(outputFolder, "mask")))
    all_image_dir = sorted(os.listdir(join(outputFolder, "image")))
    mask_num = len(all_mask_dir)
    image_num = len(all_mask_dir)
    camera_num = dataset.intrinsics_all.shape[0]
    assert mask_num == camera_num, "The number of cameras should be equal to the number of images!"
    for i in range(mask_num):
        for j in range(mult):
            rgb_dir = join("images", all_image_dir[mult*i+j])
            ixt = dataset.intrinsics_all[i]

            # add one_frame
            one_frame = {}
            one_frame["file_path"] = rgb_dir
            one_frame["transform_matrix"] = dataset.pose_all[i].tolist()
            if len(lights) != 0 :
                one_frame["light"] = lights[i][j]
            else :
                one_frame["light"] = [0,0,0]

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


def preprocess(folder):

    folderName = os.path.basename(os.path.dirname(folder))
    mainFolder = folder+"/NeuS2/"
    mainMainFolder = mainFolder + "/NeuS/"
    mainMainFolder2 = mainFolder + "/NeuS2_l60/"
    mainMainFolder3 = mainFolder + "/NeuS2_lopti/"

    if os.path.exists(mainFolder):
        shutil.rmtree(mainFolder)
    os.makedirs(mainFolder,exist_ok=True)
    os.makedirs(mainMainFolder, exist_ok=True)
    os.makedirs(mainMainFolder2, exist_ok=True)
    os.makedirs(mainMainFolder3, exist_ok=True)

    shutil.copytree(folder+"/mask/",mainMainFolder+"/mask/")

    shutil.copytree(folder+"/mask/",mainFolder+"/mask/")

    cameras_npz_to_json(folder,folder+"cameras.npz")

    shutil.copyfile(folder+"cameras.npz",mainMainFolder+"cameras.npz")

    shutil.copyfile(folder+"cameras.npz",mainFolder+"cameras.npz")

    os.makedirs(mainFolder+"/image/", exist_ok=True)
    os.makedirs(mainFolder + "/image_60/", exist_ok=True)

    print("Creating folders for Neus2...")
    print("Generating RNb images...")
    os.system("./preprocess/gen_images_light_60_opti {} {}".format(folder,mainFolder))

    shutil.copytree(mainFolder + "/image/",mainMainFolder+"/image/")
    shutil.copytree(mainFolder + "/image_60/", mainMainFolder + "/image_60/")

    print("Data convertion from NeuS to NeuS2...")
    NeuS_to_NeuS2(mainFolder,mainFolder)
    shutil.move(mainFolder + "/images/", mainMainFolder3 + "/images/")
    shutil.move(mainFolder + "/transform.json",mainMainFolder3 + "/transform.json")
    shutil.move(mainFolder + "/lights.json", mainMainFolder3 + "/lights.json")

    shutil.rmtree(mainFolder + "/image/")
    os.rename(mainFolder + "/image_60/",mainFolder + "/image/")

    NeuS_to_NeuS2(mainFolder, mainFolder)
    shutil.move(mainFolder + "/images/", mainMainFolder2 + "/images/")
    shutil.move(mainFolder + "/transform.json", mainMainFolder2 + "/transform.json")

    shutil.rmtree(mainFolder + "/image/")
    shutil.rmtree(mainFolder + "/mask/")

    os.remove(mainFolder+"/cameras.npz")
    os.remove(mainFolder+"/lights_60.json")

    print("-DONE-")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)  # Parse the argument
    args = parser.parse_args()

    folder = args.folder
    preprocess(folder)



