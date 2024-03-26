import os
import shutil


to_copy = "D:/PhD/Dropbox/CVPR_2024_2/results/"


use_normalization = False
use_geodesic_distance = False
use_L1 = True
use_L2 = False

save_mesh = True
no_gui = False
copy_result = True
max_iter = 15000
mesh_quality = 512

suffix = "_"+str(max_iter)
command = " ./build/testbed --scene ./data/NOR_INT/{}/ --mode nerf --maxiter {} ".format("{}",max_iter)
if no_gui :
	command+= "--no-gui "
if save_mesh :
	command += "--save-mesh {} ".format(mesh_quality)
if use_L1 :
    command += "--distance L1 "
    suffix += "_L1"
else :
    command += "--distance L2 "
    suffix += "_L2"
if use_normalization :
    command+= "--normalization "
    suffix += "_norm"
if use_geodesic_distance :
    command+= "--geodesic "
    suffix += "_geodesic"

cases = ["buddhaPNG"]#["bearPNG","buddhaPNG","cowPNG","pot2PNG","readingPNG"]
normals = ["GT"]# ,"SDM"]

for case in cases :
    for normal in normals :
        name = case+"_"+normal
        case_command = command.format(name)
        os.system("powershell "+case_command)
        shutil.copy("./data/NOR_INT/{}/mesh_{}_.obj".format(name,max_iter),to_copy+case+"/"+normal+suffix+".obj")
