import os
import shutil

cases = ["armadillo","bear","buddha","cow","graphosoma","pot2","reading","skull"]#""seat"

for case in cases :
    
    for normal_case in ["GT","SDM","UniMSPS"] :

        os.system("./build/testbed --mode nerf --scene ../../Blender/{}/case_{}/case_{}_NeuS/case_{}_NeuS2_l60/ ".format(case,normal_case,normal_case,normal_case)+
                  "--maxiter 5000 --save-mesh --save-snapshot --mask-weight 1.0 --no-gui")
    
        os.system("./build/testbed --mode nerf --scene ../../Blender/{}/case_{}/case_{}_NeuS/case_{}_NeuS2_l60/ ".format(case,normal_case,normal_case,normal_case)+
                  "--maxiter 15000 --save-mesh --save-snapshot --mask-weight 0.3 "+
                  "--snapshot ../../Blender/{}/case_{}/case_{}_NeuS/case_{}_NeuS2_l60/snapshot_5000.msgpack --no-gui".format(case,normal_case,normal_case,normal_case))
        
        os.system(
            "./build/testbed --mode nerf --opti-lights --scene ../../Blender/{}/case_{}/case_{}_NeuS/case_{}_NeuS2_lopti/ ".format(
                case, normal_case, normal_case, normal_case) +
            "--maxiter 25000 --save-mesh --save-snapshot --mask-weight 0.3 " +
            "--snapshot ../../Blender/{}/case_{}/case_{}_NeuS/case_{}_NeuS2_l60/snapshot_15000.msgpack --no-gui".format(
                case, normal_case, normal_case, normal_case))


        os.makedirs("../../Blender/{}/case_{}/case_{}_NeuS/result_{}/".format(case,normal_case,normal_case,normal_case),exist_ok=True)
        shutil.copyfile("../../Blender/{}/case_{}/case_{}_NeuS/case_{}_NeuS2_l60/mesh_5000_.obj".format(case,normal_case,normal_case,normal_case),"../../Blender/{}/case_{}/case_{}_NeuS/result_{}/mesh_5000.obj".format(case,normal_case,normal_case,normal_case))
        shutil.copyfile("../../Blender/{}/case_{}/case_{}_NeuS/case_{}_NeuS2_l60/mesh_15000_.obj".format(case,normal_case,normal_case,normal_case),"../../Blender/{}/case_{}/case_{}_NeuS/result_{}/mesh_15000.obj".format(case,normal_case,normal_case,normal_case))
        shutil.copyfile("../../Blender/{}/case_{}/case_{}_NeuS/case_{}_NeuS2_lopti/mesh_25000_.obj".format(case,normal_case,normal_case,normal_case),"../../Blender/{}/case_{}/case_{}_NeuS/result_{}/mesh_25000.obj".format(case,normal_case,normal_case,normal_case))
    
