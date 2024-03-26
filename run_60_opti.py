import os
import shutil

cases = ["buddhaPNG","cowPNG","pot2PNG","readingPNG"] # ["bearPNG",

for case in cases :

	os.system("./build/testbed --mode nerf --scene ./data/RNB/{}_NeuS2_l60/ ".format(case)+
			  "--maxiter 5000 --save-mesh --save-snapshot --mask-weight 1.0 --no-gui")

	os.system("./build/testbed --mode nerf --scene ./data/RNB/{}_NeuS2_l60/ ".format(case)+
			  "--maxiter 15000 --save-mesh --save-snapshot --mask-weight 0.3 "+
			  "--snapshot ./data/RNB/{}_NeuS2_l60/snapshot_5000.msgpack --no-gui".format(case))

	os.system("./build/testbed --mode nerf --opti-lights --scene ./data/RNB/{}_NeuS2_lopti/ ".format(case)+
			  "--maxiter 25000 --save-mesh --save-snapshot --mask-weight 0.3 "+
			  "--snapshot ./data/RNB/{}_NeuS2_l60/snapshot_15000.msgpack --no-gui".format(case))

	os.makedirs("./data/RNB/result_{}/".format(case),exist_ok=True)
	shutil.copyfile("./data/RNB/{}_NeuS2_l60/mesh_5000_.obj".format(case),"./data/RNB/result_{}/mesh_5000.obj".format(case))
	shutil.copyfile("./data/RNB/{}_NeuS2_l60/mesh_15000_.obj".format(case),"./data/RNB/result_{}/mesh_15000.obj".format(case))
	shutil.copyfile("./data/RNB/{}_NeuS2_lopti/mesh_25000_.obj".format(case),"./data/RNB/result_{}/mesh_25000.obj".format(case))

