import os
import shutil
import json

file_json = "./configs/nerf/base.json"
f=open(file_json,"r")
data_json = json.load(f)
f.close()

cases = ["buddhaPNG"]

data_encoding = {"otype": "HashGrid",
		"n_levels": 14,
		"n_features_per_level": 2,
		"log2_hashmap_size": 19,
		"base_resolution": 16,
		"top_resolution": 2048,
		"valid_level_scale": 0.02,
		"base_valid_level_scale": 0.2,
		"base_training_step": 100}

modif = [("base_resolution",8),("base_resolution",4),("base_resolution",32)]



for case in cases :

	for mod in modif : 
		
		data_encoding[mod[0]] = mod[1]
		data_json["encoding"] = data_encoding
		
		#print(data_json)

		f=open(file_json,"w")	
		json.dump(data_json,f,indent=4)
		f.close()
	
	

		os.system("./build/testbed --mode nerf --scene ./data/RNB/{}_NeuS2_l60/ ".format(case)+
				  "--maxiter 10000 --save-mesh --save-snapshot --mask-weight 0.3 --no-gui")

	
		os.makedirs("./data/RNB/exp_hashing/",exist_ok=True)
		shutil.copyfile("./data/RNB/{}_NeuS2_l60/mesh_10000_.obj".format(case),"./data/RNB/exp_hashing/mesh_{}_{}.obj".format(mod[0],mod[1]))

