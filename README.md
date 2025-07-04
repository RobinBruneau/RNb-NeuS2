<div align="center">
<h1>RNb-NeuS2: Multi-View Surface Reconstruction <br>
Using Normal and Reflectance Cues</h1>

[**Robin Bruneau**](https://robinbruneau.github.io/)<sup><span>&#9733;</span></sup> · [**Baptiste Brument**](https://bbrument.github.io/)<sup><span>&#9733;</span></sup>
<br>
[**Yvain Quéau**](https://yqueau.github.io/) . [**Jean Mélou**](https://www.irit.fr/~Jean.Melou/) · [**François Lauze**](https://loutchoa.github.io/) · [**Jean-Denis Durou**](https://www.irit.fr/~Jean-Denis.Durou/) · [**Lilian Calvet**](https://scholar.google.com/citations?user=6JewdrMAAAAJ&hl=en)

<span>&#9733;</span> corresponding authors

<div style="display: flex; gap: 10px; justify-content: center; align-items: center;">
    <a href='https://arxiv.org/abs/2506.04115'><img src='https://img.shields.io/badge/arXiv-RNb--NeuS2-red' alt='Paper PDF' height="30"></a>
    <a href='https://robinbruneau.github.io/publications/rnb_neus2.html'><img src='https://img.shields.io/badge/Project_Page-RNb--Neus2-green' alt='Project Page' height="30"></a>
</div>
</div>

## Table of Contents

- [Installation](#installation)
- [Training](#training)
- [Data](#data)
- [Data Convention](#data-convention)
- [Acknowledgements & Citation](#acknowledgements--citation)

## Installation

Follow the [Instant-NGP](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux) instructions for requirements and compilation. Our installation steps are similar.

Clone the repository and its submodules:
```bash
git clone https://github.com/RobinBruneau/RNb-NeuS2/
cd RNb-NeuS2
```

Build the project using CMake:
```bash
cmake . -B build
cmake --build build --config RelWithDebInfo -j 
```

Install the python requirements for the preprocess:
```bash
pip install -r requirements.txt
```

## Data Convention

Organize your data in the `./data/` folder following this structure:
```plaintext
./data/FOLDER/
    albedo/          # (Optional)
        000.png
        001.png
        002.png
    normal/          # (Mandatory)
        000.png
        001.png
        002.png
    mask/            # (Mandatory)
        000.png
        001.png
        002.png
    mask_normal_uncertainty/  # (Optional)
        000.png
        001.png
        002.png
    cameras.npz
```

## Data

We provide the [DiLiGenT-MV, LUCES-MV and Skoltech3D](https://drive.google.com/drive/folders/1TbOrB38klLpG41bXzI7B1A01qsbEbz9h?usp=sharing) datasets with normals and reflectance maps estimated using [SDM-UniPS](https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023/) and [Uni-MS-PS](https://github.com/Clement-Hardy/Uni-MS-PS). This link contains also the cleaned resulting meshs and groundtruths.


## Training

### Preprocess the Data

```bash
python script/preprocess.py --folder ./data/<FOLDER>/ --exp_name <EXP_NAME>
```

### Run Optimization

#### Our baseline
```bash
./run.sh ./data/<FOLDER>/<EXP_NAME>
```
#### Change the loss to L1
```bash
./run.sh ./data/<FOLDER>/<EXP_NAME> --lone
```
#### Only train using the normals
```bash
./run.sh ./data/<FOLDER>/<EXP_NAME> --no-albedo
```
#### The SuperNormal sub-case
```bash
./run.sh ./data/<FOLDER>/<EXP_NAME> --no-albedo --supernormal
```
#### Run Optimization with Scaled Reflectance Maps

For reflectance maps with varying scale factors, use the `--scale-albedo` flag that generates a mesh without reflectance maps first, then uses this mesh to scale the reflectance maps (pyoctree and scipy are needed in a python environment). Finally, it generates a mesh using the scaled reflectance maps. 

```bash
./run.sh ./data/<FOLDER>/<EXP_NAME> --scale-albedo
```
Results will be stored in `./data/<FOLDER>/<EXP_NAME>-albedoscaled/`.

Note: The provided datasets already have scaled reflectance maps in the `albedo` folder.

#### Other parameters to play with:
```plaintext
--maxiter INT           # Number of iterations
--resolution INT        # Resolution for marching cube (default 1024, change to 512 if memory issues occur)
--no-opti-lights        # Disable the optimal triplet of lights per pixel
--no-rgbplus            # Disable the correction the reflectance singularity
```

Results will be stored in `./data/<FOLDER>/<EXP_NAME>/`.</br>
You can also directly work with the `./build/testbed` command to do your own optimisation using the following options:

```plaintext
--scene FOLDER          # Path to your data
--maxiter INT           # Number of iterations
--mask-weight FLOAT     # Weight of the mask loss
--save-mesh             # Extract the mesh at the end
--save-snapshot         # Save the neural weights
--no-albedo             # Train only on normals
--lone                  # Apply L1 loss (L2 default)
--resolution INT        # Resolution for marching cube (default 1024, change to 512 if memory issues occur)
--no-gui                # Run optimization without GUI
--supernormal           # Apply the canonical lights (similar to the Supernormal paper)
--opti-lights           # Apply the optimal triplet of lights per pixel
--no-rgbplus            # Disable the correction the reflectance singularity
```


## Acknowledgements & Citation


- [RNb-NeuS2](https://robinbruneau.github.io/publications/rnb_neus2.html)

```bibtex
@misc{Bruneau25,
    title={Multi-view Surface Reconstruction Using Normal and Reflectance Cues},
    author={Robin Bruneau and Baptiste Brument and Yvain Quéau and Jean Mélou and François Bernard Lauze and Jean-Denis
    Durou and Lilian Calvet},
    year={2025},
    eprint={2506.04115},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.04115},
}
```


- [RNb-NeuS](https://robinbruneau.github.io/publications/rnb_neus.html)

```bibtex
@inproceedings{Brument24,
    title={RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction},
    author={Baptiste Brument and Robin Bruneau and Yvain Quéau and Jean Mélou and François Lauze and Jean-Denis Durou and Lilian Calvet},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```
