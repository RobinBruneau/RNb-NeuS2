<div align="center">
<h1>RNb-NeuS2: Multi-View Surface Reconstruction <br>
Using Normal and Reflectance Cues</h1>

[**Robin Bruneau**](https://robinbruneau.github.io/)<sup><span>&#9733;</span></sup> · [**Baptiste Brument**](https://bbrument.github.io/)<sup><span>&#9733;</span></sup> · [**Yvain Quéau**](https://yqueau.github.io/)
<br>
[**Jean Mélou**](https://www.irit.fr/~Jean.Melou/) · [**François Lauze**](https://loutchoa.github.io/) · [**Jean-Denis Durou**](https://www.irit.fr/~Jean-Denis.Durou/) · [**Lilian Calvet**](https://scholar.google.com/citations?user=6JewdrMAAAAJ&hl=en)

<span>&#9733;</span> corresponding author

<a href='https://arxiv.org/abs/2506.04115'><img src='https://img.shields.io/badge/arXiv-RNb--NeuS2-red' alt='Paper PDF' height="30"></a>

<a href='https://robinbruneau.github.io/publications/rnb_neus2.html'><img src='https://img.shields.io/badge/Project_Page-RNb--Neus2-green' alt='Project Page' height="30"></a>
</div>

## Table of Contents

- [Installation](#installation)
- [Training](#training)
- [Data](#data)
- [Data Convention](#data-convention)
- [Acknowledgements & Citation](#acknowledgements--citation)

## Installation

Follow the [Instant-NGP](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux) instructions for requirements and compilation. [NeuS2](https://github.com/19reborn/NeuS2) installation steps are similar.

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

Ensure you have Python and the following libraries installed:
- Numpy
- Scipy
- Argparse
- Json
- Cv2
- Glob
- Shutil
- PyOctree

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

We provide the [DiLiGenT-MV](https://drive.google.com/file/d/1TEBM6Dd7IwjRqJX0p8JwT9hLmy_vA5nU/view?usp=drive_link) dataset with normals and reflectance maps estimated using [SDM-UniPS](https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023/). Reflectance maps were scaled over all views, and uncertainty masks were generated from 100 normal estimations (see the paper for details).

## Training

### Preprocess the Data

```bash
python script/preprocess.py --folder ./data/<FOLDER>/ --exp_name <EXP_NAME>
```

### Run Optimization

```bash
./run.sh ./data/<FOLDER>/<EXP_NAME>
```

Results will be stored in `./data/<FOLDER>/<EXP_NAME>/`. Modify the `./build/testbed` command in the `run.sh` with the following options:

```plaintext
--scene FOLDER          # Path to your data
--maxiter INT           # Number of iterations
--mask-weight FLOAT     # Weight of the mask loss
--save-mesh             # Extract the mesh at the end
--save-snapshot         # Save the neural weights
--no-albedo             # Train only on normals
--resolution INT        # Resolution for marching cube (default 512)
--no-gui                # Run optimization without GUI
```

### Run Optimization with Scaled Reflectance Maps

For reflectance maps with varying scale factors, use the `--scale-albedo` flag that generates a mesh without reflectance maps first, then uses this mesh to scale the reflectance maps (pyoctree and scipy are needed in a python environment). Finally, it generates a mesh using the scaled reflectance maps. 

```bash
./run.sh ./data/<FOLDER>/<EXP_NAME> --scale-albedo
```
Results will be stored in `./data/<FOLDER>/<EXP_NAME>-albedoscaled/`.

Note: The provided DiLiGenT-MV dataset already has scaled reflectance maps in the `albedo` folder.

## Acknowledgements & Citation

- [RNb-NeuS](https://robinbruneau.github.io/publications/rnb_neus.html)

```bibtex
@inproceedings{Brument24,
    title={RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction},
    author={Baptiste Brument and Robin Bruneau and Yvain Quéau and Jean Mélou and François Lauze and Jean-Denis Durou and Lilian Calvet},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```

- [NeuS2](https://vcai.mpi-inf.mpg.de/projects/NeuS2/)

```bibtex
@inproceedings{neus2,
    title={NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction}, 
    author={Wang, Yiming and Han, Qin and Habermann, Marc and Daniilidis, Kostas and Theobalt, Christian and Liu, Lingjie},
    year={2023},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}
}
```

- [Instant-NGP](https://github.com/NVlabs/instant-ngp)

```bibtex
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
```

- [NeuS](https://lingjie0206.github.io/papers/NeuS/)

```bibtex
@inproceedings{wang2021neus,
    title={NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction},
    author={Wang, Peng and Liu, Lingjie and Liu, Yuan and Theobalt, Christian and Komura, Taku and Wang, Wenping},
    booktitle={Proc. Advances in Neural Information Processing Systems (NeurIPS)},
    volume={34},
    pages={27171--27183},
    year={2021}
}
