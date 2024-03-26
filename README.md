# RNb-NeuS2
This is the CUDA official implementation of **RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction**.

[Baptiste Brument*](https://bbrument.github.io/),
[Robin Bruneau*](https://robinbruneau.github.io/),
[Yvain Quéau](https://sites.google.com/view/yvainqueau),
[Jean Mélou](https://www.irit.fr/~Jean.Melou/),
[François Lauze](https://loutchoa.github.io/),
[Jean-Denis Durou](https://www.irit.fr/~Jean-Denis.Durou/),
[Lilian Calvet](https://scholar.google.com/citations?user=6JewdrMAAAAJ&hl=en)

### [Project page](https://robinbruneau.github.io/publications/rnb_neus.html) | [Paper](https://arxiv.org/abs/2312.01215)

<img src="assets/pipeline.png">

## Table Of Contents

- [Gallery](#gallery)
- [Installation](#installation)
- [Training](#training)
- [Data](#data)
- [Data Convention](#data-convention)
- [Acknowledgements \& Citation](#acknowledgements--citation)



## Installation

**Please first see [Instant-NGP](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux) for original requirements and compilation instructions. [NeuS2](https://github.com/19reborn/NeuS2) follows the installing steps of Instant-NGP.**

Clone this repository and all its submodules using the following command:
```
git clone https://github.com/RobinBruneau/RNb_NeuS2/
cd RNb_NeuS2
```

Then use CMake to build the preprocess (OpenCV and Eigen required) : 

```
cd preprocess
cmake .
make
cd ..
```

And use CMake to build the project (follow NeuS2 requirements) : 

```
cmake . -B build
cmake --build build --config RelWithDebInfo -j 
```

You will also need Python with the following libraries : 
```
- Numpy
- Scipy
- Argparse
- Json
- Cv2
- Glob
- Shutil
```

## Training

```
python ./preprocess/preprocess.py --folder ./data/FOLDER/
./run_3steps ./data/FOLDER/
```
For _l60 folder : 
```
python ./preprocess/preprocess.py --folder ./data/FOLDER/
./build/testbed --scene .data/FOLDER/NeuS2/NeuS2_l60/ --maxiter 15000 --save-mesh --mask-weight 0.3
```
For _lopti folder : 
```
python ./preprocess/preprocess.py --folder ./data/FOLDER/
./build/testbed --scene .data/FOLDER/NeuS2/NeuS2_lopti/ ----opti-lights --maxiter 15000 --save-mesh --mask-weight 0.3
```

You can use the following options :
```
--scene FOLDER (path to your data)
--maxiter INT (the number of iterations to compute)
--mask-weight FLOAT (the weight of the mask loss)
--save-mesh (extract the mesh add the end)
--save-snapshot (save the neural weights)
--no-gui (run the optimization without GUI)

```
## Data

You can download here data from DiLiGenT-MV in the expected convention after some Albedo/Normal generation with SDM-UniPS / Uni-MS-PS

## Data Convention

- You can place in the folder ./data/ your own experiments
- We expect the following convention : 
```
./data/FOLDER/
    Albedo/
        000.png
        001.png
        ...
    Normal/
        000.png
        001.png
        002.png
    Mask/
        000.png
        001.png
        002.png
    cameras.npz
```
## Acknowledgements & Citation

- [RNb-NeuS](https://robinbruneau.github.io/publications/rnb_neus.html)

```bibtex
@inproceedings{Brument23,
    title={RNb-Neus: Reflectance and normal Based reconstruction with NeuS},
    author={Baptiste Brument and Robin Bruneau and Yvain Quéau and Jean Mélou and François Lauze and Jean-Denis Durou and Lilian Calvet},
    eprint={2312.01215},
    archivePrefix={arXiv},
    year={2023}
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
```
