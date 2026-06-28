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

> [!TIP]
> **🆕 RNb-NeuS2 is now available as a Meshroom node!**
> Run the full pipeline (prepare → scale → train → mesh) directly inside
> [Meshroom](https://github.com/alicevision/Meshroom) from SfMData inputs — see the
> [Meshroom Plugin](#meshroom-plugin) section.
>
> Looking for a fully open-source, CUDA-library-free variant? Check out
> **[Open-RNb](https://github.com/meshroomHub/mrOpenRNb)** — a PyTorch /
> tiny-cuda-nn reimplementation integrated into Meshroom.

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Meshroom Plugin](#meshroom-plugin)
- [Acknowledgements & Citation](#acknowledgements--citation)

## Installation

RNb-NeuS2 builds on [Instant-NGP](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux);
its requirements (a recent CUDA toolkit, a CUDA-capable NVIDIA GPU, CMake ≥ 3.18 and
a C++14 compiler) apply here as well. Our build steps mirror theirs.
[OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) is optional — if found it
enables hardware ray tracing, but the project compiles fine without it.

**1. Clone the repository**

All C++/CUDA dependencies are vendored under `dependencies/`, so a plain clone is
enough:

```bash
git clone https://github.com/RobinBruneau/RNb-NeuS2/
cd RNb-NeuS2
```

**2. Build the CUDA testbed with CMake**

```bash
cmake . -B build
cmake --build build --config RelWithDebInfo -j
```

This produces the `./build/testbed` executable used by the training pipeline.

**3. Create the Python environment**

The Python side (data preparation, scene scaling, albedo scaling and pipeline
orchestration) lives in the `rnb_neus2` package. Use either a plain `venv` or conda.

*Option A — `venv` (no conda).* Create the environment as a `venv/` directory at the
repository root. This is also exactly what the [Meshroom plugin](#meshroom-plugin)
expects, so no extra symlink is needed later:

```bash
python3.10 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .
```

*Option B — conda.*

```bash
conda create -n rnb2 python=3.10   # pick another name if "rnb2" already exists
conda activate rnb2
pip install -e .
```

## Data

We provide the [DiLiGenT-MV, LUCES-MV and Skoltech3D](https://drive.google.com/drive/folders/1TbOrB38klLpG41bXzI7B1A01qsbEbz9h?usp=sharing)
datasets with normals and reflectance maps estimated using
[SDM-UniPS](https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023/) and
[Uni-MS-PS](https://github.com/Clement-Hardy/Uni-MS-PS). This link also contains the
cleaned resulting meshes and ground truths.

### Data Convention

Organize your data in the `./data/` folder following this structure:

```plaintext
./data/FOLDER/
    normal/          # (Mandatory)
        000.png
        001.png
        002.png
    mask/            # (Mandatory)
        000.png
        001.png
        002.png
    albedo/          # (Optional — enables albedo-based training)
        000.png
        001.png
        002.png
    cameras.npz
```

Image files are matched by name across folders (e.g. `normal/000.png`,
`mask/000.png`, `albedo/000.png` describe the same view).

`cameras.npz` follows the data format in
[IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), where
`world_mat_xx` denotes the world-to-image projection matrix and `scale_mat_xx`
denotes the normalization matrix.

## Training

Reconstruction is driven by a single Python entry point, `run_pipeline.py`, which
runs the full pipeline: load data → normalize the scene → train the CUDA testbed →
(optionally) scale albedos → extract the mesh.

### Baseline (normals only)

```bash
python run_pipeline.py --input ./data/FOLDER --testbed ./build/testbed --output ./out/FOLDER
```

The reconstructed mesh is written to `./out/FOLDER/mesh.obj`.

### Reproduce the paper results (with reflectance)

To use reflectance maps, add `--has-albedo`. This enables two-phase training and
automatically scales the reflectance maps via multi-view consistency — required to
reproduce the results of our paper:

```bash
python run_pipeline.py --input ./data/FOLDER --testbed ./build/testbed \
    --output ./out/FOLDER --has-albedo
```

### Other options

```plaintext
--max-steps INT          # Total training steps (default: 10000)
--mesh-resolution INT    # Marching cubes resolution (default: 1024; use 512 if low on memory)
--scaling-mode MODE      # auto | pcd | silhouettes | silhouettes_v2 | cameras | none (default: auto)
--sphere-scale FLOAT     # Target sphere radius after normalization (default: 1.0)
--mask-weight FLOAT      # Weight of the mask loss (default: 1.0)
--l1                     # Use L1 color loss (L2 by default)
--no-rgbplus             # Disable the reflectance-singularity (RGB+) correction
--supernormal            # SuperNormal sub-case (single-stage, normals only)
--warmup-ratio FLOAT     # Fraction of steps for the geometry-only warmup (albedo mode, default: 0.1)
```

A console entry point is also installed with the package:

```bash
rnb-neus2 --input ./data/FOLDER --testbed ./build/testbed --output ./out/FOLDER
```

### Advanced: calling the testbed directly

The compiled `./build/testbed` can be driven manually for custom experiments
(the Python pipeline wraps these same calls):

```plaintext
--scene FOLDER          # Path to the prepared data
--maxiter INT           # Number of iterations
--mask-weight FLOAT     # Weight of the mask loss
--save-mesh             # Extract the mesh at the end
--save-snapshot         # Save the neural weights
--no-albedo             # Train only on normals
--lone                  # Apply L1 loss (L2 by default)
--resolution INT        # Marching cubes resolution (default 1024)
--no-gui                # Run without GUI
--supernormal           # Apply the canonical lights (SuperNormal)
--opti-lights           # Apply the optimal triplet of lights per pixel
--no-rgbplus            # Disable the reflectance-singularity correction
```

## Meshroom Plugin

RNb-NeuS2 ships a [Meshroom](https://github.com/alicevision/Meshroom) node,
**`RNbNeuS2`**, that runs the entire pipeline (prepare → scale → train → mesh)
from AliceVision SfMData inputs — no manual data conversion required.

**Install (3 steps):**

1. Build the testbed and create the Python environment (see [Installation](#installation)).
2. Make the package importable inside Meshroom as the plugin `venv/`. Meshroom expects
   a `venv/` directory at the plugin root and adds it to its Python path.
   - If you used **Option A (`venv`)**, you already have `venv/` at the repository
     root — nothing to do.
   - If you used **Option B (conda)**, symlink the conda environment from the
     **repository root**, with it active:
     ```bash
     ln -s "$CONDA_PREFIX" venv
     ```
3. Register the plugin and start Meshroom:
   ```bash
   export MESHROOM_PLUGINS_PATH=/path/to/RNb-NeuS2
   meshroom
   ```

The path to the testbed is read from `meshroom/config.json`
(`RNB_NEUS2_TESTBED_PATH`, default `../build/testbed`); adjust it if your build
lives elsewhere.

**Use it:** drop an `RNbNeuS2` node in your graph, connect a normal-maps SfMData to
`Normal Maps SfMData` (and, optionally, albedo/mask SfMData), then compute. The
node outputs the reconstructed `mesh.obj` in world coordinates.

📖 Full node reference (all inputs/outputs, CLI equivalents): see
[`meshroom/README.md`](meshroom/README.md).

## Acknowledgements & Citation


- [RNb-NeuS2](https://robinbruneau.github.io/publications/rnb_neus2.html)

```bibtex
@article{Bruneau26,
    title={{Multi-view Surface Reconstruction Using Normal and Reflectance Cues}},
    author={Robin Bruneau and Baptiste Brument and Yvain Quéau and Jean Mélou and François Bernard Lauze and Jean-Denis Durou and Lilian Calvet},
    journal={International Journal of Computer Vision (IJCV)},
    volume={134},
    number={2},
    pages={69},
    year={2026},
    doi={10.1007/s11263-025-02628-8},
    url={https://doi.org/10.1007/s11263-025-02628-8}
}
```


- [RNb-NeuS](https://robinbruneau.github.io/publications/rnb_neus.html)

```bibtex
@inproceedings{Brument24,
    title={{RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction}},
    author={Baptiste Brument and Robin Bruneau and Yvain Quéau and Jean Mélou and François Lauze and Jean-Denis Durou and Lilian Calvet},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```

This project is built on [NeuS2](https://github.com/19reborn/NeuS2).
