from setuptools import setup, find_packages

setup(
    name="rnb-neus2-scripts",
    version="1.0.0",
    packages=find_packages(include=['scripts', 'scripts.*']),
    install_requires=[
        'numpy>=1.21,<2.0',
        'opencv-python',
        'trimesh',
        'scipy',
        'matplotlib',
        'embreex',
        'tqdm',
        'psutil',
    ],
)
