from setuptools import setup, find_packages

setup(
    name="rnb-neus2",
    version="2.0.0",
    packages=find_packages(include=[
        'rnb_neus2', 'rnb_neus2.*',
    ]),
    install_requires=[
        'numpy>=1.21,<2.0',
        'opencv-python',
        'trimesh',
        'scipy',
        'embreex',
        'tqdm',
        'psutil',
        'networkx',
    ],
    entry_points={
        'console_scripts': [
            'rnb-neus2=run_pipeline:main',
        ],
    },
)
