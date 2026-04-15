from setuptools import setup, find_packages
import os
import sys

# Check if CUDA is available
cuda_available = False
try:
    import torch
    cuda_available = torch.cuda.is_available()
except ImportError:
    pass

if cuda_available:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    
    ext_modules = [
        CUDAExtension(
            name='geotransformer.ext',
            sources=[
                'geotransformer/extensions/extra/cloud/cloud.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'geotransformer/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'geotransformer/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'geotransformer/extensions/pybind.cpp',
            ],
        ),
    ]
    cmdclass = {'build_ext': BuildExtension}
else:
    print("WARNING: CUDA not available. Building without custom C++ extensions.", file=sys.stderr)
    print("GeoTransformer requires CUDA for the custom extensions to work properly.", file=sys.stderr)
    ext_modules = []
    cmdclass = {}


setup(
    name='geotransformer',
    version='1.0.0',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
