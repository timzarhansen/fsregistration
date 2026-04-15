from setuptools import setup, find_packages
import os

# Check if CUDA is available
cuda_available = os.system('nvcc --version > /dev/null 2>&1') == 0

ext_modules = []

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
    print("WARNING: CUDA not detected. Building without custom extensions.")
    print("GeoTransformer may not work correctly without CUDA extensions.")
    cmdclass = {}

setup(
    name='geotransformer',
    version='1.0.0',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
