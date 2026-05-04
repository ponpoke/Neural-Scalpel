import os
import torch
from setuptools import setup

ext_modules = []
# Compile CUDA extension only if CUDA is available or forcefully requested
if torch.cuda.is_available() or os.environ.get('FORCE_CUDA', '0') == '1':
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    ext_modules = [
        CUDAExtension(
            name='neural_scalpel.kernel.scalpel_cuda_kernel',
            sources=[
                'neural_scalpel/kernel/csrc/bindings.cpp',
                'neural_scalpel/kernel/csrc/scalpel_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ]

setup(
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    } if ext_modules else {}
)
