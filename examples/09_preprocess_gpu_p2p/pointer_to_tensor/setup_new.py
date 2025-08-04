from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

hip_root = "/opt/rocm-6.3.3"

setup(
    name='wrap_ptr',
    ext_modules=[
        CppExtension(
            name='wrap_ptr',
            sources=['wrap_ptr.cpp'],
            include_dirs=[f"{hip_root}/include"],
            library_dirs=[f"{hip_root}/lib", f"{hip_root}/lib64"],
            libraries=['amdhip64'],  # Link HIP runtime
            extra_compile_args=[
                '-D__HIP_PLATFORM_AMD__',
                '-std=c++17'
            ],
            extra_link_args=['-lamdhip64']
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)


