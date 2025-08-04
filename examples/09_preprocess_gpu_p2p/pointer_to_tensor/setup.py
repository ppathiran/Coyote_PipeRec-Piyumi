from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

hip_include = "/opt/rocm-6.3.3/include/"

setup(
    name='wrap_ptr',
    ext_modules=[
        CppExtension(
            'wrap_ptr',
            ['wrap_ptr.cpp'],
            include_dirs=[hip_include],
            extra_compile_args=['-D__HIP_PLATFORM_AMD__'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)

