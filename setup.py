from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
setup(packages=find_packages(), ext_modules=[Pybind11Extension(
    "kv_cache._allocator", ["cpp/bindings.cpp"], include_dirs=["cpp"], cxx_std=17)],
    cmdclass={"build_ext": build_ext},
    package_data={"kv_cache": ["csrc/*.cpp", "csrc/*.cu"]})
