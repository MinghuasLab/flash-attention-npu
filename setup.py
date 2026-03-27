# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026

import sys
import functools
import warnings
import os
import re
import ast
import glob
import shutil
import sysconfig
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch_npu
import torch

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "flash_attn"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
BUILD_VERSION = os.getenv("FLASH_ATTN_BUILD_VERSION", "all").lower()

def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))

class BishengBuildExt(build_ext):
    def build_extension(self, ext):
        BASE_DIR = os.path.dirname(os.path.realpath(__file__))
        ascend_home = os.getenv("ASCEND_TOOLKIT_HOME", os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend"))
        if not os.path.exists(ascend_home):
            raise RuntimeError(f"ASCEND_TOOLKIT_HOME={ascend_home}")

        asc_include_paths = [
            os.path.join(ascend_home, "compiler/tikcpp/include"),
            os.path.join(ascend_home, "aarch64-linux/tikcpp/include"),
        ]

        asc_lib_paths = [
            os.path.join(ascend_home, "compiler/lib64"),
            os.path.join(ascend_home, "aarch64-linux/lib64"),
        ]

        asc_config =  {
            "include_dirs": asc_include_paths,
            "lib_dirs": asc_lib_paths
        }
        print("asc_config",asc_config)
        python_include = sysconfig.get_path('include')
        python_lib = sysconfig.get_path("platlib")

        torch_cmake_path = torch.utils.cmake_prefix_path
        torch_package_path = os.path.dirname(torch.__file__)
        torch_include = os.path.join(torch_cmake_path, "Torch/include")
        torch_lib = os.path.join(torch_cmake_path, "Torch/lib")

        torch_npu_path = os.path.dirname(torch_npu.__file__)
        torch_npu_include = os.path.join(torch_npu_path, "include")
        torch_npu_lib = os.path.join(torch_npu_path, "lib")

        dep_paths = {
            "python": {"include": python_include, "lib": python_lib},
            "torch": {"include": torch_include, "lib": torch_lib},
            "torch_npu": {"include": torch_npu_include, "lib": torch_npu_lib},
        }
        print("dep_paths",dep_paths)
        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)

        torch_abi = torch._C._GLIBCXX_USE_CXX11_ABI
        abi_flag = f"-D_GLIBCXX_USE_CXX11_ABI={1 if torch_abi else 0}"

        compile_cmd = [
            "bisheng",
            "-x", "asc",
            "--npu-arch=dav-2201",
            "-shared",
            "-fPIC",
            "-std=c++17",
            abi_flag,
            *[f"-I{p}" for p in asc_config["include_dirs"]],
            f"-I{dep_paths['python']['include']}",
            f"-I{dep_paths['torch_npu']['include']}",
            f"-I{dep_paths['torch']['include']}",
            f"-I{ascend_home}/include",
            f"-I{ascend_home}/runtime/include",
            f"-I{ascend_home}/include/experiment/runtime",
            f"-I{ascend_home}/include/experiment/msprof",
            f"-I{torch_package_path}/include",
            f"-I{torch_package_path}/include/torch/csrc/api/include",
            f"-I{BASE_DIR}/csrc/catlass/include",
            *[f"-L{p}" for p in asc_config["lib_dirs"]],
            f"-L{dep_paths['torch']['lib']}",
            f"-L{dep_paths['torch_npu']['lib']}",
            f"-L{torch_package_path}/lib",
            f"-L{ascend_home}/lib64",
            "-lascendcl",
            "-ltorch_npu",
            "-ltiling_api",
            "-lplatform",
            *ext.sources,
            "-o", ext_fullpath,
        ]

        print(" ".join(compile_cmd))

        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print("编译成功！输出：", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"编译失败！错误输出：{e.stderr}")
            raise e

cmdclass = {}
ext_modules = []

if os.path.isdir(".git"):
    subprocess.run(["git", "submodule", "update", "--init", "csrc/catlass"], check=True)
else:
    assert (
        os.path.exists("csrc/catlass/include/catlass/catlass.hpp")
    ), "csrc/catlass is missing, please use source distribution or git clone"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
source_files = glob.glob(os.path.join(BASE_DIR, "csrc/flash_attn_npu", "flash_api.cpp"), recursive=True)
source_files_v3 = glob.glob(os.path.join(BASE_DIR, "csrc/flash_attn_npu_v3", "flash_api.cpp"), recursive=True)

if BUILD_VERSION in ("v2", "all"):
    ext_modules.append(Extension(
        name="flash_attn_2_C",
        sources=source_files,
        language="c++",
    ))

if BUILD_VERSION in ("v3", "all"):
    ext_modules.append(Extension(
        name="flash_attn_3_C",
        sources=source_files_v3,
        language="c++",
    ))


def get_package_version():
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


def get_wheel_url():
    torch_version_raw = parse(torch.__version__)
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    flash_version = get_package_version()
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    torch_npu_version = parse(torch.__version__)
    npu_ver_tag = "80"
    wheel_filename = f"{PACKAGE_NAME}-{flash_version}+npu{npu_ver_tag}torch{torch_version}cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
   
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{flash_version}", wheel_name=wheel_filename)

    return wheel_url, wheel_filename


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            os.rename(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "flash_attn.egg-info",
        )
    ),
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BishengBuildExt}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "einops",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
