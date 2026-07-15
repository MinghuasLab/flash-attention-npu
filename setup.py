# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026

import sys
import os
import re
import ast
import glob
import sysconfig
from pathlib import Path
from packaging.version import parse
import platform

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import urllib.request
import urllib.error
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
import torch_npu

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


this_dir = os.path.dirname(os.path.realpath(__file__))

PACKAGE_NAME = "flash_attn_npu"

BASE_WHEEL_URL = (
    "https://github.com/MinghuasLab/flash-attention-npu/releases/download/{tag_name}/{wheel_name}"
)

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_NPU_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any NPU compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_NPU_BUILD = os.getenv("FLASH_ATTENTION_SKIP_NPU_BUILD", "FALSE") == "TRUE"
# FLASH_ATTN_BUILD_VERSION selects which API generations to build:
#   "v2"   build flash_attn_npu_arch22_v2     (910B/C only)
#   "v3"   build the v3 backends selected by FLASH_ATTN_BUILD_NPU:
#            flash_attn_npu_arch22_v3    (Ascend 910B/C, csrc/arch22)
#            flash_attn_npu_arch35_v3    (Ascend 950,    csrc/arch35)
#          Runtime dispatch in flash_attn_npu_v3/__init__.py picks the
#          matching backend per host via torch_npu.npu.get_device_name(),
#          so a single wheel runs on both 910 and 950.
#   "all"  build v2 + the selected v3 backends.
# FLASH_ATTN_BUILD_NPU selects which NPU hardware backends to build:
#   "910"  only Ascend 910B/C backends (flash_attn_npu_arch22_v2, flash_attn_npu_arch22_v3)
#   "950"  only the Ascend 950 backend  (flash_attn_npu_arch35_v3)
#   "all"  build every backend whose API generation is selected above
#          (default). Runtime dispatch in flash_attn_npu_v3/__init__.py
#          picks the matching backend per host via
#          torch_npu.npu.get_device_name(), so an "all" wheel runs on
#          both 910 and 950.
BUILD_VERSION = os.getenv("FLASH_ATTN_BUILD_VERSION", "all").lower()
BUILD_NPU = os.getenv("FLASH_ATTN_BUILD_NPU", "all").lower()

def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))

def get_cann_arch_dir():
    return f"{platform.machine()}-linux"  # aarch64-linux | x86_64-linux

class BishengBuildExt(build_ext):
    _toolchains = None

    def _get_toolchain(self, ext_name):
        if self._toolchains is None:
            self._toolchains = {}
        if ext_name in self._toolchains:
            return self._toolchains[ext_name]

        ascend_home = os.getenv("ASCEND_TOOLKIT_HOME", os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend"))
        if not os.path.exists(ascend_home):
            raise RuntimeError(f"ASCEND_TOOLKIT_HOME={ascend_home}")

        is_arch35 = "arch35" in ext_name
        npu_arch = "dav-3510" if is_arch35 else "dav-2201"

        extra_includes = []
        extra_defines = []
        if is_arch35:
            extra_includes.append(
                f"-I{this_dir}/csrc/arch35/flash_attn_npu_v3"
            )
            extra_defines.append("-DCATLASS_ARCH=3510")
        else:
            extra_defines.append("-DCATLASS_ARCH=2201")

        asc_include_paths = [
            os.path.join(ascend_home, "compiler/tikcpp/include"),
            os.path.join(ascend_home, get_cann_arch_dir(), "tikcpp/include"),
        ]

        asc_lib_paths = [
            os.path.join(ascend_home, "compiler/lib64"),
            os.path.join(ascend_home, get_cann_arch_dir(), "lib64"),
        ]

        python_include = sysconfig.get_path('include')

        torch_cmake_path = torch.utils.cmake_prefix_path
        torch_package_path = os.path.dirname(torch.__file__)
        torch_include = os.path.join(torch_cmake_path, "Torch/include")
        torch_lib = os.path.join(torch_cmake_path, "Torch/lib")

        torch_npu_path = os.path.dirname(torch_npu.__file__)
        torch_npu_include = os.path.join(torch_npu_path, "include")
        torch_npu_lib = os.path.join(torch_npu_path, "lib")

        torch_abi = torch._C._GLIBCXX_USE_CXX11_ABI
        abi_flag = f"-D_GLIBCXX_USE_CXX11_ABI={1 if torch_abi else 0}"

        compile_arch_flags = [
            "-x", "asc",
            f"--npu-arch={npu_arch}",
            *(["--cce-auto-infer-kernel-type=false"] if parse(torch_npu.utils.get_cann_version()) >= parse("9.0.0") else []),
            *extra_defines,
        ]
        # At link time only the target arch is needed (for device-code linking).
        link_arch_flags = [f"--npu-arch={npu_arch}"]

        include_flags = [
            *[f"-I{p}" for p in asc_include_paths],
            f"-I{python_include}",
            f"-I{torch_npu_include}",
            f"-I{torch_include}",
            f"-I{ascend_home}/include",
            f"-I{ascend_home}/pkg_inc",
            f"-I{ascend_home}/pkg_inc/profiling",
            f"-I{ascend_home}/runtime/include",
            f"-I{ascend_home}/include/experiment/runtime",
            f"-I{ascend_home}/include/experiment/msprof",
            f"-I{torch_package_path}/include",
            f"-I{torch_package_path}/include/torch/csrc/api/include",
            f"-I{this_dir}/csrc/catlass/include",
            *extra_includes,
        ]

        link_flags = [
            *[f"-L{p}" for p in asc_lib_paths],
            f"-L{torch_lib}",
            f"-L{torch_npu_lib}",
            f"-L{torch_package_path}/lib",
            f"-L{ascend_home}/lib64",
            "-lascendcl",
            "-ltorch_npu",
            "-ltiling_api",
            "-lplatform",
        ]

        # NOTE: ccache is intentionally NOT supported. bisheng requires `-x asc`
        # to enter ASC/device-compile mode (without it, kernel_operator.h and the
        # ASC headers are unresolvable). ccache hard-rejects any `-x <lang>` it
        # does not recognize, treating `-x asc` as "Unsupported language: asc" and
        # falling back to running the real compiler with zero caching — so wrapping
        # bisheng in ccache provides no benefit. If ccache ever adds ASC language
        # support, reintroduce an opt-in wrapper here.
        compiler = ["bisheng"]

        compile_common = [*compiler, "-O2", *compile_arch_flags, "-fPIC", "-std=c++17",
                          abi_flag, *include_flags]

        self._toolchains[ext_name] = (compiler, compile_common, link_arch_flags, link_flags)
        return self._toolchains[ext_name]

    def _build_aicpu_metadata(self, ext_fullpath):
        """Compile fa_metadata.aicpu (host AICPU object) for the arch22_v3 extension
        (flash_attn_npu_arch22_v3). This is a separate `bisheng -x aicpu` invocation
        (host CPU code cross-compiled with hcc, not ASC device code); the
        resulting object is linked into flash_attn_npu_arch22_v3 alongside the ASC device
        objects. Returns the .o path, or None if there is no aicpu source.
        Preserved from main's metadata feature through the parallel-pipeline
        refactor."""
        ascend_home = os.getenv("ASCEND_TOOLKIT_HOME", os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend"))
        v3_dir = os.path.join(this_dir, "csrc/arch22", "flash_attn_npu_v3")
        aicpu_src = os.path.join(v3_dir, "fa_metadata.aicpu")
        if not os.path.exists(aicpu_src):
            return None
        aicpu_obj = os.path.join(os.path.dirname(ext_fullpath), "fa_metadata.o")
        cann_arch_dir = get_cann_arch_dir()
        aicpu_inc = os.path.join(ascend_home, cann_arch_dir, "asc/include/aicpu_api")
        aicpu_lib = os.path.join(ascend_home, cann_arch_dir, "lib64/device/lib64")
        hcc = os.path.join(ascend_home, "toolkit/toolchain/hcc")
        hcc_isys = os.path.join(hcc, "aarch64-target-linux-gnu/include")
        hcc_cpp = os.path.join(hcc_isys, "c++/7.3.0")
        aicpu_cmd = [
            "bisheng",
            "-O2",
            "-std=c++17",
            "-fvisibility=default",
            "-fvisibility-inlines-hidden",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
            "-D_FORTIFY_SOURCE=2",
            "-D_GNU_SOURCE",
            f"-I{aicpu_inc}",
            f"-I{v3_dir}",  # tilingdata.h
            f"--cce-aicpu-L{aicpu_lib}",
            "--cce-aicpu-laicpu_api",
            f"--cce-aicpu-toolkit-path={os.path.join(hcc, 'bin')}",
            f"--cce-aicpu-sysroot={os.path.join(hcc, 'sysroot')}",
            "-isystem", hcc_isys,
            "-isystem", hcc_cpp,
            "-isystem", os.path.join(hcc_cpp, "aarch64-target-linux-gnu"),
            "-isystem", os.path.join(hcc_cpp, "backward"),
            "-c",
            "-o", aicpu_obj,
            "-x", "aicpu", aicpu_src,
        ]
        print("[compile-aicpu]", aicpu_src)
        print("[compile-aicpu-cmd]", " ".join(aicpu_cmd))
        try:
            result = subprocess.run(aicpu_cmd, capture_output=True, text=True, check=True)
            if result.stdout:
                print(result.stdout)
            print(f"AICPU compilation successful! output: {aicpu_obj}")
        except subprocess.CalledProcessError as e:
            print(f"AICPU compilation failed! Error output:\n{e.stderr}")
            raise e
        return aicpu_obj

    def build_extensions(self):
        toolchains = {ext.name: self._get_toolchain(ext.name) for ext in self.extensions}

        # Map every TU source -> (ext_name, obj_path). obj dir is per-extension so
        # v2/v3 .o files (same basenames, e.g. flash_api.o) never collide.
        tasks = []  # (ext_name, src, obj)
        for ext in self.extensions:
            ext_fullpath = self.get_ext_fullpath(ext.name)
            os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)
            obj_dir = os.path.join(os.path.dirname(ext_fullpath), ext.name + ".objs")
            os.makedirs(obj_dir, exist_ok=True)
            for src in ext.sources:
                obj = os.path.join(obj_dir, os.path.splitext(os.path.basename(src))[0] + ".o")
                tasks.append((ext.name, src, obj))

        def compile_one(task):
            ext_name, src, obj = task
            _compiler, compile_common, _la, _lf = toolchains[ext_name]
            cmd = [*compile_common, "-c", src, "-o", obj]
            print("[compile]", ext_name, os.path.basename(src))
            print("[compile-cmd]", " ".join(cmd))
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.stdout:
                    print(result.stdout)
                return (ext_name, obj)
            except subprocess.CalledProcessError as e:
                print(f"Compilation failed for {src}! Error output:\n{e.stderr}")
                raise

        # One shared pool across all extensions so the heaviest TUs compile
        # concurrently regardless of which extension owns them. TUs per extension
        # once autogen dispatch TUs are added: arch22_v2=12, arch22_v3=9, arch35_v3=6.
        max_workers = min(len(tasks), os.cpu_count() or 1)
        objs_by_ext = {ext.name: [] for ext in self.extensions}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(compile_one, t): t for t in tasks}
            for fut in as_completed(futures):
                ext_name, obj = fut.result()
                objs_by_ext[ext_name].append(obj)

        # AICPU metadata object for the arch22_v3 extension: compiled separately
        # (host code, `bisheng -x aicpu`) and appended to that extension's link
        # set. Built after the parallel ASC compiles, before linking.
        for ext in self.extensions:
            if ext.name == "flash_attn_npu_arch22_v3":
                ext_fullpath = self.get_ext_fullpath(ext.name)
                aicpu_obj = self._build_aicpu_metadata(ext_fullpath)
                if aicpu_obj is not None:
                    objs_by_ext[ext.name].append(aicpu_obj)

        # Link each extension from its own object files (serial; link is cheap
        # relative to compile and each ext needs only its own .o set).
        for ext in self.extensions:
            ext_fullpath = self.get_ext_fullpath(ext.name)
            objs = objs_by_ext[ext.name]
            compiler, _cc, link_arch_flags, link_flags = toolchains[ext.name]
            link_cmd = [*compiler, *link_arch_flags, "-shared", "-fPIC", *objs, *link_flags, "-o", ext_fullpath]
            print("[link]", ext_fullpath)
            print("[link-cmd]", " ".join(link_cmd))
            try:
                result = subprocess.run(link_cmd, capture_output=True, text=True, check=True)
                if result.stdout:
                    print(result.stdout)
                print(f"Link successful! output: {ext_fullpath}")
            except subprocess.CalledProcessError as e:
                print(f"Link failed! Error output:\n{e.stderr}")
                raise e

    def build_extension(self, ext):
        saved = self.extensions
        self.extensions = [ext]
        try:
            self.build_extensions()
        finally:
            self.extensions = saved

ext_modules = []

if os.path.isdir(".git"):
    subprocess.run(
        ["git", "submodule", "update", "--init", "csrc/catlass"], check=False
    )

if not os.path.exists(os.path.join(this_dir, "csrc/catlass", "include/catlass/catlass.hpp")):
    raise RuntimeError(
        f"csrc/catlass is missing its catlass headers (include/catlass/catlass.hpp). "
        f"The submodule gitlink may be unreachable. Run "
        f"`git -C csrc/catlass checkout master` (or fetch the submodule manually) "
        f"and retry."
    )

src_arch22_v2 = glob.glob(os.path.join(this_dir, "csrc/arch22/flash_attn_npu_v2", "flash_api.cpp"), recursive=True)
src_arch22_v2 += glob.glob(os.path.join(this_dir, "csrc/arch22/flash_attn_npu_v2", "fag_general_host.cpp"), recursive=True)
# arch22_v2's forward FAInfer / FAGGeneral backward / varlen-backward dispatch is split
# into per-(dtype, layout) translation units under autogen/, generated by
# autogen/generate_kernels.py, so the heavy kernel templates compile in parallel.
src_arch22_v2 += glob.glob(os.path.join(this_dir, "csrc/arch22/flash_attn_npu_v2", "autogen", "*.cpp"), recursive=True)
src_arch22_v3 = glob.glob(os.path.join(this_dir, "csrc/arch22/flash_attn_npu_v3", "flash_api.cpp"), recursive=True)
# arch22_v3's forward FAInfer / backward FAGGeneral dispatch is split into per-
# (dtype, layout) translation units under autogen/, generated by
# autogen/generate_kernels.py. flash_api.cpp keeps the fa_split host loop +
# metadata logic; the kernel templates are instantiated only in the autogen TUs.
src_arch22_v3 += glob.glob(os.path.join(this_dir, "csrc/arch22/flash_attn_npu_v3", "autogen", "*.cpp"), recursive=True)
src_arch35_v3 = glob.glob(os.path.join(this_dir, "csrc/arch35/flash_attn_npu_v3", "flash_api.cpp"), recursive=True)
src_arch35_v3 += glob.glob(os.path.join(this_dir, "csrc/arch35/flash_attn_npu_v3", "fai_host_api.cpp"), recursive=True)
# arch35_v3's forward FAInfer dispatch is split into per-(dtype, layout) translation
# units under autogen/, generated by autogen/generate_kernels.py, so the FAInfer /
# FAInferDn kernel templates compile in parallel. fai_host_api.cpp stays a
# lightweight router (BuildKernelKey + LaunchFAI); the kernel templates are
# instantiated only in the autogen TUs. head_dim is a runtime tiling axis, not a
# generation axis, so it is not enumerated here.
src_arch35_v3 += glob.glob(os.path.join(this_dir, "csrc/arch35/flash_attn_npu_v3", "autogen", "*.cpp"), recursive=True)

if not SKIP_NPU_BUILD:
    if BUILD_VERSION in ("v2", "all") and BUILD_NPU in ("910", "all"):
        ext_modules.append(Extension(
            name="flash_attn_npu_arch22_v2",
            sources=src_arch22_v2,
            language="c++",
        ))

    if BUILD_VERSION in ("v3", "all") and BUILD_NPU in ("910", "all"):
        ext_modules.append(Extension(
            name="flash_attn_npu_arch22_v3",
            sources=src_arch22_v3,
            language="c++",
        ))

    if BUILD_VERSION in ("v3", "all") and BUILD_NPU in ("950", "all"):
        if not src_arch35_v3:
            raise RuntimeError(
                "FLASH_ATTN_BUILD_NPU=950 or FLASH_ATTN_BUILD_VERSION=v3 requires csrc/arch35/flash_attn_npu_v3/flash_api.cpp;"
            )
        ext_modules.append(Extension(
            name="flash_attn_npu_arch35_v3",
            sources=src_arch35_v3,
            language="c++",
        ))

    
    if not ext_modules:
        raise RuntimeError(
            f"FLASH_ATTN_BUILD_VERSION={BUILD_VERSION!r} + "
            f"FLASH_ATTN_BUILD_NPU={BUILD_NPU!r} selects no extensions to "
            f"build (e.g. v2 has no 950 backend). Set FLASH_ATTN_BUILD_NPU "
            f"to 910, 950, or all."
        )


def get_package_version():
    with open(Path(this_dir) / "flash_attn_npu_v2" / "__init__.py", "r") as f:
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

            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            os.rename(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            super().run()

cmdclass = {"bdist_wheel": CachedWheelsCommand}
if ext_modules:
    cmdclass["build_ext"] = BishengBuildExt

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc/arch22",
            "csrc/arch35",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    author="Minghua Shen",
    author_email="shenmh6@mail.sysu.edu.cn",
    description="High-performance FlashAttention Implementation for Ascend NPU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MinghuasLab/flash-attention-npu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
    license="BSD-3-Clause",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "torch_npu",
    ],
)
