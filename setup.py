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
import hashlib
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
#   "v2"   build flash_attn_npu.flash_attn_npu     (910B/C only)
#   "v3"   build the v3 backends selected by FLASH_ATTN_BUILD_NPU:
#            flash_attn_npu_3.flash_attn_npu_3    (Ascend 910B/C, csrc/ascend910)
#            flash_attn_npu_3_950                 (Ascend 950,    csrc/ascend950)
#   "v4"   build the v4 backends selected by FLASH_ATTN_BUILD_NPU:
#            flash_attn_npu_4.flash_attn_npu_4      (Ascend 910B/C, csrc/ascend910)
#            flash_attn_npu_4_950                   (Ascend 950,    csrc/ascend950)
#          Runtime dispatch in flash_attn_npu_4/__init__.py picks the
#          matching backend per host via torch_npu.npu.get_device_name(),
#          so a single wheel runs on both 910 and 950.
#   "all"  build v2 + v3 + the v4 backends selected by FLASH_ATTN_BUILD_NPU.
# FLASH_ATTN_BUILD_NPU selects which NPU hardware backends to build:
#   "910"  only Ascend 910B/C backends (flash_attn_npu.flash_attn_npu,
#          flash_attn_npu_3.flash_attn_npu_3, flash_attn_npu_4.flash_attn_npu_4)
#   "950"  only the Ascend 950 backends (flash_attn_npu_3_950,
#          flash_attn_npu_4_950)
#   "all"  build every backend whose API generation is selected above
#          (default). Runtime dispatch in flash_attn_npu_3/__init__.py and
#          flash_attn_npu_4/__init__.py picks the matching backend per host
#          via torch_npu.npu.get_device_name(), so an "all" wheel runs on
#          both 910 and 950.
BUILD_VERSION = os.getenv("FLASH_ATTN_BUILD_VERSION", "all").lower()
BUILD_NPU = os.getenv("FLASH_ATTN_BUILD_NPU", "all").lower()
# FLASH_ATTN_ENABLE_MSSANITIZER: 使能 Ascend msSanitizer 内存异常检测。
# 开启后：
#  - 910 (dav-2201)：编译追加 "-g --cce-enable-sanitizer"，链接追加 "--cce-enable-sanitizer"（静态插桩）；
#  - 950 (dav-3510)：CANN 9.1.0 未提供 sanitizer stub（libsanitizer_stub_dav-c310），无法静态插桩，
#                    仅追加 "-g"（报告定位信息），内存检测需配合 mssanitizer 运行时注入：
#                    mssanitizer --tool=memcheck python -m pytest ...
# "-g" 用于异常报告输出文件名/行号/调用栈；静态插桩可检测越界访问、多核踩踏、非对齐访问等。
# 仅用于调试定位，会显著增大二进制并降低性能。建议通过 bdist_wheel 独立产出检测版，
# 避免覆盖正常版本（editable 安装下正常 .so 位于 flash_attn_npu/ 目录内）。
ENABLE_MSSANITIZER = os.getenv("FLASH_ATTN_ENABLE_MSSANITIZER", "FALSE") == "TRUE"

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


def _cmdhash(compile_common):
    """Stable hash of the per-extension compile flags (arch / -O / ABI / includes).
    A change here means a stored .o was produced with different flags and must be
    rebuilt, even if no source/header mtime moved."""
    return hashlib.md5("\0".join(compile_common).encode("utf-8")).hexdigest()


def _obj_is_fresh(obj, src, depfile, cmdhash_file, compile_common):
    """True iff `obj` may be reused instead of recompiled. Conservative: any
    doubt -> False (recompile). Over-recompiling is safe; reusing a stale .o is
    not. Three staleness signals:
      1. source mtime > obj mtime  (.cpp edited)
      2. any depfile-listed header newer than obj  (-MMD tracks user headers,
         incl. .cpp files included as headers such as fag_kernel.cpp; NOT
         system/CANN headers)
      3. the recorded compile-flag hash differs from the current one, or is
         missing (first time tracking this obj) -> rebuild to establish it."""
    if not os.path.exists(obj):
        return False
    obj_mtime = os.path.getmtime(obj)
    if os.path.getmtime(src) > obj_mtime:
        return False
    if os.path.exists(depfile):
        try:
            with open(depfile, "r", errors="replace") as f:
                content = f.read().replace("\\\n", " ")
            deps = content.split(":", 1)[1] if ":" in content else ""
            for tok in deps.split():
                if tok.endswith(":"):
                    break  # secondary target (-MP not used; defensive)
                if tok and os.path.exists(tok) and os.path.getmtime(tok) > obj_mtime:
                    return False
        except Exception:
            return False  # parse error -> be safe, recompile
    try:
        with open(cmdhash_file, "r") as f:
            prev = f.read().strip()
    except OSError:
        return False  # first time tracking this obj -> recompile to record hash
    return prev == _cmdhash(compile_common)


class BishengBuildExt(build_ext):
    _toolchains = None

    def _get_toolchain(self, ext):
        ext_name = ext.name
        if self._toolchains is None:
            self._toolchains = {}
        if ext_name in self._toolchains:
            return self._toolchains[ext_name]

        ascend_home = os.getenv("ASCEND_TOOLKIT_HOME", os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend"))
        if not os.path.exists(ascend_home):
            raise RuntimeError(f"ASCEND_TOOLKIT_HOME={ascend_home}")

        # Determine the target arch (3510 vs 2201) and the versioned ascend950
        # include dir from the extension's first source path. v3 modules are
        # named flash_attn_npu_3_950 (sources under csrc/ascend950/flash_attn_npu_3),
        # v4 modules are named flash_attn_npu_4_950 (sources under
        # csrc/ascend950/flash_attn_npu_4); deriving both pieces from the source
        # path keeps the v3/v4 divergence working without hardcoding.
        first_src = ext.sources[0] if ext.sources else ""
        src_norm = first_src.replace("\\", "/")
        is_ascend950 = "ascend950" in src_norm
        npu_arch = "dav-3510" if is_ascend950 else "dav-2201"

        extra_includes = []
        extra_defines = []
        if is_ascend950:
            version_dir = "flash_attn_npu_3"  # fallback
            for part in src_norm.split("/"):
                if part in ("flash_attn_npu", "flash_attn_npu_3", "flash_attn_npu_4"):
                    version_dir = part
                    break
            extra_includes.append(
                f"-I{this_dir}/csrc/ascend950/{version_dir}"
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
        # msSanitizer 静态插桩仅支持 910 (dav-c220)：CANN 9.1.0 未提供 950 的 sanitizer stub
        # （libsanitizer_stub_dav-c310 缺失），950 编译时加 "--cce-enable-sanitizer" 会在设备链接阶段
        # 报 "unable to find library -lsanitizer_stub_dav-c310"。
        # 950 (dav-3510) 请改用 mssanitizer 运行时注入（编译不加插桩，运行测试命令前加 mssanitizer 前缀）。
        use_ms_static = ENABLE_MSSANITIZER and not is_ascend950
        if use_ms_static:
            link_arch_flags = ["--cce-enable-sanitizer", *link_arch_flags]


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
        if use_ms_static:
            # msSanitizer 静态插桩：-g 生成定位信息（异常报告输出文件名/行号/调用栈），
            # --cce-enable-sanitizer 开启内存异常检测插桩。
            compile_common += ["-g", "--cce-enable-sanitizer"]
            print(f"[mssanitizer] {ext_name}: 已追加 -g（报告定位信息）；"
                  f"内存检测请用运行时注入：mssanitizer --tool=memcheck python -m pytest ...")
        elif ENABLE_MSSANITIZER:
            # 950 (dav-3510)：静态插桩 "--cce-enable-sanitizer" 在此 CANN 下不可用
            # （缺少 libsanitizer_stub_dav-c310，链接阶段报 unable to find library），
            # 内存检测改由 mssanitizer 运行时注入完成，无需重新编译算子；
            # 这里仍追加 "-g"，让运行时注入的异常报告能输出文件名/行号/调用栈。
            compile_common += ["-g", "-fno-jump-tables"]
            print(f"[mssanitizer] {ext_name}: 已追加 -g（报告定位信息）；"
                  f"内存检测请用运行时注入：mssanitizer --tool=memcheck python -m pytest ...")
                  

        self._toolchains[ext_name] = (compiler, compile_common, link_arch_flags, link_flags)
        return self._toolchains[ext_name]

    def _force_rebuild(self):
        return bool(getattr(self, "force", None)) or \
            os.getenv("FLASH_ATTN_FORCE_REBUILD", "FALSE") == "TRUE"

    def _build_aicpu_metadata(self, ext_fullpath, ext_name):
        """Compile fa_metadata.aicpu (host AICPU object) for the extensions
        that carry the scheduler-metadata feature (910 v2/v3 and 950 v3). This
        is a separate `bisheng -x aicpu` invocation (host CPU code
        cross-compiled with hcc, not ASC device code); the resulting object is
        linked into the extension alongside the ASC device objects. Returns the
        .o path, or None if there is no aicpu source."""
        ascend_home = os.getenv("ASCEND_TOOLKIT_HOME", os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend"))
        aicpu_src_dirs = {
            "flash_attn_npu.flash_attn_npu": os.path.join(this_dir, "csrc/ascend910", "flash_attn_npu"),
            "flash_attn_npu_3.flash_attn_npu_3": os.path.join(this_dir, "csrc/ascend910", "flash_attn_npu_3"),
            "flash_attn_npu_3_950": os.path.join(this_dir, "csrc/ascend950", "flash_attn_npu_3"),
            "flash_attn_npu_4.flash_attn_npu_4": os.path.join(this_dir, "csrc/ascend910", "flash_attn_npu_4"),
        }
        src_dir = aicpu_src_dirs.get(ext_name)
        if src_dir is None:
            return None
        aicpu_src = os.path.join(src_dir, "fa_metadata.aicpu")
        if not os.path.exists(aicpu_src):
            return None
        aicpu_obj = os.path.join(
            os.path.dirname(ext_fullpath),
            "fa_metadata_950.o" if ext_name == "flash_attn_npu_3_950" else "fa_metadata.o",
        )
        # Incremental: aicpu is a host-code cross-compile (hcc) with no depfile,
        # so mtime-on-source only. Skip if the object is already up-to-date.
        if not self._force_rebuild() and os.path.exists(aicpu_obj) and \
                os.path.getmtime(aicpu_src) <= os.path.getmtime(aicpu_obj):
            print("[compile-aicpu-skip]", aicpu_src, "(obj up-to-date)")
            return aicpu_obj
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
            f"-I{src_dir}",  # tilingdata.h
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
        toolchains = {ext.name: self._get_toolchain(ext) for ext in self.extensions}
        force = self._force_rebuild()

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
            depfile = obj + ".d"
            cmdhash_file = obj + ".cmdhash"
            if not force and _obj_is_fresh(obj, src, depfile, cmdhash_file, compile_common):
                print("[compile-skip]", ext_name, os.path.basename(src), "(obj up-to-date)")
                return (ext_name, obj)
            cmd = [*compile_common, "-c", src, "-o", obj, "-MMD", "-MF", depfile]
            print("[compile]", ext_name, os.path.basename(src))
            print("[compile-cmd]", " ".join(cmd))
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.stdout:
                    print(result.stdout)
                with open(cmdhash_file, "w") as f:
                    f.write(_cmdhash(compile_common))
                return (ext_name, obj)
            except subprocess.CalledProcessError as e:
                print(f"Compilation failed for {src}! Error output:\n{e.stderr}")
                raise

        # One shared pool across all extensions so the heaviest TUs compile
        # concurrently regardless of which extension owns them. TUs per extension
        # once autogen dispatch TUs are added: ascend910_v2=12, ascend910_v3=9,
        # ascend950_v3=6, ascend910_v4=5, ascend950_v4=6.
        max_workers = min(len(tasks), os.cpu_count() or 1)
        objs_by_ext = {ext.name: [] for ext in self.extensions}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(compile_one, t): t for t in tasks}
            for fut in as_completed(futures):
                ext_name, obj = fut.result()
                objs_by_ext[ext_name].append(obj)

        # AICPU metadata object for the extensions carrying the
        # scheduler-metadata feature: compiled separately (host code,
        # `bisheng -x aicpu`) and appended to each extension's link set. Built
        # after the parallel ASC compiles, before linking.
        for ext in self.extensions:
            ext_fullpath = self.get_ext_fullpath(ext.name)
            aicpu_obj = self._build_aicpu_metadata(ext_fullpath, ext.name)
            if aicpu_obj is not None:
                objs_by_ext[ext.name].append(aicpu_obj)

        # Link each extension from its own object files (serial; link is cheap
        # relative to compile - measured ~1s/ext - so parallelizing would save
        # only ~5s total and stays serial). Skip an ext's link when its .so
        # already exists and is no older than every .o: that means no object
        # changed since the last link, so relinking would be a no-op.
        for ext in self.extensions:
            ext_fullpath = self.get_ext_fullpath(ext.name)
            objs = objs_by_ext[ext.name]
            if not force and os.path.exists(ext_fullpath) and \
                    all(os.path.exists(o) and os.path.getmtime(o) <= os.path.getmtime(ext_fullpath)
                        for o in objs):
                print("[link-skip]", ext_fullpath, "(.so up-to-date)")
                continue
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

src_ascend910_v2 = glob.glob(os.path.join(this_dir, "csrc/ascend910/flash_attn_npu", "flash_api.cpp"), recursive=True)
src_ascend910_v2 += glob.glob(os.path.join(this_dir, "csrc/ascend910/flash_attn_npu", "fag_general_host.cpp"), recursive=True)
# ascend910 v2's forward FAInfer / FAGGeneral backward / varlen-backward dispatch is split
# into per-(dtype, layout) translation units under autogen/, generated by
# autogen/generate_kernels.py, so the heavy kernel templates compile in parallel.
src_ascend910_v2 += glob.glob(os.path.join(this_dir, "csrc/ascend910/flash_attn_npu", "autogen", "*.cpp"), recursive=True)
src_ascend910_v3 = glob.glob(os.path.join(this_dir, "csrc/ascend910/flash_attn_npu_3", "flash_api.cpp"), recursive=True)
# ascend910 v3's forward FAInfer / backward FAGGeneral dispatch is split into per-
# (dtype, layout) translation units under autogen/, generated by
# autogen/generate_kernels.py. flash_api.cpp keeps the fa_split host loop +
# metadata logic; the kernel templates are instantiated only in the autogen TUs.
src_ascend910_v3 += glob.glob(os.path.join(this_dir, "csrc/ascend910/flash_attn_npu_3", "autogen", "*.cpp"), recursive=True)
src_ascend950_v3 = glob.glob(os.path.join(this_dir, "csrc/ascend950/flash_attn_npu_3", "flash_api.cpp"), recursive=True)
# ascend950 v3's forward FAInfer dispatch is split into per-(dtype, layout) translation
# units under autogen/, generated by autogen/generate_kernels.py, so the FAInfer /
src_ascend950_v3 += glob.glob(os.path.join(this_dir, "csrc/ascend950/flash_attn_npu_3", "autogen", "*.cpp"), recursive=True)
src_ascend910_v4 = glob.glob(os.path.join(this_dir, "csrc/ascend910/flash_attn_npu_4", "flash_api.cpp"), recursive=True)
# ascend910's forward FAInfer / backward FAGGeneral dispatch is split into
# per-(dtype, layout) translation units under autogen/, generated by
# autogen/generate_kernels.py. flash_api.cpp keeps the fa_split host loop and
# mha_bwd tiling; the kernel templates are instantiated only in the autogen TUs.
src_ascend910_v4 += glob.glob(os.path.join(this_dir, "csrc/ascend910/flash_attn_npu_4", "autogen", "*.cpp"), recursive=True)
src_ascend950_v4 = glob.glob(os.path.join(this_dir, "csrc/ascend950/flash_attn_npu_4", "flash_api.cpp"), recursive=True)
# ascend950 v4's forward FAInfer dispatch is split into per-(dtype, layout) translation
# units under autogen/, generated by autogen/generate_kernels.py, so the FAInfer /
src_ascend950_v4 += glob.glob(os.path.join(this_dir, "csrc/ascend950/flash_attn_npu_4", "autogen", "*.cpp"), recursive=True)

if not SKIP_NPU_BUILD:
    if BUILD_VERSION in ("v2", "all") and BUILD_NPU in ("910", "all"):
        # Nested under the Python package so the extension name does not collide
        # with the flash_attn_npu package itself.
        ext_modules.append(Extension(
            name="flash_attn_npu.flash_attn_npu",
            sources=src_ascend910_v2,
            language="c++",
        ))

    if BUILD_VERSION in ("v3", "all") and BUILD_NPU in ("910", "all"):
        ext_modules.append(Extension(
            name="flash_attn_npu_3.flash_attn_npu_3",
            sources=src_ascend910_v3,
            language="c++",
        ))

    if BUILD_VERSION in ("v3", "all") and BUILD_NPU in ("950", "all"):
        if not src_ascend950_v3:
            raise RuntimeError(
                "FLASH_ATTN_BUILD_NPU=950 or FLASH_ATTN_BUILD_VERSION=v3 requires csrc/ascend950/flash_attn_npu_3/flash_api.cpp;"
            )
        ext_modules.append(Extension(
            name="flash_attn_npu_3_950",
            sources=src_ascend950_v3,
            language="c++",
        ))

    if BUILD_VERSION in ("v4", "all") and BUILD_NPU in ("910", "all"):
        if not src_ascend910_v4:
            raise RuntimeError(
                "FLASH_ATTN_BUILD_VERSION=v4 requires csrc/ascend910/flash_attn_npu_4/flash_api.cpp;"
            )
        ext_modules.append(Extension(
            name="flash_attn_npu_4.flash_attn_npu_4",
            sources=src_ascend910_v4,
            language="c++",
        ))

    if BUILD_VERSION in ("v4", "all") and BUILD_NPU in ("950", "all"):
        if not src_ascend950_v4:
            raise RuntimeError(
                "FLASH_ATTN_BUILD_NPU=950 or FLASH_ATTN_BUILD_VERSION=v4 requires csrc/ascend950/flash_attn_npu_4/flash_api.cpp;"
            )
        ext_modules.append(Extension(
            name="flash_attn_npu_4_950",
            sources=src_ascend950_v4,
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
    with open(Path(this_dir) / "flash_attn_npu" / "__init__.py", "r") as f:
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
            "csrc/ascend910",
            "csrc/ascend950",
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
