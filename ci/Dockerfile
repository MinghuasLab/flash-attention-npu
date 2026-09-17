#
# 统一 CI 镜像 (参数化): 指定 CANN 基础镜像 + torch + torch_npu 版本。
# 910B / 950 等所有架构均用本 Dockerfile 构建, 由 ci/build_matrix_images.sh 读取
# ci/build_matrix.tsv 为每个 combo 构建一次, 镜像名统一为:
#   fa-npu-ci:<name>   其中 <name> = <arch>-cann<cann>-torch<torch>
#   例如 fa-npu-ci:910b-cann9.1-torch2.9 / fa-npu-ci:950-cann9.1-torch2.9
#
# 手动构建单个 combo:
#   docker build -t fa-npu-ci:910b-cann9.1-torch2.9 \
#     --build-arg BASE_IMAGE=<cann image> \
#     --build-arg PY_TAG=cp312 \
#     --build-arg TORCH_VER=2.9.0 \
#     --build-arg TORCH_NPU_VER=2.9.0.post2 \
#     --build-arg TORCH_NPU_RELEASE=v26.0.0-pytorch2.9.0 \
#     --build-arg ARCH=aarch64 \
#     -f ci/Dockerfile ci/
#
# 镜像内只装工具链 (CANN 来自基础镜像 + torch/torch_npu), 不含本仓源码;
# 源码在 ci/run_ci_container.sh / ci/run_ci_matrix.sh 运行时挂载。

ARG BASE_IMAGE
ARG PY_TAG
ARG TORCH_VER
ARG TORCH_NPU_VER
ARG TORCH_NPU_RELEASE
FROM ${BASE_IMAGE}

# FROM 之前声明的 ARG 只能用于 FROM 本身; 构建阶段内使用需重新声明 (不带值,
# 会继承 FROM 之前传入的同名 ARG 值)。否则 RUN 指令里这些变量会是空字符串。
ARG PY_TAG
ARG TORCH_VER
ARG TORCH_NPU_VER
ARG TORCH_NPU_RELEASE
# ARCH 在 FROM 之前未声明 (FROM 不需要), 这里直接给默认值 x86_64;
# --build-arg ARCH=... 会覆盖默认值。取值: x86_64 / aarch64。
ARG ARCH=x86_64

# CANN toolkit home (官方镜像一般已设; 显式声明防缺失)。8.5.0/9.0.0 镜像若路径不同,
# 在此调整或改用基础镜像自带的 set_env.sh。
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/runtime/lib64:${LD_LIBRARY_PATH:-}
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${PATH:-}

# pip: 基础镜像配置的镜像源可能缺包 (实测 950 镜像装 setuptools 报 "from versions: none"),
# 加阿里云源作为额外索引兜底; pip 会合并所有索引的可用版本, 后续 pip 操作一并受益。
ENV PIP_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# 1. 基础工具
RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
      apt-get update -qq && apt-get install -y -qq \
        git curl wget jq build-essential cmake ninja-build \
        && rm -rf /var/lib/apt/lists/*; \
    elif command -v yum >/dev/null 2>&1; then \
      yum install -y -q git curl wget jq gcc-c++ make cmake ninja-build \
        && yum clean all; \
    fi

# 2. Python 包管理工具
RUN python3 -m pip install --upgrade pip wheel setuptools

# 2.5 拷入预下载的 wheel 缓存 (来自构建上下文的 wheels/ 目录; 空目录也无妨)。
#     step 3/4 会先查 /wheels/ 有无目标 wheel, 有则用, 无则 wget。
COPY wheels/ /wheels/

# 3. torch (cpu 版, x86_64/aarch64 统一从 pytorch.org 下载 +cpu manylinux wheel)。
#    优先用 /wheels/ 下预下载的 wheel (省下载), 没有再 wget。
#    注意: 真实文件名含 '+' (如 torch-2.9.0+cpu-...), URL 里 '+' 编码为 '%2B'。
#    pip 按文件名解析 wheel 元数据, 必须保留原始文件名, 不能改名成 torch.whl。
RUN cd /tmp && \
    torch_fname="torch-${TORCH_VER}+cpu-${PY_TAG}-${PY_TAG}-manylinux_2_28_${ARCH}.whl"; \
    if [ -f "/wheels/${torch_fname}" ]; then \
      echo "[torch] using cached /wheels/${torch_fname}"; \
      python3 -m pip install --no-cache-dir "/wheels/${torch_fname}"; \
    else \
      echo "[torch] downloading (not in cache)"; \
      wget -q -O "${torch_fname}" \
        "https://download.pytorch.org/whl/cpu/torch-${TORCH_VER}%2Bcpu-${PY_TAG}-${PY_TAG}-manylinux_2_28_${ARCH}.whl" && \
      python3 -m pip install --no-cache-dir "${torch_fname}" && \
      rm -f "${torch_fname}"; \
    fi

# 4. torch_npu (按 PY_TAG + ARCH + gitcode release tag); 优先用 /wheels/ 下缓存。
#    同理保留原始文件名供 pip 解析。
RUN cd /tmp && \
    npu_whl="torch_npu-${TORCH_NPU_VER}-${PY_TAG}-${PY_TAG}-manylinux_2_28_${ARCH}.whl"; \
    if [ -f "/wheels/${npu_whl}" ]; then \
      echo "[torch_npu] using cached /wheels/${npu_whl}"; \
      python3 -m pip install --no-cache-dir "/wheels/${npu_whl}"; \
    else \
      echo "[torch_npu] downloading (not in cache)"; \
      wget -q -O "${npu_whl}" \
        "https://gitcode.com/Ascend/pytorch/releases/download/${TORCH_NPU_RELEASE}/${npu_whl}" && \
      python3 -m pip install --no-cache-dir "${npu_whl}" && \
      rm -f "${npu_whl}"; \
    fi

# 5. 编译/测试所需 Python 包 (本仓不依赖 triton)
RUN python3 -m pip install --no-cache-dir \
      pybind11 \
      einops \
      packaging \
      psutil \
      pytest \
      numpy

# 6. (已移除 import 校验) 镜像构建环境无 NPU 硬件, torch_npu import 会因缺
#    libascend_hal.so 等 runtime 库失败。模块可用性留给 run_ci_matrix.sh
#    编译阶段自然暴露, 届时若缺库再针对性处理。

WORKDIR /workspace/flash-attention-npu
