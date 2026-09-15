# 为 flash-attention-npu 贡献代码

<div align="center">
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/English-CONTRIBUTING.md-blue?style=flat-square" alt="English"></a> <a href="#"><img src="https://img.shields.io/badge/中文-CONTRIBUTING.zh.md-green?style=flat-square" alt="中文"></a>
</div>

感谢你为 flash-attention-npu 做出贡献。

本文档介绍仓库的代码质量检查流程，包括 pre-commit 钩子、本地质量检查、格式化工具行为和 Pull Request 检查。构建、单元测试和 NPU 测试说明不在本文档范围内。

仓库使用 [`pre-commit`](https://pre-commit.com/) 作为统一的质量检查编排工具。本地钩子和 CI 使用同一个 [`.pre-commit-config.yaml`](.pre-commit-config.yaml)，因此工具版本和文件选择规则保持一致。

## 想要贡献补丁？

创建 Pull Request 之前，请暂存准备提交的文件，并运行本地质量检查。同时检查工作区差异和已暂存差异。

```bash
git diff
git add <files>
git diff --cached
git commit
```

安装完成后，`git commit` 会自动运行一次 pre-commit。如果格式化工具修改了文件，本次提交会停止。请检查修改、重新暂存结果，然后再次提交：

```bash
git diff
git add <fixed-files>
git commit
```

这是预期行为。格式化工具可以自动修复机械性问题，但贡献者必须在修改进入提交前检查这些改动。不要使用 `git commit --no-verify` 绕过钩子来隐藏质量检查失败。

## 配置质量检查

### 本地 pre-commit 钩子

本地配置需要 Git、支持的 Python 3（包含 `pip`）以及首次下载钩子环境所需的网络连接。本地钩子使用主机上的 `python3` 解释器；质量检查 Docker 镜像则固定自己的 Python 版本，以保证 CI 可复现。

```bash
make quality-install
```

该命令会安装固定版本的 pre-commit 并运行 `pre-commit install`。完成后，每次 `git commit` 都会自动检查本次提交中已暂存的文件。

该命令等价于：

```bash
python3 -m pip install -r ci/quality-requirements.txt
python3 -m pre_commit install
```

如果 `python3 -m pip` 不可用，请先使用操作系统的软件包管理器安装 Python 3、pip 和 venv，再运行 `make quality-install`。例如，在 Ubuntu 或 Debian 上：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
make quality-install
```

虚拟环境还可以避免修改系统 Python 安装。

pre-commit 会创建并缓存配置的钩子所需环境。无需在主机上单独安装 Ruff、clang-format、yamllint、ShellCheck 或 actionlint。第一次运行时需要下载钩子环境，耗时可能较长。

### 质量检查 Docker

当主机 Python 环境不完整，或希望复现 CI 工具环境时，可以使用质量检查 Docker 镜像：

```bash
git add <files>
make quality-docker
```

该镜像基于 [`ci/Dockerfile.code_quality`](ci/Dockerfile.code_quality) 构建，并挂载当前代码仓库。它检查已暂存的项目文件，主机无需安装 Python、pip、CANN、NPU、`torch_npu` 或 lint 工具。Docker 入口会读取 Git 暂存区，因此运行前请先执行 `git add`。

## 手动运行检查

已安装的 Git 钩子是常规入口。提交前如需运行相同的暂存文件检查，可以使用：

```bash
make quality
make quality-docker
```

两个命令都只检查已暂存的项目文件，不会将未暂存文件加入检查范围。

维护或基线检查可以显式运行整个仓库的扫描：

```bash
make quality-all-docker
```

完整扫描可能暴露历史格式问题，不是常规开发流程。`quality-fix` 和 `quality-fix-docker` 目标作为兼容别名保留；格式化工具已经在正常检查流程中运行。

## 检查内容

当前配置包括：

- 对 Python 和 `.pyi` 文件运行 Ruff lint 和 Ruff format；
- 对项目 C/C++ 和 AscendC 文件运行 clang-format；
- 对 YAML 文件运行 yamllint；
- 对 Shell 脚本运行 ShellCheck；
- 对 GitHub Actions 工作流运行 actionlint；
- 检查合并冲突标记、行尾空格和文件末尾换行。

格式化钩子遵循 Apache Arrow 的模式：在正常提交钩子期间运行，并可能重写文件。如果钩子重写了文件，本次提交或检查会失败，以便贡献者检查并重新暂存结果。

## 仓库边界

部分路径有明确的检查边界：

- `csrc/catlass` 是第三方子模块，不作为普通项目代码检查；
- `csrc/*/autogen/` 下的生成 C/C++ 文件不参与常规 clang-format 和空白检查；
- 生成器的 Python 源码仍属于项目代码，会接受 Ruff 检查；
- 不要在无关功能改动中重新生成或重新格式化生成文件。

## Pull Request 质量检查

Code Quality 工作流会在 Pull Request 创建、更新或重新打开时运行。它不使用贡献者本地的暂存区，而是检查 Pull Request 基准提交和头提交之间发生变化的文件：

```text
基准提交 ... Pull Request 头提交
```

因此：

- 本地 `pre-commit`、`make quality` 和 `make quality-docker` 检查已暂存文件；
- Pull Request CI 检查 `base...HEAD` 中新增、复制、修改或重命名的文件；
- 手动触发的维护运行是唯一常规的全仓库扫描；
- CI 在质量检查 Docker 镜像中运行，不依赖 GitHub runner 预装的工具；
- CI 不会将格式化修改提交到 Pull Request 分支。检查并暂存修复后，请推送新的提交。

如需在本地复现 Pull Request 范围：

```bash
BASE_SHA=<base-commit> make quality-changed
```

如果省略 `BASE_SHA`，`quality-changed` 会比较 `HEAD^...HEAD`。

## 检查清单

请求评审前，请确认：

1. `git diff --cached` 只包含与本次改动相关的文件；
2. 所有格式化修改都经过人工检查；
3. 被格式化工具修改过的文件已经重新暂存；
4. 构建目录、日志、wheel、缓存和临时文件没有被暂存；
5. 工作流、CI 脚本和 pre-commit 修改已通过 `make quality-docker`；
6. Pull Request 中的 `Code Quality / lint / format / workflow checks` 结果对应最新推送的提交。

请将大范围格式化、工具升级和大规模基线修改放在专门的 Pull Request 中。不要将它们与无关的功能改动混在一起。
