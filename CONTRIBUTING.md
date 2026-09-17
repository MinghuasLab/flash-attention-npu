# Contributing to flash-attention-npu

<div align="center">
  <a href="#"><img src="https://img.shields.io/badge/English-CONTRIBUTING.md-blue?style=flat-square" alt="English"></a> <a href="CONTRIBUTING.zh.md"><img src="https://img.shields.io/badge/中文-CONTRIBUTING.zh.md-green?style=flat-square" alt="中文"></a>
</div>

Thank you for contributing to flash-attention-npu.

This guide describes the repository's code-quality review workflow. It covers
the pre-commit hook, local quality checks, formatter behavior, and pull-request
checks. Build, unit-test, and NPU-test instructions are outside the scope of
this document.

The repository uses [`pre-commit`](https://pre-commit.com/) as its single
quality-check orchestration layer. Local hooks and CI read the same
[`.pre-commit-config.yaml`](.pre-commit-config.yaml), so tool versions and
file-selection rules remain consistent.

## Do you want to contribute a patch?

Before opening a pull request, stage the files you intend to submit and run the
local quality checks. Review both the working-tree diff and the staged diff.

```bash
git diff
git add <files>
git diff --cached
git commit
```

The `git commit` command automatically runs pre-commit once it has been
installed. If a formatter changes a file, the commit is stopped. Review the
change, stage the result, and commit again:

```bash
git diff
git add <fixed-files>
git commit
```

This is intentional. Formatters may repair mechanical issues automatically,
but contributors must review those edits before they become part of a commit.
Do not bypass the hook with `git commit --no-verify` to hide a quality failure.

## Setting up the quality checks

### Local pre-commit hook

The local setup requires Git, a supported Python 3 installation with `pip`, and
network access for the first download of hook environments. The local hooks use
the `python3` interpreter available on the host; the quality Docker image pins
its own Python version for CI reproducibility.

```bash
make quality-install
```

This installs the pinned pre-commit version and runs `pre-commit install`.
Afterwards, every `git commit` automatically checks the staged files for that
commit.

The command is equivalent to:

```bash
python3 -m pip install -r ci/quality-requirements.txt
python3 -m pre_commit install
```

If `python3 -m pip` is unavailable, install the generic Python 3 pip and venv
packages using your operating system's package manager before running
`make quality-install`. For example, on Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
make quality-install
```

The virtual environment also avoids modifying the system Python installation.

pre-commit creates and caches the environments required by the configured hooks.
You do not need to install Ruff, clang-format, yamllint, ShellCheck, or
actionlint separately on the host. The first invocation can take longer while
the hook environments are downloaded.

### Quality Docker

Use the quality Docker image when the host Python environment is incomplete or
when you want to reproduce the CI tool environment:

```bash
git add <files>
make quality-docker
```

The image is built from [`ci/Dockerfile.code_quality`](ci/Dockerfile.code_quality)
and mounts the checkout. It checks staged project files and does not require
Python, pip, CANN, an NPU, `torch_npu`, or lint tools installed on the host. The
Docker entry point reads the Git index, so run `git add` before invoking it.

## Running the checks manually

The installed Git hook is the normal entry point. To run the same staged-file
check before committing, use either:

```bash
make quality
make quality-docker
```

Both commands inspect only staged project files. Unstaged files are not added
to the check scope.

For maintenance or baseline work, an explicit full-repository scan is available:

```bash
make quality-all-docker
```

The full scan can expose historical formatting issues and is not the normal
developer workflow. The `quality-fix` and `quality-fix-docker` targets are kept
as compatibility aliases; formatters already run during the normal check.

## What the hooks check

The current configuration includes:

- Ruff lint and Ruff format for Python and `.pyi` files;
- clang-format for project C/C++ and AscendC files;
- yamllint for YAML;
- ShellCheck for shell scripts;
- actionlint for GitHub Actions workflows;
- merge-conflict, trailing-whitespace, and end-of-file checks.

The formatter hooks follow the Apache Arrow model: they run during the normal
commit hook and may rewrite files. A hook that rewrites a file causes the
current commit/check to fail so that the contributor can inspect and stage the
result.

## Repository boundaries

Some paths have explicit review boundaries:

- `csrc/catlass` is a third-party submodule and is not checked as ordinary
  project code;
- generated C/C++ files under `csrc/*/autogen/` are excluded from ordinary
  clang-format and whitespace hooks;
- generator Python sources remain project code and are checked by Ruff;
- generated files should not be regenerated or reformatted as an unrelated
  part of a functional change.

## Pull request quality review

The Code Quality workflow runs when a pull request is opened, updated, or
reopened. It does not use the contributor's local staging area. Instead, it
checks the files changed between the pull request base and head commits:

```text
base commit ... pull request head commit
```

Consequently:

- local `pre-commit`, `make quality`, and `make quality-docker` check staged
  files;
- pull-request CI checks added, copied, modified, or renamed files in
  `base...HEAD`;
- a manually dispatched maintenance run is the only normal full-repository
  scan;
- CI runs the checks in the quality Docker image and does not depend on tools
  preinstalled on the GitHub runner;
- CI never commits formatter changes to a pull-request branch. Push a new
  commit after reviewing and staging fixes.

To reproduce a local pull-request range:

```bash
BASE_SHA=<base-commit> make quality-changed
```

If `BASE_SHA` is omitted, `quality-changed` compares `HEAD^...HEAD`.

## Review checklist

Before requesting review, confirm that:

1. `git diff --cached` contains only files related to the change;
2. all formatter edits have been reviewed by a human;
3. files modified by a formatter have been staged again;
4. build directories, logs, wheels, caches, and temporary files are not staged;
5. workflow, CI-script, and pre-commit changes have passed `make quality-docker`;
6. the pull request's `Code Quality / lint / format / workflow checks` result is
   for the latest pushed commit.

Keep formatter-wide rewrites, tool upgrades, and large baseline changes in
focused pull requests. Do not mix them with an unrelated functional change.
