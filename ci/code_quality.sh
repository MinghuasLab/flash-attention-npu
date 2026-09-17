#!/usr/bin/env bash
#
# Common quality-check entry point.
#
# Local staged scan (the normal developer entry point):
#   ci/code_quality.sh --staged
#
# Local full scan:
#   ci/code_quality.sh --all-files
#
# CI/incremental scan:
#   ci/code_quality.sh --base <base-commit>
#
# Formatters follow the Apache Arrow model: normal pre-commit execution runs
# them automatically. If a hook changes a file, the command fails so the
# developer can review, stage, and commit the change explicitly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"

MODE="staged"
BASE_SHA=""
FIX="false"
SCOPE_SET="false"

usage() {
    cat >&2 <<'EOF'
usage: ci/code_quality.sh [--staged]
       ci/code_quality.sh --all-files
       ci/code_quality.sh --base <commit>
       ci/code_quality.sh --fix [--staged | --all-files | --base <commit>]
EOF
}

while (($#)); do
    case "$1" in
        --fix)
            [[ "$FIX" == false ]] || { usage; exit 2; }
            FIX="true"
            shift
            ;;
        --all-files)
            [[ "$SCOPE_SET" == false ]] || { usage; exit 2; }
            MODE="all"
            SCOPE_SET="true"
            shift
            ;;
        --staged)
            [[ "$SCOPE_SET" == false ]] || { usage; exit 2; }
            MODE="staged"
            SCOPE_SET="true"
            shift
            ;;
        --base)
            [[ "$SCOPE_SET" == false ]] || { usage; exit 2; }
            [[ $# -ge 2 && -n "$2" ]] || { usage; exit 2; }
            MODE="changed"
            BASE_SHA="$2"
            SCOPE_SET="true"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if ! python3 -m pre_commit --version >/dev/null 2>&1; then
    echo "[code-quality] pre-commit is not installed" >&2
    echo "[code-quality] run: python3 -m pip install -r ci/quality-requirements.txt" >&2
    exit 1
fi

filter_project_files() {
    local path
    CHECKED=()
    for path in "$@"; do
        case "$path" in
            csrc/catlass|csrc/catlass/*)
                continue
                ;;
            csrc/*/autogen/*.[ch]|csrc/*/autogen/*.cc|csrc/*/autogen/*.cpp|\
            csrc/*/autogen/*.cxx|csrc/*/autogen/*.hh|csrc/*/autogen/*.hpp|\
            csrc/*/autogen/*.cuh)
                continue
                ;;
            *)
                CHECKED+=("$path")
                ;;
        esac
    done
}

run_files() {
    if (($# == 0)); then
        echo "[code-quality] no project files selected; passed"
        return 0
    fi

    echo "[code-quality] checking $# project file(s):"
    printf '  %s\n' "$@"
    if [[ "$FIX" == true ]]; then
        echo "[code-quality] --fix is retained for compatibility; configured formatters run automatically"
    fi
    python3 -m pre_commit run --show-diff-on-failure --files "$@"
}

if [[ "$MODE" == all ]]; then
    if [[ "$FIX" == true ]]; then
        echo "[code-quality] --fix is retained for compatibility; configured formatters run automatically"
    fi
    exec python3 -m pre_commit run --all-files --show-diff-on-failure
fi

if [[ "$MODE" == staged ]]; then
    mapfile -d '' STAGED < <(
        git diff --cached --name-only -z --diff-filter=ACMR
    )
    filter_project_files "${STAGED[@]}"
    if ((${#CHECKED[@]} == 0)); then
        echo "[code-quality] no staged project files; passed"
        exit 0
    fi
    run_files "${CHECKED[@]}"
    exit 0
fi

if ! git cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null; then
    echo "[code-quality] base commit does not exist: ${BASE_SHA}" >&2
    echo "[code-quality] check checkout history and the workflow event SHA" >&2
    exit 1
fi

mapfile -d '' CHANGED < <(
    git diff --name-only -z --diff-filter=ACMR "${BASE_SHA}"...HEAD
)

filter_project_files "${CHANGED[@]}"
run_files "${CHECKED[@]}"
