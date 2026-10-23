#!/usr/bin/env bash
# Build the A5 (Ascend 950) delivery bundle for flash_attn_combine.
#
#   bash package_a5_delivery.sh [output_dir]
#
# Produces <output_dir>/flash_attn_combine_a5_delivery_<date>.tar.gz containing:
#   README_FIRST.md, flash-attention-npu/, atk/, docs/analysis/,
#   run_a5_verify.sh, MANIFEST.txt
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
ATK_SRC="/data/w00896777/ATK/flash_attn_combine"
OUT_DIR="${1:-/data/w00896777/ATK}"
STAMP="$(date +%Y%m%d)"
PKG="flash_attn_combine_a5_delivery_${STAMP}"
WORK="$(mktemp -d)"
DEST="$WORK/$PKG"

echo "repo    : $REPO"
echo "atk src : $ATK_SRC"
echo "output  : $OUT_DIR/$PKG.tar.gz"
mkdir -p "$DEST/flash-attention-npu" "$DEST/atk" "$DEST/docs/analysis"

# ---------- 1. source tree (no build artifacts, no VCS, trimmed catlass) ----------
rsync -a \
  --exclude '.git/' --exclude 'build/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '*.so' --exclude '*.o' --exclude '*.a' --exclude '.pytest_cache/' \
  --exclude 'atk_output/' --exclude 'result/' --exclude 'atk.log' \
  --exclude 'extra-info/data-dump/' \
  --exclude 'csrc/catlass/' \
  "$REPO"/ "$DEST/flash-attention-npu"/

# catlass: only what setup.py needs to compile (-I csrc/catlass/include)
mkdir -p "$DEST/flash-attention-npu/csrc/catlass"
rsync -a "$REPO/csrc/catlass/include/" "$DEST/flash-attention-npu/csrc/catlass/include/"
for extra in 3rdparty LICENSE README.md CMakeLists.txt cmake; do
  [ -e "$REPO/csrc/catlass/$extra" ] && rsync -a "$REPO/csrc/catlass/$extra" "$DEST/flash-attention-npu/csrc/catlass/"
done

# ---------- 2. ATK assets ----------
cp "$ATK_SRC"/all_test_flash_attn_combine_bf16fp16.json "$DEST/atk/"
cp "$ATK_SRC"/all_test_flash_attn_combine_v4_bf16fp16.json "$DEST/atk/"
cp "$ATK_SRC"/accuracy-all_test_flash_attn_combine_bf16fp16_reports_*.xlsx "$DEST/atk/" 2>/dev/null || true
cp "$ATK_SRC"/accuracy-all_test_flash_attn_combine_v4_bf16fp16_reports_*.xlsx "$DEST/atk/" 2>/dev/null || true
cp "$ATK_SRC"/memory-all_test_flash_attn_combine_bf16fp16_reports_*.xlsx "$DEST/atk/" 2>/dev/null || true
cp "$ATK_SRC"/memory-all_test_flash_attn_combine_v4_bf16fp16_reports_*.xlsx "$DEST/atk/" 2>/dev/null || true
cp "$REPO"/tests/st/atk_flash_attn_combine/{execute_flash_attn_combine.py,flash_attn_combine_constraint.py,test_flash_attn_combine.yaml,test_flash_attn_combine_v4.yaml,run_atk.sh,extend_cases_to_200.py,renumber_case_ids.py,rekey_report_ids.py,bench_combine.py} "$DEST/atk/"

# ---------- 3. docs ----------
cp "$REPO"/docs/analysis/*.md "$DEST/docs/analysis/"
cp "$REPO/README.zh.md" "$DEST/README_FIRST.md"

# ---------- 4. one-click script ----------
cp "$REPO/tests/st/atk_flash_attn_combine/run_a5_verify.sh" "$DEST/run_a5_verify.sh"
chmod +x "$DEST/run_a5_verify.sh"

# ---------- 5. manifest ----------
(
  cd "$DEST"
  echo "# flash_attn_combine A5 delivery, $(date '+%F %T')"
  echo "# columns: sha256  size(bytes)  path"
  find . -type f ! -name MANIFEST.txt -print0 \
    | sort -z | xargs -0 sha256sum \
    | while read -r sum path; do printf '%s  %10d  %s\n' "$sum" "$(stat -c %s "$path")" "${path#./}"; done
) > "$DEST/MANIFEST.txt"

# ---------- 6. tarball ----------
mkdir -p "$OUT_DIR"
tar -C "$WORK" -czf "$OUT_DIR/$PKG.tar.gz" "$PKG"
rm -rf "$WORK"

echo "built: $OUT_DIR/$PKG.tar.gz"
ls -lh "$OUT_DIR/$PKG.tar.gz"
