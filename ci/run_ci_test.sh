#!/usr/bin/env bash
#
# 阶段2: NPU 自检 + 安装 + 测试 (容器内执行, 需要 NPU, 已加锁)
#   1. NPU 可用性自检
#   2. python setup.py install (复用阶段1 build/ 产物, 快速安装)
#   3. import 校验
#   4. 扫描 tests/ 下可跑的 test_*.py, 按文件 pytest
#      (quick 每测试函数随机采样至多 N 个, full 全量)
#
# 由 ci/run_ci_container.sh 阶段2通过 docker run 调用 (绑卡 + 加锁)。
#
# 环境变量 (由 run_ci_container.sh 注入):
#   ASCEND_RT_VISIBLE_DEVICES   宿主机物理卡号
#   CI_MODE                     quick|full
#   CI_RUN_EXAMPLE_ST           true|false
#   CI_TEST_WORKERS             xdist -n (默认 2)
#   CI_QUICK_SAMPLE             quick 每函数采样数 (默认 30)
#   CI_RANDOM_SEED              采样 seed (默认 0, 可复现)
#   CI_TEST_DIRECT_FILE         指定文件时只跑该文件; 指定目录时同样按文件扫描
#   CI_TEST_DIRECT_FILTER       直接模式的 -k 过滤
#   CI_CONTAINER_DEVICE         容器内逻辑设备号 (默认 0)
#   GOLDEN_CACHE_MODE/DIR       golden reference cache mode and container path
#   GOLDEN_CACHE_STATS_FILE     xdist-safe tab-separated cache event output

set -euo pipefail

REPO_ROOT="$(pwd)"
DEVICE="${CI_CONTAINER_DEVICE:-0}"

# git safe.directory (容器内 root 操作宿主机 runner 用户的目录, 会触发 dubious ownership)
git config --global --add safe.directory "$REPO_ROOT"

log() { printf '[CI-test] %s\n' "$*"; }
die() { printf '[CI-test][ERROR] %s\n' "$*" >&2; exit 1; }

LOG_DIR="${CI_TEST_LOG_DIR:-/tmp/ci_test_logs}"
# per-file logs go into logs/ subdir, top level keeps summary files only
RUN_LOG_DIR="$LOG_DIR/logs"
SLOW_CASES_FILE="$LOG_DIR/slow_cases.csv"
mkdir -p "$LOG_DIR" "$RUN_LOG_DIR"
printf 'nodeid,duration_s,verdict,rand,h2d,pack,cache,index,postprocess,forward,backward,ref,golden,compare,other,events\n' > "$SLOW_CASES_FILE"
TEST_PHASE_START="$(date +%s)"
export GOLDEN_CACHE_STATS_FILE="${GOLDEN_CACHE_STATS_FILE:-$LOG_DIR/golden_cache_events.tsv}"
CACHE_STATS_FILE="$GOLDEN_CACHE_STATS_FILE"

cache_artifact_count() {
  if [ -n "${GOLDEN_CACHE_DIR:-}" ] && [ -d "$GOLDEN_CACHE_DIR" ]; then
    { find "$GOLDEN_CACHE_DIR" -type f -name 'case_*.tar.gz' 2>/dev/null || true; } | wc -l | tr -d ' '
  else
    printf '0\n'
  fi
}

CACHE_ARTIFACTS_BEFORE=0
if [ "${GOLDEN_CACHE_MODE:-off}" != "off" ] && [ "${GOLDEN_CACHE_MODE:-off}" != "0" ]; then
  : > "$CACHE_STATS_FILE"
  CACHE_ARTIFACTS_BEFORE="$(cache_artifact_count)"
  log "golden cache: mode=${GOLDEN_CACHE_MODE} dir=${GOLDEN_CACHE_DIR:-<unset>} artifacts_before=$CACHE_ARTIFACTS_BEFORE stats=$CACHE_STATS_FILE"
else
  : > "$CACHE_STATS_FILE"
  log "golden cache: disabled"
fi

log "repo=$REPO_ROOT device=$DEVICE mode=${CI_MODE:-quick}"
log "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
log "test phase start: $(date '+%Y-%m-%d %H:%M:%S')"

# ---------- 1. NPU 自检 (需要卡) ----------
command -v python3 >/dev/null 2>&1 || die "python3 not found in container"
python3 - <<'PY' || die "torch_npu not functional inside container (check --privileged / driver mount)"
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("torch_npu device_count:", torch_npu.npu.device_count())
assert torch_npu.npu.device_count() >= 1, "device_count==0; --privileged or driver mount missing?"
PY

# ---------- 2. 安装 (复用 build/ 产物, 不重新编译) ----------
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
log "python setup.py install --skip-build (reuse build/ artifacts)"
python3 setup.py install --skip-build

log "import check"
python3 - <<'PY'
import flash_attn_npu_3
print("flash_attn_npu_3", flash_attn_npu_3.__version__)
PY

# ---------- 3. 按文件 pytest ----------
if [ "${CI_RUN_EXAMPLE_ST:-true}" != "true" ]; then
  log "CI_RUN_EXAMPLE_ST!=true, skip tests"
  exit 0
fi

command -v pytest >/dev/null 2>&1 || pip install pytest --quiet
python3 -c "import xdist" 2>/dev/null || pip install pytest-xdist --quiet

MODE="${CI_MODE:-quick}"
TEST_WORKERS="${CI_TEST_WORKERS:-2}"
# quick 模式: 每个测试函数随机采样至多 CI_QUICK_SAMPLE 个 item (固定 seed 可复现,
# 采样逻辑在 tests/conftest.py; CI_RANDOM_SEED 改 seed 可换一组子集)
export CI_RANDOM_SEED="${CI_RANDOM_SEED:-0}"
CI_QUICK_SAMPLE="${CI_QUICK_SAMPLE:-30}"

# quick 采样, full 全量
SAMPLE_ARG=""
if [ "$MODE" = "quick" ]; then
  SAMPLE_ARG="--random-sample=${CI_QUICK_SAMPLE}"
fi

# Set to true for detailed slow-case timing output during CI debugging
VERBOSE="${CI_VERBOSE:-false}"

FAILED_FILE="$LOG_DIR/failed_cases.txt"
: > "$FAILED_FILE"
SUMMARY_FILE="$LOG_DIR/summary.txt"
: > "$SUMMARY_FILE"
# per-file table and totals for the final summary
SUMMARY_ROWS=()
TOTAL_P=0; TOTAL_F=0; TOTAL_S=0
TOTAL_FILES_RAN=0; TOTAL_FILES_SKIP=0; TOTAL_FILES_CF=0; TOTAL_FILES_DISC=0

# extract failed test nodeids from pytest short summary into failed_cases.txt
append_failed_nodeids() {
  local logfile="$1" target="$2"
  echo "[FILE] $target" >> "$FAILED_FILE"
  sed -n 's/^FAILED[[:space:]]\+\(.*\)$/\1/p; s/^ERROR[[:space:]]\+\(.*\)$/\1/p' \
    "$logfile" 2>/dev/null >> "$FAILED_FILE" || true
}

# parse timing string "total=57.4s ref=31.2s ..." for total seconds
timing_total() {
  local t="${1#total=}"
  printf '%s' "${t%% *}" | tr -d 's'
}

# top 3 nonzero phases from timing string
timing_top_phases() {
  printf '%s\n' $1 | sed 's/s$//' \
    | awk -F= '$1 != "total" && $2+0 > 0 {printf "%s=%.1fs\n", $1, $2}' \
    | sort -t= -k2 -rn | head -3 | awk '{printf "%s%s", sep, $0; sep = " "}'
}

# csv tail for one timing summary, 12 phase seconds then quoted event tags
timing_csv_fields() {
  printf '%s\n' $1 \
    | awk -F= 'BEGIN{split("rand h2d pack cache index postprocess forward backward ref golden compare other", o, " ")}
        NF < 2 {next}
        $2 ~ /^[0-9]+(\.[0-9]+)?s?$/ {v[$1] = $2 + 0; next}
        {tags = tags (tags == "" ? "" : ";") $0}
        END{
          for (i = 1; i <= 12; i++) printf "%s%s", v[o[i]] + 0, (i < 12 ? "," : "")
          printf ",\"%s\"\n", tags
        }'
}

# slowest 5 rows of slow_cases.csv, tab keyed by duration so nodeid commas stay quoted
top5_slow_lines() {
  tail -n +2 "$SLOW_CASES_FILE" \
    | awk -F'",' '{
        nodeid = $1; sub(/^"/, "", nodeid)
        split($2, a, ",")
        split("rand h2d pack cache index postprocess forward backward ref golden compare other", ph, " ")
        m1 = 0; m2 = 0
        for (i = 3; i <= 14; i++) {
          v = a[i] + 0
          if (v > m1) {m2 = m1; n2 = n1; m1 = v; n1 = ph[i - 2]}
          else if (v > m2) {m2 = v; n2 = ph[i - 2]}
        }
        tag = ""
        if (m1 > 0) {tag = n1 "=" m1 "s"; if (m2 > 0) tag = tag " " n2 "=" m2 "s"}
        printf "%.1f\t%s\t%s\t%s\n", a[1] + 0, a[2], nodeid, tag
      }' \
    | sort -t"$(printf '\t')" -k1,1 -rn | head -5 \
    | awk -F'\t' '{line = sprintf("[CI-test]   %10ss  %s [%s]", $1, $3, $2); if ($4 != "") line = line " (" $4 ")"; print line}'
}

# extract first failure traceback from pytest FAILURES section, 60 lines max
print_first_failure() {
  local logfile="$1"
  awk '/^=+ FAILURES =+/{flag=1; next}
       flag && /^_+ .* _+$/{c++; if (c == 2) exit}
       flag' "$logfile" 2>/dev/null | head -n 60 | sed 's/^/    /'
}

# tally finished run results, structured events first, xdist log grep fallback
tally_finished_run() {
  local pf="$1" lf="$2"
  TALLY_P=0; TALLY_F=0; TALLY_S=0
  if [ -f "$pf" ] && grep -q '^result|' "$pf" 2>/dev/null; then
    TALLY_P="$(awk -F'|' '$1 == "result" && $2 == "passed"' "$pf" | wc -l)"
    TALLY_F="$(awk -F'|' '$1 == "result" && ($2 == "failed" || $2 == "error")' "$pf" | wc -l)"
    TALLY_S="$(awk -F'|' '$1 == "result" && $2 != "passed" && $2 != "failed" && $2 != "error"' "$pf" | wc -l)"
  elif [ -f "$lf" ]; then
    TALLY_P="$(grep -cE '^\[gw[0-9]+\] PASSED' "$lf" || true)"
    TALLY_F="$(grep -cE '^\[gw[0-9]+\] (FAILED|ERROR)' "$lf" || true)"
    TALLY_S="$(grep -cE '^\[gw[0-9]+\] SKIPPED' "$lf" || true)"
  fi
}

run_pytest() {
  local target="$1" logfile="$2"; shift 2
  local progress_file="$LOG_DIR/pytest_progress.events"
  local short="${FILE_NAME:-$target}"
  if [ -n "${FILE_IDX:-}" ]; then
    echo "::group::[group $FILE_IDX/$FILE_TOT] ${short##*/} (${FILE_CASES:-?} cases)"
  else
    echo "::group::$target"
  fi
  : > "$progress_file"
  export CI_PROGRESS_FILE="$progress_file"
  (
    local total=0 done=0 passed=0 failed=0 skipped=0 events_seen=0 last_line=0
    local started_at="$(date +%s)" running="" running_since=0
    local last_progress=$(( $(date +%s) - 20 ))
    local kind value duration encoded details nodeid timing dur_s
    local log_total log_pass log_fail log_skip now
    while :; do
      if [ -f "$progress_file" ]; then
        while IFS='|' read -r kind value duration encoded details; do
          case "$kind" in
            total) total="$value" ;;
            start)
              running="$(printf '%s' "$value" | base64 -d 2>/dev/null || true)"
              running_since="$(date +%s)"
              ;;
            result)
              events_seen=1
              nodeid="$(printf '%s' "$encoded" | base64 -d 2>/dev/null || true)"
              done=$((done + 1))
              timing="$(printf '%s' "$details" | base64 -d 2>/dev/null || true)"
              dur_s="${duration%s}"
              [ -z "$dur_s" ] && dur_s="$(timing_total "$timing")"
              if [ "$value" = "passed" ]; then
                passed=$((passed + 1))
                # Skip slow-PASS for golden miss, slowness is expected there
                if [ "$VERBOSE" = "true" ] && \
                   awk -v v="$dur_s" 'BEGIN{exit !(v>=3)}' && \
                   ! printf '%s' "$timing" | grep -q 'golden=miss'; then
                  printf '[slow-PASS] %s | %ss (%s)\n' \
                    "${nodeid#tests/}" "$dur_s" "$(timing_top_phases "$timing")"
                fi
                # Always record to slow_cases file for the final top-5
                if awk -v v="$dur_s" 'BEGIN{exit !(v>=3)}'; then
                  printf '"%s",%s,pass,%s\n' "$nodeid" "$dur_s" "$(timing_csv_fields "$timing")" >> "$SLOW_CASES_FILE"
                fi
              elif [ "$value" = "failed" ] || [ "$value" = "error" ]; then
                failed=$((failed + 1))
                printf '[FAIL] %s | %ss\n' "${nodeid#tests/}" "$dur_s"
                printf '"%s",%s,fail,%s\n' "$nodeid" "$dur_s" "$(timing_csv_fields "")" >> "$SLOW_CASES_FILE"
              else
                skipped=$((skipped + 1))
              fi
              [ "$running" = "$nodeid" ] && running=""
              ;;
          esac
        done < <(tail -n +$((last_line + 1)) "$progress_file")
        last_line="$(wc -l < "$progress_file")"
      fi
      if [ "$total" -eq 0 ] 2>/dev/null && [ -f "$logfile" ]; then
        log_total="$(sed -n 's/.*workers \[\([0-9][0-9]*\) items\].*/\1/p' "$logfile" | head -n 1)"
        [ -n "$log_total" ] && total="$log_total"
      fi
      if [ "$events_seen" -eq 0 ] && [ "$total" -gt 0 ] && [ -f "$logfile" ]; then
        log_pass="$(grep -cE '^\[gw[0-9]+\] PASSED' "$logfile" || true)"
        log_fail="$(grep -cE '^\[gw[0-9]+\] (FAILED|ERROR)' "$logfile" || true)"
        log_skip="$(grep -cE '^\[gw[0-9]+\] SKIPPED' "$logfile" || true)"
        passed="$log_pass"; failed="$log_fail"; skipped="$log_skip"
        done=$((passed + failed + skipped))
      fi
      if [ -z "$running" ] && [ -f "$logfile" ]; then
        running="$(grep -E '^tests/' "$logfile" | tail -n 1 | sed 's/[[:space:]]*$//' | cut -c1-110 || true)"
        running_since=0
      fi
      now="$(date +%s)"
      if [ $((now - last_progress)) -ge "${CI_PROGRESS_INTERVAL:-30}" ]; then
        local log_size=0
        [ -f "$logfile" ] && log_size="$(stat -c%s "$logfile" 2>/dev/null || echo 0)"
        if [ "$total" -gt 0 ] 2>/dev/null; then
          local overall_seg=""
          if [ "${OVERALL_TOTAL:-0}" -gt 0 ] 2>/dev/null; then
            local overall_done=$(( ${OVERALL_BASE:-0} + done ))
            local pct=$(( overall_done * 100 / OVERALL_TOTAL ))
            overall_seg="overall $overall_done/$OVERALL_TOTAL (${pct}%) | "
          fi
          local overall_fail=$(( ${CUM_F_FAIL:-0} + failed ))
          printf '[progress] %sgroup %s/%s %s: %s/%s | %ss\n' \
            "overall $overall_done/$OVERALL_TOTAL (${pct}%) fail=$overall_fail | " \
            "${FILE_IDX:-?}" "${FILE_TOT:-1}" "${short##*/}" \
            "$done" "$total" "$((now - started_at))"
          if [ -n "$running" ]; then
            local run_seg=""
            if [ "$running_since" -gt 0 ]; then
              run_seg=" ($((now - running_since))s"
              [ $((now - running_since)) -ge "${CI_PROGRESS_SLOW_SEC:-120}" ] && run_seg="$run_seg SLOW"
              run_seg="$run_seg)"
            fi
            printf '[progress] running %s%s\n' "$running" "$run_seg"
          fi
        else
          printf '[progress] waiting for collection (log=%sB, %ss)\n' \
            "${log_size:-0}" "$((now - started_at))"
        fi
        last_progress="$now"
      fi
      if [ "$events_seen" -gt 0 ] && [ "$total" -gt 0 ] && [ "$done" -ge "$total" ] 2>/dev/null; then
        break
      fi
      sleep 1
    done
  ) &
  local watcher_pid=$!
  set +e
  # shellcheck disable=SC2086
  python3 -m pytest $target -vs -n "$TEST_WORKERS" --dist=loadscope \
    -p ci.pytest_ci_reporter $SAMPLE_ARG "$@" >"$logfile" 2>&1
  local rc=$?
  sleep 1
  kill "$watcher_pid" 2>/dev/null || true
  wait "$watcher_pid" 2>/dev/null || true
  set -e
  if [ $rc -ne 0 ]; then
    printf 'first failure detail:\n'
    print_first_failure "$logfile"
    append_failed_nodeids "$logfile" "$target"
  fi
  echo "::endgroup::"
}



# Discover test files, probe with collect-only, run per file, skip means no runnable cases
pytest_log_name() {
  local rel="${1#./}"
  rel="${rel%.py}"
  printf '%s\n' "${rel//\//_}.log"
}

file_is_runnable() {
  local file="$1" collect_log="$2" rc=0
  shift 2
  # shellcheck disable=SC2086
  python3 -m pytest "$file" --collect-only -q "$@" >"$collect_log" 2>&1 || rc=$?
  COLLECTED_COUNT="$(grep -c '::' "$collect_log" 2>/dev/null || true)"
  if grep -q '::' "$collect_log"; then
    return 0
  fi
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
    return 2
  fi
  return 1
}

run_pytest_scanned() {
  local root="$1"; shift
  local files=() file collect_log logfile probe_rc
  while IFS= read -r file; do
    [ -n "$file" ] && files+=("$file")
  done < <(find "$root" -type f -name 'test_*.py' | sort)
  if [ "${#files[@]}" -eq 0 ]; then
    log "no test_*.py under $root"
    echo "$root" >> "$FAILED_FILE"
    return
  fi
  log "scan $root: discovered ${#files[@]} test file(s): ${files[*]}"

  # Group files by imported flash_attn version to avoid aicpu fold
  local common_files=() v2_files=() v3_files=() v4_files=()
  local skipped_files=() collect_failed_files=()
  local runnable_common=() runnable_v2=() runnable_v3=() runnable_v4=()
  local common_count=0 v2_count=0 v3_count=0 v4_count=0

  for file in "${files[@]}"; do
    local has_v2=0 has_v3=0 has_v4=0
    grep -q "flash_attn_npu_4" "$file" 2>/dev/null && has_v4=1
    grep -q "flash_attn_npu_3" "$file" 2>/dev/null && has_v3=1
    grep -q "flash_attn_npu" "$file" 2>/dev/null && ! grep -q "flash_attn_npu_[34]" "$file" 2>/dev/null && has_v2=1
    if [ $((has_v2 + has_v3 + has_v4)) -gt 1 ]; then
      log "WARNING: $file imports multiple flash_attn versions, isolating in common group"
      common_files+=("$file")
    elif [ "$has_v4" = "1" ]; then
      v4_files+=("$file")
    elif [ "$has_v3" = "1" ]; then
      v3_files+=("$file")
    elif [ "$has_v2" = "1" ]; then
      v2_files+=("$file")
    else
      common_files+=("$file")
    fi
  done

  # Probe collect-only per group, count total cases
  local group_names=()
  local group_files_str=()
  local group_counts=()
  local group_runnable=()

  for gname in common v2 v3 v4; do
    local gfiles=""
    case "$gname" in
      common) gfiles="${common_files[*]}" ;;
      v2)     gfiles="${v2_files[*]}" ;;
      v3)     gfiles="${v3_files[*]}" ;;
      v4)     gfiles="${v4_files[*]}" ;;
    esac
    [ -z "$gfiles" ] && continue
    group_names+=("$gname")
    group_files_str+=("$gfiles")
    group_counts+=(0)
    local gi=$(( ${#group_names[@]} - 1 ))
    local gtotal=0
    for file in $gfiles; do
      collect_log="$RUN_LOG_DIR/collect_$(pytest_log_name "$file")"
      probe_rc=0
      # shellcheck disable=SC2086
      file_is_runnable "$file" "$collect_log" $SAMPLE_ARG "$@" || probe_rc=$?
      case "$probe_rc" in
        0) gtotal=$((gtotal + COLLECTED_COUNT)) ;;
        2)
          collect_failed_files+=("$file")
          log "collect failed: $file"
          tail -n 20 "$collect_log" 2>/dev/null | sed 's/^/    /'
          echo "[COLLECT-FAILED] $file" >> "$FAILED_FILE"
          ;;
        *) skipped_files+=("$file") ;;
      esac
    done
    group_counts[$gi]=$gtotal
    if [ "$gtotal" -gt 0 ]; then
      group_runnable+=("$gi")
    fi
  done

  local total_run=0
  for gi in "${group_runnable[@]+"${group_runnable[@]}"}"; do
    total_run=$((total_run + 1))
  done
  TOTAL_FILES_RAN=$total_run
  TOTAL_FILES_SKIP=${#skipped_files[@]}
  TOTAL_FILES_CF=${#collect_failed_files[@]}
  TOTAL_FILES_DISC=${#files[@]}

  log "groups: common=${#common_files[@]} v2=${#v2_files[@]} v3=${#v3_files[@]} v4=${#v4_files[@]} skip=${#skipped_files[@]}"

  if [ "${#group_runnable[@]}" -eq 0 ]; then
    return
  fi

  local overall_total=0
  for gi in "${group_runnable[@]}"; do
    overall_total=$((overall_total + group_counts[$gi]))
  done

  local group_idx=0 group_tot=${#group_runnable[@]}
  local cum_p=0 cum_f=0 cum_s=0 overall_base=0

  for gi in "${group_runnable[@]}"; do
    group_idx=$((group_idx + 1))
    local gname="${group_names[$gi]}"
    local gfiles="${group_files_str[$gi]}"
    [ -z "$gfiles" ] && continue
    logfile="$RUN_LOG_DIR/${gname}_group.log"
    local short="${gname}_group"
    FILE_IDX="$group_idx"; FILE_TOT="$group_tot"; FILE_NAME="$short"
    FILE_CASES="${group_counts[$gi]}"
    OVERALL_TOTAL="$overall_total"; OVERALL_BASE="$overall_base"; CUM_F_FAIL="$cum_f"
    local file_start file_dur dur_h
    file_start="$(date +%s)"
    run_pytest "$gfiles" "$logfile" "$@"
    file_dur=$(( $(date +%s) - file_start ))
    unset FILE_IDX FILE_TOT FILE_NAME FILE_CASES OVERALL_TOTAL OVERALL_BASE CUM_F_FAIL
    tally_finished_run "$LOG_DIR/pytest_progress.events" "$logfile"
    cum_p=$((cum_p + TALLY_P))
    cum_f=$((cum_f + TALLY_F))
    cum_s=$((cum_s + TALLY_S))
    overall_base=$((cum_p + cum_f + cum_s))
    if [ "$file_dur" -ge 60 ]; then
      dur_h="$((file_dur / 60))m$((file_dur % 60))s"
    else
      dur_h="${file_dur}s"
    fi
    log "$group_idx/$group_tot $short done: pass=$TALLY_P fail=$TALLY_F skip=$TALLY_S ($dur_h) | overall $overall_base/$overall_total"
    SUMMARY_ROWS+=("$short|$TALLY_P|$TALLY_F|$TALLY_S|$dur_h")
  done
  TOTAL_P=$cum_p; TOTAL_F=$cum_f; TOTAL_S=$cum_s
}

# Final summary: per-file table, slowest cases, one-line verdict
print_final_summary() {
  local wall=$(( $(date +%s) - TEST_PHASE_START ))
  local row name p f s d verdict
  emit() { printf '%s\n' "$1"; printf '%s\n' "$1" >> "$SUMMARY_FILE"; }
  emit ""
  emit "[CI-test] ---- per-file summary ----"
  printf -v row '%-34s %6s %6s %6s %8s' "file" "pass" "fail" "skip" "time"; emit "$row"
  for row in ${SUMMARY_ROWS+"${SUMMARY_ROWS[@]}"}; do
    IFS='|' read -r name p f s d <<< "$row"
    printf -v row '%-34s %6s %6s %6s %8s' "$name" "$p" "$f" "$s" "${d:-?}"; emit "$row"
  done
  printf -v row '%-34s %6s %6s %6s %8s' "TOTAL" "$TOTAL_P" "$TOTAL_F" "$TOTAL_S" "${wall}s"; emit "$row"
  emit "[CI-test] files: ran=$TOTAL_FILES_RAN skip=$TOTAL_FILES_SKIP collect-failed=$TOTAL_FILES_CF discovered=$TOTAL_FILES_DISC"
  if [ "$VERBOSE" = "true" ] && [ -s "$SLOW_CASES_FILE" ]; then
    emit "[CI-test] slowest cases:"
    top5_slow_lines | while IFS= read -r line; do emit "$line"; done
  fi
  if [ "$TOTAL_F" -gt 0 ]; then verdict="FAILED"; else verdict="PASSED"; fi
  emit "[CI-test] ============ $TOTAL_P passed, $TOTAL_F failed, $TOTAL_S skipped in ${wall}s ($verdict) ============"
}

log "running pytest (mode=$MODE workers=$TEST_WORKERS sample=${SAMPLE_ARG:-<none>})"

# Direct mode: run only the file. Directory: scan then run per file.
if [ -n "${CI_TEST_DIRECT_FILE:-}" ] && [ ! -d "${CI_TEST_DIRECT_FILE:-}" ]; then
  log "single file (no scan): $CI_TEST_DIRECT_FILE"
  run_pytest "$CI_TEST_DIRECT_FILE" "$RUN_LOG_DIR/direct.log" ${CI_TEST_DIRECT_FILTER:+-k "$CI_TEST_DIRECT_FILTER"}
  tally_finished_run "$LOG_DIR/pytest_progress.events" "$RUN_LOG_DIR/direct.log"
  TOTAL_P=$TALLY_P; TOTAL_F=$TALLY_F; TOTAL_S=$TALLY_S
  TOTAL_FILES_RAN=1; TOTAL_FILES_DISC=1
  SUMMARY_ROWS+=("$(basename "$CI_TEST_DIRECT_FILE" .py)|$TALLY_P|$TALLY_F|$TALLY_S|?")
else
  run_pytest_scanned "${CI_TEST_DIRECT_FILE:-tests}" ${CI_TEST_DIRECT_FILTER:+-k "$CI_TEST_DIRECT_FILTER"}
fi

echo ""
print_final_summary

# golden cache summary to console and summary.txt
summarize_golden_cache() {
  local artifacts_after
  artifacts_after="$(cache_artifact_count)"
  if [ ! -s "$CACHE_STATS_FILE" ]; then
    log "[golden-cache-summary] no events recorded artifacts_before=$CACHE_ARTIFACTS_BEFORE artifacts_after=$artifacts_after"
    return
  fi
  awk -F '\t' '
    $2 == "test" { count[$1]++ }
    END {
      printf "[CI-test] [golden-cache-summary] hit=%d miss=%d refresh=%d read_error=%d write_ok=%d write_error=%d disabled=%d\n",
        count["hit"], count["miss"], count["refresh"], count["read_error"],
        count["write_ok"], count["write_error"], count["disabled"]
    }
  ' "$CACHE_STATS_FILE"
  log "[golden-cache-summary] artifacts_before=$CACHE_ARTIFACTS_BEFORE artifacts_after=$artifacts_after"
}
summarize_golden_cache 2>&1 | tee -a "$SUMMARY_FILE"

FAILED_CASES="$(tr '\n' ' ' < "$FAILED_FILE" 2>/dev/null || true)"
if [ -n "$FAILED_CASES" ]; then
  die "pytest FAILED targets:$FAILED_CASES"
fi

log "all tests passed"
log "test phase end: $(date '+%Y-%m-%d %H:%M:%S')"
