#!/usr/bin/env bash
#
# 扫描宿主机 Ascend NPU，选一张可用于 CI 的卡。
#
# 用法:
#   bash ci/detect_npu.sh --summary      # 打印所有 NPU 与选中卡
#   bash ci/detect_npu.sh --candidates   # 打印 CI 选卡顺序 (按 task_count 升序)
#   bash ci/detect_npu.sh --env          # 打印 "export NPU_SELECTED_DEVICE=<id>"
#   eval "$(bash ci/detect_npu.sh --env)"
#
# 选卡策略 (动态选卡, 不锁卡):
#   1. 过滤: health=OK (CI_REQUIRE_HEALTHY_NPU=true 时) + free>MIN_FREE + task_count<MAX_TASKS
#   2. 排序: task_count 升序 (负载均衡, 优先选最空闲的卡), task_count 相同按 id 升序
#   3. 没有候选时返回非 0
#
# 兼容两种 npu-smi info 卡信息表格式:
#   910B (25.5.0): "| NPU   Name | Health |"   -> id+name 在 f[2], health 在 f[3]
#   950  (25.7.x): "| NPU ID | Name   | Health |" -> id 在 f[2], name 在 f[3], health 在 f[4]
# 自适应: 在含 health 关键词的行内用正则抓取 health, 再从行首的数字字段取 id。
#
# 环境变量:
#   CI_REQUIRE_HEALTHY_NPU  (默认 false)  为 true 时只接受 health=OK 的卡
#   CI_NPU_MAX_TASKS        (默认 4)      卡上进程数 < 此值才选用
#   CI_NPU_MIN_FREE_MB      (默认 1024)   HBM 剩余(MB) > 此值才选用
#   NPU_SELECTED_DEVICE     可手动指定，跳过自动选卡

set -euo pipefail

NPU_SMI_BIN="${NPU_SMI_BIN:-npu-smi}"
REQUIRE_HEALTHY="${CI_REQUIRE_HEALTHY_NPU:-false}"
MAX_TASKS="${CI_NPU_MAX_TASKS:-4}"
MIN_FREE_MB="${CI_NPU_MIN_FREE_MB:-1024}"

log() { printf '[detect_npu] %s\n' "$*"; }
die() { printf '[detect_npu][ERROR] %s\n' "$*" >&2; exit 1; }

command -v "$NPU_SMI_BIN" >/dev/null 2>&1 || die "npu-smi not found in PATH; is Ascend driver installed?"

# 解析 npu-smi info 输出，得到 "id|name|health|free|task_count|soc" 行。
# 兼容 910B 和 950 两种输出格式。
# 卡信息表: 每张卡两行
#   NPU 行: 含 health (OK/Alarm/Warning/Critical) + id + name
#   Chip 行: 含 Bus-Id + HBM-Usage "used / total" (取最后一个数字/数字段)
# 进程表 (卡信息表下方):
#   进程行: "| <npu_id> ... | <pid> ... | <name> | <mem> |"
#   空卡行: "| No running processes found in NPU <npu_id> |"
# 两表都解析, 合并出 task_count。
parse_npu_smi() {
  "$NPU_SMI_BIN" info 2>/dev/null | awk '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s; }
    BEGIN {
      next_id=""; next_name=""; next_health="";
      delete task_count;   # task_count[npu_id] = 进程数
    }
    # NPU 行: 含 health 关键词 (OK/Alarm/Warning/Critical) 且不是表头
    /\| (OK|Alarm|Warning|Critical)[ \t]*\|/ {
      line=$0;
      n=split(line, f, /[|]/);
      if (n >= 4) {
        # 950 格式: f[2]=id, f[3]=name, f[4]=health
        # 910B 格式: f[2]="id name", f[3]=health
        h4=trim(f[4]);
        if (h4 ~ /^(OK|Alarm|Warning|Critical)$/) {
          # 950: health 在 f[4]
          next_id=trim(f[2]);
          next_name=trim(f[3]);
          next_health=h4;
        } else {
          # 910B: id+name 在 f[2], health 在 f[3]
          idname=trim(f[2]);
          split(idname, idname_parts, /[ \t]+/);
          next_id=idname_parts[1];
          next_name=idname_parts[2];
          health_field=trim(f[3]);
          split(health_field, hp, /[ \t]+/);
          next_health=hp[1];
        }
      }
      next;
    }
    # chip 行: 含 Bus-Id (形如 0000:XX:00.0) 和 "used / total" 段
    # 取最后一个 "数字 / 数字" = HBM-Usage (在行尾)
    /0000:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9]/ {
      line=$0;
      seg="";
      s=line;
      while (match(s, /[0-9]+[ \t]*\/[ \t]*[0-9]+/)) {
        seg=substr(s, RSTART, RLENGTH);
        s=substr(s, RSTART+RLENGTH);
      }
      if (seg != "" && next_id != "") {
        gsub(/[ \t]/, "", seg);
        split(seg, parts, "/");
        used=parts[1]+0; total=parts[2]+0;
        free=total-used;
        # 暂存, 等进程表统计完 task_count 再输出
        npu_free[next_id]=free;
        npu_name[next_id]=next_name;
        npu_health[next_id]=next_health;
        npu_soc[next_id]="ascend910b";
        next_id="";
      }
      next;
    }
    # 进程表 - 进程行: 行首 "|" 后跟数字 (npu_id), 后续有 pid 数字
    # 格式: | <npu_id> ... | <pid> ... | <name> | <mem> |
    # 注意排除表头行 (含 "Process id" 文字) 和分隔线
    /^\|[ \t]+[0-9]+[ \t]+.*\|[ \t]+[0-9]+[ \t]+/ {
      line=$0;
      n=split(line, pf, /[|]/);
      if (n >= 3) {
        id_field=trim(pf[2]);
        split(id_field, idp, /[ \t]+/);
        nid=idp[1]+0;
        if (nid != "") {
          task_count[nid]++;
        }
      }
      next;
    }
    # 进程表 - 空卡行: "No running processes found in NPU <id>"
    /No running processes found in NPU [0-9]+/ {
      match($0, /NPU [0-9]+/);
      seg=substr($0, RSTART+4, RLENGTH-4);
      nid=seg+0;
      if (!(nid in task_count)) {
        task_count[nid]=0;
      }
      next;
    }
    END {
      # 输出所有解析到的卡 (已从卡信息表拿到 free/name/health)
      for (nid in npu_free) {
        tc = (nid in task_count) ? task_count[nid] : 0;
        printf "%d|%s|%s|%d|%d|%s\n", nid, npu_name[nid], npu_health[nid], npu_free[nid], tc, npu_soc[nid];
      }
    }
  '
}

# 生成候选列表: "id|name|health|free|task_count|soc"
# 过滤: free > MIN_FREE_MB, task_count < MAX_TASKS, health (REQUIRE_HEALTHY 时)
# 排序: task_count 升序, 相同按 id 升序
build_candidates() {
  local rows
  rows="$(parse_npu_smi)"
  if [ -z "$rows" ]; then
    return 0
  fi

  # 过滤 + 排序 (sort -t'|' -k5,5n -k1,1n)
  echo "$rows" | while IFS='|' read -r id name health free task_count soc; do
    [ -z "$id" ] && continue
    if [ "${free:-0}" -le "$MIN_FREE_MB" ]; then
      continue
    fi
    if [ "${task_count:-0}" -ge "$MAX_TASKS" ]; then
      continue
    fi
    if [ "$REQUIRE_HEALTHY" = "true" ] && [ "$health" != "OK" ]; then
      continue
    fi
    printf '%s|%s|%s|%s|%s|%s\n' "$id" "$name" "$health" "$free" "$task_count" "$soc"
  done | sort -t'|' -k5,5n -k1,1n
}

select_device() {
  # 手动指定优先
  if [ -n "${NPU_SELECTED_DEVICE:-}" ]; then
    echo "$NPU_SELECTED_DEVICE"
    return 0
  fi
  local first
  first="$(build_candidates | head -n1 | cut -d'|' -f1)"
  if [ -z "$first" ]; then
    return 1
  fi
  echo "$first"
  return 0
}

case "${1:---summary}" in
  --summary)
    echo "Detected NPU devices:"
    rows="$(parse_npu_smi)"
    if [ -z "$rows" ]; then
      die "no NPU detected by npu-smi"
    fi
    # 按 id 升序展示全量
    echo "$rows" | sort -t'|' -k1,1n | while IFS='|' read -r id name health free task_count soc; do
      [ -z "$id" ] && continue
      printf '  - id=%s name=%s health=%s free=%sMB tasks=%s soc=%s\n' \
        "$id" "$name" "$health" "$free" "$task_count" "$soc"
    done
    echo ""
    echo "Selection criteria: free>${MIN_FREE_MB}MB tasks<${MAX_TASKS} require_healthy=${REQUIRE_HEALTHY}"
    sel="$(select_device || true)"
    if [ -n "$sel" ]; then
      meta="$(build_candidates | grep "^${sel}|" || true)"
      if [ -n "$meta" ]; then
        IFS='|' read -r id name health free task_count soc <<< "$meta"
        printf 'Selected NPU: id=%s name=%s health=%s free=%sMB tasks=%s soc=%s\n' \
          "$id" "$name" "$health" "$free" "$task_count" "$soc"
      else
        printf 'Selected NPU: id=%s\n' "$sel"
      fi
    else
      die "no selectable NPU (free>${MIN_FREE_MB}MB and tasks<${MAX_TASKS}) found"
    fi
    ;;
  --candidates)
    cands="$(build_candidates)"
    if [ -z "$cands" ]; then
      die "no candidate NPU (free>${MIN_FREE_MB}MB and tasks<${MAX_TASKS})"
    fi
    echo "CI will try NPU in this order (sorted by task_count asc, id asc):"
    while IFS='|' read -r id name health free task_count soc; do
      [ -z "$id" ] && continue
      printf '  - id=%s name=%s health=%s free=%sMB tasks=%s\n' \
        "$id" "$name" "$health" "$free" "$task_count"
    done <<< "$cands"
    ;;
  --env)
    sel="$(select_device || true)"
    if [ -z "$sel" ]; then
      die "no selectable NPU"
    fi
    printf 'export NPU_SELECTED_DEVICE=%s\n' "$sel"
    ;;
  *)
    die "unknown argument: $1 (use --summary|--candidates|--env)"
    ;;
esac
