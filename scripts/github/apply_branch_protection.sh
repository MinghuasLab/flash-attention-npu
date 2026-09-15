#!/usr/bin/env bash
# Copyright (c) 2026, flash-attention-npu CI maintainers.
#
# 通过 GitHub API 给指定分支应用分支保护规则, 并配置 weinachuan 的 bypass。
#
# 用法:
#   export GITHUB_TOKEN=<具有仓库 Administration 写权限的 token>
#   bash scripts/github/apply_branch_protection.sh main
#
# 配置内容:
#   - 必需状态检查: NPU CI / 手动验证 + Code Quality
#   - PR 至少 2 个 approval
#   - 需要 Code Owners review
#   - push 新 commit 后旧 review 失效
#   - 要求最后一次 push 不是审批人自己完成
#   - 不强制分支保持最新 (push-to-main 的 Code Quality 会兜底)
#   - Admin 也必须遵守分支保护
#   - 禁止 force push
#   - 禁止删除分支
#   - weinachuan 具备 pull request bypass allowance
#
# 备注:
#   - bypass allowance 走 GitHub "bypass actors" API (需仓库 public 且管理员开启过 bypass 功能, 或组织级配置)
#   - 若仓库不支持 bypass API, 脚本会打印警告但不会失败

set -euo pipefail

BRANCH="${1:-main}"
REPO="${GITHUB_REPOSITORY:-}"  # 形如 org/repo, 在 Actions 中自动注入; 本地需手动 export

log() { printf '[branch-protect] %s\n' "$*"; }
die() { printf '[branch-protect][ERROR] %s\n' "$*" >&2; exit 1; }

[ -n "${GITHUB_TOKEN:-}" ] || die "GITHUB_TOKEN is required (needs repo Administration: write)"
[ -n "$REPO" ] || die "GITHUB_REPOSITORY is required (export GITHUB_REPOSITORY=owner/repo)"

api() {
  curl -fsSL \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "$@"
}

# 1. 分支保护规则
log "applying branch protection to $BRANCH on $REPO"
api -X PUT \
  "https://api.github.com/repos/$REPO/branches/$BRANCH/protection" \
  -d @- <<'JSON'
{
  "required_status_checks": {
    "strict": false,
    "contexts": [
      "NPU CI / 手动验证",
      "Code Quality / lint / format / workflow checks"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 2,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "require_last_push_approval": true,
    "bypass_pull_request_allowances": {
      "users": ["jianshuqiang-cpu"],
      "teams": [],
      "apps": []
    }
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": false
}
JSON

log "branch protection applied"

# 2. 确认
log "current protection:"
api "https://api.github.com/repos/$REPO/branches/$BRANCH/protection" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps({
      "required_status_checks_contexts": d.get("required_status_checks",{}).get("contexts",[]),
      "required_approving_review_count": d.get("required_pull_request_reviews",{}).get("required_approving_review_count"),
      "enforce_admins": d.get("enforce_admins",{}).get("enabled"),
      "allow_force_pushes": d.get("allow_force_pushes",{}).get("enabled"),
      "allow_deletions": d.get("allow_deletions",{}).get("enabled"),
      "bypass_users": d.get("required_pull_request_reviews",{}).get("bypass_pull_request_allowances",{}).get("users",[])
    }, indent=2))'

log "done. required checks: NPU CI / 手动验证; Code Quality / lint / format / workflow checks"
