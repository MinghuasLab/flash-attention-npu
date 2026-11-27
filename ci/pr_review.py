#!/usr/bin/env python3
# Copyright (c) 2026, Minghua Shen.

"""Trusted PR review entrypoint: snapshot preparation, Cursor review, and publication."""

import argparse
import codecs
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


AUDIT_MARKER = re.compile(
    r"<!-- pr-review-agent:v1 head=([0-9a-f]{40}) base=([0-9a-f]{40}) audit=(allow|deny|manual) -->"
)


def gh_json(*args):
    return json.loads(subprocess.check_output(["gh", "api", *args], text=True))


def prepare_main():
    repo = os.environ["GITHUB_REPOSITORY"]
    event_name = os.environ["GITHUB_EVENT_NAME"]
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    output = Path(os.environ["GITHUB_OUTPUT"])

    should_run = True
    if event_name == "pull_request_target":
        pr_event = event.get("pull_request", {})
        pr_number = pr_event.get("number")
        should_run = pr_number is not None and not pr_event.get("draft", False)
    else:
        raise ValueError(f"unsupported event: {event_name}")

    if not should_run:
        values = {
            "should_run": "false",
            "pr_number": "",
            "head_sha": "",
            "base_sha": "",
            "audit_required": "false",
            "previous_audit": "",
        }
    else:
        pr = gh_json(f"repos/{repo}/pulls/{pr_number}")
        head_sha = pr["head"]["sha"]
        base_sha = pr["base"]["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", head_sha) or not re.fullmatch(
            r"[0-9a-f]{40}", base_sha
        ):
            raise ValueError("GitHub returned an invalid PR SHA")

        query = (
            f'.[] | select(.commit_id == "{head_sha}" and '
            '.user.login == "github-actions[bot]") | .body'
        )
        bodies = subprocess.check_output(
            [
                "gh",
                "api",
                "--paginate",
                "--jq",
                query,
                f"repos/{repo}/pulls/{pr_number}/reviews?per_page=100",
            ],
            text=True,
        )
        previous_audit = ""
        for body in bodies.splitlines():
            match = AUDIT_MARKER.search(body)
            if match and match.group(1) == head_sha and match.group(2) == base_sha:
                previous_audit = match.group(3)
        values = {
            "should_run": "true",
            "pr_number": str(pr_number),
            "head_sha": head_sha,
            "base_sha": base_sha,
            "audit_required": "false" if previous_audit else "true",
            "previous_audit": previous_audit,
        }

    with output.open("a") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")
    print("Review event prepared: " + json.dumps(values, ensure_ascii=False))


SHA = re.compile(r"^[0-9a-f]{40}$")
PROMPT = Path(__file__).resolve().parent.parent / ".github" / "prompts" / "pr-review.prompt.md"


def progress(message, started):
    elapsed = time.monotonic() - started
    print(f"{elapsed:6.1f}s  {message}", file=sys.stderr, flush=True)


def clean_terminal_text(value):
    value = codecs.decode(value, "unicode_escape") if "\\x1b" in value else value
    return "".join(
        char for char in value if char in "\n\r\t" or ord(char) >= 32 and ord(char) != 127
    )


def shell_command_summary(command, checkout):
    if not isinstance(command, str):
        return "shell inspection"
    command = command.replace(str(checkout), ".")
    command = re.sub(
        r"(?i)(CURSOR_API_KEY|GITHUB_TOKEN|GH_TOKEN|API_KEY|PASSWORD)(\s*=\s*)\S+",
        r"\1\2[redacted]",
        command,
    )
    command = re.sub(r"^\s*cd\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s*&&\s*", "", command)
    command = " ".join(clean_terminal_text(command).split())
    return command[:180] + ("…" if len(command) > 180 else "")


def run(command, *, cwd=None, env=None, text=True):
    return subprocess.run(
        command, cwd=cwd, env=env, text=text, check=True, capture_output=True
    ).stdout


def snapshot_from_github(number, env):
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("GITHUB_REPOSITORY must be OWNER/REPO")
    data = json.loads(run(["gh", "api", f"repos/{repository}/pulls/{number}"], env=env))
    if data.get("state") != "open":
        raise ValueError("PR must be open")
    snapshot = {
        "repository": repository,
        "number": number,
        "head_sha": data["head"]["sha"],
        "base_sha": data["base"]["sha"],
        "base_ref": data["base"]["ref"],
    }
    diff = run(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/vnd.github.v3.diff",
            f"repos/{repository}/pulls/{number}",
        ],
        env=env,
    )
    return snapshot, diff


def validate_snapshot(snapshot):
    for key in ("head_sha", "base_sha"):
        if not isinstance(snapshot.get(key), str) or not SHA.fullmatch(snapshot[key]):
            raise ValueError(f"invalid {key}")
    if snapshot["head_sha"] == snapshot["base_sha"]:
        raise ValueError("head and base SHAs must differ")


def added_lines(diff):
    lines = set()
    path = None
    new_line = None
    for row in diff.splitlines():
        if row.startswith("+++ b/"):
            path = row[6:]
        elif row.startswith("@@ "):
            match = re.search(r"\+(\d+)(?:,\d+)? @@", row)
            new_line = int(match.group(1)) if match else None
        elif new_line is not None and row.startswith("+") and not row.startswith("+++ "):
            if path is not None:
                lines.add((path, new_line))
            new_line += 1
        elif new_line is not None and row.startswith(" "):
            new_line += 1
    return lines


def extract_json_object(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
                if isinstance(value, dict):
                    return json.dumps(value, ensure_ascii=False)
            except json.JSONDecodeError:
                continue
    raise ValueError("Cursor output did not contain a JSON object")


def validate_result(result, snapshot, audit_required, diff):
    if result.get("schema_version") != 1 or result.get("audit_required") is not audit_required:
        raise ValueError("schema_version or audit_required mismatch")
    pr = result.get("pr")
    for key in ("number", "head_sha", "base_sha", "base_ref"):
        if not isinstance(pr, dict) or pr.get(key) != snapshot.get(key):
            raise ValueError(f"PR snapshot mismatch: {key}")
    review = result.get("review")
    if not isinstance(review, dict) or not isinstance(review.get("summary"), str):
        raise ValueError("invalid review summary")
    for key in ("cross_cutting", "inline_comments"):
        if not isinstance(review.get(key), list):
            raise ValueError(f"invalid {key}")
    valid_lines = added_lines(diff)
    for item in review["inline_comments"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("line"), int)
            or item["line"] < 1
            or item.get("side") != "RIGHT"
            or item.get("severity") not in {"high", "medium", "low"}
            or item.get("category") not in {"security", "review"}
            or not isinstance(item.get("body"), str)
        ):
            raise ValueError("invalid inline comment")
        if (item["path"], item["line"]) not in valid_lines:
            raise ValueError("inline comment does not point to an added diff line")
    for item in review["cross_cutting"]:
        if (
            not isinstance(item, dict)
            or item.get("severity") not in {"high", "medium", "low"}
            or not isinstance(item.get("title"), str)
            or not isinstance(item.get("detail"), str)
            or not isinstance(item.get("related_paths"), list)
        ):
            raise ValueError("invalid cross-cutting finding")
    audit = result.get("security_audit")
    if not isinstance(audit, dict):
        raise ValueError("missing security_audit")
    if audit_required:
        if audit.get("status") != "completed" or audit.get("recommendation") not in {
            "allow",
            "deny",
            "manual",
        }:
            raise ValueError("invalid required audit")
        if not isinstance(audit.get("rationale"), str) or not isinstance(
            audit.get("evidence"), list
        ):
            raise ValueError("invalid audit rationale or evidence")
        for item in audit["evidence"]:
            if (
                not isinstance(item, dict)
                or item.get("path") is not None
                and not isinstance(item.get("path"), str)
                or item.get("line") is not None
                and (not isinstance(item.get("line"), int) or item["line"] < 1)
                or not isinstance(item.get("note"), str)
            ):
                raise ValueError("invalid audit evidence item; expected path/line/note object")
        for field in ("entry_bypass_found", "residual_risks"):
            if field not in audit:
                raise ValueError(f"missing security audit field: {field}")
        if not isinstance(audit["entry_bypass_found"], bool) or not isinstance(
            audit["residual_risks"], list
        ):
            raise ValueError("invalid security audit bypass flag or residual risks")
    elif (
        audit.get("status") != "skipped"
        or audit.get("recommendation") is not None
        or audit.get("rationale") is not None
        or audit.get("evidence") != []
    ):
        raise ValueError("audit should have been skipped")


def review_main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pr", type=int)
    source.add_argument("--snapshot", type=Path)
    parser.add_argument("--source-repo", help="Local Git repository for a snapshot fixture")
    parser.add_argument("--audit-required", choices=("true", "false"), default="true")
    parser.add_argument("--model", help="Cursor model name; omit to use Cursor's default")
    parser.add_argument("--expected-head-sha")
    parser.add_argument("--expected-base-sha")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.pr is not None and args.pr < 1:
        parser.error("--pr must be positive")
    if args.snapshot and not args.source_repo:
        parser.error("--source-repo is required with --snapshot")
    if (args.output_dir / "review_result.json").exists():
        parser.error(
            "output directory already contains review_result.json; use a fresh run directory"
        )

    cursor_key = os.environ.get("CURSOR_API_KEY")
    if not cursor_key:
        parser.error("CURSOR_API_KEY is required")
    safe_env = os.environ.copy()
    safe_env.pop("CURSOR_API_KEY", None)
    audit_required = args.audit_required == "true"
    if args.pr is not None:
        progress(f"Fetching PR #{args.pr} metadata and diff from GitHub API", started)
        snapshot, diff = snapshot_from_github(args.pr, safe_env)
        source_repo = f"https://github.com/{snapshot['repository']}.git"
    else:
        snapshot = json.loads(args.snapshot.read_text())
        source_repo = args.source_repo
        diff = None
    validate_snapshot(snapshot)
    for name, expected in (
        ("head_sha", args.expected_head_sha),
        ("base_sha", args.expected_base_sha),
    ):
        if expected is not None and (not SHA.fullmatch(expected) or snapshot[name] != expected):
            raise ValueError(f"PR {name} changed after the workflow snapshot was fixed")
    progress(
        f"Snapshot fixed   head {snapshot['head_sha'][:12]} · base {snapshot['base_sha'][:12]} · audit {audit_required}",
        started,
    )

    with tempfile.TemporaryDirectory(prefix="pr-review-") as temporary:
        checkout = Path(temporary) / "source"
        progress("Git             preparing temporary checkout", started)
        run(
            ["git", "clone", "--no-checkout", "--filter=blob:none", source_repo, str(checkout)],
            env=safe_env,
        )
        if args.pr is not None:
            progress("Git             fetching PR head and verifying SHA", started)
            run(["git", "fetch", "origin", f"refs/pull/{args.pr}/head"], cwd=checkout, env=safe_env)
            if (
                run(["git", "rev-parse", "FETCH_HEAD"], cwd=checkout, env=safe_env).strip()
                != snapshot["head_sha"]
            ):
                raise ValueError("PR head changed during fetch")
        for sha in (snapshot["base_sha"], snapshot["head_sha"]):
            progress(f"Git             checking commit {sha[:12]}", started)
            run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=checkout, env=safe_env)
        progress("Git             checking out fixed head and preparing diff", started)
        run(
            ["git", "-c", "core.hooksPath=/dev/null", "checkout", "--detach", snapshot["head_sha"]],
            cwd=checkout,
            env=safe_env,
        )
        if diff is None:
            diff = run(
                ["git", "diff", "--no-ext-diff", snapshot["base_sha"], snapshot["head_sha"]],
                cwd=checkout,
                env=safe_env,
            )
        if not diff.strip() or not diff.startswith("diff --git "):
            raise ValueError("empty or invalid diff")
        (checkout / ".review-snapshot.json").write_text(
            json.dumps({"pr": snapshot, "audit_required": audit_required}, indent=2)
        )
        (checkout / ".review.diff").write_text(diff)
        # PR-controlled Cursor config must not relax the trusted review tool policy.
        project_cursor_config = checkout / ".cursor"
        if project_cursor_config.is_symlink() or project_cursor_config.is_file():
            project_cursor_config.unlink()
        project_cursor_config.mkdir(exist_ok=True)
        for config_name in ("cli.json", "sandbox.json"):
            untrusted_config = project_cursor_config / config_name
            if untrusted_config.is_symlink() or untrusted_config.is_file():
                untrusted_config.unlink()
            elif untrusted_config.exists():
                shutil.rmtree(untrusted_config)
        cursor_permissions = {
            "permissions": {
                "allow": ["Read(**)"],
                "deny": ["Write(**)", "Shell(*)", "WebFetch(*)", "Mcp(*:*)"],
            }
        }
        (project_cursor_config / "cli.json").write_text(json.dumps(cursor_permissions, indent=2))
        instruction = (
            PROMPT.read_text()
            + "\n\nReview this fixed snapshot. Read .review-snapshot.json and .review.diff first. "
            "Use this checkout only for necessary read-only context. Do not use shell, web, or MCP tools. "
            "Return the JSON to stdout.\n"
        )
        cursor_env = safe_env.copy()
        cursor_env.pop("GH_TOKEN", None)
        cursor_env.pop("GITHUB_TOKEN", None)
        cursor_env["CURSOR_API_KEY"] = cursor_key
        cursor_home = Path(temporary) / "cursor-home"
        (cursor_home / ".cursor").mkdir(parents=True)
        cursor_cli_config = {
            "version": 1,
            "editor": {"vimMode": False},
            **cursor_permissions,
            "sandbox": {
                "readBoundary": "workspace",
            },
        }
        (cursor_home / ".cursor" / "cli-config.json").write_text(
            json.dumps(cursor_cli_config, indent=2)
        )
        cursor_env["HOME"] = str(cursor_home)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_label = args.model or "default"
        progress(
            f"Agent           starting read-only review · shell/web/MCP/write denied · model {model_label}",
            started,
        )
        command = [
            "agent",
            "-p",
            "--trust",
            "--mode=ask",
            "--output-format",
            "stream-json",
            "--stream-partial-output",
        ]
        if args.model:
            command.extend(["--model", args.model])
        command.append(instruction)
        cursor_log_path = args.output_dir / "cursor.log.txt"
        with cursor_log_path.open("w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=checkout,
                env=cursor_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_parts = []
            final_result = None
            final_assistant = None
            active_tools = {}
            next_progress_chars = 4000
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    cleaned = clean_terminal_text(line).strip()
                    if cleaned:
                        progress(f"Agent           {cleaned[:120]}", started)
                    continue
                if not isinstance(event, dict):
                    cleaned = clean_terminal_text(line).strip()
                    if cleaned:
                        progress(f"Agent           {cleaned[:120]}", started)
                    continue
                event_type = event.get("type")
                subtype = event.get("subtype", "")
                if event_type == "system" and subtype == "init":
                    progress(
                        f"Agent           initialized · model {event.get('model', 'unknown')}",
                        started,
                    )
                elif event_type == "tool_call":
                    tool_call = event.get("tool_call", {})
                    call_type = next((key for key in tool_call if key.endswith("ToolCall")), None)
                    details = tool_call.get(call_type, {}) if call_type else {}
                    args_info = details.get("args", {})
                    tool_id = event.get("toolCallId") or event.get("call_id")
                    if subtype == "completed":
                        started_tool = active_tools.pop(tool_id, None)
                        if started_tool and call_type == "shellToolCall":
                            summary, tool_started = started_tool
                            elapsed = time.monotonic() - tool_started
                            progress(
                                f"Agent shell     finished · {elapsed:.1f}s · {summary}", started
                            )
                        continue
                    if subtype != "started":
                        continue
                    target = args_info.get("path") or ""
                    if target.startswith(str(checkout)):
                        target = "." + target[len(str(checkout)) :]
                    if call_type == "readToolCall":
                        progress(f"Agent           reading {target or 'file'}", started)
                    elif call_type in {"globToolCall", "grepToolCall"}:
                        progress("Agent           searching relevant files", started)
                    elif call_type == "shellToolCall":
                        summary = shell_command_summary(args_info.get("command"), checkout)
                        active_tools[tool_id] = (summary, time.monotonic())
                        progress(f"Agent shell     {summary}", started)
                    else:
                        progress("Agent           inspecting relevant context", started)
                elif event_type == "assistant":
                    message = event.get("message", {})
                    message_text = "".join(
                        block.get("text", "")
                        for block in message.get("content", [])
                        if isinstance(block, dict) and isinstance(block.get("text"), str)
                    )
                    for block in message.get("content", []):
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            output_parts.append(block["text"])
                    if "timestamp_ms" not in event and message_text:
                        final_assistant = message_text
                    total_chars = sum(map(len, output_parts))
                    if total_chars >= next_progress_chars:
                        progress("Agent           still analyzing the snapshot", started)
                        next_progress_chars += 4000
                elif event_type == "result":
                    final_result = event.get("result")
                    progress(
                        f"Agent           response ready · {event.get('duration_ms', 'unknown')} ms",
                        started,
                    )
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Cursor exited {return_code}; see {cursor_log_path}")
        cursor_output = final_assistant or "".join(output_parts)
        if not cursor_output.strip() and isinstance(final_result, str):
            cursor_output = final_result
        cursor_output = extract_json_object(cursor_output)
        (args.output_dir / "review_result.raw.txt").write_text(cursor_output)
        progress("Validator       parsing and checking response", started)
        result = json.loads(cursor_output)
        validate_result(result, snapshot, audit_required, diff)
        if args.pr is not None:
            progress("GitHub          rechecking PR snapshot for updates", started)
            current, _ = snapshot_from_github(args.pr, safe_env)
            if current != snapshot:
                raise ValueError("PR snapshot changed during review; result is stale")
        (args.output_dir / "review_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        (args.output_dir / "snapshot.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
        )
        progress(
            f"Complete        validated result saved · {args.output_dir / 'review_result.json'}",
            started,
        )


def publish_security_status(repository, sha, verdict):
    status_by_verdict = {
        "allow": ("success", "Security audit allows the fixed snapshot."),
        "deny": ("failure", "Security audit denied the fixed snapshot."),
        "manual": ("error", "Security audit requires manual review."),
    }
    state, description = status_by_verdict[verdict]
    subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{repository}/statuses/{sha}",
            "-f",
            f"state={state}",
            "-f",
            "context=Trusted PR Security Audit",
            "-f",
            f"description={description}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def render_body(result, audit_verdict):
    review = result["review"]
    audit = result["security_audit"]
    sha = result["pr"]["head_sha"]
    base_sha = result["pr"]["base_sha"]
    lines = [
        f"<!-- pr-review-agent:v1 head={sha} base={base_sha} audit={audit_verdict} -->",
        f"## Security audit: `{audit_verdict.upper()}`",
        "",
    ]
    if audit["status"] == "completed":
        lines.append(audit["rationale"])
        for item in audit["evidence"]:
            location = ""
            if item.get("path"):
                location = f"`{item['path']}" + (f":{item['line']}`" if item.get("line") else "`")
            lines.append(f"- {location + ': ' if location else ''}{item['note']}")
    else:
        lines.append(
            f"Reused the existing `{audit_verdict.upper()}` audit conclusion for this head SHA."
        )
    lines += ["", "## Review summary", "", review["summary"]]
    if review["cross_cutting"]:
        lines += ["", "## Overall findings", ""]
        for finding in review["cross_cutting"]:
            lines += [
                f"### [{finding['severity'].upper()}] {finding['title']}",
                "",
                finding["detail"],
            ]
    return "\n".join(lines)


def render_inline_comment(item):
    category = item["category"].upper()
    severity = item["severity"].upper()
    label = f"[{category}][{severity}]"
    if item["category"] == "security":
        label = f"🔴 **{label}**"
    else:
        label = f"🟠 **{label}**"
    return f"{label} {item['body']}"


def publish_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--expected-head-sha", required=True)
    parser.add_argument("--expected-base-sha", required=True)
    parser.add_argument("--previous-audit", default="")
    args = parser.parse_args()

    result = json.loads(open(args.result, encoding="utf-8").read())
    snapshot = result["pr"]
    if snapshot["number"] != args.pr:
        raise ValueError("review result PR number does not match the requested PR")
    if (
        snapshot["head_sha"] != args.expected_head_sha
        or snapshot["base_sha"] != args.expected_base_sha
    ):
        raise ValueError("review result SHA does not match the workflow snapshot")

    audit = result["security_audit"]
    if audit["status"] == "completed":
        audit_verdict = audit["recommendation"]
    elif audit["status"] == "skipped" and args.previous_audit in {"allow", "deny", "manual"}:
        audit_verdict = args.previous_audit
    else:
        raise ValueError("missing current or previously recorded audit conclusion")

    repository = os.environ["GITHUB_REPOSITORY"]
    current = gh_json(f"repos/{repository}/pulls/{args.pr}")
    if (
        current["head"]["sha"] != args.expected_head_sha
        or current["base"]["sha"] != args.expected_base_sha
    ):
        raise ValueError("PR head or base changed during review; refusing to publish stale output")

    payload = {
        "commit_id": args.expected_head_sha,
        "event": "COMMENT",
        "body": render_body(result, audit_verdict),
        "comments": [
            {
                "path": item["path"],
                "line": item["line"],
                "side": "RIGHT",
                "body": render_inline_comment(item),
            }
            for item in sorted(
                result["review"]["inline_comments"],
                key=lambda item: item["category"] != "security",
            )
        ],
    }
    response = subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            "--input",
            "-",
            f"repos/{repository}/pulls/{args.pr}/reviews",
        ],
        input=json.dumps(payload),
        text=True,
        check=True,
        capture_output=True,
    )
    review = json.loads(response.stdout)
    publish_security_status(repository, args.expected_head_sha, audit_verdict)
    print(f"Published review {review.get('id')} for PR #{args.pr} at {args.expected_head_sha}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: pr_review.py {prepare|review|publish} [args...]")
    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if command == "prepare":
        prepare_main()
    elif command == "review":
        review_main()
    elif command == "publish":
        publish_main()
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"pr_review failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
