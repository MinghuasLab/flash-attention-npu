#!/usr/bin/env python3
"""Resolve a PR snapshot and decide whether this head SHA already has an audit."""

import json
import os
from pathlib import Path
import re
import subprocess


AUDIT_MARKER = re.compile(
    r"<!-- pr-review-agent:v1 head=([0-9a-f]{40}) base=([0-9a-f]{40}) audit=(allow|deny|manual) -->"
)


def gh_json(*args):
    return json.loads(subprocess.check_output(["gh", "api", *args], text=True))


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    event_name = os.environ["GITHUB_EVENT_NAME"]
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    output = Path(os.environ["GITHUB_OUTPUT"])

    should_run = True
    if event_name == "issue_comment":
        pr = event.get("issue", {}).get("pull_request")
        should_run = bool(pr) and re.fullmatch(r"/review", event.get("comment", {}).get("body", "").strip()) is not None
        pr_number = event.get("issue", {}).get("number") if pr else None
    elif event_name == "pull_request_target":
        pr_event = event.get("pull_request", {})
        pr_number = pr_event.get("number")
        should_run = pr_number is not None and not pr_event.get("draft", False)
    else:
        raise ValueError(f"unsupported event: {event_name}")

    if not should_run:
        values = {"should_run": "false", "pr_number": "", "head_sha": "",
                  "base_sha": "", "audit_required": "false", "previous_audit": ""}
    else:
        pr = gh_json(f"repos/{repo}/pulls/{pr_number}")
        head_sha = pr["head"]["sha"]
        base_sha = pr["base"]["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", head_sha) or not re.fullmatch(r"[0-9a-f]{40}", base_sha):
            raise ValueError("GitHub returned an invalid PR SHA")

        query = (f'.[] | select(.commit_id == "{head_sha}" and '
                 '.user.login == "github-actions[bot]") | .body')
        bodies = subprocess.check_output(
            ["gh", "api", "--paginate", "--jq", query,
             f"repos/{repo}/pulls/{pr_number}/reviews?per_page=100"], text=True)
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


if __name__ == "__main__":
    main()
