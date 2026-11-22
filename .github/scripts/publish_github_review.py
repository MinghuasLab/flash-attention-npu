#!/usr/bin/env python3
"""Publish a validated review result against the exact PR snapshot it reviewed."""

import argparse
import json
import os
import subprocess
import sys


def gh_json(*args):
    return json.loads(subprocess.check_output(["gh", "api", *args], text=True))


def render_body(result, audit_verdict):
    review = result["review"]
    audit = result["security_audit"]
    sha = result["pr"]["head_sha"]
    base_sha = result["pr"]["base_sha"]
    lines = [
        f"<!-- pr-review-agent:v1 head={sha} base={base_sha} audit={audit_verdict} -->",
        "## Review summary",
        "",
        review["summary"],
    ]
    if review["cross_cutting"]:
        lines += ["", "## Overall findings", ""]
        for finding in review["cross_cutting"]:
            lines += [f"### [{finding['severity'].upper()}] {finding['title']}", "", finding["detail"]]
    lines += ["", f"## Security audit: `{audit_verdict.upper()}`", ""]
    if audit["status"] == "completed":
        lines.append(audit["rationale"])
        for item in audit["evidence"]:
            location = ""
            if item.get("path"):
                location = f"`{item['path']}" + (f":{item['line']}`" if item.get("line") else "`")
            lines.append(f"- {location + ': ' if location else ''}{item['note']}")
    else:
        lines.append(f"Reused the existing `{audit_verdict.upper()}` audit conclusion for this head SHA.")
    return "\n".join(lines)


def main():
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
    if snapshot["head_sha"] != args.expected_head_sha or snapshot["base_sha"] != args.expected_base_sha:
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
    if current["head"]["sha"] != args.expected_head_sha or current["base"]["sha"] != args.expected_base_sha:
        raise ValueError("PR head or base changed during review; refusing to publish stale output")

    payload = {
        "commit_id": args.expected_head_sha,
        "event": "COMMENT",
        "body": render_body(result, audit_verdict),
        "comments": [
            {"path": item["path"], "line": item["line"], "side": "RIGHT", "body": item["body"]}
            for item in result["review"]["inline_comments"]
        ],
    }
    response = subprocess.run(
        ["gh", "api", "--method", "POST", "--input", "-",
         f"repos/{repository}/pulls/{args.pr}/reviews"],
        input=json.dumps(payload), text=True, check=True, capture_output=True,
    )
    review = json.loads(response.stdout)
    print(f"Published review {review.get('id')} for PR #{args.pr} at {args.expected_head_sha}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"publish failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
