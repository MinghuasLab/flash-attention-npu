#!/usr/bin/env bash
# Upload a release asset idempotently: identical assets are reused, conflicts fail.

set -euo pipefail

[ "$#" -eq 2 ] || { echo "usage: $0 <tag> <asset>" >&2; exit 2; }
tag="$1"
asset="$2"
name="$(basename "$asset")"
[ -f "$asset" ] || { echo "asset not found: $asset" >&2; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "gh not found" >&2; exit 1; }

existing_url="$(gh release view "$tag" --json assets --jq '.assets[] | select(.name == "'"$name"'") | .url' | head -n1)"
if [ -z "$existing_url" ]; then
  gh release upload "$tag" "$asset"
  exit 0
fi

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT
gh release download "$tag" --pattern "$name" --dir "$temp_dir"
current_sha="$(sha256sum "$asset" | awk '{print $1}')"
existing_sha="$(sha256sum "$temp_dir/$name" | awk '{print $1}')"
if [ "$current_sha" = "$existing_sha" ]; then
  echo "release asset already exists with matching SHA256: $name"
  exit 0
fi
echo "release asset $name already exists with a different SHA256" >&2
exit 1
