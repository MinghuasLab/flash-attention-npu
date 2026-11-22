# Trusted PR Review trial (fork)

This fork trial runs the trusted review code on GitHub-hosted `ubuntu-latest` runners.
It automatically reviews a non-draft PR when opened (or when marked ready), and
does not auto-run on `synchronize`; after pushing a new head, comment exactly
`/review` to request another review.

## Setup

1. Add the repository Actions secret `CURSOR_API_KEY` in the fork settings.
2. Enable GitHub Actions for the fork. The workflow uses the repository
   `GITHUB_TOKEN` with `contents: read` and `pull-requests: write`.
3. Keep the fork's self-hosted NPU workflows disabled during this trial. Their
   files are preserved under `.github/disabled-workflows/` and are not loaded by
   GitHub Actions.

The PR review workflow checks out the trusted workflow revision, snapshots the
PR's current head and base SHA, clones that exact head into a temporary directory,
and posts only if both SHAs are still current. It posts a GitHub review with the
validated summary and inline comments. The security verdict is recorded in that
review against the head and base SHA; a later `/review` on the same snapshot
reuses the verdict while generating a fresh code review.

Cursor runs with a temporary CLI configuration that allows workspace reads and
denies writes, shell, web fetches, and MCP tools. `CURSOR_API_KEY` is present only
in the review step; GitHub tokens are removed from the Cursor process environment.

Commenter permission filtering and NPU CI admission are not part of this trial
stage. Do not attach or enable NPU runners until the SHA-bound admission gate is
implemented and tested.
