#!/usr/bin/env bash

set -euo pipefail

remote="${1:-origin}"
branch="${2:-xinghao-codex}"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "error: GITHUB_TOKEN is not set" >&2
    echo "usage: GITHUB_TOKEN=<token> $0 [remote] [branch]" >&2
    exit 2
fi

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
remote_url="$(git -C "$repo_root" remote get-url "$remote")"

if [[ "$remote_url" != https://github.com/* ]]; then
    echo "error: remote '$remote' is not an HTTPS GitHub URL: $remote_url" >&2
    exit 2
fi

askpass_dir="$(mktemp -d /tmp/loopaware-git-askpass.XXXXXX)"
askpass="$askpass_dir/askpass.sh"

cleanup() {
    rm -f -- "$askpass"
    rmdir -- "$askpass_dir"
}
trap cleanup EXIT HUP INT TERM

cat >"$askpass" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
    *Username*) printf '%s\n' "${GITHUB_USERNAME:-x-access-token}" ;;
    *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
    *) exit 1 ;;
esac
ASKPASS
chmod 700 "$askpass"

echo "Pushing '$branch' to '$remote' ($remote_url) ..."
GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
    git -C "$repo_root" push "$remote" "$branch"
