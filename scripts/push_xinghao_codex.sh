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

case "$remote_url" in
    https://github.com/*)
        push_url="$remote_url"
        ;;
    git@github.com:*)
        push_url="https://github.com/${remote_url#git@github.com:}"
        ;;
    ssh://git@github.com/*)
        push_url="https://github.com/${remote_url#ssh://git@github.com/}"
        ;;
    *)
        echo "error: remote '$remote' is not a recognized GitHub URL: $remote_url" >&2
        exit 2
        ;;
esac

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

echo "Pushing '$branch' to '$remote' ($push_url) ..."
if [[ "$push_url" == "$remote_url" ]]; then
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
        git -C "$repo_root" push "$remote" "$branch"
else
    # Override only the push URL for this command. The configured SSH remote is
    # left unchanged, while authentication is performed over HTTPS via askpass.
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
        git -C "$repo_root" \
        -c "remote.${remote}.pushurl=${push_url}" \
        push "$remote" "$branch"
fi
