#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: ./publish.sh \"commit message\""
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: this script must run inside a Git repository."
  exit 1
fi

BRANCH="$(git symbolic-ref --quiet --short HEAD)" || {
  echo "Error: detached HEAD; check out a branch before publishing."
  exit 1
}

git add -A

if git diff --cached --quiet; then
  echo "No changes to publish."
  exit 0
fi

git commit -m "$*"
git push origin "$BRANCH"

echo "Published branch '$BRANCH'."
