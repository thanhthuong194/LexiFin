#!/usr/bin/env bash
# Quality gate: the single entry point for pre-commit, CI, and agent Stop hooks.
#
#   scripts/check.sh changed   files changed vs HEAD, plus untracked files (agent Stop hook)
#   scripts/check.sh staged    staged files (pre-commit)
#   scripts/check.sh all       whole repository, duplication, and tests (CI)
#   scripts/check.sh install   install project dependencies (CI)
#
# This file is the same in every project. Language-specific checks live in
# scripts/checks/<stack>.sh; each profile defines <stack>_install,
# <stack>_changed, and <stack>_all, and this script runs every profile present.
# Every check runs and all failures are reported together.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mode="${1:-changed}"
failed=()

# Run one named check; record a failure without stopping the remaining checks.
step() {
  local name="$1"
  shift
  echo "==> $name"
  if ! "$@"; then
    failed+=("$name")
  fi
}

# Run a command inside a directory relative to the repository root.
in_dir() {
  local dir="$1"
  shift
  (cd "$dir" && "$@")
}

# Print files changed in the current mode that lie under DIR and match REGEX,
# as paths relative to DIR.
changed_files_in() {
  local dir="${1%/}" pattern="$2" prefix="" file
  [[ "$dir" == "." ]] || prefix="$dir/"
  case "$mode" in
    changed) { git diff --name-only --diff-filter=ACMR HEAD; git ls-files --others --exclude-standard; } ;;
    staged) git diff --cached --name-only --diff-filter=ACMR ;;
  esac | sort -u | while IFS= read -r file; do
    if [[ "$file" == "$prefix"* && "$file" =~ $pattern ]]; then
      echo "${file#"$prefix"}"
    fi
  done
}

profiles=()
for profile_file in scripts/checks/*.sh; do
  [[ -e "$profile_file" ]] || continue
  # shellcheck source=/dev/null
  source "$profile_file"
  profiles+=("$(basename "$profile_file" .sh)")
done
if ((${#profiles[@]} == 0)); then
  echo "No profiles found in scripts/checks/. Copy one from the kit's profiles/ folder." >&2
  exit 1
fi

case "$mode" in
  install)
    for profile in "${profiles[@]}"; do "${profile}_install"; done
    ;;
  changed | staged)
    for profile in "${profiles[@]}"; do "${profile}_changed"; done
    ;;
  all)
    for profile in "${profiles[@]}"; do "${profile}_all"; done
    step "duplication" npx --yes jscpd@4
    ;;
  *)
    echo "usage: $0 [changed|staged|all|install]" >&2
    exit 64
    ;;
esac

if ((${#failed[@]} > 0)); then
  echo "FAILED: ${failed[*]}" >&2
  exit 1
fi
echo "All checks passed."
