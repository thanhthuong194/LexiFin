# Python checks for scripts/check.sh. Copy to scripts/checks/python.sh and
# adjust the settings below; nothing else in this file needs to change.
# Tools: ruff (format, lint), mypy (types), vulture (dead code), pytest (tests).
# Their rules live in pyproject.toml (see profiles/python.pyproject.toml).

# Settings.
PYTHON_ROOT=.                    # folder with pyproject.toml, relative to the repo root (e.g. backend)
PYTHON_SRC_DIRS=(src)            # production code under PYTHON_ROOT: type-checked, scanned for dead code
PYTHON_RUN=(uv run)              # prefix for project tools: (uv run), (poetry run), or ()
PYTHON_INSTALL=(uv sync --frozen)
PYTHON_FAST_TESTS=()             # tests cheap enough for every agent stop, e.g. (uv run pytest -q -x -m "not slow")

python_install() {
  step "python install" in_dir "$PYTHON_ROOT" "${PYTHON_INSTALL[@]}"
}

python_dead_code() {
  local -a whitelist=()
  [[ -f "$PYTHON_ROOT/vulture_whitelist.py" ]] && whitelist=(vulture_whitelist.py)
  step "python dead code" in_dir "$PYTHON_ROOT" "${PYTHON_RUN[@]}" vulture "${PYTHON_SRC_DIRS[@]}" "${whitelist[@]}"
}

python_changed() {
  local -a py_files src_files
  mapfile -t py_files < <(changed_files_in "$PYTHON_ROOT" '\.pyi?$')
  ((${#py_files[@]} > 0)) || return 0
  local src_pattern
  src_pattern="^($(IFS='|'; echo "${PYTHON_SRC_DIRS[*]}"))/"
  mapfile -t src_files < <(printf '%s\n' "${py_files[@]}" | grep -E "$src_pattern" || true)

  step "python format" in_dir "$PYTHON_ROOT" "${PYTHON_RUN[@]}" ruff format --force-exclude "${py_files[@]}"
  step "python lint" in_dir "$PYTHON_ROOT" "${PYTHON_RUN[@]}" ruff check --fix --force-exclude --output-format=concise "${py_files[@]}"
  if ((${#src_files[@]} > 0)); then
    step "python types" in_dir "$PYTHON_ROOT" "${PYTHON_RUN[@]}" mypy --follow-imports=silent "${src_files[@]}"
  fi
  python_dead_code
  if ((${#PYTHON_FAST_TESTS[@]} > 0)); then
    step "python fast tests" in_dir "$PYTHON_ROOT" "${PYTHON_FAST_TESTS[@]}"
  fi
}

python_all() {
  step "python format" in_dir "$PYTHON_ROOT" "${PYTHON_RUN[@]}" ruff format --check .
  step "python lint" in_dir "$PYTHON_ROOT" "${PYTHON_RUN[@]}" ruff check .
  step "python types" in_dir "$PYTHON_ROOT" "${PYTHON_RUN[@]}" mypy "${PYTHON_SRC_DIRS[@]}"
  python_dead_code
  step "python tests" in_dir "$PYTHON_ROOT" "${PYTHON_RUN[@]}" pytest -q
}
