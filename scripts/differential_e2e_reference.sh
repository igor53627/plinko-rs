#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WORK_DIR="${WORK_DIR:-$(mktemp -d /tmp/plinko-diff-e2e.XXXXXX)}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
REF_REPO_URL="${REF_REPO_URL:-https://github.com/keewoolee/rms24-plinko-spec}"
REF_REPO_DIR="${REF_REPO_DIR:-$WORK_DIR/rms24-plinko-spec}"
PY_VENV="${PY_VENV:-$WORK_DIR/.venv}"
REF_TRACE_JSON="${REF_TRACE_JSON:-$WORK_DIR/reference_trace.json}"

if [[ "$KEEP_ARTIFACTS" != "1" ]]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

log() {
  printf '[diff-e2e] %s\n' "$*"
}

clone_or_update_ref_repo() {
  if [[ -d "$REF_REPO_DIR/.git" ]]; then
    log "using existing reference repo: $REF_REPO_DIR"
  elif [[ -e "$REF_REPO_DIR" ]]; then
    echo "ERROR: REF_REPO_DIR exists but is not a git repo: $REF_REPO_DIR" >&2
    exit 1
  else
    log "cloning reference repo into: $REF_REPO_DIR"
    git clone --depth 1 "$REF_REPO_URL" "$REF_REPO_DIR" >/dev/null
  fi
}

apply_py39_annotations_compat() {
  # Reference repo uses PEP 604 type unions (`T | None`) in annotations.
  # Python 3.9 evaluates these eagerly unless postponed via __future__ import.
  # Patch temp clone only; do not touch the upstream repository.
  "$PY_VENV/bin/python" - <<'PY' "$REF_REPO_DIR"
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])

for path in repo.rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    if "from __future__ import annotations" in text:
        continue

    lines = text.splitlines(keepends=True)
    i = 0

    if lines and lines[0].startswith("#!"):
        i = 1

    # Skip leading blank/comment lines.
    while i < len(lines) and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
        i += 1

    insert_at = i
    if i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                insert_at = i + 1
            else:
                j = i + 1
                while j < len(lines):
                    if quote in lines[j]:
                        insert_at = j + 1
                        break
                    j += 1
                else:
                    insert_at = i + 1

    lines.insert(insert_at, "from __future__ import annotations\n")
    path.write_text("".join(lines), encoding="utf-8")
PY
}

log "work dir: $WORK_DIR"
clone_or_update_ref_repo

log "reference head: $(git -C "$REF_REPO_DIR" rev-parse --short HEAD)"

log "setting up Python venv"
python3 -m venv "$PY_VENV"
"$PY_VENV/bin/python" -m pip install --quiet --upgrade pip
"$PY_VENV/bin/python" -m pip install --quiet -r "$REF_REPO_DIR/requirements.txt"

log "applying Python 3.9 annotations compatibility patch in temp clone"
apply_py39_annotations_compat

log "running deterministic reference scenario"
"$PY_VENV/bin/python" scripts/reference_plinko_scenario.py \
  --repo "$REF_REPO_DIR" \
  --output "$REF_TRACE_JSON" >/dev/null

log "running focused reference pytest checks"
(
  cd "$REF_REPO_DIR"
  "$PY_VENV/bin/python" -m pytest -q \
    tests/test_plinko_pir.py::TestPlinkoPIR::test_basic_query \
    tests/test_plinko_pir.py::TestPlinkoPIR::test_database_update >/dev/null
)

log "running local Rust checks aligned with hint/iPRF semantics"
cargo test --manifest-path plinko/Cargo.toml --test ct_hintinit_test \
  test_ct_hintinit_matches_fast_path_tiny -- --nocapture >/dev/null
cargo test --manifest-path plinko/Cargo.toml --test ct_hintinit_test \
  test_ct_hintinit_matches_fast_path_small -- --nocapture >/dev/null
cargo test --manifest-path plinko/Cargo.toml --test ct_hintinit_test \
  test_iprf_tee_inverse_ct_coverage -- --nocapture >/dev/null
cargo test --manifest-path plinko/Cargo.toml \
  test_iprf_inverse_contains_preimage -- --nocapture >/dev/null

log "reference trace summary:"
"$PY_VENV/bin/python" - <<'PY' "$REF_TRACE_JSON"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
print(f"  blocks={data['params']['num_blocks']}, block_size={data['params']['block_size']}")
print(f"  num_reg_hints={data['params']['num_reg_hints']}, num_backup_hints={data['params']['num_backup_hints']}")
print(f"  rounds={len(data['rounds'])}, updates={data['updates']}, remaining_queries={data['remaining_queries']}")
PY

log "differential e2e baseline checks passed"
log "artifacts: $WORK_DIR"
