#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WORK_DIR="${WORK_DIR:-$(mktemp -d /tmp/plinko-protocol-diff.XXXXXX)}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"
REF_REPO_URL="${REF_REPO_URL:-https://github.com/keewoolee/rms24-plinko-spec}"
REF_REPO_DIR="${REF_REPO_DIR:-$WORK_DIR/rms24-plinko-spec}"
PY_VENV="${PY_VENV:-$WORK_DIR/.venv}"
REF_JSON="${REF_JSON:-$WORK_DIR/reference_summary.json}"
RUST_JSON="${RUST_JSON:-$WORK_DIR/rust_summary.json}"

if [[ "$KEEP_ARTIFACTS" != "1" ]]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

log() {
  printf '[protocol-diff] %s\n' "$*"
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

log "running reference scenario (entry_size=32)"
"$PY_VENV/bin/python" scripts/reference_plinko_scenario.py \
  --repo "$REF_REPO_DIR" \
  --entry-size 32 \
  --output "$REF_JSON" >/dev/null

log "running Rust protocol scenario"
cargo run --manifest-path plinko/Cargo.toml --bin protocol_scenario -- \
  --output "$RUST_JSON" >/dev/null

log "comparing reference and Rust summaries"
"$PY_VENV/bin/python" - <<'PY' "$REF_JSON" "$RUST_JSON"
import json
import pathlib
import sys

ref = json.loads(pathlib.Path(sys.argv[1]).read_text())
rust = json.loads(pathlib.Path(sys.argv[2]).read_text())

param_keys = (
    "num_entries",
    "entry_size",
    "block_size",
    "num_blocks",
    "num_reg_hints",
    "num_backup_hints",
)
for key in param_keys:
    if ref["params"][key] != rust["params"][key]:
        raise SystemExit(
            f"parameter mismatch for {key}: ref={ref['params'][key]} rust={rust['params'][key]}"
        )

if ref["updates"] != rust["updates"]:
    raise SystemExit(f"updates mismatch: ref={ref['updates']} rust={rust['updates']}")

if ref["remaining_queries"] != rust["remaining_queries"]:
    raise SystemExit(
        f"remaining_queries mismatch: ref={ref['remaining_queries']} rust={rust['remaining_queries']}"
    )

if len(ref["rounds"]) != len(rust["rounds"]):
    raise SystemExit(
        f"round count mismatch: ref={len(ref['rounds'])} rust={len(rust['rounds'])}"
    )

for round_idx, (rr, rs) in enumerate(zip(ref["rounds"], rust["rounds"]), start=1):
    if rr["indices"] != rs["indices"]:
        raise SystemExit(
            f"round {round_idx} indices mismatch: ref={rr['indices']} rust={rs['indices']}"
        )
    if rr["result_sha256"] != rs["result_sha256"]:
        raise SystemExit(
            f"round {round_idx} result hashes mismatch:\n"
            f"  ref={rr['result_sha256']}\n"
            f"  rust={rs['result_sha256']}"
        )

print("reference and Rust summaries match")
PY

log "protocol transcript differential check passed"
log "artifacts: $WORK_DIR"
