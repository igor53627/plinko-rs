#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WORK_DIR="${WORK_DIR:-$(mktemp -d /tmp/plinko-e2e.XXXXXX)}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"

if [[ "$KEEP_ARTIFACTS" != "1" ]]; then
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

# Tiny defaults keep CT mode runtime reasonable on local machines.
ACCOUNTS="${ACCOUNTS:-2}"
STORAGE="${STORAGE:-2}"
LAMBDA="${LAMBDA:-1}"
ENTRIES_PER_BLOCK="${ENTRIES_PER_BLOCK:-2}"
MASTER_SEED="${MASTER_SEED:-00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff}"
SYNTH_SEED="${SYNTH_SEED:-11}"

GEN_BIN="./target/debug/gen_synthetic"
HINTS_BIN="./target/debug/plinko_hints"
COST_BIN="./target/debug/cost_estimate"

DATA_DIR="$WORK_DIR/data"
FAST_LOG="$WORK_DIR/hints_fast.log"
CT_LOG="$WORK_DIR/hints_ct.log"
GEN_LOG="$WORK_DIR/gen.log"
COST_JSON="$WORK_DIR/cost_estimate.json"

log() {
  printf '[local-e2e] %s\n' "$*"
}

require_metric() {
  local file="$1"
  local key="$2"
  local value

  value="$(rg -o "${key}=[0-9]+" "$file" | tail -n1 | cut -d= -f2 || true)"
  if [[ -z "$value" ]]; then
    printf 'ERROR: metric %s not found in %s\n' "$key" "$file" >&2
    printf '--- log head (%s) ---\n' "$file" >&2
    sed -n '1,120p' "$file" >&2 || true
    printf '--- log tail (%s) ---\n' "$file" >&2
    tail -n 120 "$file" >&2 || true
    exit 1
  fi
  printf '%s' "$value"
}

compare_metric() {
  local key="$1"
  local fast ct
  fast="$(require_metric "$FAST_LOG" "$key")"
  ct="$(require_metric "$CT_LOG" "$key")"

  if [[ "$fast" != "$ct" ]]; then
    printf 'ERROR: %s mismatch: fast=%s ct=%s\n' "$key" "$fast" "$ct" >&2
    exit 1
  fi
  log "metric OK: ${key}=${fast}"
}

log "work dir: $WORK_DIR"
log "building required binaries"
cargo build --manifest-path plinko/Cargo.toml \
  --bin gen_synthetic \
  --bin plinko_hints \
  --bin cost_estimate >/dev/null

mkdir -p "$DATA_DIR"

log "generating synthetic dataset (accounts=$ACCOUNTS storage=$STORAGE)"
"$GEN_BIN" \
  --output-dir "$DATA_DIR" \
  --accounts "$ACCOUNTS" \
  --storage "$STORAGE" \
  --seed "$SYNTH_SEED" 2>&1 | tee "$GEN_LOG" >/dev/null

DB_PATH="$DATA_DIR/database.bin"
if [[ ! -s "$DB_PATH" ]]; then
  echo "ERROR: expected non-empty database at $DB_PATH" >&2
  exit 1
fi

log "running plinko_hints (fast path)"
RUST_LOG="${RUST_LOG:-info}" RUST_LOG_STYLE=never "$HINTS_BIN" \
  --db-path "$DB_PATH" \
  --lambda "$LAMBDA" \
  --entries-per-block "$ENTRIES_PER_BLOCK" \
  --seed "$MASTER_SEED" 2>&1 | tee "$FAST_LOG" >/dev/null

log "running plinko_hints (constant-time path)"
RUST_LOG="${RUST_LOG:-info}" RUST_LOG_STYLE=never "$HINTS_BIN" \
  --db-path "$DB_PATH" \
  --lambda "$LAMBDA" \
  --entries-per-block "$ENTRIES_PER_BLOCK" \
  --constant-time \
  --seed "$MASTER_SEED" 2>&1 | tee "$CT_LOG" >/dev/null

log "comparing key result metrics between fast and CT runs"
for key in \
  total_regular \
  non_zero_regular \
  zero_regular_seeds \
  total_backup \
  non_zero_parity_in \
  non_zero_parity_out \
  zero_backup_seeds; do
  compare_metric "$key"
done

log "running cost_estimate smoke"
"$COST_BIN" --entries 1000 --json > "$COST_JSON"
python3 -m json.tool "$COST_JSON" >/dev/null

log "local e2e pipeline passed"
log "artifacts: $WORK_DIR"
