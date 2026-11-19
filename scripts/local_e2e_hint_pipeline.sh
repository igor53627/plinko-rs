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
SEED_FILE="$WORK_DIR/master_seed.hex"
FAST_HINTS="$WORK_DIR/hints_fast.bin"
CT_HINTS="$WORK_DIR/hints_ct.bin"
COST_JSON="$WORK_DIR/cost_estimate.json"

log() {
  printf '[local-e2e] %s\n' "$*"
}

require_metric() {
  local file="$1"
  local key="$2"
  local value
  local esc

  esc="$(printf '\033')"
  value="$(
    sed -E "s/${esc}\\[[0-9;]*[[:alpha:]]//g" "$file" \
      | grep -Eo "${key}=[0-9]+" \
      | tail -n1 \
      | cut -d= -f2 || true
  )"
  if [[ -z "$value" ]]; then
    printf 'ERROR: metric %s not found in %s\n' "$key" "$file" >&2
    printf '%s\n' "--- log head (${file}) ---" >&2
    sed -n '1,120p' "$file" >&2 || true
    printf '%s\n' "--- log tail (${file}) ---" >&2
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
# The constant-time path is behind the experimental `tee` feature.
log "building required binaries (features: tee)"
cargo build --manifest-path plinko/Cargo.toml \
  --features tee \
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

# A file from an earlier run keeps its mode, so the seed goes to a new file.
rm -f "$SEED_FILE"
(umask 077 && set -o noclobber && printf '%s\n' "$MASTER_SEED" > "$SEED_FILE")

log "running plinko_hints (fast path)"
RUST_LOG="${RUST_LOG:-info}" RUST_LOG_STYLE=never "$HINTS_BIN" \
  --db-path "$DB_PATH" \
  --lambda "$LAMBDA" \
  --entries-per-block "$ENTRIES_PER_BLOCK" \
  --seed-file "$SEED_FILE" \
  --output "$FAST_HINTS" 2>&1 | tee "$FAST_LOG" >/dev/null

log "running plinko_hints (constant-time path)"
RUST_LOG="${RUST_LOG:-info}" RUST_LOG_STYLE=never "$HINTS_BIN" \
  --db-path "$DB_PATH" \
  --lambda "$LAMBDA" \
  --entries-per-block "$ENTRIES_PER_BLOCK" \
  --constant-time \
  --seed-file "$SEED_FILE" \
  --output "$CT_HINTS" 2>&1 | tee "$CT_LOG" >/dev/null

log "comparing key result metrics between fast and CT runs"
for key in \
  total_regular \
  non_zero_regular \
  invalid_regular \
  total_backup \
  non_zero_parity_in \
  non_zero_parity_out \
  invalid_backup; do
  compare_metric "$key"
done

log "comparing hint files of the fast and CT runs"
if [[ ! -s "$FAST_HINTS" || ! -s "$CT_HINTS" ]]; then
  echo "ERROR: a hint file is missing or empty" >&2
  exit 1
fi
if ! cmp -s "$FAST_HINTS" "$CT_HINTS"; then
  echo "ERROR: fast and constant-time hint files differ" >&2
  cmp "$FAST_HINTS" "$CT_HINTS" >&2 || true
  exit 1
fi
log "hint files are byte-identical ($(wc -c < "$FAST_HINTS" | tr -d ' ') bytes)"

log "running cost_estimate smoke"
"$COST_BIN" --entries 1000 --json > "$COST_JSON"
python3 -m json.tool "$COST_JSON" >/dev/null

log "local e2e pipeline passed"
log "artifacts: $WORK_DIR"
