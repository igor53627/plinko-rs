#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT_DIR/target/wasm-keyword-smoke}"

echo "Building plinko::keyword KV-SCF WASM smoke module (wasm32-unknown-unknown, --no-default-features)..."
CARGO_TARGET_DIR="$TARGET_DIR" cargo build \
  --manifest-path scripts/wasm_keyword_smoke/Cargo.toml \
  --target wasm32-unknown-unknown \
  --release

WASM_PATH="$TARGET_DIR/wasm32-unknown-unknown/release/wasm_keyword_smoke.wasm"
if [[ ! -f "$WASM_PATH" ]]; then
  echo "ERROR: expected wasm artifact at $WASM_PATH" >&2
  exit 1
fi

WASM_BYTES="$(wc -c < "$WASM_PATH" | tr -d ' ')"
echo "Built $WASM_PATH (${WASM_BYTES} bytes)"

node - "$WASM_PATH" <<'EOF'
const fs = require("fs");
const wasmPath = process.argv[2];
const bytes = fs.readFileSync(wasmPath);

(async () => {
  const mod = await WebAssembly.compile(bytes);
  const imports = WebAssembly.Module.imports(mod);
  if (imports.length !== 0) {
    console.error("FAIL: expected 0 WASM host imports, got:", imports);
    process.exit(1);
  }
  const inst = await WebAssembly.instantiate(mod, {});
  const t0 = performance.now();
  const rc = inst.exports.run_keyword_kvscf_smoke();
  const dt = performance.now() - t0;
  if (rc !== 0) {
    console.error(`FAIL: run_keyword_kvscf_smoke returned ${rc}`);
    process.exit(1);
  }
  console.log(
    `PASS: plinko::keyword KV-SCF WASM E2E (0 imports, ${bytes.length} B wasm, ${dt.toFixed(1)} ms)`
  );
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
EOF
