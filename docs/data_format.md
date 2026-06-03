# Data Format

Canonical v3 on-disk layout for Plinko PIR. Implementation: [`plinko/src/schema40.rs`](../plinko/src/schema40.rs) (types), [`src/main.rs`](../src/main.rs) (extractor).

## Artifacts

| File | Format | Produced by extractor |
|------|--------|----------------------|
| `database.bin` | `total_entries` × 40-byte records | yes |
| `account-mapping.bin` | `accounts` × (`Address(20)` \|\| `Index(4)` LE) | yes |
| `storage-mapping.bin` | `storage_slots` × (`Address(20)` \|\| `SlotKey(32)` \|\| `Index(4)` LE) | yes |
| `code_store.bin` | `[count: u32 LE][hash₀: 32B]…` | yes (if any contract code) |
| `metadata.json` | JSON snapshot descriptor | yes |

### `database.bin` ordering

Flat mmap file used by `plinko_hints` ([`Database40`](../plinko/src/db.rs)):

1. Indices `0 .. accounts-1`: account entries (Reth `PlainAccountState` walk order).
2. Indices `accounts .. total_entries-1`: storage entries (Reth storage walk order).

`Index` in mapping files is the u32 little-endian offset into this array (same index for both mapping types).

### Size formulas

Let `A` = accounts, `S` = storage slots, `T = A + S`, `H` = unique bytecode hashes (non-zero code hashes).

| File | Bytes |
|------|------:|
| `database.bin` | `T × 40` |
| `account-mapping.bin` | `A × 24` |
| `storage-mapping.bin` | `S × 56` |
| `code_store.bin` | `4 + H × 32` |

## Entry layout (40 bytes)

Source of truth: [`schema40.rs`](../plinko/src/schema40.rs).

### Account entry

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 16 | Balance | Lower 128 bits of 256-bit balance, LE |
| 16 | 4 | Nonce | u32 LE (extractor errors if `nonce > u32::MAX`) |
| 20 | 4 | CodeID | u32 LE; `0` = EOA; `k ≥ 1` → `code_store` hash at index `k-1` |
| 24 | 8 | TAG | `keccak256(address)[0:8]` |
| 32 | 8 | Padding | Zero |

### Storage entry

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 32 | Value | Full slot word, LE |
| 32 | 8 | TAG | `keccak256(address ‖ slot_key)[0:8]` |

## `metadata.json` (extractor)

Written by [`src/main.rs`](../src/main.rs):

```json
{
  "schema_version": 3,
  "entry_size_bytes": 40,
  "block": 23876768,
  "accounts": 351681953,
  "storage_slots": 1482413924,
  "total_entries": 1834095877,
  "unique_bytecode_hashes": 2064516,
  "generated_at": "2026-01-24 10:47:10"
}
```

Synthetic datasets ([`gen_synthetic`](../plinko/src/bin/gen_synthetic.rs)) add `"synthetic": true`, `"seed"`, `"scale_percent"`, `"unique_bytecodes"`, and `"size_bytes"`.

## Mainnet v3 reference counts

Used by `cost_estimate --mainnet` and benchmarks. Also in [`plinko_paper_index.json`](plinko_paper_index.json) → `mainnet_v3_snapshot`.

| Field | Value |
|-------|------:|
| Block | 23,876,768 |
| Accounts | 351,681,953 |
| Storage slots | 1,482,413,924 |
| Total entries | 1,834,095,877 |
| Unique bytecode hashes | 2,064,516 |
| `database.bin` size | 73,363,835,080 (= `T × 40`) |
| Generated (UTC local string) | 2026-01-24 10:47:10 |

## Obtaining data

For smaller runs:

```bash
cargo build --release
./target/release/plinko --db-path <reth-db> --output-dir ./data --limit <N>
```

Or synthetic scale data:

```bash
cargo build --release -p plinko --bin gen_synthetic
./target/release/gen_synthetic --output-dir ./data/synthetic --scale-percent 0.1
```

In-repo tests: `cargo test -p plinko`.