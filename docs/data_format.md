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
| `hints.bin` | Hint set of one client (see [Hint file](#hint-file-hintsbin)) | no (`plinko_hints --output`) |

### `database.bin` ordering

Flat mmap file used by `plinko_hints` (see [`hint_generation.md`](hint_generation.md)). The hint generator and the protocol work with any array of fixed-size records (`--record-size`); the v3 layout below is the output of the extractor and of `gen_synthetic`, with 40-byte records:

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
| 32 | 8 | TAG | `keccak256(address || slot_key)[0:8]` |

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

## Hint file (`hints.bin`)

Written by `plinko_hints --output` and read by the protocol client
(`plinko::hints::HintSet`). All integers are little-endian. Version 2, where
`r` is the record size of the database in bytes (`plinko_hints --record-size`,
default 40):

| Offset   | Size          | Field                                                  |
|----------|---------------|--------------------------------------------------------|
| 0        | 8             | magic `PLNKHINT`                                       |
| 8        | 4             | version (`2`)                                          |
| 12       | 4             | `record_size` (`r`), 1 to 1,048,576                    |
| 16       | 8             | `num_entries`: real records of the database            |
| 24       | 8             | `block_size` (`w`)                                     |
| 32       | 8             | `num_blocks` (`c`), padding included                   |
| 40       | 8             | `num_regular`                                          |
| 48       | 8             | `num_backup`                                           |
| 56       | 8             | seed check: `SHA-256(master_seed \|\| "plinko_seed_check")[0..8]` |
| 64       | `4 + r` each  | regular hints: cutoff (`u32`), parity (`r` bytes)      |
| ...      | `4 + 2r` each | backup hints: cutoff (`u32`), parity in, parity out    |
| end - 32 | 32            | BLAKE3 hash of all bytes before it                     |

- Regular hint `j` has hint ID `j`; backup hint `b` has hint ID `num_regular + b`.
- A cutoff defines the block subset of a hint ([`hint_generation.md`](hint_generation.md#block-subsets)). Cutoff `0` marks an invalid hint with zero parities.
- A parity is the XOR of the selected records, over all `r` bytes of a record.
- Size: `96 + num_regular × (4 + r) + num_backup × (4 + 2r)` bytes. Mainnet v3 (`r` = 40) with lambda = 128 and w = 65536 (8,388,608 regular and 8,388,608 backup hints): 1,073,741,920 bytes (computed).
- The reader rejects a file of another version and a record size outside the range. `Client::from_hints` rejects a hint set whose record size differs from the parameters of the client.
- Version 1 had 32-byte parities over the first 32 bytes of a 40-byte record. No code reads it; run `plinko_hints` again.
- The file holds no master seed. The file and the seed are client-secret.

## Limits of the extractor

The extractor (`src/main.rs`) is a source of test data. The full Ethereum
state is not a target ([protocol_overview.md](protocol_overview.md#scope)),
and these defects are known and not planned (TASK-25, closed without
implementation):

- **No single snapshot.** Each batch (`--batch-size`, 100,000 items by default)
  uses a new read transaction. On a node that runs, the output mixes the state
  of several blocks.
- **Block number.** `"block"` is the last key of the `CanonicalHeaders` table,
  read once before the first batch. It is not the best block of the Reth
  provider, and an empty table gives `0` with no error.
- **No link to a state root.** `metadata.json` has no block hash, no state root
  and no hash of `database.bin`.
- **Lossy records.** The balance is the low 128 bits, with no error for a
  larger value. A nonce above `u32::MAX` stops the run.
- **CodeIDs** follow the first appearance in address order, so they change
  between two extractions.
- **Failed run.** The extractor does not remove partial output files.

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