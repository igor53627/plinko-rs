# Plinko PIR Hint Generator

Implements Plinko's HintInit matching Fig. 7 of the paper and Plinko.v Coq spec.

## Key Design (per paper and Coq formalization)

- Generates c iPRF keys (one per block), not one global key
- Regular hints: block subset of size c/2+1, single parity
- Backup hints: block subset of size c/2, dual parities (in/out)
- iPRF domain = total hints (λw + q), range = w (block size)

## Module Structure

| File | Size (bytes) | Purpose |
|------|--------------|---------|
| mod.rs | 790 | Module wiring and re-exports |
| types.rs | 2560 | Core data types (`Args`, hints, constants) |
| bitset.rs | 1822 | `BlockBitset` for constant-time membership checks |
| keys.rs | 3885 | iPRF block key and subset-seed derivation |
| subsets.rs | 2875 | Regular/backup subset block computation |
| driver.rs | 9113 | Geometry, validation, seed handling, hint initialization, and result summaries |
| fast_path.rs | 2644 | Standard streaming path (non-constant-time), accepts `FastProcessInput` |
| ct_path.rs | 8849 | Constant-time streaming path for TEE mode, accepts `CtProcessInput` |

## API Notes

- The streaming processors now take typed input structs instead of long argument lists:
  - `ct_path::process_entries_ct(CtProcessInput, ...)`
  - `fast_path::process_entries_fast(FastProcessInput, ...)`
- `driver::print_results` now takes only the parameters it uses:
  - `print_results(duration, file_len, regular_hints, backup_hints, params)`
- Related CPU benchmark/generator path in `plinko/src/gpu.rs` now uses `CpuHintInput` for `generate_hints`.

## Usage

```bash
# Build
cd plinko && cargo build --release --bin plinko_hints

# Generate hints (standard mode)
./target/release/plinko_hints \
  --db-path ./database.bin \
  --lambda 128

# Generate hints (constant-time mode for TEE)
./target/release/plinko_hints \
  --db-path ./database.bin \
  --lambda 128 --constant-time
```

See [constant_time_mode.md](constant_time_mode.md) for TEE security details.

## Block Set Memory Optimization (#116)

Block subsets (which blocks each hint covers) are stored as packed bitmaps
(`BlockBitset`) instead of sorted `Vec<usize>` arrays. This reduces block
set memory by up to 31x during hint generation:

| c (blocks) | `Vec<usize>` (old) | `BlockBitset` (new) | Reduction |
|------------|-------------------:|--------------------:|----------:|
| 100 | 54 KB | 5 KB | 10.8x |
| 1,000 | 504 KB | 19 KB | 26.5x |
| 10,000 | 5,004 KB | 160 KB | 31.3x |

*128 hints per measurement. Ratio converges to ~32x as c grows.*

At Ethereum Mainnet scale (c ~ 16,404), block set storage drops from
~4 GB to ~130 MB for 33.5M hints. This is a pure memory win; the
compute bottleneck remains the Swap-Or-Not rounds in the iPRF.

### Impact by execution path

| Path | Membership check | Memory impact | Compute impact |
|------|-----------------|---------------|----------------|
| **CPU fast path** | `contains()` — O(1) bit test | 31x less block storage; fits in L2 cache | Slight speedup from cache locality; replaces O(log c) binary search |
| **CPU CT path** | `contains_ct()` — constant-time bit test | Same reduction | No change (already used `BlockBitset`) |
| **GPU path** | Raw byte bitsets in CUDA kernel | No intermediate `Vec<usize>` spike on host | No kernel change (GPU already operates on packed bits) |

### Protocol client (online queries)

The protocol `Client` caches sorted block vectors on `HintSlot` and
`BackupHint` (computed once from seed during `offline_init`). Promoted
backup hints use a complement flag (`complemented: bool`) instead of
materializing the complement set, making `promote_backup` zero-allocation.

Run the memory benchmark: `cargo bench --bench memory_bench`
