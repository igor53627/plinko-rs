# Plinko PIR Hint Generator

Implements Plinko's HintInit matching Fig. 7 of the paper and Plinko.v Coq spec.

Reads v3 `database.bin` (40-byte records). Geometry uses the record count (`file_size / 40`).
The CPU streaming paths XOR the first 32 bytes of each record into hint parity; the CUDA path
expands each record to 48 bytes in VRAM (see `plinko/src/gpu.rs`).

## Key Design (per paper and Coq formalization)

- Generates c iPRF keys (one per block), not one global key
- Regular hints: block subset of size c/2+1, single parity
- Backup hints: block subset of size c/2, dual parities (in/out)
- A block subset is one 32-bit cutoff per hint (see [Block Subsets](#block-subsets))
- iPRF domain = total hints (λw + q), range = w (block size)
- `w` must be a power of two. The default is the smallest power of two >= sqrt(N), the same rule as the protocol client (`plinko::protocol::default_block_size`). Mainnet v3: w = 65536, c = 27988

## Module Structure

| File         | Purpose                                                                        |
|--------------|--------------------------------------------------------------------------------|
| mod.rs       | Module wiring and re-exports                                                   |
| types.rs     | Core data types (`Args`, hints, constants)                                     |
| keys.rs      | iPRF block key derivation                                                      |
| subsets.rs   | Parity XOR helper (block subsets are in the library module `plinko::subsets`)  |
| driver.rs    | Geometry, validation, seed handling, hint initialization, and result summaries |
| fast_path.rs | Standard streaming path (non-constant-time), accepts `FastProcessInput`        |
| ct_path.rs   | Constant-time streaming path for TEE mode, accepts `CtProcessInput`            |

## API Notes

- The streaming processors now take typed input structs instead of long argument lists:
  - `ct_path::process_entries_ct(CtProcessInput, ...)`
  - `fast_path::process_entries_fast(FastProcessInput, ...)`
- `driver::print_results` now takes only the parameters it uses:
  - `print_results(duration, file_len, regular_hints, backup_hints, params)`
- Related CPU benchmark/generator path in `plinko/src/gpu.rs` now uses `CpuHintInput` for `generate_hints`.

## Usage

```bash
# Build (from repo root)
cargo build --release -p plinko --bin plinko_hints

# Generate hints (standard mode)
./target/release/plinko_hints \
  --db-path ./data/database.bin \
  --lambda 128

# Generate hints (constant-time mode for TEE)
./target/release/plinko_hints \
  --db-path ./data/database.bin \
  --lambda 128 --constant-time
```

See [constant_time_mode.md](constant_time_mode.md) for TEE security details.

## Block Subsets

Each hint stores its block subset as one 32-bit cutoff. The scheme is the
median-cutoff representation of RMS24 (Ren, Mughees, Sun, CCS 2024, section 3.2),
with the subset sizes of Plinko Fig. 7. The code is in the library module
`plinko::subsets`; `plinko_hints` and the protocol client both use it.

- `v[j][k] = PRF("select", j, k)` is a 32-bit value for hint `j` and block `k`.
- The cutoff of hint `j` has exactly `c/2 + 1` values of `v[j]` below it for a
  regular hint, and exactly `c/2` for a backup hint.
- Block `k` is in the subset of hint `j` when `v[j][k] < cutoff`.
- Hint IDs are global: regular hints are `0..num_regular`, and backup hint `b`
  is `num_regular + b`. The iPRF uses the same IDs.

The rank order of independent PRF values is a uniform permutation. The subset
is thus uniform among the subsets of that size, as Fig. 7 requires.

PRF: the key is `SHA-256(master_seed || "plinko_select")[0..16]`. One AES-128
block `AES(key, hint_be64 || (block / 4)_be64)` gives four big-endian `u32`
values; block `k` uses word `k % 4`. The key is client-secret.

### Invalid hints

If the two values at the subset boundary are equal, no cutoff selects the exact
number of blocks. The cutoff is then `0` (`INVALID_CUTOFF`), and the hint is
invalid: its parities stay zero, and the client does not use it for a query, a
promotion, or an update. S3PIR (the RMS24 reference code) uses the same rule.
The probability is about `c / 2^33` per hint. For mainnet (c = 27988,
16,777,216 hints) that is about 55 invalid hints (computed, not measured).
`plinko_hints` logs the counts as `invalid_regular` and `invalid_backup`.

### Memory

| Representation                       | Per hint          | Mainnet, 16,777,216 hints |
|--------------------------------------|-------------------|---------------------------|
| Sorted block list (client, removed)  | about `4c` bytes  | about 1.9 TB              |
| Packed bitmap (generator, removed)   | `ceil(c/64) * 8`  | about 59 GB               |
| Cutoff                               | 4 bytes           | 67 MB                     |

The values are computed from c = 27988, not measured.

### Cost

A membership test is one AES-128 block. A cutoff needs `c/4` AES blocks and a
selection (`select_nth_unstable`) over `c` values. The compute bottleneck
remains the Swap-Or-Not rounds in the iPRF.

Measured at c = 27988 on one thread (Apple M3 Ultra, hardware AES through
`--cfg aes_armv8`):

| Operation                         | Time                                   |
|-----------------------------------|----------------------------------------|
| Membership test (`contains`)      | 1.4 ns                                 |
| Cutoff of one hint                | 44 us                                  |
| Cutoffs of all 16,777,216 hints   | about 740 s (extrapolated from 20,000) |
