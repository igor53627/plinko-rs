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
| types.rs | 2608 | Core data types (`Args`, hints, constants) |
| bitset.rs | 1822 | `BlockBitset` for constant-time membership checks |
| keys.rs | 3885 | iPRF block key and subset-seed derivation |
| subsets.rs | 2875 | Regular/backup subset block computation |
| driver.rs | 8980 | Geometry, validation, seed handling, and hint initialization |
| fast_path.rs | 2452 | Standard streaming path (non-constant-time) |
| ct_path.rs | 8638 | Constant-time streaming path for TEE mode |

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
