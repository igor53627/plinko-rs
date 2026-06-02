# Plinko RS

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/igor53627/plinko-rs)
[![Interactive Visualization](https://img.shields.io/badge/Demo-Protocol%20Visualization-blue)](https://igor53627.github.io/plinko-rs/protocol-visualization.html)

Rust implementation of [Plinko](https://eprint.iacr.org/2024/318) single-server PIR for Ethereum-scale state: Reth extraction, invertible PRF (iPRF) primitives, hint generation (CPU / TEE / optional GPU), and Rocq formal specs.

## Repository layout

| Path | Role |
|------|------|
| `src/main.rs` | **Extractor** — reads Reth MDBX, writes flat `database.bin` + mappings |
| `plinko/` | **Core crate** — iPRF, PMNS binomial sampling, hint generator, optional CUDA |
| `plinko/formal/` | **Rocq** specs and proofs (`plinko/formal/README.md`) |
| `docs/` | Protocol, data format, benchmarks, verification |
| `tee-test/` | SEV-SNP TEE benchmark notes |

Canonical protocol details: [`docs/plinko_paper_index.json`](docs/plinko_paper_index.json).

Contributing: use feature branches and PRs — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Why this exists

- **Speed**: Reads Reth's MDBX via `reth-db` (no RPC).
- **Efficiency**: Streams extraction with bounded memory.
- **Scale**: Flat 40-byte entries (v3 schema) for PIR-friendly layout.
- **Correctness**: HintInit aligned with paper Fig. 7 and [`docs/Plinko.v`](docs/Plinko.v); binomial CDF stable at mainnet PMNS parameters ([#94](https://github.com/igor53627/plinko-rs/issues/94)).

## Quick start

### Build

```bash
# Extractor (workspace root package)
cargo build --release

# Hint generator and cost estimator (`plinko` crate)
cargo build --release -p plinko --bin plinko_hints --bin cost_estimate
```

### Extract state (Reth DB)

```bash
./target/release/plinko \
  --db-path /path/to/reth/db \
  --output-dir ./data \
  --limit 1000000   # optional: subset for testing
```

| Flag | Description |
|------|-------------|
| `--db-path` | Reth database path (default: `/var/lib/reth/mainnet/db`) |
| `--output-dir` | Output directory (default: `data`) |
| `--limit` | Optional cap on accounts/slots extracted |

### Generate hints

```bash
./target/release/plinko_hints \
  --db-path ./data/database.bin \
  --lambda 128

# Constant-time path (TEE)
./target/release/plinko_hints \
  --db-path ./data/database.bin \
  --lambda 128 --constant-time
```

See [`docs/hint_generation.md`](docs/hint_generation.md) and [`docs/constant_time_mode.md`](docs/constant_time_mode.md).

### Cost estimate

```bash
cargo run -p plinko --bin cost_estimate -- --mainnet
cargo run -p plinko --bin cost_estimate -- --entries 100000000 --gpus 2 --tee --json
```

## Extractor output

| File | Contents |
|------|----------|
| `database.bin` | Flat 40-byte entries (v3 schema) |
| `account-mapping.bin` | `Address(20) \|\| Index(4)` |
| `storage-mapping.bin` | `Address(20) \|\| SlotKey(32) \|\| Index(4)` |
| `code_store.bin` | `[count: u32][hash0: 32B]...` |
| `metadata.json` | Snapshot metadata and schema version |

Layout and dataset sizes: [`docs/data_format.md`](docs/data_format.md).

## Core primitives (`plinko` crate)

- **iPRF**: Swap-or-Not PRP + PMNS ([`plinko/src/iprf.rs`](plinko/src/iprf.rs))
- **Binomial / PMNS**: Derandomized `Binomial(n,p; r)` via inverse CDF; large `n` uses a continued-fraction regularized incomplete beta (mainnet-safe; `puruspe::betai` approx path is not used when both shape parameters exceed 3000)
- **TEE**: `IprfTee` + constant-time binomial ([`docs/constant_time_mode.md`](docs/constant_time_mode.md))
- **GPU** (optional): `cargo build --release --features cuda` — see [`docs/gpu_benchmark_commands.md`](docs/gpu_benchmark_commands.md)

Reference Python spec (small-scale, O(n) binomial): [keewoolee/rms24-plinko-spec](https://github.com/keewoolee/rms24-plinko-spec).

## Protocol highlights

1. O(1) hint search via iPRF preimage structure  
2. Query cost trade-off O~(n/r)  
3. O(1) updates via XOR deltas  
4. Ethereum mainnet-scale parameters (λ=128, w≈√N)  
5. Optional GPU hint throughput  

Details: [`docs/protocol_overview.md`](docs/protocol_overview.md). Benchmarks: [`docs/benchmarks.md`](docs/benchmarks.md).

## Verification

- **Rocq**: `plinko/formal/` — specs for iPRF, Swap-or-Not, binomial ([`docs/verification.md`](docs/verification.md))
- **Kani / proptest**: `cd plinko && cargo test --lib`
- **Regression**: mainnet-scale binomial / PMNS tests in `plinko/src/binomial.rs`

## Documentation

| Topic | Doc |
|-------|-----|
| Architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Hint generation | [`docs/hint_generation.md`](docs/hint_generation.md) |
| Data format | [`docs/data_format.md`](docs/data_format.md) |
| Updates | [`docs/update_strategy.md`](docs/update_strategy.md) |
| Feature flags | [`docs/FEATURE_FLAGS.md`](docs/FEATURE_FLAGS.md) |
| Deployment | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |

**Paper**

- PDF: [`docs/2024-318.pdf`](docs/2024-318.pdf)
- Parsed JSON: [`docs/plinko_paper_index.json`](docs/plinko_paper_index.json), parts 2–3, 6 in `docs/plinko_paper_part*.json`
- Coq reference: [`docs/Plinko.v`](docs/Plinko.v)

## References

- [Plinko: Single-Server PIR with Efficient Updates](https://vitalik.eth.limo/general/2025/11/25/plinko.html) — Vitalik Buterin's overview  
- [Plinko (ePrint 2024/318)](https://eprint.iacr.org/2024/318) — Mughees, Shi, Chen  
- [Morris–Rogaway 2013](https://eprint.iacr.org/2013/560.pdf) — Swap-or-Not PRP  
- [rms24-plinko-spec](https://github.com/keewoolee/rms24-plinko-spec) — readable Python reference (RMS24 + Plinko)

---

[Alex Hoover: Plinko - Single-Server PIR with Efficient Updates via Invertible PRFs](https://www.youtube.com/watch?v=okJaBn7ZXnc)