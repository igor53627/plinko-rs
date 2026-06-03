# Plinko RS

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/igor53627/plinko-rs)
[![Interactive Visualization](https://img.shields.io/badge/Demo-Protocol%20Visualization-blue)](https://igor53627.github.io/plinko-rs/protocol-visualization.html)

Rust implementation of [Plinko](https://eprint.iacr.org/2024/318) single-server PIR for Ethereum-scale state: Reth extraction, invertible PRF (iPRF) primitives, hint generation (CPU / TEE / optional GPU), and Rocq formal specs.

## Repository layout

| Path | Role |
|------|------|
| `src/main.rs` | **Extractor** — reads Reth on-disk DB via `reth-db` (MDBX-backed); writes flat `database.bin` + mappings |
| `plinko/` | **Core crate** — iPRF, PMNS binomial sampling, hint generator (`src/bin/hint_gen/`), optional CUDA |
| `plinko/formal/` | **Rocq** specs and proofs (`plinko/formal/README.md`) |
| `docs/` | Protocol, data format, benchmarks, verification |
| `tee-test/` | SEV-SNP TEE benchmark notes |

**Where to look for protocol / implementation truth** (in order):

1. [`docs/hint_generation.md`](docs/hint_generation.md) — HintInit behavior (Fig. 7), module map  
2. [`docs/data_format.md`](docs/data_format.md) — v3 on-disk layout, last mainnet snapshot metadata
3. [`docs/protocol_overview.md`](docs/protocol_overview.md) — end-to-end pipeline  
4. [`docs/Plinko.v`](docs/Plinko.v) — Coq reference spec  
5. [`plinko/formal/`](plinko/formal/) — Rocq proofs and machine-checked specs  

Parsed paper JSON ([`docs/plinko_paper_part*.json`](docs/plinko_paper_part6_algorithms.json), catalog [`docs/plinko_paper_index.json`](docs/plinko_paper_index.json)) is useful for agents and pseudocode lookup. Public HTTP (`pir.53627.org`) and R2 regression buckets are **not** available; see [`docs/data_format.md`](docs/data_format.md) for layout and local extract.

Contributing: feature branches and PRs — [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Why this exists

- **Speed**: Extractor reads Reth state through `reth-db` (no RPC); hint generation uses flat `database.bin` only (no MDBX).
- **Efficiency**: Streams extraction with bounded memory.
- **Scale**: Flat 40-byte entries (v3 schema) for PIR-friendly layout.
- **Correctness**: HintInit aligned with paper Fig. 7 and [`docs/Plinko.v`](docs/Plinko.v); binomial inverse CDF stable at mainnet PMNS parameters ([`plinko/src/binomial.rs`](plinko/src/binomial.rs)).

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
| `--db-path` | Reth node data directory (`reth-db` / MDBX-backed; default: `/var/lib/reth/mainnet/db`) |
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

Layout and last published mainnet v3 counts: [`docs/data_format.md`](docs/data_format.md).

## Core primitives (`plinko` crate)

- **iPRF**: Swap-or-Not PRP + PMNS ([`plinko/src/iprf.rs`](plinko/src/iprf.rs))
- **Binomial / PMNS**: Derandomized `Binomial(n,p; r)` via inverse CDF; large `n` uses a continued-fraction regularized incomplete beta (mainnet-safe; `puruspe::betai` approx path is not used when both shape parameters exceed 3000)
- **TEE**: `IprfTee` + constant-time binomial ([`docs/constant_time_mode.md`](docs/constant_time_mode.md))
- **GPU** (optional): `cargo build --release -p plinko --features cuda` — [`docs/gpu_benchmark_commands.md`](docs/gpu_benchmark_commands.md)

Reference Python spec (small-scale, O(n) binomial): [keewoolee/rms24-plinko-spec](https://github.com/keewoolee/rms24-plinko-spec).

## Protocol highlights

1. O(1) hint search via iPRF preimage structure  
2. Query cost trade-off O~(n/r)  
3. O(1) updates via XOR deltas  
4. Ethereum mainnet-scale parameters (λ=128, w≈√N)  
5. Optional GPU hint throughput  

Details: [`docs/protocol_overview.md`](docs/protocol_overview.md). Benchmarks: [`docs/benchmarks.md`](docs/benchmarks.md) (index to TEE + GPU docs).

## Verification

- **Rocq**: `plinko/formal/` — specs for iPRF, Swap-or-Not, binomial ([`docs/verification.md`](docs/verification.md))
- **Kani / proptest**: `cd plinko && cargo test --lib`
- **Regression**: mainnet-scale binomial / PMNS tests in `plinko/src/binomial.rs`

## Documentation

| Topic | Doc |
|-------|-----|
| **Full map** | [`docs/README.md`](docs/README.md) |
| Architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Hint generation | [`docs/hint_generation.md`](docs/hint_generation.md) |
| Data format / mainnet snapshot | [`docs/data_format.md`](docs/data_format.md) |
| Updates | [`docs/update_strategy.md`](docs/update_strategy.md) |
| Feature flags | [`docs/FEATURE_FLAGS.md`](docs/FEATURE_FLAGS.md) |
| Deployment | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| TEE (SEV-SNP) | [`tee-test/SEV-SNP-BENCHMARK.md`](tee-test/SEV-SNP-BENCHMARK.md) |
| GPU benchmarks | [`docs/gpu_benchmark_commands.md`](docs/gpu_benchmark_commands.md), [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md) |
| Archived / historical | [`docs/archived/README.md`](docs/archived/README.md) |

### Paper and formal references

| Resource | Use |
|----------|-----|
| [`docs/2024-318.pdf`](docs/2024-318.pdf) | Plinko paper (ePrint 2024/318) |
| [`docs/Plinko.v`](docs/Plinko.v) | Coq specification (HintInit, hints, updates) |
| [`docs/plinko_paper_part6_algorithms.json`](docs/plinko_paper_part6_algorithms.json) | Parsed Fig. 7 / algorithm pseudocode |
| [`docs/plinko_paper_part2_technical.json`](docs/plinko_paper_part2_technical.json), [`docs/plinko_paper_part3_scheme.json`](docs/plinko_paper_part3_scheme.json) | iPRF / scheme narrative |
| [`docs/plinko_paper_index.json`](docs/plinko_paper_index.json) | Catalog of parts 1–6 (agent index; verify live data against `data_format.md`) |

## References

- [Plinko: Single-Server PIR with Efficient Updates](https://vitalik.eth.limo/general/2025/11/25/plinko.html) — Vitalik Buterin's overview  
- [Plinko (ePrint 2024/318)](https://eprint.iacr.org/2024/318) — Mughees, Shi, Chen  
- [Morris–Rogaway 2013](https://eprint.iacr.org/2013/560.pdf) — Swap-or-Not PRP  
- [rms24-plinko-spec](https://github.com/keewoolee/rms24-plinko-spec) — readable Python reference (RMS24 + Plinko)  
- [Alex Hoover — Plinko talk (YouTube)](https://www.youtube.com/watch?v=okJaBn7ZXnc)