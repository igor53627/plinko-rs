# plinko-rs

[![CI](https://github.com/igor53627/plinko-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/igor53627/plinko-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper: ePrint 2024/318](https://img.shields.io/badge/paper-ePrint%202024%2F318-blue.svg)](https://eprint.iacr.org/2024/318)
[![Ask DeepWiki](https://img.shields.io/badge/DeepWiki-ask-blue.svg)](https://deepwiki.com/igor53627/plinko-rs)
[![Demo: protocol tutorial](https://img.shields.io/badge/demo-protocol%20tutorial-blue.svg)](https://igor53627.github.io/plinko-rs/)

Rust implementation of [Plinko](https://eprint.iacr.org/2024/318) single-server PIR for a database of fixed-size records: invertible PRF (iPRF) primitives, hint generation by the client, the online protocol, hint updates, and Rocq models.

The full Ethereum state is not a target: each client must download the database (73.4 GB at mainnet) and run HintInit (about 300 core-hours, extrapolated). The Reth extractor and the mainnet numbers stay as a source of large test data and a scale reference. See [scope and measured costs](docs/protocol_overview.md#scope).

## Repository layout

| Path | Role |
|------|------|
| `src/main.rs` | **Extractor** — reads Reth on-disk DB via `reth-db` (MDBX-backed); writes flat `database.bin` + mappings |
| `plinko/` | **Core crate** — iPRF, PMNS binomial sampling, HintInit (`src/hint_init.rs`, binary `plinko_hints`), 1-level protocol client/server (`src/protocol.rs`), experimental multi-level PIR prototype (`src/multilevel.rs`, binary `bench_multilevel`), `wasm32-unknown-unknown` support (`--no-default-features`), optional CUDA |
| `plinko/formal/` | **Rocq** models of the algorithms and proofs about the models, not about the Rust code (`plinko/formal/README.md`) |
| `docs/` | Protocol, data format, benchmarks, verification |
| `tee-test/` | SEV-SNP TEE benchmark notes (historical) |

**Where to look for protocol / implementation truth** (in order):

1. [`docs/hint_generation.md`](docs/hint_generation.md) — HintInit behavior (Fig. 7), module map  
2. [`docs/data_format.md`](docs/data_format.md) — v3 on-disk layout, last mainnet snapshot metadata
3. [`docs/protocol_overview.md`](docs/protocol_overview.md) — end-to-end pipeline  
4. [`plinko/formal/`](plinko/formal/) — Rocq models and proofs about the models; assumptions in [`plinko/formal/docs/TRUST_BASE.md`](plinko/formal/docs/TRUST_BASE.md)  
5. [`docs/Plinko.v`](docs/Plinko.v) — older Coq sketch (HintInit, GetHint, UpdateHint)  

Parsed paper JSON ([`docs/plinko_paper_part*.json`](docs/plinko_paper_part6_algorithms.json), catalog [`docs/plinko_paper_index.json`](docs/plinko_paper_index.json)) is useful for agents and pseudocode lookup. Datasets: local Reth extract or synthetic — [`docs/data_format.md`](docs/data_format.md).

Contributing: feature branches and PRs — [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Why this exists

- **Speed**: Extractor reads Reth state through `reth-db` (no RPC); hint generation uses flat `database.bin` only (no MDBX).
- **Efficiency**: Streams extraction with bounded memory.
- **Scale**: Flat 40-byte entries (v3 schema) for PIR-friendly layout.
- **Correctness**: HintInit aligned with paper Fig. 7 and [`docs/Plinko.v`](docs/Plinko.v); integer-only PMNS sampler ([`plinko/src/binomial.rs`](plinko/src/binomial.rs)).

## Trust model

The client runs HintInit (`plinko_hints`) and holds the master seed; no other party gets the seed or the hints. The server publishes `database.bin` and answers queries. The constant-time (TEE) mode and the GPU path are experimental and are not part of this model. Cost for the client, and what is not covered yet: [`docs/trust_model.md`](docs/trust_model.md).

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
| `--db-path` | Reth MDBX database folder, for example `/var/lib/reth/mainnet/db` |
| `--output-dir` | Output directory (default: `data`) |
| `--limit` | Optional cap on accounts/slots extracted |

### Generate hints

```bash
./target/release/plinko_hints \
  --db-path ./data/database.bin \
  --lambda 128

# Constant-time path (experimental, see docs/trust_model.md).
# Needs: cargo build --release -p plinko --bin plinko_hints --features tee
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

The CPU section is the HintInit cost of `plinko_hints`, extrapolated from the
measurements in [`hint_generation.md`](docs/hint_generation.md#mainnet-cost): about 300
core-hours for mainnet with hardware AES. `--tee` selects the constant-time path
(about 8,500 core-hours); it adds no overhead of a TEE. The GPU section is the
throughput benchmark, which does not compute Plinko hints.

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

- **iPRF**: Swap-or-Not PRP + PMNS ([`plinko/src/iprf.rs`](plinko/src/iprf.rs)), with 8-way interleaved `forward_many`
- **Binomial / PMNS**: integer-only `Binomial(count, 1/2)` sampler, the bit count of an AES-128 counter keystream per tree node (exact for a uniform keystream) ([`plinko/src/binomial.rs`](plinko/src/binomial.rs)); the iPRF range `w` must be a power of two
- **Multi-level PIR (experimental)**: 2-level and `d`-level waterfall prototype based on [ePrint 2026/1273](https://eprint.iacr.org/2026/1273) with dynamic `UpdateHint`, failure-private `Refresh`, `PLNKMLH1` checkpoints, and zero-import `wasm32` support ([`plinko/src/multilevel.rs`](plinko/src/multilevel.rs), [`docs/multilevel_pir.md`](docs/multilevel_pir.md), `./scripts/test_wasm_multilevel.sh`)
- **TEE** (opt-in feature `tee`): `IprfTee` + constant-time binomial ([`docs/constant_time_mode.md`](docs/constant_time_mode.md)); experimental, not part of the [trust model](docs/trust_model.md#tee-mode)
- **GPU** (opt-in features `gpu`, `cuda`): `cargo build --release -p plinko --features cuda` — [`docs/gpu_benchmark_commands.md`](docs/gpu_benchmark_commands.md); a throughput benchmark, not Plinko hints ([trust model](docs/trust_model.md#gpu-path))

Reference Python spec (small-scale, O(n) binomial): [keewoolee/rms24-plinko-spec](https://github.com/keewoolee/rms24-plinko-spec).

## Protocol costs

Measured on 16,777,216 records of 40 bytes (`w = 4096`, λ=128, Apple M3 Ultra, hardware AES; client and server in one process):

| Step | Cost |
|------|------|
| HintInit (client) | 0.57 ms CPU for each record: 408 s on 24 threads |
| Hint file | 67.1 MB (`4 + r` bytes for a regular hint and `4 + 2r` bytes for a backup hint; `r` is the record size) |
| Query (client) | 53 ms CPU (median), 8.7 KB message; `c/2` iPRF forward calls |
| Update (client) | 0.53 ms CPU for each changed record; one iPRF inverse |

The record size is a parameter (`plinko_hints --record-size`); runs with 32-byte and 256-byte records gave CPU times within 1% of these. Conditions, scaling rules and the mainnet extrapolation: [`docs/protocol_overview.md`](docs/protocol_overview.md#cost). Index of benchmarks, with the historical TEE and GPU documents: [`docs/benchmarks.md`](docs/benchmarks.md).

## Verification

- **Rocq**: `plinko/formal/` — models of the iPRF, Swap-or-Not and the coin sampler, with proofs about the models. The Rust code is not formally verified ([`docs/verification.md`](docs/verification.md))
- **Proptest**: `cd plinko && cargo test --lib`; **Kani**: one harness, see [`docs/verification.md`](docs/verification.md)
- **Regression**: sampler distribution and PMNS bin-load tests in `plinko/src/binomial.rs` and `plinko/src/iprf.rs`

## Documentation

| Topic | Doc |
|-------|-----|
| **Full map** | [`docs/README.md`](docs/README.md) |
| Architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Trust model | [`docs/trust_model.md`](docs/trust_model.md) |
| Integrity (what the client must trust) | [`docs/integrity.md`](docs/integrity.md) |
| Online protocol | [`docs/online_protocol.md`](docs/online_protocol.md) |
| Multi-level PIR prototype (experimental) | [`docs/multilevel_pir.md`](docs/multilevel_pir.md) |
| Hint generation | [`docs/hint_generation.md`](docs/hint_generation.md) |
| Data format / mainnet snapshot | [`docs/data_format.md`](docs/data_format.md) |
| Updates | [`docs/update_strategy.md`](docs/update_strategy.md) |
| Feature flags | [`docs/FEATURE_FLAGS.md`](docs/FEATURE_FLAGS.md) |
| Deployment | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| Benchmarks | [`docs/benchmarks.md`](docs/benchmarks.md) |
| TEE (SEV-SNP), historical | [`tee-test/SEV-SNP-BENCHMARK.md`](tee-test/SEV-SNP-BENCHMARK.md) |
| GPU benchmarks, historical | [`docs/gpu_benchmark_commands.md`](docs/gpu_benchmark_commands.md), [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md) |
| Archived / historical | [`docs/archived/README.md`](docs/archived/README.md) |

### Paper and formal references

| Resource | Use |
|----------|-----|
| [`docs/2024-318.pdf`](docs/2024-318.pdf) | Plinko paper (ePrint 2024/318) |
| [`docs/Plinko.v`](docs/Plinko.v) | Coq sketch (HintInit, hints, updates); not a specification of the Rust code |
| [`docs/plinko_paper_part6_algorithms.json`](docs/plinko_paper_part6_algorithms.json) | Parsed Fig. 7 / algorithm pseudocode |
| [`docs/plinko_paper_part2_technical.json`](docs/plinko_paper_part2_technical.json), [`docs/plinko_paper_part3_scheme.json`](docs/plinko_paper_part3_scheme.json) | iPRF / scheme narrative |
| [`docs/plinko_paper_index.json`](docs/plinko_paper_index.json) | Catalog of parts 1–6 (agent index; verify live data against `data_format.md`) |

## References

- [Plinko: Single-Server PIR with Efficient Updates](https://vitalik.eth.limo/general/2025/11/25/plinko.html) — Vitalik Buterin's overview  
- [Plinko (ePrint 2024/318)](https://eprint.iacr.org/2024/318) — Hoover, Patel, Persiano, Yeo  
- [Multi-level PIR (ePrint 2026/1273)](https://eprint.iacr.org/2026/1273) — basis of the experimental `plinko::multilevel` prototype  
- [Morris–Rogaway 2013](https://eprint.iacr.org/2013/560.pdf) — Swap-or-Not PRP  
- [rms24-plinko-spec](https://github.com/keewoolee/rms24-plinko-spec) — readable Python reference (RMS24 + Plinko)  
- [Alex Hoover — Plinko talk (YouTube)](https://www.youtube.com/watch?v=okJaBn7ZXnc)
