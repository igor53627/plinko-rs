# Plinko RS

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/igor53627/plinko-rs)
[![Interactive Visualization](https://img.shields.io/badge/Demo-Protocol%20Visualization-blue)](https://igor53627.github.io/plinko-rs/protocol-visualization.html)

A Rust workspace for the Plinko PIR stack: Ethereum state extraction, iPRF + hint generation, and formal verification artifacts.

## Why this exists?
- **Speed**: Directly reads the MDBX database using `reth-db`, bypassing RPC overhead.
- **Efficiency**: Streams data to the output files with minimal memory footprint.
- **State Tree Support**: Extracts both Accounts and Storage Slots into a unified flat database format suitable for PIR.

## Usage

```bash
# Build all release binaries
cargo build --release

# Run plinko
./target/release/plinko \
  --db-path /path/to/reth/db \
  --output-dir ./data \
  --limit 1000000 # Optional: Extract subset for testing
```

### Options
- `--db-path`: Path to the Reth database (default: `/var/lib/reth/mainnet/db`).
- `--output-dir`: Output directory (default: `data`).
- `--limit`: (Optional) Limit the number of accounts/slots extracted (for testing).

### Cost Estimator CLI

```bash
cargo run -p plinko --bin cost_estimate -- --mainnet
cargo run -p plinko --bin cost_estimate -- --entries 100000000 --gpus 2 --tee --json
```

## Output Artifacts (Extractor)

The extractor produces five files:

- `database.bin`: Flat array of 40-byte entries (v3 schema)
- `account-mapping.bin`: `Address(20) || Index(4)`
- `storage-mapping.bin`: `Address(20) || SlotKey(32) || Index(4)`
- `code_store.bin`: `[count: u32][hash0: 32B]...`
- `metadata.json`: snapshot metadata and schema version

Full layout details and size breakdowns: `docs/data_format.md`.

## Protocol Overview (Top-5 Features)

1) O(1) hint search via iPRF
2) Optimal query time trade-off O~(n/r)
3) O(1) update time via XOR deltas
4) Ethereum-scale practicality
5) GPU-accelerated hint generation

More detail: `docs/protocol_overview.md`.

## Datasets

- Mainnet v3 snapshot + sizes: `docs/data_format.md`
- Regression data (legacy v2): `docs/data_format.md`

## Docs

- Protocol overview: `docs/protocol_overview.md`
- Data format and datasets: `docs/data_format.md`
- Hint generation: `docs/hint_generation.md`
- Update strategy: `docs/update_strategy.md`
- Benchmarks: `docs/benchmarks.md`
- Verification: `docs/verification.md`

### Additional Documentation

- **Paper**: [docs/2024-318.pdf](docs/2024-318.pdf) - Original academic paper
- **Parsed Paper** (machine-readable JSON):
  - [plinko_paper_index.json](docs/plinko_paper_index.json) - Document structure
  - [plinko_paper_part2_technical.json](docs/plinko_paper_part2_technical.json) - Technical foundations
  - [plinko_paper_part3_scheme.json](docs/plinko_paper_part3_scheme.json) - Plinko scheme details
  - [plinko_paper_part6_algorithms.json](docs/plinko_paper_part6_algorithms.json) - Pseudocode
- **Coq Formalization**: [docs/Plinko.v](docs/Plinko.v) - Mechanized proofs and reference implementation
- **Ops Docs**:
  - [docs/FEATURE_FLAGS.md](docs/FEATURE_FLAGS.md)
  - [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
  - [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
  - [docs/API_ENDPOINTS.md](docs/API_ENDPOINTS.md)

## References

- [Plinko: Single-Server PIR with Efficient Updates](https://vitalik.eth.limo/general/2025/11/25/plinko.html) - Vitalik Buterin's overview
- [Plinko Paper (ePrint 2024/318)](https://eprint.iacr.org/2024/318.pdf) - Academic paper by Mughees, Shi, and Chen
- [Morris-Rogaway 2013](https://eprint.iacr.org/2013/560.pdf) - Swap-or-Not small-domain PRP construction

---

[Alex Hoover: Plinko - Single-Server PIR with Efficient Updates via Invertible PRFs](https://www.youtube.com/watch?v=okJaBn7ZXnc)
