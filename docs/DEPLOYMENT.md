# Deployment

CLI-driven flows (no HTTP API). See [`README.md`](../README.md) for the entry point.

## Build

```bash
# Extractor (workspace root)
cargo build --release

# Hint generator and cost estimator
cargo build --release -p plinko --bin plinko_hints --bin cost_estimate
```

## Estimate resource cost (optional)

```bash
cargo run -p plinko --bin cost_estimate -- --mainnet
cargo run -p plinko --bin cost_estimate -- --entries 100000000 --gpus 2 --tee --json
```

## Extract state

Reth node data directory (`reth-db`, MDBX on disk):

```bash
./target/release/plinko \
  --db-path /path/to/reth/db \
  --output-dir ./data
```

## Generate hints

Reads flat `database.bin` only (not MDBX):

```bash
./target/release/plinko_hints \
  --db-path ./data/database.bin \
  --lambda 128
```

## Constant-time mode (TEE)

```bash
./target/release/plinko_hints \
  --db-path ./data/database.bin \
  --lambda 128 --constant-time
```

## Production notes

- Mainnet snapshot needs hundreds of GB free under `--output-dir` / hint output paths.
- Long-running jobs: `systemd-run` examples in [`AGENTS.md`](../AGENTS.md).
- GPU benchmarks: [`gpu_benchmark_commands.md`](gpu_benchmark_commands.md).