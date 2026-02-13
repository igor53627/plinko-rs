# Deployment

This project is primarily CLI-driven. Typical flows are:

## Build

```bash
cargo build --release
```

## Estimate Resource Cost (Optional Preflight)

```bash
cargo build --release --manifest-path plinko/Cargo.toml --bin cost_estimate
./target/release/cost_estimate --mainnet
./target/release/cost_estimate --entries 100000000 --gpus 2 --tee --json
```

## Extract State

```bash
./target/release/plinko \
  --db-path /path/to/reth/db \
  --output-dir ./data
```

## Generate Hints

```bash
cd plinko && cargo build --release --bin plinko_hints
./target/release/plinko_hints \
  --db-path ../data/database.bin \
  --lambda 128
```

## Constant-Time Mode (TEE)

```bash
./target/release/plinko_hints \
  --db-path ../data/database.bin \
  --lambda 128 --constant-time
```

## Production Notes

- Ensure the output directory has sufficient disk space (hundreds of GB for mainnet).
- For long-running jobs, consider systemd-run for logging (see AGENTS.md for examples).
