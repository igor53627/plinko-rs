# Deployment

This project is primarily CLI-driven. Typical flows are:

## Build

```bash
cargo build --release
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
