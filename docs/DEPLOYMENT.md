# Deployment

CLI-driven flows (no HTTP API). See [`README.md`](../README.md) and [`docs/README.md`](README.md).

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

The client runs this step on its own machine; the master seed must not leave the client ([`trust_model.md`](trust_model.md)). Reads flat `database.bin` only (not MDBX):

```bash
(umask 077 && openssl rand -hex 32 > ./data/master_seed.hex)

./target/release/plinko_hints \
  --db-path ./data/database.bin \
  --lambda 128 \
  --seed-file ./data/master_seed.hex \
  --output ./data/hints.bin
```

The seed and `hints.bin` are client-secret. Whoever holds both can link each query of the client to its hint. Format: [`data_format.md`](data_format.md#hint-file-hintsbin).

## Constant-time mode (TEE)

Experimental, and not part of the [trust model](trust_model.md#tee-mode): there is no attestation, no seed channel into an enclave, and no hint export. The mode uses one thread unless `--threads` is given ([`constant_time_mode.md`](constant_time_mode.md#worker-threads)). The default build does not contain it; build with the `tee` feature ([`FEATURE_FLAGS.md`](FEATURE_FLAGS.md)).

```bash
cargo build --release -p plinko --bin plinko_hints --features tee

./target/release/plinko_hints \
  --db-path ./data/database.bin \
  --lambda 128 --constant-time \
  --seed-file ./data/master_seed.hex \
  --output ./data/hints.bin
```

## Production notes

- Full mainnet v3 `database.bin` is ~73 GB ([`data_format.md`](data_format.md)); plan disk for extract output, hint files, and (if using GPU) expanded VRAM uploads separately.
- Long-running jobs: `systemd-run` examples in [`AGENTS.md`](../AGENTS.md).
- HintInit for mainnet is about 300 core-hours with hardware AES (extrapolated, see [`hint_generation.md`](hint_generation.md#mainnet-cost)). On arm64, build with `RUSTFLAGS="--cfg aes_armv8"`.
- GPU benchmarks: [`gpu_benchmark_commands.md`](gpu_benchmark_commands.md). The GPU path is a throughput benchmark; it does not write Plinko hints ([`trust_model.md`](trust_model.md#gpu-path)).