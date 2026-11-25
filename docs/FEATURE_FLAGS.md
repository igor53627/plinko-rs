# Feature Flags

This project does not use runtime feature flags. Behavior is controlled via CLI options and build features.

## CLI Options

- Extractor options are documented in `README.md` under "Usage".
- Hint generation options are documented in `docs/hint_generation.md`.

## Build Features

Default features (`cli`, `logging`) build the `plinko` library and its CLI
binaries (`plinko_hints`, `cost_estimate`, `gen_synthetic`,
`protocol_scenario`, `bench_multilevel`). They contain only the code of the
[trust model](trust_model.md): the client runs HintInit. Worker threads of
`plinko_hints` (`--threads`) and of `protocol::compute_hints` need no feature.

Disable default features (`default-features = false`) for a library-only build
with no CLI or memory-map dependencies, including `wasm32-unknown-unknown`
(where worker counts clamp to 1 so no thread is spawned).

| Feature    | Default | Adds                                                                                                                                                                                                                                                                                                       |
|------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cli`      | yes     | CLI binaries (`plinko_hints`, `cost_estimate`, `gen_synthetic`, `protocol_scenario`, `bench_multilevel`), `plinko::db::Database40` (memory-mapped file helper), and `clap`/`eyre`/`indicatif`/`memmap2`/`serde` dependencies |
| `logging`  | yes     | `tracing` and `tracing-subscriber` (required by `plinko_hints` and `gen_synthetic`)                                                                                                                                                                                                                        |
| `tee`      | no      | Experimental, outside the trust model: `plinko::constant_time`, `IprfTee`, `SwapOrNotSrTee`, `SwapOrNotTee`, `coin_binomial_ct` (`plinko/src/iprf/tee.rs`), `subsets::find_cutoff_ct`, `SubsetPrf::cutoff_ct`, `hint_init::hint_cutoffs_ct` and `hint_init::fill_parities_ct` (`plinko/src/hint_init/tee.rs`), and `plinko_hints --constant-time` |
| `gpu`      | no      | Experimental, outside the trust model: `plinko::gpu` and the `bench_gpu_hints` binary (with `cli`) with its CPU baseline (a throughput benchmark, not Plinko hints)                                                                                                                                        |
| `cuda`     | no      | The CUDA kernel of `bench_gpu_hints` (`plinko/cuda/`); implies `gpu`                                                                                                                                                                                                                                       |
| `parallel` | no      | Rayon in the CPU baseline of `bench_gpu_hints`; implies `gpu`                                                                                                                                                                                                                                              |

Build examples:

```bash
# Model only (default: cli + logging)
cargo build --release -p plinko --bin plinko_hints

# Library only, no CLI dependencies
cargo check -p plinko --lib --no-default-features

# WebAssembly library (wasm32-unknown-unknown)
rustup target add wasm32-unknown-unknown
cargo build --release -p plinko --lib --no-default-features --target wasm32-unknown-unknown

# Constant-time mode
cargo build --release -p plinko --bin plinko_hints --features tee

# GPU benchmark
cargo build --release -p plinko --bin bench_gpu_hints --features cuda,parallel

# All tests
cargo test --release -p plinko
cargo test --release -p plinko --no-default-features
cargo test --release -p plinko --features tee,gpu
```
