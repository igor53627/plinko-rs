# Feature Flags

This project does not use runtime feature flags. Behavior is controlled via CLI options and build features.

## CLI Options

- Extractor options are documented in `README.md` under "Usage".
- Hint generation options are documented in `docs/hint_generation.md`.

## Build Features

A build of the `plinko` crate without features contains only the code of the
[trust model](trust_model.md): the client runs HintInit. Worker threads of
`plinko_hints` need no feature (`--threads`).

Two opt-in features add experimental code that is not part of the trust model:

| Feature    | Adds                                                                                                  |
|------------|-------------------------------------------------------------------------------------------------------|
| `tee`      | `plinko::constant_time`, `IprfTee`, `SwapOrNotSrTee`, `SwapOrNotTee`, `coin_binomial_ct` (`plinko/src/iprf/tee.rs`), and `plinko_hints --constant-time` |
| `gpu`      | `plinko::gpu` and the `bench_gpu_hints` binary with its CPU baseline (a throughput benchmark, not Plinko hints) |
| `cuda`     | The CUDA kernel of `bench_gpu_hints` (`plinko/cuda/`); implies `gpu`                                  |
| `parallel` | Rayon in the CPU baseline of `bench_gpu_hints`; implies `gpu`                                         |

Build examples:

```bash
# Model only
cargo build --release -p plinko --bin plinko_hints

# Constant-time mode
cargo build --release -p plinko --bin plinko_hints --features tee

# GPU benchmark
cargo build --release -p plinko --bin bench_gpu_hints --features cuda,parallel

# All tests
cargo test --release -p plinko
cargo test --release -p plinko --features tee,gpu
```
