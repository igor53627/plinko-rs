# Feature Flags

This project does not use runtime feature flags. Behavior is controlled via CLI options and build features.

## CLI Options

- Extractor options are documented in `README.md` under "Usage".
- Hint generation options are documented in `docs/hint_generation.md`.

## Build Features

- `cuda`: Enable GPU hint generation (see `plinko/cuda/`).
- `parallel`: Enable parallel CPU execution paths where available.

Build example:

```bash
cargo build --release -p plinko --bin bench_gpu_hints --features cuda,parallel
```
