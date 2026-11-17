# Benchmarks

Full doc map: [`README.md`](README.md).

## Current Code

These numbers are from the code of November 2026. All are from one machine
(Apple M3 Ultra, 24 performance cores, hardware AES with
`RUSTFLAGS="--cfg aes_armv8"`). No number is from a mainnet-size run.

| Workload                                         | Result                                                  | Doc                                                        |
|--------------------------------------------------|---------------------------------------------------------|------------------------------------------------------------|
| HintInit, 16,777,216 records, 24 threads         | 412 s wall, 9,682 s CPU, 0.58 ms for each record        | [`protocol_overview.md`](protocol_overview.md#measured-16777216-records) |
| HintInit, mainnet hint count, 24 blocks          | 0.59 ms for each record                                 | [`hint_generation.md`](hint_generation.md#worker-threads)  |
| HintInit, mainnet v3                             | About 300 core-hours (extrapolated)                     | [`hint_generation.md`](hint_generation.md#mainnet-cost)    |
| Query, client CPU, 16,777,216 records            | 56.0 ms median                                          | [`protocol_overview.md`](protocol_overview.md#measured-16777216-records) |
| Update, client CPU                               | 0.51 to 0.53 ms for each changed record                 | [`update_strategy.md`](update_strategy.md#cost)            |
| Constant-time path (feature `tee`), one thread   | 16.7 ms for each record                                 | [`trust_model.md`](trust_model.md#tee-mode)                |

Synthetic database: `cargo run --release -p plinko --bin gen_synthetic -- --help`.

## Historical

These documents measured code that is not in the repository today, or code
that does not compute Plinko hints. Each has a note at its top with the
details. Do not use their numbers for the current code.

| Document                                                             | Date     | What it measured                                                                                   |
|----------------------------------------------------------------------|----------|----------------------------------------------------------------------------------------------------|
| [`tee-test/SEV-SNP-BENCHMARK.md`](../tee-test/SEV-SNP-BENCHMARK.md)  | Dec 2025 | A generator with regular hints only and no iPRF (revision `0b3fccb` or earlier), in KVM and SEV-SNP |
| [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)                       | Jan 2026 | GPU throughput kernel; the output is not a Plinko hint set (TASK-21)                               |
| [`gpu_benchmark_commands.md`](gpu_benchmark_commands.md)             | Jan 2026 | Commands and reports of the same GPU benchmark                                                     |
| [`archived/gpu_runs/`](archived/gpu_runs/)                           | Jan 2026 | Modal run logs                                                                                     |

The SEV-SNP document reports 19 min 35 s (KVM) and 19 min 57 s (SEV-SNP) on
32 vCPUs for a 73 GB database. The ratio (about 1.8% overhead) is a result
about SEV-SNP for a compute-bound workload. The absolute time is not the time
of the current HintInit, which is about 300 core-hours for the v3 snapshot.

Older write-ups (other vCPU counts and database paths):

- https://gist.github.com/igor53627/44f237c4f89fb6dcf20a58d71af0d048
- https://gist.github.com/igor53627/4c21ea3ea9d8963d4d20c9277cc45754
