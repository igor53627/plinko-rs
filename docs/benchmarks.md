# Benchmarks

Canonical sources (prefer these over older gists or GPU run notes):

| Workload | Doc |
|----------|-----|
| CPU / SEV-SNP TEE HintInit | [`tee-test/SEV-SNP-BENCHMARK.md`](../tee-test/SEV-SNP-BENCHMARK.md) |
| GPU hint generation | [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md), [`gpu_benchmark_commands.md`](gpu_benchmark_commands.md) |
| Historical GPU runs | [`gpu_hint_benchmark_2026-01-23.md`](gpu_hint_benchmark_2026-01-23.md) (superseded by later runs in `gpu_hint_benchmark_2026-01-24*.md`) |

## CPU / TEE (mainnet-scale, λ=128)

From [`tee-test/SEV-SNP-BENCHMARK.md`](../tee-test/SEV-SNP-BENCHMARK.md) (AMD EPYC 9375F, ~2.4B entries / 73 GB DB):

| Environment | vCPUs | Wall time | Throughput |
|-------------|-------|-----------|------------|
| KVM (no SEV) | 32 | 19m 35s | 62.90 MB/s |
| SEV-SNP TEE | 32 | 19m 57s | 61.80 MB/s |

SEV-SNP overhead vs KVM on the same VM shape: **~1.8%** wall clock (not the older ~2.6× figures from an earlier benchmark configuration).

Older write-ups (may use different vCPU counts or DB paths):

- https://gist.github.com/igor53627/44f237c4f89fb6dcf20a58d71af0d048
- https://gist.github.com/igor53627/4c21ea3ea9d8963d4d20c9277cc45754

## GPU

See [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) and [`gpu_benchmark_commands.md`](gpu_benchmark_commands.md).

Synthetic DB for GPU dev: `cargo run -p plinko --bin gen_synthetic -- --help`.