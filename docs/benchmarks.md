# Benchmarks

## CPU / TEE Benchmarks (Mainnet, λ=128, w=49177)

*AMD EPYC 9375F, 1.1TB RAM*

| Environment | vCPUs | Time | Throughput | XOR ops/s | Hint Storage |
|-------------|-------|------|------------|-----------|--------------|
| Bare metal | 64 | 22 min | 55.8 MB/s | 117M/s | 192 MB |
| SEV-SNP TEE | 32 | 57 min | 21.5 MB/s | 45M/s | 192 MB |

SEV-SNP overhead: ~2.6x (with half vCPUs). Normalized for vCPUs: ~1.3x.

Full results:
- https://gist.github.com/igor53627/44f237c4f89fb6dcf20a58d71af0d048
- https://gist.github.com/igor53627/4c21ea3ea9d8963d4d20c9277cc45754

## GPU Benchmarks

See:
- `docs/BENCHMARK_RESULTS.md`
- `docs/gpu_hint_benchmark_2026-01-24_chacha12.md`
