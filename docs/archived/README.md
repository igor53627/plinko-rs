# Archived Documentation

> [!WARNING]
> Files here are **obsolete** or **historical**. Do not use as implementation source of truth.

| File / path | Notes | Prefer instead |
|-------------|-------|----------------|
| `data-oblivious-iprf-tee.md` | Pre-CT design | [`constant_time_mode.md`](../constant_time_mode.md) |
| `FLAKY_TESTS.md` | 2025 incident note | N/A |
| `findings_2025-11-23.md` | Early perf notes | [`tee-test/SEV-SNP-BENCHMARK.md`](../../tee-test/SEV-SNP-BENCHMARK.md) |
| `Plinko PIR tutorial.md` | Draft with embedded images | Paper PDF + [`protocol_overview.md`](../protocol_overview.md) |
| `xof-optimization.md` | XOF experiment notes | [`hint_generation.md`](../hint_generation.md) |
| `KANBAN_OPTIMIZATION.md` | Old GPU kanban (pre-v3) | [`BENCHMARK_RESULTS.md`](../BENCHMARK_RESULTS.md) |
| `perf_kanban.md` | Duplicate kanban | same |
| `2026-01-29-readme-split-design.md` | Completed README split plan | [`README.md`](../../README.md) |
| `API_ENDPOINTS.md` | No HTTP API | [`README.md`](../../README.md), [`DEPLOYMENT.md`](../DEPLOYMENT.md) |
| `compression_benchmark.md` | One-off compression study | [`data_format.md`](../data_format.md) |
| `CUDA_OPTIMIZATION.md` | 40B GPU compaction notes | [`gpu_benchmark_commands.md`](../gpu_benchmark_commands.md) |
| `optimization/EXPANSION_40B_48B.md` | VRAM 40→48B expansion design | v3 40B on disk; see CUDA kernel in `plinko/cuda/` |
| `optimization/sort_stream_calc.md` | Theoretical sort-and-stream model | [`BENCHMARK_RESULTS.md`](../BENCHMARK_RESULTS.md) |
| `gpu_runs/*.md` | Modal GPU run logs (Jan 2026) | [`BENCHMARK_RESULTS.md`](../BENCHMARK_RESULTS.md), [`gpu_benchmark_commands.md`](../gpu_benchmark_commands.md) |
| `kani-verification-results.md` | Kani run snapshot (2025-12-06) | `cd plinko && cargo kani` per [`verification.md`](../verification.md) |