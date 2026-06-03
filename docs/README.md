# Documentation map

Entry point for everything under `docs/`. The repository root [`README.md`](../README.md) is the quickstart; this page lists all maintained docs and where archived material lives.

## Canonical (use for implementation)

| Doc | Topic |
|-----|--------|
| [`hint_generation.md`](hint_generation.md) | HintInit (Fig. 7), `plinko/src/bin/hint_gen/` layout |
| [`data_format.md`](data_format.md) | v3 `database.bin`, mappings, snapshot metadata (HTTP host retired) |
| [`protocol_overview.md`](protocol_overview.md) | End-to-end pipeline (extract → hints → query/update) |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Components and data flow |
| [`constant_time_mode.md`](constant_time_mode.md) | TEE / constant-time HintInit |
| [`update_strategy.md`](update_strategy.md) | Incremental updates; see also [`../plinko/docs/delta-format.md`](../plinko/docs/delta-format.md) |
| [`verification.md`](verification.md) | Rocq, Kani, proptest entry points |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Build and run commands |
| [`FEATURE_FLAGS.md`](FEATURE_FLAGS.md) | `cuda`, `parallel` build features |

## Benchmarks

| Doc | Topic |
|-----|--------|
| [`benchmarks.md`](benchmarks.md) | Index (CPU TEE + GPU) |
| [`../tee-test/SEV-SNP-BENCHMARK.md`](../tee-test/SEV-SNP-BENCHMARK.md) | SEV-SNP HintInit numbers |
| [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) | **Canonical GPU** optimization summary |
| [`gpu_benchmark_commands.md`](gpu_benchmark_commands.md) | Modal / multi-GPU command templates |

## Paper and formal reference

| Resource | Topic |
|----------|--------|
| [`2024-318.pdf`](2024-318.pdf) | Plinko paper |
| [`Plinko.v`](Plinko.v) | Coq spec |
| [`plinko_paper_index.json`](plinko_paper_index.json) | Catalog of parsed paper parts 1–6 |
| [`plinko_paper_part6_algorithms.json`](plinko_paper_part6_algorithms.json) | Fig. 7 / algorithm pseudocode |
| [`plinko_paper_part2_technical.json`](plinko_paper_part2_technical.json), [`plinko_paper_part3_scheme.json`](plinko_paper_part3_scheme.json) | iPRF / scheme |
| [`../plinko/formal/`](../plinko/formal/) | Rocq proofs |
| [`codex-cloud-verification.md`](codex-cloud-verification.md) | Remote Rocq verify (agents) |

## Other assets

| Path | Topic |
|------|--------|
| [`protocol-visualization.html`](protocol-visualization.html) | Interactive protocol demo |
| [`alex_hoover_plinko_talk_transcript.json`](alex_hoover_plinko_talk_transcript.json) | Talk transcript (reference) |

## Archived

Obsolete or historical material: [`archived/README.md`](archived/README.md).

- GPU experiment notes: [`archived/CUDA_OPTIMIZATION.md`](archived/CUDA_OPTIMIZATION.md), [`archived/optimization/`](archived/optimization/)
- **Modal run logs**: [`archived/gpu_runs/`](archived/gpu_runs/)
- Old kanban, compression study, tutorial draft, etc.