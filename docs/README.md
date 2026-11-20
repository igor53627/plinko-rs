# Documentation map

Entry point for everything under `docs/`. The repository root [`README.md`](../README.md) is the quickstart; this page lists all maintained docs and where archived material lives.

| Link | URL |
|------|-----|
| DeepWiki (agent index) | [deepwiki.com/igor53627/plinko-rs](https://deepwiki.com/igor53627/plinko-rs) |
| Protocol demo (GitHub Pages) | [igor53627.github.io/plinko-rs](https://igor53627.github.io/plinko-rs/): the page [`protocol-visualization.html`](protocol-visualization.html) |

## Canonical (use for implementation)

| Doc | Topic |
|-----|--------|
| [`hint_generation.md`](hint_generation.md) | HintInit (Fig. 7), `plinko/src/bin/hint_gen/` layout |
| [`data_format.md`](data_format.md) | v3 `database.bin`, mappings, mainnet reference counts |
| [`protocol_overview.md`](protocol_overview.md) | Scope, measured costs, phases (HintInit → query → update) |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Components and data flow |
| [`trust_model.md`](trust_model.md) | Who runs HintInit, who holds the seed, what each party learns |
| [`integrity.md`](integrity.md) | What the client can check and what it must trust; the library gives no integrity of an answer |
| [`online_protocol.md`](online_protocol.md) | Query, decoy, query format, retry rule, query limit; deviations from the paper |
| [`keyword_lookup.md`](keyword_lookup.md) | Proposal: lookup by key through a cuckoo table (not implemented) |
| [`constant_time_mode.md`](constant_time_mode.md) | TEE / constant-time HintInit |
| [`update_strategy.md`](update_strategy.md) | Database versions, deltas, restart, cost, rule for a database that grows, what is not implemented |
| [`verification.md`](verification.md) | Rocq, Kani, proptest entry points |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Build and run commands |
| [`FEATURE_FLAGS.md`](FEATURE_FLAGS.md) | `cuda`, `parallel` build features |

## Benchmarks

| Doc | Topic |
|-----|--------|
| [`benchmarks.md`](benchmarks.md) | Index: numbers of the current code, and the historical documents |
| [`../tee-test/SEV-SNP-BENCHMARK.md`](../tee-test/SEV-SNP-BENCHMARK.md) | Historical (Dec 2025): SEV-SNP overhead with an earlier generator |
| [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) | Historical (Jan 2026): GPU throughput kernel, not Plinko hints |
| [`gpu_benchmark_commands.md`](gpu_benchmark_commands.md) | Historical (Jan 2026): Modal / multi-GPU command templates |

## Paper and formal reference

| Resource | Topic |
|----------|--------|
| [`2024-318.pdf`](2024-318.pdf) | Plinko paper |
| [`Plinko.v`](Plinko.v) | Coq sketch (not a specification of the Rust code) |
| [`plinko_paper_index.json`](plinko_paper_index.json) | Catalog of parsed paper parts 1–6 |
| [`plinko_paper_part6_algorithms.json`](plinko_paper_part6_algorithms.json) | Fig. 7 / algorithm pseudocode |
| [`plinko_paper_part2_technical.json`](plinko_paper_part2_technical.json), [`plinko_paper_part3_scheme.json`](plinko_paper_part3_scheme.json) | iPRF / scheme |
| [`../plinko/formal/`](../plinko/formal/) | Rocq models and proofs about the models |
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