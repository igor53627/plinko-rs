---
id: TASK-21
title: Make the GPU path compute Plinko hints or mark it as a benchmark
status: Done
assignee: []
created_date: '2026-11-17T11:36:00Z'
updated_date: '2026-11-28T13:29:30Z'
labels:
  - gpu
  - hint-gen
dependencies:
  - TASK-19
references:
  - plinko/src/bin/bench_gpu_hints.rs
  - plinko/cuda/hint_kernel.cu
  - plinko/src/gpu.rs
  - plinko/src/db.rs
  - docs/gpu_benchmark_commands.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
bench_gpu_hints uses public block keys (key[0] = i), two fixed subsets ((hint + block) % 2), a plain Swap-or-Not with ChaCha12 on domain w, no PMNS, no backup hints and no inverse. Its output is not a hint set. hint_kernel.cu line 679 overflows u32 at mainnet, and gpu.rs asserts size 32 for a 48-byte struct. db.rs, cost_estimate and the GPU use w = 131072 where plinko_hints and the client use w = 65536. docs/gpu_benchmark_commands.md calls the run a production run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The GPU path writes a hint file that is byte-identical to the plinko_hints file on a test database, or the binary and the documents state that it is a throughput benchmark and not Plinko
- [x] #2 db.rs and cost_estimate derive w and c with plinko::protocol::default_block_size
- [x] #3 The u32 overflow in hint_kernel.cu and the size assertion in gpu.rs are fixed
- [x] #4 The gpu.rs doctest passes under --ignored
- [x] #5 docs/gpu_benchmark_commands.md does not name a non-Plinko run a production run
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-19: cost_estimate now derives w, c and the hint counts with ProtocolParams::new (the half of criterion 2 for cost_estimate). Its GPU section and VRAM lines keep db::derive_plinko_params, because bench_gpu_hints uses it; they are labelled as a throughput benchmark. Open for this task: db.rs, and then the removal of the second geometry from cost_estimate. Also open: the CPU section of cost_estimate uses 55.8 MB/s on 64 vCPUs and a TEE factor of 2.6 from the generator of December 2025 (20.9 min for mainnet); the current code needs about 300 core-hours (docs/hint_generation.md).

2026-11-19, later: the CPU section of cost_estimate is fixed (0.59 ms per record in the standard path, 16.7 ms in the constant-time path; 301 and 8,509 core-hours for mainnet). The item 'Also open' of the note above is closed.

2026-11-28: Marked gpu.rs, bench_gpu_hints.rs, hint_kernel.cu, and docs/gpu_benchmark_commands.md consistently as a throughput benchmark (not Plinko hints), removing remaining production-run wording. Unified db::derive_plinko_params and cost_estimate on plinko::protocol::default_block_size (even c >= 2, matching ProtocolParams::new) and updated plinko/formal/docs/TRUST_BASE.md via trust_base_notes.toml. Fixed 64-bit byte_idx/idx indexing in hint_kernel.cu, fixed PlinkoParams (40 B) and HintOutput (48 B) layout assertions in gpu.rs, and made the gpu.rs doctest runnable under --features gpu (plus a no_run GpuHintGenerator doctest under --features cuda).
<!-- SECTION:NOTES:END -->
