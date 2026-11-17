---
id: TASK-21
title: Make the GPU path compute Plinko hints or mark it as a benchmark
status: To Do
assignee: []
created_date: '2026-11-17T11:36:00Z'
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
- [ ] #1 The GPU path writes a hint file that is byte-identical to the plinko_hints file on a test database, or the binary and the documents state that it is a throughput benchmark and not Plinko
- [ ] #2 db.rs and cost_estimate derive w and c with plinko::protocol::default_block_size
- [ ] #3 The u32 overflow in hint_kernel.cu and the size assertion in gpu.rs are fixed
- [ ] #4 The gpu.rs doctest passes under --ignored
- [ ] #5 docs/gpu_benchmark_commands.md does not name a non-Plinko run a production run
<!-- AC:END -->
