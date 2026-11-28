---
id: TASK-11
title: Fix bench_size_comparison allocation inside timed region
status: Done
assignee: []
created_date: '2026-03-31 08:00'
labels:
  - testing
dependencies: []
references:
  - 'https://github.com/igor53627/plinko-rs/issues/116'
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`bench_size_comparison` in `memory_bench.rs` calls `generate_subsets` inside the benchmark loop, meaning it measures allocation time for generating subsets rather than just the conversion to `BlockBitset`. The `bench_vec_allocation` group already pre-generates subsets outside the timed region.

Either label `bench_size_comparison` clearly as a "combined allocation" benchmark or restructure it like `bench_vec_allocation` to separate setup from measurement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 bench_size_comparison separates setup from measurement, or is clearly labelled
- [x] #2 Benchmark still compiles and runs
<!-- AC:END -->
