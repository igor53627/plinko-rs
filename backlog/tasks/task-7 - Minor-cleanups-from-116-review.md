---
id: TASK-7
title: "Minor cleanups from #116 review"
status: Done
assignee: []
created_date: '2026-03-31 06:00'
labels:
  - tech-debt
dependencies:
  - TASK-1
  - TASK-2
references:
  - 'https://github.com/igor53627/plinko-rs/issues/116'
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Minor issues identified during code review of the compact block set PR:

1. **`memory_bench.rs`**: `bench_size_comparison` has `println!` outside `b.iter()` that may emit duplicate output across Criterion warmup runs. Move print block entirely outside the benchmark function.

2. **`fast_path.rs`**: Missing `debug_assert_eq!(input.num_backup, input.backup_hint_blocks.len())` to document the invariant that these must match.

3. **`protocol.rs:promote_backup`**: No comment explaining why any available backup can serve any queried block (the complement-adaptation invariant — backup has c/2 blocks, so whether the queried block is in or out, the complement trick always produces a valid c/2 subset).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 bench_size_comparison prints results without duplication
- [x] #2 fast_path.rs has debug assertion for num_backup == backup_hint_blocks.len()
- [x] #3 promote_backup has a comment explaining the complement-adaptation invariant
<!-- AC:END -->
